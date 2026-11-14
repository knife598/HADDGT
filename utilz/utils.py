# This file will generate dataset for Scene centric models
from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import yaml
import re
from shapely.geometry import Point, Polygon

# 特征 / 通道布局 / 解码 的唯一出处（本文件只保留"调用"，实现都在 utilz/ 下）
from utilz.features import build_features
from utilz.layout import validate as validate_layout
from utilz.decode import find_topk_selected_words as _find_topk_impl

class Scenes(Dataset): # 数据集容器
    def __init__(self, config): 
        self.Nnodes = config['Nnodes']
        self.NFeatures = config['NFeatures'] # initial number of features which the Vx, Vy and heading are added
        self.input_size = config['input_size'] # fullscene shape
        self.NZones = config['NZones']
        self.sl = config['sl']
        self.shift = config['sw']
        self.sw = config['sw']
        self.sn = config['sn']
        self.future = config['future']
        self.device = config['device']
        self.framelen = 2*self.sl # number of frames used for training
        self.prohibitedZones = [200, 100] # These are the zones that are not allowed to be in the scene, such as parking lots and sidewalks, etc. This zones will be ignored
        self.xyidx = config['xy_indx']
        self.Centre = config['Centre']
        self.dwn_smple = config['dwn_smple']
        self.config = config  # 特征工程 / 布局自检都要读配置（use_xxx、NZones、Centre…）
        self.eyemat = torch.eye(self.Nnodes, self.Nnodes, device=self.device)
        self.epsilon = -1e-16
        self.ID = [] # We keep the track of the object IDs
        self.Fr = [] # Same as ID but for the frame number
        self.Zones = []
        self.NUsers = [] # Number of users in each scene, this helps decreasing the preprocessings

        self.Scene = torch.empty(0, self.sl//self.dwn_smple, self.Nnodes, self.input_size, dtype=torch.long, device=self.device) #[Sampeles, Sequence Length, Nnodes, NFeatures], Scene is the input to the model
        self.Adj_Mat_Scene = torch.empty(0, self.sl//self.dwn_smple, self.Nnodes, self.Nnodes, device=self.device) #[Sampeles, Sequence Length, Nnodes, Nnodes], Adjacency matrix for the Scene
        self.Target = torch.empty(0, self.future//self.dwn_smple, self.Nnodes, self.input_size, device=self.device, dtype= torch.long) #[Sampeles, Sequence Length, Nnodes, NFeatures], Target is the output of the model
        # 创建空张量容器Scene、Adj_Mat_Scene和Target，分别用于存储场景数据、场景的邻接矩阵和目标数据。它们的形状为[Sampeles, Sequence Length, Nnodes, NFeatures]，并且使用long类型和指定的设备(device)进行初始化。
        # torch.empty(0, ...)：创建一个在第一个维度为 0 的空张量，后续用 torch.cat 追加时，初始空张量不会造成问题

    def addnew(self, Scene, Target, Zones, ID, Fr, NObjs):
        self.Scene = torch.cat((self.Scene, Scene.unsqueeze(0)), dim=0)
        self.Adj_Mat_Scene = torch.cat((self.Adj_Mat_Scene, self.Adjacency(Scene, Zones[:self.sl:self.dwn_smple], NObjs).unsqueeze(0)), dim=0)
        self.Zones.append(Zones) 
        self.ID.append(ID)
        self.Fr.append(Fr)
        self.NUsers.append(NObjs)
        self.Target = torch.cat((self.Target, Target.unsqueeze(0)), dim=0)
        
    def Slide_(self, fullscene, Zones, IDs, Fr, NObjs):
        # 滑窗采样 + 特征衍生
        fullscene= self.ExtractFeatures(fullscene) # Extract more features from the fullscene
        # 通道数自检：账本（features.REGISTRY → layout.resolve）与实际产出必须一致
        validate_layout(self.config, fullscene.size(-1))
        for j in range(0, self.sn*self.sw, self.sw):
            final_indx = j + self.sl + self.future
            start_indx = j + self.sl
            scene = fullscene[j : j+ self.sl: self.dwn_smple]
            no_show_frames = (scene[:,:,self.xyidx[0]] == 0).sum(dim=0) # The number of zeros in the x direction, which means they are out of camera sight
            # scene[:,:,0] 取BBx列，统计每辆车在15帧中"无车"的帧数
            no_show_frames = (no_show_frames < self.sl//2//self.dwn_smple).view(1, self.Nnodes, 1) # At least the vehicle must be half of the sequence, unless we ignore the scene
            # 如果某辆车超过一半帧数在画面外（BBx==0），标记为不可靠
            target = fullscene[start_indx : final_indx:self.dwn_smple] * no_show_frames
            scene = scene* no_show_frames
            # 不可靠的车辆 → 全0
            self.addnew(scene, target, 
                        Zones[j : final_indx: self.dwn_smple], IDs[j:final_indx: self.dwn_smple],
                        Fr[j:final_indx: self.dwn_smple], NObjs)
            # 调用addnew，把滑窗得到的 (scene, target, zones, ids, fr) 组成一条样本，追加到数据集中
        

    def augmentation(self, num): # 数据增强（打乱顺序） it works with permuting the order of the users in the scene  "num" times
        len = self.Scene.size(0)
        extenstion = torch.arange(self.NUsers, self.Nnodes, device=self.device)
        for i in range(num):
            # Now we need to permute the rows of the Scene and the Target based on the order
            for n in range(len):
                order0 = torch.randperm(self.NUsers,device=self.device)
                # 随机排列实际车辆的序号
                order = torch.cat((order0, extenstion), dim=0)
                # 后面接上空位
                newscene = self.Scene[n] # 按新顺序重排
                zones = [z[order0] for z in self.Zones[n]]
                newtarget = self.Target[n]
                newscene = newscene[:,order]
                newtarget = newtarget[:,order]
                self.addnew(newscene, newtarget, zones, [], self.Fr[n], self.NUsers[n])
        
    def addnoise(self, multiply, mx, ratio): # 数据增强（加噪声） this adds noise "multiply" times to the scene and the target, it works by mx amplitude and probability of ratio
        ss = self.Scene.size()
        ts = self.Target.size()
        init_scene = self.Scene.clone()
        init_Adj_Mat_Scene = self.Adj_Mat_Scene.clone()
        init_Target = self.Target.clone()
        mask = torch.zeros_like(init_scene)
        mask[:,:,:, self.xyidx] = 1
        mask = mask * (init_scene!= 0)

        trmask = torch.zeros_like(init_Target)
        trmask[:,:,:, self.xyidx] = 1
        trmask = trmask * (init_Target!= 0)

        for i in range(multiply):
            noise = torch.randint(0, mx, ss, device=self.device)
            rand = torch.rand(ss, device=self.device) > ratio
            noisy_scene = init_scene + noise * rand * mask
            noise = torch.randint(0, mx, ts, device=self.device)
            rand = torch.rand(ts, device=self.device) > ratio
            noisy_tar = init_Target + noise * rand * trmask
            self.Scene = torch.cat((self.Scene, noisy_scene), dim=0)
            self.Adj_Mat_Scene = torch.cat((self.Adj_Mat_Scene, init_Adj_Mat_Scene), dim=0)
            self.Target = torch.cat((self.Target, init_Target), dim=0) # For now let's not add noise to the target; noisy_tar



    def ExtractFeatures(self, fullscene): # 特征工程（通道由 utilz/features.py 决定）
        """返回 (T, N, 7 + Σ启用组通道数)。

        默认 = 7 原始 + 8 base + 7 kinematics = 22；config 打开 use_road_state 再 +8 = 30。
        新增/删除特征请改 utilz/features.py 的 REGISTRY（含通道数），这里不用动。
        """
        return build_features(fullscene, self.config)


    def Adjacency(self, Scene, Zone, NObjs): # 计算车辆之间的空间关系矩阵 Create the adjacency matrix for the Scene agents, the values are the normalized inverse of the distance between the agents
        # 每两辆车之间的距离用一个 0~1 之间的值表示，值越大表示关系越紧密
        Adj_mat = torch.zeros(Scene.size(0), self.Nnodes, self.Nnodes, device=self.device) + self.epsilon
        # 初始化邻接矩阵

        Kmat0 = ((Zone[:, :NObjs]!=200) & (Zone[:, :NObjs]!=100)).float()
        # 找出哪些车辆不在禁区内
        Kmat = torch.mul(Kmat0.unsqueeze(2), Kmat0.unsqueeze(1)) 

        diff = Scene[:, :NObjs, self.xyidx].unsqueeze(2) - Scene[:, :NObjs, self.xyidx].unsqueeze(1)
        # 计算每两辆车之间的坐标差向量

        # Invsqr_dist = torch.sum(diff**2, dim=3).pow(-1).tanh().pow(0.5)
        Invsqr_dist = torch.sum((diff/1024)**2, dim=3).sqrt() # 计算归一化距离
        Invsqr_dist = torch.exp(-Invsqr_dist)
        Adj_mat[:, :NObjs, :NObjs] = torch.mul(Invsqr_dist, Kmat)
        # torch.mul：逐元素相乘——>都在正常区域则正常计算邻接值，至少一个在禁区（Kmat=0）则屏蔽
        return Adj_mat


    def load_class(self, path, cmnt): # 硬盘存取 It usually takes time preparing the dataset, so it's better to save it and load it later
        self.Scene = torch.load(os.path.join(path, cmnt, 'Scene.pt'), weights_only=True).to(self.device)
        self.Zones = torch.load(os.path.join(path, cmnt, 'Zones.pt'), weights_only=True)
        self.ID = torch.load(os.path.join(path, cmnt, 'ID.pt'), weights_only=True)
        self.Fr = torch.load(os.path.join(path, cmnt, 'Fr.pt'), weights_only=True)
        self.Adj_Mat_Scene = torch.load(os.path.join(path, cmnt, 'Adj_Mat_Scene.pt'), weights_only=True).to(self.device)
        self.NUsers = torch.load(os.path.join(path, cmnt, 'NUsers.pt'), weights_only=True)
        self.Target = torch.load(os.path.join(path, cmnt, 'Target.pt'), weights_only=True).to(self.device) # 修复：与 Scene/Adj_Mat_Scene 一致，加载后搬到目标设备

    def save_class(self, path, cmnt): # 硬盘存取 
        if not os.path.exists(os.path.join(path, cmnt)): # Create the directory if it doesn't exist
            os.makedirs(os.path.join(path, cmnt))
        else:
            #remove the existing files
            for file in os.listdir(os.path.join(path, cmnt)):
                os.remove(os.path.join(path, cmnt, file))
        torch.save(self.Scene, os.path.join(path, cmnt, 'Scene.pt'))
        torch.save(self.Zones, os.path.join(path, cmnt, 'Zones.pt'))
        torch.save(self.ID, os.path.join(path, cmnt, 'ID.pt'))
        torch.save(self.Fr, os.path.join(path, cmnt, 'Fr.pt'))
        torch.save(self.Adj_Mat_Scene, os.path.join(path, cmnt, 'Adj_Mat_Scene.pt'))
        torch.save(self.NUsers, os.path.join(path, cmnt, 'NUsers.pt'))
        torch.save(self.Target, os.path.join(path, cmnt,  'Target.pt'))

    def __len__(self):
        return self.Scene.size(0)
    
    def __getitem__(self, idx):
        return self.Scene[idx], self.Target[idx], self.Adj_Mat_Scene[idx] #, self.Adj_Mat_Target[idx] , self.ID[idx], self.Fr[idx], self.Zones[idx]
    
    

def Scene_Process(Scenetr, Scenetst, Sceneval, Traffic_data, config):
# 把 CSV 数据切分成训练/测试/验证三个数据集
# Scenetr：训练集容器，Scenetst：测试集容器，Sceneval：验证集容器
    Nnodes = config['Nnodes']
    NFeatures = config['NFeatures']
    sl = config['sl']
    future = config['future']
    sw = config['sw']
    sn = config['sn']
    Columns_to_keep = config['Columns_to_keep']
    Seed = not config['generate_data']
    ct = config['ct']
    device = config['device']
    Nusers = config['Nusers']
    
    scenelet_len = sl + future + sn*sw # Maximum number of frames in a scenelet
    scenelets = int((max(Traffic_data['Frame']) - min(Traffic_data['Frame']) + 1) // scenelet_len)
    train_size = int(0.9 * scenelets)
    test_size = int(0.1 * scenelets)
    val_size = scenelets - train_size - test_size
    savelog(f"Train size: {train_size}, Test size: {test_size}, Validation size: {val_size} ", ct)
    if Seed: # generate_data = false → 加载之前的索引（可复现）
        savelog("Using seed for random split", ct)
        indices = torch.load(os.path.join(os.getcwd(),'Pickled',f'indices.pt'))
    else: # generate_data = true → 随机生成新索引
        savelog("Randomly splitting the data", ct)
        indices = torch.randperm(scenelets) # To avoid data leakage, seed selects between the scenelet_len frames, so sliding window does not affect it
        

    Traffic_data = torch.tensor(Traffic_data.values, dtype=torch.float, device=device)
    DFindx, DFindxtmp = 0, 0 # We sweep through the first frame to the last frame
    IDs = []
    Zones = []
    Fr = []
    Scene = []
    for fr in range(int(min(Traffic_data[:,0])), int(max(Traffic_data[:,0]))): # 遍历每一帧
        DFindxtmp = DFindx # 记录当前帧的起始位置
        rows = torch.empty(0, NFeatures, device=device) # Nnodes and NFeatures are predefined fixed values
        while Traffic_data[DFindx,0] == fr:
            DFindx += 1 # 帧号在 CSV 的第 0 列-跳过所有属于当前帧的行

        rows = Traffic_data[DFindxtmp:DFindx, Columns_to_keep] # We only keep the columns that we need
        # 提取当前帧所有车辆的 7 个保留列
        Fr.append(fr)  # 记录帧号 We keep the track of the frame number
        IDs.append(Traffic_data[DFindxtmp:DFindx, 1]) # 记录车辆 ID
        Zones.append(Traffic_data[DFindxtmp:DFindx, 11]) # 记录区域
        while rows.size()[0] < Nusers:  # # 如果当前帧车辆数 < 32-This is to make sure that rows is [Nnodes, NFeatures]
            rows = torch.cat((rows, torch.zeros(1, NFeatures, device=device)), dim=0)
            # 用全 0 行填充，直到凑满 32 行

        Scene.append(rows) # 把当前帧存入缓冲区
        if len(Scene) % scenelet_len == 0: # # 累计满帧 → 开始处理-This only happens after each scenelet_len frames
            trORtst = checktstvstr(fr, indices, scenelet_len, train_size, test_size) # Find if the Scene belongs to train, test, or validation
            # 根据帧号判断当前场景属于训练集、测试集还是验证集

            Scene = torch.stack(Scene, dim=0).to(device=device)
            # [100 个 [32, 24] 张量] → [100, 32, 24]

            Scene, Zones, IDs, NObjs = ConsistensyCheck(Scene, Zones, IDs, Nnodes, NFeatures) # Now we sort the agents to be on the same row in the whole Scene
            # 联动下方函数ConsistensyCheck，对齐车辆 ID → 同一车在不同帧排在同一行
            if trORtst == 1: # test
                Scenetst.Slide_(Scene, Zones, IDs, Fr, NObjs)
            if trORtst == 0: # train
                Scenetr.Slide_(Scene, Zones, IDs, Fr, NObjs)
            elif trORtst == 2: # validation
                Sceneval.Slide_(Scene, Zones, IDs, Fr, NObjs)

            Scene = []
            Fr = []
            IDs = []
            Zones = []
            # 清空缓冲区，准备下一组 100 帧

    torch.save(indices, os.path.join(os.getcwd(),'Pickled',f'indices.pt'))
    return Scenetr, Scenetst, Sceneval

def ConsistensyCheck(Scene, Zones, IDs, Nnodes, NFeatures): # 车辆 ID 对齐-Makes sure that the rows related to the same ID are in the same order in the  whole lists
    globlist = torch.cat(IDs, dim=0).unique() # Base list to compare the rest of the lists
    # cat → 合并所有帧的ID → unique → 去重 → [车A, 车B, 车C]
    len_globlist = max(len(globlist), Nnodes)
    device = Scene.device
    SortedScene = []
    SortedZones = []
    SortedIDs = []

    for k, ids in enumerate(IDs): # 遍历每一帧-IDs are spread over the frames, we need to sort them frame by frame
        Sc = torch.zeros(len_globlist, NFeatures, device=device) # tmp Scene which will be sorted
        Zonelist = 200*torch.ones(len_globlist, device=device) # 默认 Zone=200（禁区）-Zone 200 is the default zone that meant to be ignored
        _, indices = torch.where(ids.unsqueeze(1)==globlist) 
        Sc[indices] = Scene[k, :len(indices)] # 把原始帧中每辆车的特征放到正确的行-Swap the rows based on the indices
        Zonelist[indices] = Zones[k] # Zone 也同步重排-Swap the zones based on the indices
        SortedZones.append(Zonelist[:Nnodes]) # Sort the Zones based on the indxlist
        SortedScene.append(Sc[:Nnodes]) # 丢弃超出的车辆-We dump the rest of the agents that are over Nnodes
    SortedIDs = globlist.repeat(k, 1) # Sort the IDs based on the indxlist
    SortedScene = torch.stack(SortedScene, dim= 0).to(device=device)
    SortedZones = torch.stack(SortedZones, dim=0).to(device=device)
    return SortedScene, SortedZones, SortedIDs, len(globlist)



def checktstvstr(fr, indices, scenelet_len, train_size, test_size):
    for indx , i in enumerate(indices): # 遍历随机排列的索引
        if i == fr//scenelet_len-1: # 找到当前帧属于第几个 scenelet
            break
    if indx <= train_size: # 前 90% → 训练
        return 0 # train
    elif indx > train_size and indx <= train_size + test_size: # 接着 5% → 测试
        return 1 # test
    else: # 最后 5% → 验证
        return 2 # validation

def DataLoader_Scene(datasettr, datasettst, datasetval, batch_size):
    train_loader = DataLoader(datasettr, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasettst, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(datasetval, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader
# 三个 Scenes 实例（训练/测试/验证）包装成 PyTorch 的 DataLoader，让模型可以批量读取数据
# batch_size：从 config 读取，默认 128	每次取出 128 个样本
# shuffle False：不打乱数据


def loadcsv(frmpath, Header, trjpath = None):
    df = pd.read_csv(frmpath, dtype =float) # 用 pandas 读 CSV
    df.columns = Header
    if trjpath is not None:
        trj = pd.read_csv(trjpath, dtype =float)
        return df, trj
    return df
# dtype=float：把所有列读成浮点数。这是为了后续能直接转成 PyTorch 张量
# df.columns = Header：CSV 文件本身没有列名，这里手动指定。


def savelog(log, ct): # appends the log to the existing log file while keeping the old logs
    # if the log file does not exist, create one
    print(log) # 先打印到控制台
    if not os.path.exists(os.path.join(os.getcwd(),'logs')): # 如果 logs/ 目录不存在，就创建它
        os.mkdir(os.path.join(os.getcwd(),'logs'))
    with open(os.path.join(os.getcwd(),'logs', f'log-{ct}.txt'), 'a') as file:
        file.write('\n' + log) # 追加写入到 logs/log-{时间戳}.txt
        file.close()


def Zoneconf(path = '/utilz/ZoneConf.yaml'): # 解析区域配置
    ZoneConf = []
    with open(path) as file:
        ZonesYML = yaml.load(file, Loader=yaml.FullLoader)
        #convert the string values to float-解析 YAML 文件，得到字典
        for _, v in ZonesYML.items():
            lst = []
            for _, p  in v.items():    
                for x in p[0]:
                    b = re.split(r'[,()]',p[0][x])
                    lst.append((float(b[1]), float(b[2])))
            ZoneConf.append(lst)
    return ZoneConf
# 把configs/ZoneConf.yaml 中的区域多边形顶点坐标，从字符串解析为 Python 浮点数元组


def zonefinder(BB, Zones):
    B, Nnodes,_ = BB.size()
    BB = BB.reshape(-1,2).cpu()
    PredZone = torch.zeros(B*Nnodes, device=BB.device)
    for n , bb in enumerate(BB.int()): # 遍历每个坐标点
        for i, zone in enumerate(Zones): # 遍历每个区域
            Poly = Polygon(zone)
            if Poly.contains(Point(bb[0], bb[1])): # 判断点是否在多边形内
                PredZone[n] = i+1 # 记录区域编号（从1开始）
                break
        
    
    return PredZone.reshape(B, Nnodes)


def Find_topk_selected_words (Pred_target, Target, mode=None, k=5, ignore_classes=None): # 评估函数-连接模型输出和评估指标的关键函数
# Pred_target：模型输出的 logits（未 softmax）
# Target：真实坐标值
# 解码方式与指标口径全部在 utilz/decode.py；mode 不填 = 'topk5'，与改造前逐位一致
    return _find_topk_impl(Pred_target, Target, mode=mode or 'topk5', k=k,
                           ignore_classes=ignore_classes)


def count_parameters(model): # 统计参数数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# main.py 中，被调用于训练前打印模型参数量


def load_model(model, weight_path, ct): # 加载模型权重-从 .pth 文件加载模型权重
        savelog("Loading model from the checkpoint", ct)
        savelog('=> Loading checkpoint', ct)
        checkpoint = torch.load(weight_path) # 加载权重文件
        model.load_state_dict(checkpoint['state_dict'])
        return model

def Scene_Process_plotting(Scenetst, Traffic_data, config): # 可视化版数据预处理
# 与 Scene_Process 功能类似，但只处理测试集，且不使用滑窗重叠——每个 scenelet 只取一个样本
    Nnodes = config['Nnodes']
    NFeatures = config['NFeatures']
    sl = config['sl']
    future = config['future']
    Columns_to_keep = config['Columns_to_keep']

    device = config['device']
    Nusers = config['Nusers']
    dwn_smple = config['dwn_smple']
    scenelet_len = sl + future # Maximum number of frames in a scenelet
    # scenelet_len = 10*60 # 60 seconds * 10 frames per second
    Traffic_data = torch.tensor(Traffic_data.values, dtype=torch.float, device=device)
    DFindx, DFindxtmp = 0, 0 # We sweep through the first frame to the last frame
    IDs = []
    Zones = []
    Fr = []
    Scene = []
    for fr in range(int(min(Traffic_data[:,0])), int(max(Traffic_data[:,0]))):
        DFindxtmp = DFindx
        rows = torch.empty(0, NFeatures, device=device) # Nnodes and NFeatures are predefined fixed values
        while Traffic_data[DFindx,0] == fr:
            DFindx += 1

        rows = Traffic_data[DFindxtmp:DFindx, Columns_to_keep] # We only keep the columns that we need
        Fr.append(fr)  # We keep the track of the frame number
        IDs.append(Traffic_data[DFindxtmp:DFindx, 1])
        Zones.append(Traffic_data[DFindxtmp:DFindx, 11])
        while rows.size()[0] < Nusers:  # This is to make sure that rows is [Nnodes, NFeatures]
            rows = torch.cat((rows, torch.zeros(1, NFeatures, device=device)), dim=0)

        Scene.append(rows)
        if len(Scene) % scenelet_len == 0: # This only happens after each scenelet_len frames
            Scenex = torch.stack(Scene, dim=0).to(device=device)
            Scenex, Zonesx, IDsx, NObjs = ConsistensyCheck(Scenex, Zones, IDs, Nnodes, NFeatures)
            Scenex= Scenetst.ExtractFeatures(Scenex) # Now we sort the agents to be on the same row in the whole Scene
            Scenetst.addnew(Scenex[:sl:dwn_smple], Scenex[sl::dwn_smple], Zonesx, IDsx, Fr, NObjs) # We add the scenelet to the dataset
            Scene = Scene[sl:] # We pass the second half frames
            Fr = Fr[sl:] # We pass the second half frames
            IDs = IDs[sl:]
            Zones = Zones[sl:]
    return Scenetst



if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")