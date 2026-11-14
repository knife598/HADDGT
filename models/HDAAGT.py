
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *

# 第一层 ==================================
# 说明：早前尝试过"运动历史编码器"（把最近 K 帧相对位移编码后与位置嵌入相加），
# 因位置表示随后进入 DAAG 的 exp(score)（modules.py 里 score 近似随输入范数平方增长）
# 而两次在训练中途梯度爆炸成 NaN，已整体删除，改用输入特征通道（utilz/features.py）。
# 第二层
# 处理车辆的位置坐标（BBx、BBy）
# 用 DAAG 建模车辆间空间交互，用时序卷积捕捉时间运动模式
class Positional_Encoding_Layer(nn.Module):
    def __init__(self, hidden_size, num_att_heads, xy_indx, pos_embedding_dim,pos_embedding_dict_size, nnodes, sl, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.xy_indx = xy_indx

        # Embeddings of the positional features 坐标 Embedding 列表
        self.Postional_embeddings = nn.ModuleList() # nn.ModuleList：专门管理多个子模块的列表
        for i in range(len(xy_indx)):
            self.Postional_embeddings.append(nn.Embedding(pos_embedding_dict_size[i], pos_embedding_dim[i], padding_idx=0))
        # nn.Embedding：将离散整数映射为稠密向量，两个embedding层，对应BBX与BBy

        # Temporal Convolution and Positional Attention layers 时间卷积层和位置注意力层
        self.TemporalConv = TemporalConv(2*hidden_size, sl)
        # 时序卷积层-TemporalConv 定义在 modules.py。输入 [B*32, 15, 256]，三个并行的 Conv1d 在不同时间分辨率下提取运动模式，三路相加后输出 [B*32, 15, 256]
        self.Position_Att = nn.MultiheadAttention(embed_dim= 3*hidden_size, num_heads=num_att_heads, batch_first=True)
        # 时间维度自注意力：每个时间步看到所有时间步
        self.Pos_FF = FeedForwardNetwork(d_model= 3*hidden_size, out_dim=3*hidden_size)
        # FFN层，调用自modules.py，对每个时间步独立做非线性变换，捕捉特征之间的复杂关系
        self.Position_Rezero = nn.Parameter(torch.zeros(3*hidden_size))
        # 控制自注意力残差的权重。初始为 0 → 注意力输出完全被忽略 → 只有原始信号通过。随着训练逐渐增大 → 注意力逐渐参与

        # The recent results showed that placing DAAG layer here is more efficient than putting it after the Encoder-实验证明，DAAG 放在 Embedding 之后、Encoder 之前，比放在 Encoder 之后效果更好
        # DAAG 图注意力层-定义在 modules.py，是模型的核心创新
        self.DAAG = DAAG_Layer(in_features=2*hidden_size, out_features=2*hidden_size, n_heads= num_att_heads, n_nodes = nnodes, concat=True)
        # 对于每个时间步的 32 辆车，DAAG 计算它们之间的注意力分数，考虑邻接矩阵中的距离信息，生成融合了周围车辆信息的节点表示
        
        # 位置嵌入本身恒等于 2*hidden_size（连续特征走 Traffic 分支，不占这里）
        assert sum(pos_embedding_dim) == 2 * hidden_size, (
            '位置嵌入维度必须等于 2*hidden_size=%d，当前 sum(pos_embedding_dim)=%d'
            % (2 * hidden_size, sum(pos_embedding_dim)))


        self.dropout = nn.Dropout(dropout)

    def forward(self, Scene, Scene_mask, adj):
        B, SL, Nnodes, _ = Scene.size()

        # Embedding the positional features
        positional_embedding = []
        for n, indx in enumerate(self.xy_indx):
            positional_embedding.append(self.Postional_embeddings[n](Scene[:,:,:,indx].long()))
        Pos_embd = torch.cat(positional_embedding, dim=-1)



        # DAAG layer
        Pos_embd = self.DAAG(Pos_embd, adj) 
        # 在 32 辆车之间做图注意力

        Pos_embd = Pos_embd.permute(0,2,1,3).reshape(B*Nnodes, SL, 2*self.hidden_size) 
        # 重塑形状 + 时序卷积-为什么合并 B 和 Nnodes？
        # 因为每个节点的时间序列是独立的，卷积操作可以在合并后的维度上进行，节省计算资源，同时保持每个节点的时间特征。
        # 后续的 TemporalConv 和 MultiheadAttention 都是在每辆车的时间序列上独立操作的。合并后，把 32 辆车当作独立的训练样本处理，batch size 从 B 变成 B×32


        # Leaky_Residual 
        Leaky_residual = self.TemporalConv(Pos_embd)  # 时序卷积
       # Leaky ：时序卷积的输出不是替代原始特征，而是泄漏到旁边
        Pos_embd= torch.cat((Pos_embd,Leaky_residual), dim=-1)

        # Positional Encoding + Attention 位置编码 + 自注意力
        Pos_embd = Pos_embd + positional_encoding(Pos_embd, 3*self.hidden_size)
        # positional_encoding() 定义在 modules.py
        Pos_embd_att = self.Position_Att(Pos_embd , Pos_embd , Pos_embd, need_weights=False, key_padding_mask = Scene_mask)[0] #key_padding_mask = src_mask
        # 三个输入都是 Pos_embd 自身，所以是自注意力——每个时间步关注其他所有时间步
        # Scene_mask 来自 modules.py 的 create_src_mask()，标记哪些时间步在该车辆上是空位，让注意力机制忽略它们


        Pos_embd = self.Position_Rezero*Pos_embd_att + Pos_embd
        # Rezero 残差：注意力输出 × 可学习系数 + 原始输入


        Pos_embd = self.dropout(Pos_embd)
        return Pos_embd, Leaky_residual
# =========================================

# 第三层 ==================================
class Traffic_Encoding_Layer(nn.Module):
    def __init__(self, hidden_size, num_att_heads, Traffic_indx, trf_embedding_dim, trf_embedding_dict_size, Num_linear_inputs, Linear_indx, dropout = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.Traffic_indx = Traffic_indx
        self.Linear_indx = Linear_indx

        self.Linear_Embedding_dim = 2*hidden_size - sum(trf_embedding_dim)

        # Embeddings of the Traffic features
        self.Traffic_embeddings = nn.ModuleList()
        for i in range(len(Traffic_indx)):
            self.Traffic_embeddings.append(nn.Embedding(trf_embedding_dict_size[i], trf_embedding_dim[i], padding_idx=0))
         # i = 0~4，对应5个特征：车辆所在区域+东、西、南、北四个方向的红绿灯
        
        # For the Linear features, we consider the followinng fully connected layers
        self.LE_LN1 = nn.LayerNorm(Num_linear_inputs)
        self.Linear_Embedding1 = nn.Linear(Num_linear_inputs, self.Linear_Embedding_dim)
        self.LE_LN2 = nn.LayerNorm(self.Linear_Embedding_dim)
        self.Linear_Embedding2 = nn.Linear(self.Linear_Embedding_dim, self.Linear_Embedding_dim)
        self.LE_LN3 = nn.LayerNorm(self.Linear_Embedding_dim)
        self.Linear_Embedding3 = nn.Linear(self.Linear_Embedding_dim, self.Linear_Embedding_dim)
        # 3 层 MLP，每层结构：LayerNorm → Linear → LeakyReLU → Rezero 残差
        
        # Traffic Attention
        self.Traffic_Att = nn.MultiheadAttention(embed_dim= 3*hidden_size, num_heads=num_att_heads, batch_first=True)
        self.Traffic_FF = FeedForwardNetwork(d_model=3* hidden_size, out_dim=3* hidden_size)
        # FFN，来自modules.py
        self.LE_Rezero2 = nn.Parameter(torch.zeros(self.Linear_Embedding_dim))
        self.LE_Rezero3 = nn.Parameter(torch.zeros(self.Linear_Embedding_dim))

        self.Traffic_Rezero = nn.Parameter(torch.zeros(3*hidden_size)) # 可训练残差系数，初始全零
        self.dropout = nn.Dropout(dropout)

    def forward(self, Scene, Scene_mask, Leaky_residual):
        B, SL, Nnodes, _ = Scene.size()

        # Embedding the Traffic features
        Traffic_embedding = []
        for n, indx in enumerate(self.Traffic_indx):
            Traffic_embedding.append(self.Traffic_embeddings[n](Scene[:,:,:,indx].long()))
        Trf_embd = torch.cat(Traffic_embedding, dim=-1)
        # Linear Embedding 线性嵌入
        Lin_embd = self.LE_LN1(Scene[:,:,:,self.Linear_indx])
        Lin_embd1 = self.Linear_Embedding1(Lin_embd)
        Lin_embd2 = self.LE_Rezero2* self.Linear_Embedding2(F.leaky_relu(Lin_embd1)) + Lin_embd1
        Lin_embd = self.LE_Rezero3* self.Linear_Embedding3(F.leaky_relu(Lin_embd2)) + Lin_embd2

        Trf_embd = torch.cat((Trf_embd, Lin_embd), dim=-1).reshape(B*Nnodes, SL, 2*self.hidden_size)
        Trf_embd = torch.cat((Trf_embd, Leaky_residual), dim =-1)
        # 拼接+重塑-Leaky_residual 来自第一层的 TemporalConv 输出，它包含了：
        # 位置信息：通过 TemporalConv 从坐标中提取的运动模式
        # 速度信息：时间卷积捕捉的连续帧间变化

        
        # Traffic Attention
        Trf_embd = Trf_embd + positional_encoding(Trf_embd, 3*self.hidden_size)
        # 源自modules.py的positional_encoding() 生成 [15, 384] 的正余弦位置编码矩阵，广播加到 Trf_embd 上
        Trf_embd_att = self.Traffic_Att(Trf_embd, Trf_embd, Trf_embd, key_padding_mask = Scene_mask)[0] #key_padding_mask = src_mask
        # Q=K=V → 自注意力
        Trf_embd = self.dropout(self.Traffic_Rezero*Trf_embd_att + Trf_embd)
        # Rezero 残差：注意力贡献 × 可学习系数（初始0）+ 原始信号
        Trf_embd = self.Traffic_FF(Trf_embd)
        # 定义自FFN（modules.py）：对每个时间步独立做非线性变换，捕捉特征之间的复杂关系
        Trf_embd = self.dropout(Trf_embd)
        return Trf_embd
# =========================================

# 第三层 ==================================
class Mixed_Attention_Layer(nn.Module): # 任务：把前两层的输出融合起来
    def __init__(self, hidden_size, num_att_heads):
        super().__init__()
        self.Mixed_Att = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_att_heads, batch_first=True)
        self.Mixed_FF = FeedForwardNetwork(d_model=hidden_size, out_dim=hidden_size)
        self.Mixed_Rezero = nn.Parameter(torch.zeros(hidden_size)) # 注意力残差系数
        self.Mixed_Rezero2 = nn.Parameter(torch.zeros(hidden_size)) # FFN 残差系数
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, Pos_embd, Traffic_embd, Scene_mask):
        # Pos_embd：第一层输出：位置+运动信息
        # Traffic_embd：第二层输出：交通+速度信息
        # Scene_mask：掩码，标记哪些时间步是空位
        # Cross Attention
        Att_mixed = self.Mixed_Att(Pos_embd, Traffic_embd, Traffic_embd, key_padding_mask = Scene_mask, need_weights=False)[0] #key_padding_mask = src_mask

        Mixed = self.Mixed_Rezero*Att_mixed + Pos_embd # 第一个残差连接
        Mixed_FF = self.Mixed_FF(Mixed) # FFN
        Mixed = Mixed + self.Mixed_Rezero2*Mixed_FF # 第二个残差连接
        Mixed = self.dropout(Mixed)
        return Mixed

class Encoder_DAAG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        self.xy_indx = config['xy_indx']
        self.Traffic_indx = config['Traffic_indx']
        # self.Linear_indx = [x for x in list(range(config['input_size'])) if x not in self.xy_indx + self.Traffic_indx]
        self.Linear_indx = config['Linear_indx']
        # 这六行把配置参数读到类里，供后续使用

        # 三个子层
        self.Positional_Encoding_Layer = Positional_Encoding_Layer(hidden_size=self.hidden_size, num_att_heads=num_heads,
                                                                xy_indx=self.xy_indx, pos_embedding_dim=config['pos_embedding_dim'],
                                                                pos_embedding_dict_size=config['pos_embedding_dict_size'],
                                                                nnodes=config['Nnodes'], sl=config['sl'] // config['dwn_smple'])
        self.Traffic_Encoding_Layer = Traffic_Encoding_Layer(hidden_size = self.hidden_size,
                                                             num_att_heads = num_heads, Traffic_indx = self.Traffic_indx, trf_embedding_dim = config['trf_embedding_dim'],
                                                             trf_embedding_dict_size = config['trf_embedding_dict_size'], Num_linear_inputs = len(self.Linear_indx),
                                                             Linear_indx = self.Linear_indx)
        self.Mixed_Att_Layer = Mixed_Attention_Layer(hidden_size = 3*self.hidden_size, num_att_heads = num_heads)


    def forward(self, Scene, Scene_mask, Adj_mat):
        Pos_embd, Leaky_Res = self.Positional_Encoding_Layer(Scene, Scene_mask, Adj_mat)
        # 第一层：位置编码
        Traffic_embd = self.Traffic_Encoding_Layer(Scene, Scene_mask, Leaky_Res)
        # 第二层：交通编码（接收第一层的 Leaky_Res）
        Mixed = self.Mixed_Att_Layer(Pos_embd, Traffic_embd, Scene_mask)
        # 第三层：交叉注意力融合
        return Mixed


class Projection(nn.Module):
    def __init__(self, hidden_size, output_size, output_dict_size, Nnodes):
        super(Projection, self).__init__()
        self.output_dict_size = output_dict_size
        self.Nnodes = Nnodes
        self.LN = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size*output_dict_size)
        self.output_size = output_size
        

    def forward(self, x):
        B, SL, _ = x.size()
        x = self.LN(x) # 数值被归一化
        x = self.linear2(x).reshape(B//self.Nnodes, self.Nnodes,  SL, self.output_size, self.output_dict_size).permute(0,2,1,3,4) # [B, SL, Nnodes, output_size, output_dict_size]
        
        return x


class HDAAGT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        output_size = len(config['xy_indx'])
        Nnodes = config['Nnodes']
        self.encoder = Encoder_DAAG(config)
        self.proj = Projection(3*self.hidden_size, output_size, output_dict_size= config['output_dict_size'], Nnodes= Nnodes)
        # 温度：原代码写死 /20，现在从配置读；不写这个键时仍然等于 20，行为不变
        self.temperature = float(config.get('Temperature', 20.0))
        if self.temperature <= 0:
            raise ValueError("config['Temperature'] 必须 > 0，当前 = %r" % (self.temperature,))
    def forward(self, scene: torch.Tensor, src_mask, adj_mat: torch.Tensor):
        enc_out = self.encoder(scene, src_mask, adj_mat)
        proj = self.proj(enc_out) / self.temperature  # 原版写死 /20
        return proj

if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")