
# export cuda_visible_devices=2,3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np
import os
import yaml
import datetime
import argparse
# The rest is HDAAGT specific imports
from utilz.utils import *
from utilz.layout import resolve as resolve_layout   # 通道布局账本（input_size / Linear_indx）
from utilz.losses import build_criterion, describe as describe_loss  # 损失统一出口
from train import train_model           # 训练循环（train.py）
from test import test_model              # 测试/评估循环（test.py）
from models.HDAAGT import *

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 初始配置
if __name__ == '__main__':
    # If the model is run through terminal
    parser = argparse.ArgumentParser(description="Run the main script with a config file.")
    parser.add_argument("-c", "--config", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()
    torch.cuda.empty_cache()
    # argparse 的作用：让用户通过命令行指定配置文件
    # Otherwise we use the default directory to the config path
    if args.config is None:
        args.config = "configs/config.yaml"
        print(f"Using the '{args.config}' file")
        # 如果用户没有通过 -c 指定配置文件，就使用默认路径 configs/config.yaml

    with  open(args.config ) as file:
        config = yaml.safe_load(file) # 把 config.yaml 中的内容读取为 Python 字典 config
    cwd = os.getcwd() # 获取当前工作目录
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu') # 设备设置， config['device'] 原为 'cuda:3' 
    ct = datetime.datetime.now().strftime("%m-%d-%H-%M") # 生成时间戳-Current time, used for saving the log
    savelog(f"Total  # of GPUs {torch.cuda.device_count()}", ct)
    savelog(f"Using {device} device", ct)
    savelog(f' Model description: {config["description"]}', ct=ct)
    # 记录初始日志-savelog 来自 utils.py，打印到控制台 + 写入 logs/log-{ct}.txt。

    
    ZoneConf = Zoneconf(config['Zoneconf_path']) # This is the setting for zone configuration, you can find the image file at 'data/Zone Configuration.jpg'
    # config non-indipendent parameters
    config['NFeatures']= len(config["Columns_to_keep"]) # This goes to the data preparation process
    config["output_size"] = len(config['xy_indx']) # The number of columns to predict
    config['device'] = device
    config['ct'] = ct
    # 通道布局（input_size / Linear_indx / Num_linear_inputs）由 features.REGISTRY + 配置开关数出来，
    # 不再手写常量：加一组特征只需在 utilz/features.py 注册，这里自动跟着变。
    _layout = resolve_layout(config)
    savelog("[layout] " + _layout.summary(), ct)

    set_seed(int(config.get('seed', 42)))
    savelog(f"Random seed = {config.get('seed', 42)}", ct) # 种子

    log_code = f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}"
    if config['verbal']: # true-把所有配置参数逐条打印到日志文件
        for arg in config.items(): # Print the arguments to the log file
            savelog(f"{arg[0]} = {arg[1]}", ct) # log_code 保存了关键超参数的字符串，用于后续的摘要日志

    # 构造 SOS / EOS 张量：补齐到 input_size 维
    _n_pad = config["input_size"] - len(config['sos'])
    config['sos'] = torch.cat((torch.tensor(config['sos']), torch.zeros(_n_pad)), dim=0).repeat(config['Nnodes'], 1).to(device)
    config['eos'] = torch.cat((torch.tensor(config['eos']), torch.zeros(_n_pad)), dim=0).repeat(config['Nnodes'],1).to(device)
    assert config['sos'].size(-1) == config['input_size'] and config['eos'].size(-1) == config['input_size'], \
        'sos/eos 维度 %d 与 input_size %d 不一致' % (config['sos'].size(-1), config['input_size'])
    savelog(f"sos/eos padded to input_size={config['input_size']} (pad={_n_pad})", ct)
    
    # Before Everything, make sure to assert all the parameters are correct, implement later
        # Later


    # 数据准备-Create datasets, we split the datasets to make sure there is no data leakage between training and test samples
    Scenetr = Scenes(config) # 训练集
    Scenetst = Scenes(config) # 测试集
    Sceneval = Scenes(config) # 验证集
    savelog(f'Starting at:  {datetime.datetime.now().strftime("%H-%M-%S")}', ct)
    # 记录精确的开始时刻（时-分-秒），用于追踪数据预处理耗时

    if not config['generate_data']: # 分支：从硬盘加载已处理的数据
        savelog("Loading data ...", ct)
        Scenetr.load_class('Pickled', cmnt = 'Train')
        Scenetst.load_class('Pickled', cmnt = 'Test')
        Sceneval.load_class('Pickled', cmnt = 'Validation')

    else: # 分支：从 CSV 重新处理数据
        savelog("Loading CSV file ...", ct)
        Traffic_data = loadcsv(config['detection_path'], config['Headers'])
        # 读取 Fisheye-MARC.csv
        Scenetr, Scenetst, Sceneval = Scene_Process(Scenetr, Scenetst, Sceneval, Traffic_data, config)
        print(f"Augmenting the data {config['noise_multiply']} times")
        Scenetr.addnoise(config['noise_multiply'], config['noise_amp'], config['noise_ratio'])
        # addnoise：只在训练集上加噪声，测试集和验证集保持原始数据
        Scenetr.save_class('Pickled', cmnt = 'Train')
        Scenetst.save_class('Pickled', cmnt = 'Test')
        Sceneval.save_class('Pickled', cmnt = 'Validation')
        # 保存到硬盘，下次运行就可以走分支 A 直接加载
    savelog(f"Train size: {len(Scenetr)}, Test size: {len(Scenetst)}, Validation size: {len(Sceneval)}", ct)
    # 打印数据集大小

    
    train_loader, test_loader, val_loader = DataLoader_Scene(Scenetr, Scenetst, Sceneval, config['batch_size'])
    # 创建 DataLoader-DataLoader_Scene 来自 utils.py

    model = HDAAGT(config).to(device) # Here we define our model-定义模型model，与其他模块联动
    
    # 损失统一由 utilz/losses.py 构造（详见该文件顶部说明）：
    #   loss_type: 'ce'    纯交叉熵；mask_empty_in_loss=true 时忽略空槽（实验 A）
    #              'ce_l1' CE + λ·L1(期望位置, 真值)/scale（即"实验二"，给交叉熵补距离感）
    #   兼容旧配置：只写 expect_l1_lambda>0 而不写 loss_type 时自动当作 'ce_l1'
    criterion = build_criterion(config)
    savelog("[loss] " + describe_loss(config), ct)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate']) # 创建优化器
    scheduler = StepLR(optimizer, step_size=config['schd_stepzise'], gamma=config['gamma']) # 创建调度器

    if not config['Load_Model']: # false-随机初始化权重，从头训练
        savelog("Creating model from scratch",ct)
    if config['Load_Model']: # true-加载已有模型权重
        savelog("Loading model from the checkpoint", ct)
        model.load_state_dict(torch.load(config['Load_Model_Path'], map_location=device))

    if config['Train']: # 训练
        savelog(f"The number of learnable parameter is {count_parameters(model=model)} !", ct)
        print(f"Learning rate: {config['learning_rate']}, Hidden size: {config['hidden_size']}, Batch size: {config['batch_size']}")
        Best_Model, train_loss, trainADE, trainFDE = train_model(model, optimizer, criterion, scheduler, train_loader,test_loader, config)
        # train_model 来自 train.py-执行训练循环
        savelog(f"{log_code} with Avg loss of {train_loss[-1]}, ADE of {trainADE[-1]}, FDE of {trainFDE[-1]}", f"summary {ct}")
        if train_loss[-1] < float(config.get('save_loss_threshold', 1.5)): # 最终 loss 达标才保存（默认 1.5 = 原版）
            savelog(f"Saving result of {log_code}", ct)
            torch.save(Best_Model.state_dict(), os.path.join(cwd,'Pickled', f'best_trained_model{ct}.pth'))
            torch.save(train_loss, os.path.join(cwd,'Pickled', f'epoch_losses{ct}.pt'))
        else: # 仅提示日志，不改变其它行为
            savelog(f"[注意] 最终 loss={train_loss[-1]:.4f} ≥ {float(config.get('save_loss_threshold', 1.5))}，未保存模型", ct)
        savelog(f"Training finished for {ct}!", ct)

    if config['Test']: # 测试-If not training, then test the model
        Topk_Selected_words, ADE, FDE = test_model(model, test_loader, config)
        # test_model 来自 train.py，遍历测试集的所有 batch，计算整个测试集上的平均 ADE 和 FDE
        savelog(f"Average Displacement Error: {ADE}, Final Displacement Error: {FDE}", ct)