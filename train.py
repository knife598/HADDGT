"""
train.py —— 训练循环（训练相关的代码都在这里）

    train_model(...)   完整的训练流程：前向 → 损失 → 反传 → 更新参数 → 记录 → 早停

配套文件：
    test.py         测试/评估循环  test_model(...)
    utilz/prep.py       两者共用的输入预处理  prep_model_input(...)

拆分理由：训练需要反传与优化器，评估只需 no_grad 前向并计时，两者生命周期不同；
分开后各自依赖更清楚，main.py 按需导入。
"""
import torch

from utilz.utils import Find_topk_selected_words, savelog
from utilz.losses import compute_loss          # 损失统一出口（CE 或 CE+λ·L1）
from utilz.prep import prep_model_input        # 输入预处理（test.py 也用同一个）
from test import test_model                # 训练中每 5 轮要调用一次评估


def train_model(model, optimizer, criterion, scheduler, train_loader, test_loader, config):
    print("Training the model ...")
    patience_limit = config['patience_limit']
    sos = config['sos']
    eos = config['eos']
    # 读取配置

    train_loss, prev_average_loss, patience= [],  10000000.0, 0
    trainFDE, trainADE = [], []
    Best_Model = []
    # 初始化记录变量

    for epoch in range(config['epochs']): # 外层循环，总共训练 config['epochs'] 轮
        model.train() # 切换训练模式
        epoch_losses, epoch_ADE, epoch_FDE = [], [], [] # 初始化本轮记录
        epoch_ce, epoch_l1 = [], [] # 【方案二】分类项与距离项分开记录（只在 λ>0 时填充）
        for Scene, Target, Adj_Mat_Scene in train_loader: # Scene & Taget => [B, SL0, Nnodes, Features], Adj_Mat_Scene => [B, SL, Nnodes, Nnodes]
        # 每次从 train_loader 中取出一个 batch 的训练数据

            Scene = Scene.to(config['device'])
            Target = Target.to(config['device'])
            Adj_Mat_Scene = Adj_Mat_Scene.to(config['device'])

            optimizer.zero_grad() # 梯度清零
            Scene, Scene_mask, Adj_Mat, Target = prep_model_input(Scene, Adj_Mat_Scene,Target, sos, eos, config)
            # prep_model_input调用自下方函数，数据输入预处理
            outputs = model(Scene, Scene_mask, Adj_Mat)

            # 损失统一出口（utilz/losses.py）：
            #   loss_type='ce'    → 纯交叉熵（mask_empty_in_loss 决定是否忽略空槽）
            #   loss_type='ce_l1' → CE + λ·L1(期望位置, 真值)/scale
            loss, _ce, _l1 = compute_loss(outputs, Target, config, criterion)
            if _ce is not None:
                epoch_ce.append(float(_ce))
                epoch_l1.append(float(_l1))
            loss.backward() # 求偏导数 把梯度写进param.grad 
            # PyTorch 从 loss 开始，沿着计算图反向传播，计算每个可训练参数的梯度（存到参数的 .grad 属性中）

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            # 梯度裁剪，防止梯度爆炸
            optimizer.step() # 参数更新 param = param - lr * param.grad 

            # 计算 ADE/FDE（不计算梯度）
            with torch.no_grad():
                _, ADE, FDE = Find_topk_selected_words(outputs[:,1:-1], Target[:,1:-1],
                                                       mode=config.get('decode_mode')) # sos/eos 帧不参与指标
            epoch_losses.append(loss.item()) # 记录本 batch 的损失
            epoch_ADE.append(ADE) # 记录本 batch 的 ADE
            epoch_FDE.append(FDE) # 记录本 batch 的 FDE

        trainADE.append(sum(epoch_ADE)/ len(train_loader))
        trainFDE.append(sum(epoch_FDE)/ len(train_loader))
        train_loss.append(sum(epoch_losses) / len(train_loader))
        # 平均值：batch 的损失、ADE、FDE，存入 train_loss、trainADE、trainFDE
        
        scheduler.step()# 学习率调度器更新学习率
        
        

        log = f"Epoch [{epoch+1}/{config['epochs']}], Loss: {train_loss[-1]:.2f}, ADE: {trainADE[-1]:.3f}, FDE: {trainFDE[-1]:.2f}"
        if epoch_ce: # 【方案二】把分类项与距离项分开打出来，便于判断 λ 是否压过头
            log += f", CE: {sum(epoch_ce)/len(epoch_ce):.3f}, 期望偏差: {sum(epoch_l1)/len(epoch_l1):.1f}px"
        savelog(log, config['ct'])
        # Saving the best model to the file 日志记录
        
        if train_loss[-1] < prev_average_loss: # checkpoint update
            prev_average_loss = train_loss[-1]  # Update previous average loss
            patience = 0
            Best_Model = model
        elif patience > patience_limit:
                savelog(f"early stopping, Patience lvl1 , lvl2 {patience}", config['ct'])
                break
        patience += 1
        # 早停机制 本轮损失 < 历史最低损失->更新历史最低损失，patience=0；否则，patience+1，若patience>patience_limit，则早停

        if config['Test_during_training'] and epoch % 5 == 3: # 训练中测试（可选）-每 5 轮（第 3, 8, 13, 18, ... 轮）在测试集上评估一次
            _, ADE, FDE = test_model(model, test_loader, config)
            savelog(f"During Training, Test ADE: {ADE :.2f}, FDE: {FDE :.2f}", config['ct'])
            model.train() # 切换回训练模式
            # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.5f} MB")
            # print(f"Reserved memory:  {torch.cuda.memory_reserved() / 1024**2:.5f} MB")
    return Best_Model, train_loss, trainADE, trainFDE


if __name__ == "__main__":
    print("Yohoooo, Ran a Wrong Script!")
