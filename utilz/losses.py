"""
utilz/losses.py —— 损失函数的唯一出口

三类损失都可在这里选择，train_test.py 只调用 compute_loss()：

    loss_type = 'ce'     纯交叉熵（原版行为：mask_empty_in_loss 决定是否忽略空槽）
    loss_type = 'ce_l1'  交叉熵 + λ·L1(期望位置, 真值)/scale（"实验二"）
    兼容旧写法：不写 loss_type，只写 expect_l1_lambda > 0 → 自动视为 'ce_l1'

为什么要有 L1 项
---------------------------
逐轴 1024 类交叉熵只判断"真值那格的概率够不够高"，对"错 1 格"与"错 400 格"
给同样的惩罚；模型一旦没把握就把概率押到远处的峰上，解码出来离真值几百像素
"""
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 交叉熵（分类项）
# ----------------------------------------------------------------------------
def build_criterion(config):
    """ignore_index=0：空槽（target 全 0）不进 loss。真实坐标 >= 1"""
    if config.get('mask_empty_in_loss', False):
        return nn.CrossEntropyLoss(ignore_index=0)
    return nn.CrossEntropyLoss()
"""nn.CrossEntropyLoss()为torch内部函数"""

# ----------------------------------------------------------------------------
# 交叉熵 + λ·L1(期望位置, 真值)
# ----------------------------------------------------------------------------
def expect_distance_loss(outputs, Target, lam, scale=50.0, ignore_empty=True):
    """分类交叉熵 + λ·L1(期望位置, 真值)／scale

    参数
    ----
    outputs : [B, SL, N, 2, 1024]  每个槽、每个轴（x/y）的 1024 类 logits
    Target  : [B, SL, N, 2]        真值格子编号（空槽 = 0）
    lam     : 距离项权重；=0 时调用方应走纯 CE 分支
    scale   : 把像素折算成与交叉熵同量纲的常数（默认 50 px）
    ignore_empty : True = 空槽（target==0）不参与 CE，等价于实验 A 的开关

    SOS/EOS 两帧由 compute_loss 统一切掉，这里不管。
    返回 (loss, ce, l1_px)：后两个是 detach 过的标量，只用于日志观察
    """
    n_cls = outputs.size(-1)
    idx = torch.arange(n_cls, device=outputs.device, dtype=outputs.dtype)

    logp = torch.log_softmax(outputs, dim=-1)                  # [B,SL,N,2,1024]
    tgt = Target.long().unsqueeze(-1)                          # [B,SL,N,2,1]
    tgt_safe = tgt.clamp(0, n_cls - 1)
    valid = tgt.squeeze(-1) != 0                               # [B,SL,N,2] 真车才算数

    # ---- 1) 交叉熵 ----
    ce_pos = -logp.gather(-1, tgt_safe).squeeze(-1)            # [B,SL,N,2]
    w_ce = valid.to(ce_pos.dtype) if ignore_empty else torch.ones_like(ce_pos)
    ce = (ce_pos * w_ce).sum() / w_ce.sum().clamp(min=1.0)     # 加权平均，避免整批全空时 nan

    # ---- 2) 期望位置：按类别概率加权求和（分块算）----
    expect = torch.zeros(Target.shape, device=outputs.device, dtype=outputs.dtype)
    for s in range(0, n_cls, 256):
        e = min(s + 256, n_cls)
        expect += (logp[..., s:e].exp() * idx[s:e]).sum(-1)

    # ---- 3) 距离项：只对真车生效----
    l1 = (expect - Target.float()).abs()
    w_l1 = valid.to(l1.dtype)
    l1_px = (l1 * w_l1).sum() / w_l1.sum().clamp(min=1.0)

    loss = ce + lam * (l1_px / scale)
    return loss, ce.detach(), l1_px.detach()


# ----------------------------------------------------------------------------
# 统一入口
# ----------------------------------------------------------------------------
def loss_type(config):
    t = config.get('loss_type')
    if t:
        return str(t).strip().lower()
    if float(config.get('expect_l1_lambda', 0) or 0) > 0:
        return 'ce_l1'
    return 'ce'


def compute_loss(outputs, Target, config, criterion):
    """返回 (loss, ce, l1_px)；纯 CE 分支的 ce/l1_px 为 None。

    统一切掉 SOS/EOS 两帧：它们的目标是哨兵值 1008/1016（借用了坐标词表
    的两个格子），是无意义的分类项。原码评测时用 outputs[:, 1:-1]，训练里却
    算进了 loss，这里对齐
    """
    o, t = (outputs[:, 1:-1], Target[:, 1:-1]) if outputs.size(1) > 2 else (outputs, Target)
    if loss_type(config) == 'ce_l1':
        return expect_distance_loss(o, t,
                                    float(config.get('expect_l1_lambda', 0.3)),
                                    float(config.get('expect_l1_scale', 50.0)),
                                    ignore_empty=bool(config.get('mask_empty_in_loss', False)))
    loss = criterion(o.reshape(-1, o.size(-1)), t.reshape(-1).long())
    return loss, None, None


def describe(config):
    """一行摘要，写进日志方便回溯本实验用了什么损失。"""
    if loss_type(config) == 'ce_l1':
        return "loss = CE%s + λ·L1(期望位置)/scale  [λ=%g, scale=%g]" % (
            "(忽略空槽)" if config.get('mask_empty_in_loss', False) else "",
            float(config.get('expect_l1_lambda', 0.3)),
            float(config.get('expect_l1_scale', 50.0)))
    return "loss = CrossEntropy%s" % ("(忽略空槽)" if config.get('mask_empty_in_loss', False) else "")
