"""
utilz/decode.py —— 类别 logits → 连续坐标 的唯一步骤，以及 ADE/FDE 的定义

----------------------------------
模型的输出是"每个轴 1024 个格子"的分类分布（x 轴一个 1024 类，y 轴一个 1024 类），
必须先把分布"解码"成坐标才能算误差。解码方式直接决定指标：

    topk5  (top_indices*top_values).sum()/top_values.sum()   原版做法，概率加权
    argmax 取概率最大的那一格                                 实测比 topk5 好约 17%
    expect 全分布期望 Σ(i·p_i)                              实测更差，不推荐

注意：训练用 CE 只优化"真值那格的概率"，并不约束"概率质量离真值多远"；而
ADE/FDE 却是欧氏距离。这个错位是本项目误差的主要来源（见 losses.py 的说明）。

指标口径（务必分清）
--------------------
Find_topk_selected_words 里把空槽的预测乘 0（`*flg`），于是空槽误差恒为 0：

    ADE(原始口径) = ADE(只算真车) × 填充率

所以不同数据集/不同窗口长度的"原始口径 ADE"不可直接比较（SinD 填充率≈0.33、
Fisheye-MARC≈0.44）。tools/ 里的分析脚本请用 agent_metrics() 取车口径。
"""
import torch


DEFAULT_K = 5


def decode_positions(pred_logits, mode='topk5', k=DEFAULT_K, ignore_classes=None):
    """把 [..., 2, 1024] 的 logits 解码成 [..., 2] 的坐标。

    pred_logits  : 未 softmax 的 logits，最后一维是 1024 个整数格子
    mode         : 'topk5' | 'argmax' | 'expect'
    ignore_classes : 参与解码前先置零概率的类别列表（例如 SOS/EOS 的 1008/1016）
    """
    probs = pred_logits.softmax(dim=-1)
    if ignore_classes:
        idx = torch.as_tensor(list(ignore_classes), dtype=torch.long, device=probs.device)
        probs = probs.clone()
        probs[..., idx] = 0.0

    mode = str(mode or 'topk5').strip().lower()
    if mode == 'topk5':
        top_values, top_indices = torch.topk(probs, k=k, dim=-1)
        # 只用前 k 个峰做概率加权平均：峰值附近取连续坐标，避免量化误差
        return (top_indices * top_values).sum(-1) / top_values.sum(-1).clamp(min=1e-12)
    if mode == 'argmax':
        return probs.argmax(dim=-1).to(probs.dtype)
    if mode == 'expect':
        n_cls = probs.size(-1)
        idx = torch.arange(n_cls, device=probs.device, dtype=probs.dtype)
        return (probs * idx).sum(-1)
    raise ValueError("未知的 decode_mode: %r（可选 topk5 / argmax / expect）" % (mode,))


def _errors(words, target):
    """返回逐 (样本, 帧, 槽) 的欧氏误差；空槽误差为 0（与原版定义一致）。"""
    flg = target[:, :, :, :1] != 0                  # 空位标记
    diff = words * flg - target
    return torch.sqrt(torch.pow(diff, 2).sum(-1))   # [B, SL, N]


def find_topk_selected_words(pred_logits, target, mode='topk5', k=DEFAULT_K,
                             ignore_classes=None):
    """评估函数：模型输出 + 真值 → (解码坐标, ADE, FDE)。

    默认参数（mode='topk5'）与改造前逐位一致，因此历史结果仍可复现。
    """
    words = decode_positions(pred_logits, mode=mode, k=k, ignore_classes=ignore_classes)
    err = _errors(words, target)
    ade = err.mean()
    fde = err[:, -1].mean()
    return words, ade, fde


def agent_metrics(pred_logits, target, mode='topk5', k=DEFAULT_K, ignore_classes=None,
                  skip_edge_frames=True):
    """车口径指标：只对真实车求平均，并给出填充率，便于跨数据集比较。

    返回 dict: ade / fde（只算真车）、ade_raw / fde_raw（原始口径）、fill_ratio
    """
    tgt = target
    prd = pred_logits
    if skip_edge_frames and tgt.size(1) > 2:        # 训练/测试调用时通常已切掉 SOS/EOS
        tgt, prd = tgt[:, 1:-1], prd[:, 1:-1]
    words = decode_positions(prd, mode=mode, k=k, ignore_classes=ignore_classes)
    flg = (tgt[:, :, :, :1] != 0).to(words.dtype)
    err = torch.sqrt(torch.pow(words - tgt, 2).sum(-1))
    valid = flg.expand_as(err)
    ade_agent = (err * valid).sum() / valid.sum().clamp(min=1.0)
    last = err[:, -1]
    last_valid = valid[:, -1]
    fde_agent = (last * last_valid).sum() / last_valid.sum().clamp(min=1.0)
    fill = flg.mean()
    return dict(ade=float(ade_agent), fde=float(fde_agent),
                ade_raw=float(ade_agent * fill), fde_raw=float(fde_agent * fill),
                fill_ratio=float(fill))
