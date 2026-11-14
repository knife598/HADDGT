"""
utilz/layout.py —— 通道布局账本（谁占几号通道、input_size 是多少、连续分支读哪些）

改造前，"通道数"这件事散落在四处，改一处忘三处就会崩：
    main.py            input_size = NFeatures + 8 + 7        （手写常量）
    config.yaml        Linear_indx: [7..21]                  （手写列表）
    utilz/utils.py     ExtractFeatures 里 cat 了几段          （手写顺序）
    Slide_()           自己跟自己比一下通道数                 （事后自检）

改造后只有 features.REGISTRY 一个事实来源，本文件负责把它翻译成需要的东西：
    input_size   = NFeatures + Σ(启用组的通道数)
    Linear_indx  = [NFeatures, input_size)                   连续分支读全部派生通道
    Traffic_indx = config 里那一行（Zone + 4 路灯，共 5 个嵌入表）
调用方式：main.py 加载 config 之后立刻 resolve(config)，之后所有地方都读 config。
"""
from utilz.features import REGISTRY, enabled_groups, n_derived


class Layout(object):
    """一次实验的通道布局快照。"""

    def __init__(self, n_raw, groups, counts, xy_indx, traffic_indx):
        self.n_raw = n_raw              # 原始列数（Columns_to_keep 的长度）
        self.groups = groups            # 启用的特征组，按拼接顺序
        self.counts = counts            # 每组占几个通道
        self.n_derived = sum(counts.values())
        self.n_channels = n_raw + self.n_derived
        self.input_size = self.n_channels
        self.xy_indx = list(xy_indx)    # 坐标在原始列里的位置（基础数据里固定 [0,1]）
        self.traffic_indx = list(traffic_indx)
        self.linear_indx = list(range(n_raw, self.n_channels))

    # 每个通道在原始 7 列之后的位置：{组名: (起始, 结束)}
    def ranges(self):
        out, cur = {}, self.n_raw
        for g in self.groups:
            n = self.counts[g]
            out[g] = (cur, cur + n)
            cur += n
        return out

    def summary(self):
        parts = ["原始 %d" % self.n_raw]
        for g in self.groups:
            a, b = self.ranges()[g]
            parts.append("%s %d (%d~%d)" % (g, self.counts[g], a, b - 1))
        return ("input_size=%d = " % self.input_size) + " + ".join(parts) + \
               "；连续特征 Linear_indx=[%d..%d] 共 %d 个" % (
                   self.linear_indx[0], self.linear_indx[-1], len(self.linear_indx))

    def __repr__(self):
        return "<Layout %s>" % self.summary()


def resolve(config, fix_config=True):
    """按 features.REGISTRY + config 开关算出布局；默认顺手把 config 里的
    input_size / Linear_indx / Num_linear_inputs 写成一致的值（避免手写漂移）。"""
    n_raw = int(config['NFeatures'])
    groups = enabled_groups(config)
    counts = dict((g, REGISTRY[g][1]) for g in groups)
    lay = Layout(n_raw, groups, counts,
                 config['xy_indx'], config['Traffic_indx'])
    if fix_config:
        config['input_size'] = lay.input_size
        config['Linear_indx'] = lay.linear_indx
        config['Num_linear_inputs'] = len(lay.linear_indx)
    return lay


def validate(config, actual_channels):
    """自检：实际产出的通道数必须与账本一致（不一致时给出可读的报错）。"""
    lay = resolve(config, fix_config=False)
    if int(actual_channels) != lay.input_size:
        raise ValueError(
            "特征通道数不一致：实际产出 %d，账本 input_size=%d。\n"
            "  %s\n"
            "常见原因：① config 的 Columns_to_keep 与 NFeatures 不符；"
            "② ExtractFeatures 里手工 cat 了未注册的通道；"
            "③ 开了 use_xxx 但没在 features.REGISTRY 登记通道数。"
            % (actual_channels, lay.input_size, lay.summary()))
    return lay
