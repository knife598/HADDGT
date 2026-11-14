# HDAAGT 改造版（structure）

基于 IEEE T-ITS 2025 论文 *HDAAGT*（Heterogeneous Decision-Aware Attention Graph Transformer）的改造副本。

**模型的网络结构一行没动**（DAAG / TemporalConv / Traffic_Encoding / Mixed_Attention / Projection 全部保留原样），改的是四件事：

1. **喂什么**——输入特征从 24 通道（其中 16 个算了不用）改为 22 通道，派生特征全部接入连续分支；
2. **优化什么**——损失从纯交叉熵改为「交叉熵 + 距离项」，空槽位不参与损失；
3. **怎么读出来**——解码方式从写死 top-5 改为可配置（topk5 / argmax / expect）；
4. **配置与代码对齐**——修掉 5 处写死的常量。

---

## 1. 环境

```bash
conda activate hdaagt310          # 本项目实测环境
python -c "import torch; print(torch.__version__)"   # 期望 2.4.1+cu121
```

依赖：`torch`、`numpy`、`pandas`、`pyyaml`、`matplotlib`。

## 2. 数据

需要一个 CSV，13 列，列名为：

```
Frame, ID, BBx, BBy, W, L, Cls, Tr1, Tr2, Tr3, Tr4, Zone, Xreal, Yreal
```

- `BBx / BBy`：**整数**坐标，取值 `1 ~ 1023`（直接作为 1024 类别的索引，0 保留给"空槽"）
- `Tr1 ~ Tr4`：信号灯状态，`0~8`
- `Zone`：`1 ~ 9`（`0` 表示无效）
- `Frame`：连续整数帧号，10 fps

配置里 `detection_path` 请写**绝对路径**，例如：

```yaml
detection_path: '/home/ky1/sjkx/data/Fisheye-sind-big.csv'
```

## 3. 快速开始

```bash
# ① 10 秒自检：不需要数据、不占显存，检查通道数与模型是否自洽
python tools/check_layout.py

# ② 训练 + 测试（以 SinD-big 为例）
python -u main.py -c configs/config_sind_big.yaml

# ③ 只看测试（不重训）
#    把配置里的 Train 改 false、Test 改 true、Load_Model 改 true、generate_data 改 false
python -u main.py -c configs/config_sind_l1_eval.yaml
```

产物：

| 文件 | 内容 |
|---|---|
| `logs/log-<时间码>.txt` | 完整日志（训练曲线、每 5 轮测试、最终指标） |
| `logs/log-summary <时间码>.txt` | 一两行的总览（**注意：里面是训练集指标**） |
| `Pickled/best_trained_model<时间码>.pth` | 权重 |
| `Pickled/Train, Test, Validation` | 预处理缓存（换配置后要先删掉重建） |

## 4. 配置文件怎么改

所有配置都在 `configs/` 下。常用的几个：

| 文件 | 用途 |
|---|---|
| `config.yaml` | Fisheye-MARC，22 通道，纯交叉熵（基线） |
| `config_fisheye_l1.yaml` | Fisheye-MARC + 距离项 |
| `config_sind_base.yaml` | SinD，纯交叉熵 |
| `config_sind_l1.yaml` | SinD + 距离项（T=5） |
| `config_sind_big.yaml` | **SinD-big（天津+西安合并），当前主结果** |
| `config_sind_l1_eval.yaml` | 只评测（不训练），用于换解码方式做对照 |

关键键：

| 键 | 含义 |
|---|---|
| `detection_path` | 数据 CSV 绝对路径 |
| `generate_data` | `true` = 重建预处理缓存；`false` = 读已有缓存（**换配置必须重新设 true 一次**） |
| `Nnodes` / `Nusers` | 每帧的最大槽位数（SinD-big 用 48） |
| `sl` / `future` / `dwn_smple` | 历史帧数 / 未来帧数 / 采样步长（3s-3s 即 30/30/2） |
| `Columns_to_keep` | 原始 CSV 里参与模型的列（本版本固定 7 个） |
| `Temperature` | logits 的除数（默认 20；实测 5 明显更好） |
| `mask_empty_in_loss` | `true` = 空槽位不参与交叉熵 |
| `loss_type` | `ce` 或 `ce_l1` |
| `expect_l1_lambda` / `expect_l1_scale` | 距离项权重 / 量纲（默认 0.3 / 50） |
| `decode_mode` | 评测解码：`topk5` / `argmax` / `expect` |
| `Train` / `Test` / `Load_Model` | 训练 / 测试 / 加载权重 |
| `device` | 如 `cuda:0` |

## 5. 目录结构

```
.
├── main.py                 入口：读配置 → 建数据 → 建模型 → 训练/测试
├── train_test.py           训练循环、测试循环、prep_model_input
├── utilz/
│   ├── utils.py            数据管线（loadcsv / Scene_Process / Slide_ / Scene 缓存）
│   ├── features.py         派生特征（base 8 个 + 运动学 7 个），单一事实来源
│   ├── layout.py           通道布局账本：input_size / Linear_indx 自动推导
│   ├── losses.py           损失：ce / ce_l1，空槽忽略
│   ├── decode.py           解码：topk5 / argmax / expect + ADE/FDE
│   └── kalman.py           匀速模型的卡尔曼滤波（对照基线用）
├── models/
│   ├── HDAAGT.py           编码器主干（Positional / Traffic / Mixed / Projection）
│   └── modules.py          DAAG、TemporalConv、位置编码、mask 工具
├── configs/                配置 + ZoneConf + requirements
├── tools/
│   ├── check_layout.py     10 秒自检
│   ├── probe_model.py      诊断脚本（解码方式对比、误差分位数）
│   └── run_*.sh            超参扫描模板
├── docs/                   改进章程、损失函数方案
└── Pickled/                预处理缓存（首次运行自动生成）
```

## 6. 相对原始 HDAAGT 的改动

| # | 位置 | 原版 | 本版 |
|---|---|---|---|
| 1 | `main.py` | `input_size = 7 + 17` 写死 | 由 `utilz/layout.py` 自动推导（22） |
| 2 | `main.py` | `torch.zeros(17)` 补 SOS/EOS | `pad = input_size - len(sos)` |
| 3 | `models/HDAAGT.py` | `proj / 20` 写死 | `proj / config['Temperature']` |
| 4 | `main.py` | `if loss < 1.5` 才存模型 | `config['save_loss_threshold']` |
| 5 | `utilz/utils.py` | 24 通道，`Linear_indx:[9]` 只喂 1 个 | 22 通道，15 个派生特征全进连续分支 |
| 6 | `utilz/utils.py` | 空槽位参与交叉熵 | `ignore_index=0`（可关） |
| 7 | `utilz/losses.py` | 纯 CE | CE + λ·L1(期望位置)/scale |
| 8 | `utilz/decode.py` | 只有 top-5 | topk5 / argmax / expect 可选 |

参数量：原版 `3,947,038` → 本版 `3,956,506`（+0.24%，来自连续分支拓宽）。

## 7. 常见问题

**Q：`generate_data: true` 但没重新生成缓存？**
先删掉 `Pickled/` 下的 `Train`、`Test`、`Validation` 三个文件，再跑。

**Q：报 `特征通道数 22 与 config 的 input_size N 不一致`？**
`utilz/utils.py` 里的通道自检发现配置与数据不匹配——检查 `configs/*.yaml` 是不是和代码同一版本。

**Q：日志里的 ADE/FDE 怎么和论文对不上？**
本项目指标是**原始口径**：`ADE(原始) = ADE(只算真车) × 填充率`，因为空槽的预测被 `*flg` 乘 0。跨数据集比较请用 `utilz/decode.py::agent_metrics()`。

**Q：`Validation size: 0`？**
划分逻辑有 off-by-one，那一段被并进了测试集；验证集全程未被使用，不影响结果。
