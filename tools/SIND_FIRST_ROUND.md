# SinD 适配第一轮说明

本轮以官方 `Veh_smoothed_tracks.csv` 为输入，预测车辆，不读取 `Ped_smoothed_tracks.csv`。默认保留 `car/truck/bus/motorcycle`，排除 `tricycle/bicycle` 等未明确属于机动车的类别；如需纳入三轮车可显式传 `--agent-types car,truck,bus,motorcycle,tricycle`。`agent_type` 仅在转换时用于筛选，不作为模型特征。速度、朝向、加速度和时间间隔也都不输入模型。

## 缺失值与编码

- 有效位置量化到 `1..1023`，`0` 专用于 padding。
- SinD 灯色原始值 `0/1/3`（红/绿/黄）转换为 `1/2/3`；灯状态 `0` 专用于缺失/padding。
- SinD 交通灯文件记录状态变更事件。转换器按 `timestamp_ms` 将最近一次事件向前填充到车辆帧；第一次事件之前状态未知。
- HDAAGT 输入额外携带 `agent_valid`、`ZoneMask`、`Tr1Mask..Tr4Mask`。模型注意力、类别嵌入门控、训练损失和 ADE/FDE 使用这些 mask，不再靠坐标或类别值判断 padding。
- Changchun 文件提供两个车辆信号灯通道；另外两个通道的类别保持 0、mask 恒为 0。SinD 的道路 Zone 本轮不映射，ZoneMask 恒为 0。
- CSV 的有效位置仍是离散网格，米制 ADE/FDE 用同目录生成的 `.meta.json` 仿射映射计算。

## 生成数据

在 `HADDGT` 目录下运行：

```bash
python tools/convert_sind_to_hdaagt.py \
  --vehicle ../SinD/Data/Changchun/changchun_pudong_507_009/Veh_smoothed_tracks.csv \
  --traffic ../SinD/Data/Changchun/changchun_pudong_507_009/Traffic_Lights.csv \
  --out data/SinD/Changchun_507_009_masked.csv
```

然后用新配置启动数据生成和训练：

```bash
python main.py --config configs/config_SinD_vehicle.yaml
```

服务器上的 SinD 路径与本地不同的话，只需把转换命令里的 `--vehicle` 和 `--traffic` 改为服务器 `Data` 目录下的实际路径。CSV 与 `.meta.json` 属于原始数据转换产物，仓库忽略了 `data/SinD/`，不会由 Git 上传；应在服务器从官方文件重新生成。

## 本轮范围与限制

当前配置为 2 秒历史、2 秒预测，沿用 HDAAGT 的离散坐标输出。交通灯只有在第一次已观测状态之后才有 mask=1。SinD 没有对应的 Zone 区域映射，因此模型本轮不会得到有效 Zone 信息。配置调小了类别嵌入总维度，使 HDAAGT 原有线性特征分支不再是 0 维；它仍只使用模型从位置序列推导的既有特征，不会读取 SinD 的速度/朝向/加速度字段。

第一次用新配置 `generate_data: true` 会重建 `Pickled/Train`、`Pickled/Test`、`Pickled/Validation`。旧 `.pt` 文件不兼容此 mask-aware 输入格式，不应设置 `generate_data: false` 复用旧缓存或旧 checkpoint。
