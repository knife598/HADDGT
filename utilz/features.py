"""
utilz/features.py —— 特征工程的唯一出口（注册表 + 按注册顺序拼接）

设计约定
--------
1. 通道 0/1 = x/y（由 Columns_to_keep 决定），通道 2..6 = Zone + 4 路灯。
2. 每个 build_xxx(c, cfg) 返回 (T, N, n) 的新通道，n 必须与 REGISTRY 里登记的一致。
3. 所有新特征必须：因果（不用预测期）、量级 O(1)、空槽（y==0）置零。
4. 加一组新特征 = 写一个 build_xxx + REGISTRY 加一行 + config 加一个 use_xxx 开关。
   模型、main.py、train_test.py 都不需要动（input_size / Linear_indx 由 layout.py 数出来）。

通道编号（启用 road_state 时）
-----------------------------
0..6   原始 7 列（x, y, Zone, Tr1..Tr4）
7..14  base：dx, dy, heading, xc, yc, Rc, xc², yc²
15..21 kinematics：ax, ay, vx3, vy3, turn_sin, turn_cos, dspeed
22..29 road_state：occ, zone_cnt, zone_stop, zone_speed, near_cnt, age, is_new, frame_frac
"""
import torch


# ----------------------------------------------------------------------------
# 公共中间量：只算一次，三组特征共用（与旧版 ExtractFeatures 里的局部变量完全一致）
# ----------------------------------------------------------------------------
def _common(fs, cfg):
    i0, i1 = cfg['xy_indx'][0], cfg['xy_indx'][1]
    x = fs[:, :, i0].unsqueeze(2)
    y = fs[:, :, i1].unsqueeze(2)
    dx = torch.diff(x, n=1, append=x[-1:, :], dim=0)   # 与旧版一致（前向差分）
    dy = torch.diff(y, n=1, append=y[-1:, :], dim=0)
    flg = y != 0                                       # 空位标记（BBy 为 0）
    return dict(fs=fs, x=x, y=y, dx=dx, dy=dy, flg=flg)


# ----------------------------------------------------------------------------
# 组 1：base —— 8 个坐标派生特征（原版逻辑，逐位搬过来）
# ----------------------------------------------------------------------------
def build_base(c, cfg):
    x, y, flg = c['x'], c['y'], c['flg']
    centre_x, centre_y = cfg['Centre'][0], cfg['Centre'][1]
    xc = ((x - centre_x) / centre_x) * flg      # 归一化 x 坐标
    yc = ((y - centre_y) / centre_y) * flg      # 归一化 y 坐标
    xc2 = xc ** 2
    yc2 = yc ** 2
    Rc = torch.sqrt(xc ** 2 + yc ** 2) * flg
    heading = torch.atan2(xc, yc) * flg
    return torch.cat((c['dx'], c['dy'], heading, xc, yc, Rc, xc2, yc2), dim=2)


# ----------------------------------------------------------------------------
# 组 2：kinematics —— 7 个运动学特征（原版逻辑，逐位搬过来）
#   ax/ay 单帧加速度、vx3/vy3 三帧平均速度、turn_sin/turn_cos 转向率、dspeed 速度变化
# ----------------------------------------------------------------------------
def build_kinematics(c, cfg):
    x, y, dx, dy, flg = c['x'], c['y'], c['dx'], c['dy'], c['flg']
    eps = 1e-6
    # (1) 单帧加速度 = 速度的差分
    ax = torch.zeros_like(dx)
    ay = torch.zeros_like(dy)
    ax[1:] = dx[1:] - dx[:-1]
    ay[1:] = dy[1:] - dy[:-1]
    # (2) k=3 帧平均速度（历史不足则填 0）
    k3 = 3
    x3 = torch.zeros_like(x)
    y3 = torch.zeros_like(y)
    ok3 = torch.zeros_like(x)
    if k3 < x.shape[0]:
        x3[k3:] = x[:-k3]
        y3[k3:] = y[:-k3]
        ok3[k3:] = 1.0
    vx3 = (x - x3) / k3 * ok3 * flg
    vy3 = (y - y3) / k3 * ok3 * flg
    # (3) 转向率：当前速度方向 vs 3 帧前速度方向
    px = torch.zeros_like(dx)
    py = torch.zeros_like(dy)
    if k3 < dx.shape[0]:
        px[k3:] = dx[:-k3]
        py[k3:] = dy[:-k3]
    cross = dx * py - dy * px
    dot = dx * px + dy * py
    nrm = torch.sqrt(cross ** 2 + dot ** 2) + eps
    turn_sin = cross / nrm * flg
    turn_cos = dot / nrm * flg
    # (4) 速度大小变化，归一化到 [-1, 1]
    s_now = torch.sqrt(dx ** 2 + dy ** 2)
    s_past = torch.sqrt(px ** 2 + py ** 2)
    dspeed = (s_now - s_past) / (s_now + s_past + eps) * flg
    return torch.cat((ax, ay, vx3, vy3, turn_sin, turn_cos, dspeed), dim=2)


# ----------------------------------------------------------------------------
# 组 3：road_state —— 8 个道路/场景状态特征（本次新增）
#   都是"场景级聚合"，无法从单车特征推出来（这是它与 E2 灯语义的本质区别）
# ----------------------------------------------------------------------------
def build_road_state(c, cfg):
    fs, flg = c['fs'], c['flg']
    T, N = fs.shape[0], fs.shape[1]
    dt = fs.dtype
    dev = fs.device

    nz = int(cfg.get('NZones', 10))
    stop_px = float(cfg.get('stop_speed_px', 0.5))
    near_px = float(cfg.get('near_radius_px', 100.0))

    occ2 = (flg > 0).to(dt).squeeze(-1)                  # (T,N) 0/1
    z = fs[:, :, 2].long().clamp(0, nz - 1)              # (T,N) Zone 索引

    # --- (1) occ -------------------------------------------------------------
    occ = occ2.unsqueeze(-1)                             # (T,N,1)

    # --- (2)(3)(4) 同 zone 聚合：车辆数 / 停驶数 / 平均速度 ---------------------
    p = torch.cat((c['x'], c['y']), dim=2)               # (T,N,2)
    speed2 = torch.sqrt(c['dx'] ** 2 + c['dy'] ** 2).squeeze(-1) * occ2   # (T,N) px/frame
    stop2 = ((speed2 < stop_px) & (occ2 > 0)).to(dt)     # (T,N)

    def zone_scatter(src):
        acc = torch.zeros(T, nz, dtype=dt, device=dev)
        acc.scatter_add_(1, z, src)
        return acc

    cnt = zone_scatter(occ2)                             # (T,nz)
    nstop = zone_scatter(stop2)
    ssum = zone_scatter(speed2)

    cnt_slot = torch.gather(cnt, 1, z).unsqueeze(-1)     # (T,N,1)
    nstop_slot = torch.gather(nstop, 1, z).unsqueeze(-1)
    ssum_slot = torch.gather(ssum, 1, z).unsqueeze(-1)
    zone_cnt = cnt_slot / 10.0 * occ
    zone_stop = nstop_slot / 10.0 * occ
    zone_speed = (ssum_slot / cnt_slot.clamp(min=1.0)) / 10.0 * occ

    # --- (5) 局部拥挤：半径 near_px 内的车辆数 --------------------------------
    d = torch.cdist(p, p)                                # (T,N,N)
    near_mask = (d <= near_px) & (occ2.unsqueeze(-1) * occ2.unsqueeze(-2) > 0)
    near_cnt = (near_mask.sum(-1).to(dt) - 1.0).clamp(min=0.0).unsqueeze(-1)
    near_cnt = near_cnt / 10.0 * occ

    # --- (6) age：该槽连续被占用的帧数（累计重置，严格因果） -------------------
    idx = torch.arange(T, dtype=dt, device=dev).view(T, 1)
    last_zero = torch.where(occ2 < 0.5, idx.expand(T, N), torch.zeros(T, N, dtype=dt, device=dev))
    last_zero = torch.cummax(last_zero, dim=0)[0]
    age = torch.where(occ2 > 0.5, idx.expand(T, N) - last_zero + 1.0,
                      torch.zeros(T, N, dtype=dt, device=dev))
    age = (age / 30.0).unsqueeze(-1) * occ

    # --- (7) is_new：本帧刚出现 -------------------------------------------------
    prev = torch.cat((torch.zeros(1, N, dtype=dt, device=dev), occ2[:-1]), dim=0)
    is_new = (occ2 * (1.0 - prev)).unsqueeze(-1)

    # --- (8) frame_frac：当前帧在 scenelet 内的相对位置 -------------------------
    frac = (torch.arange(T, dtype=dt, device=dev).view(T, 1) / max(T - 1, 1)).expand(T, N)
    frac = frac.unsqueeze(-1) * occ

    return torch.cat((occ, zone_cnt, zone_stop, zone_speed, near_cnt, age, is_new, frac), dim=2)


# ----------------------------------------------------------------------------
# 注册表：通道数必须与 build 函数实际返回的宽度一致（build_features 会断言）
# 顺序 = 通道顺序，新增组请加在末尾（保证老通道号不漂）
# ----------------------------------------------------------------------------
REGISTRY = {
    'base':       (build_base,       8),
    'kinematics': (build_kinematics, 7),
    'road_state': (build_road_state, 8),
}

# 各组默认开关：base 永远开；其余不写 key 时用这里的默认值
DEFAULT_ON = {
    'base':       True,
    'kinematics': True,
    'road_state': False,   # 新特征默认关 → 不写 key 的配置行为与旧版完全一致
}


def is_enabled(name, cfg):
    if name == 'base':
        return True
    return bool(cfg.get('use_' + name, DEFAULT_ON.get(name, False)))


def enabled_groups(cfg):
    return [g for g in REGISTRY if is_enabled(g, cfg)]


def n_derived(cfg):
    return sum(REGISTRY[g][1] for g in enabled_groups(cfg))


def build_features(fs, cfg):
    """按注册顺序拼接：原始列 + 各启用组。返回 (T, N, 7 + Σn)。"""
    c = _common(fs, cfg)
    out = [fs]
    for name, (fn, n) in REGISTRY.items():
        if not is_enabled(name, cfg):
            continue
        ch = fn(c, cfg)
        if ch.size(-1) != n:
            raise RuntimeError(
                "特征组 '%s' 声明 %d 个通道，实际返回 %d 个（请同步修改 REGISTRY）"
                % (name, n, ch.size(-1)))
        out.append(ch)
    return torch.cat(out, dim=2)
