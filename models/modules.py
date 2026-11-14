import torch
from torch import nn
import torch.nn.functional as F


################################
###         FFN LAYER       ### 前馈神经网络层
################################

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, out_dim): # nn.Module 要求把"这个层有什么组件"在 __init__ 里声明好
            super().__init__() 
            # 初始化nn.Module中的注册表，记录有哪些子层、哪些可训练参数
            # 只有调用了super().__init__()，才会将子类的参数注册到父类中，否则无法被优化器识别

            self.linear1 = nn.Linear(d_model, out_dim) # 全连接层
            self.relu = nn.LeakyReLU(negative_slope=0.01)
            # LeakyReLU 是带负斜率的 ReLU-普通 ReLU 在x<0时输出恒为 0，导致神经元死亡（再也不更新）-LeakyReLU 保留了一小部分负值梯度（0.01倍），让信息仍然可以流过。
            self.linear2 = nn.Linear(out_dim, out_dim)
            # 两个linear输出维度都为 out_dim，保证输入输出维度一致，便于残差连接

    def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))
    

################################
###         DAAG LAYER       ### 决策感知图注意力层
################################
class DAAG_Layer(nn.Module):
    r"""
    ## Decision-Aware Attention Graph Transformer (DAAGT) Layer
    This is a single graph attention v2 layer.
    author: @Behzad Abdi
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int, n_nodes: int,
                concat: bool = True,
                dropout: float = 0.01,
                leaky_relu_negative_slope: float = 0.01,
                share_weights: bool = False): # 初始化
        r"""
        * `in_features`, $F$, is the number of input features per node
        输入特征维度（每个节点的特征数）
        * `out_features`, $F'$, is the number of output features per node
        输出特征维度（每个节点的特征数）
        * `n_heads`, $K$, is the number of attention heads
        注意力头的数量
        * `is_concat` whether the multi-head results should be concatenated or averaged
        是否将多头结果连接或平均
        * `dropout` is the dropout probability
        Dropout概率
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        Leaky ReLU 激活函数的负斜率
        * `share_weights` if set to `True`, the same matrix will be applied to the source and the target node of every edge
        Q、K 是否共享同一个线性变换权重
        """
        super().__init__()
        # 基础缓存-把外部传入超参保存为类内部变量，前向传播 forward 里随时调用
        self.is_concat = concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        self.n_nodes = n_nodes

        # Calculate the number of dimensions per head 每个头的维度数
        if concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features
            # concat=True 时，每头的维度=256 ÷ 4=64 维，4个头拼接回256维

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        # linear_l：Query (Q) 变换，自身节点特征映射
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        # linear_r：Key (K) 变换，邻居节点特征映射
        if share_weights: # share_weights为true时，Key 的线性层复用 Query 的权重
            self.linear_r = self.linear_l
        else: # Q/K使用各自的权重
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads,  bias=False)
        # linear_v：Value (V) 变换，用于加权聚合邻居信息
        self.linear_v = nn.Linear(in_features, self.n_hidden * n_heads,  bias=False)
        # Linear layer to compute attention score $e_{ij}$
        # 三个线性变换层：输出维度统一为 n_hidden * n_heads，方便拆分成多头并行计算
        # Q/K/V 每个都是 256 → 256

        # =====图注意力特有的层=====
        self.att = nn.Linear(n_nodes + self.n_hidden, self.n_hidden)
        # 融合图节点数量信息 + 单头特征的注意力映射层
        self.att_mh_1 = nn.Parameter(torch.Tensor(1, n_heads, n_nodes))
        # 告诉pytorch“这个张量要在训练中更新”
        self.scoreatt = nn.Linear(1,n_nodes) # 把 1 个标量映射到 32 个节点的权重
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.FF = FeedForwardNetwork(self.n_hidden, self.n_hidden) # 内部 FFN
        self.Rezero = nn.Parameter(torch.zeros(self.n_hidden)) # 可学习残差系数，初始为 0
        self.LN = nn.LayerNorm(n_nodes)
        # 层归一化，在节点维度做标准化，稳定图注意力训练分布
        # =========================

    def forward(self, h: torch.Tensor, Adj_mat: torch.Tensor):

        # Number of nodes
        B, SL, n_nodes, _ = h.shape # 获取形状
        Adj_mat = Adj_mat - torch.eye(n_nodes).to(Adj_mat.device).repeat(B, SL, 1, 1)
        # 移除自连接（对角线）
        
        # 原始的邻接矩阵对角线是 e^{-0} = 1（自己到自己的距离为 0）。
        # 减去 1 后对角线 ≈ 0。后面有一步 masked_fill(score, -1e6) 会屏蔽值小于 0.25 的位置
        # 如果不移除自连接，节点自己会被屏蔽掉，自己不能关注自己。

        Adj_mat = Adj_mat.unsqueeze(-2) < 0.25 # 距离较远的车辆对，它们的注意力分数被屏蔽为极小的值
        q = self.linear_l(h).view(B, SL, n_nodes, self.n_heads, self.n_hidden)
        k = self.linear_r(-h).view(B, SL, n_nodes, self.n_heads, self.n_hidden)
        # Key 的输入是 -h（取负），因为在图注意力中，邻居节点的特征是负相关的，取负可以增强注意力机制的区分度
        v = self.linear_v(h).view(B, SL, n_nodes, self.n_heads, self.n_hidden)
        
        score = torch.einsum("bsihf, bsjhf->bsihj", [q, k])/ ((self.n_hidden*self.n_heads*n_nodes)**0.5)
        # 计算原始注意力分数
        score = score.masked_fill(Adj_mat, -1e6) # 应用邻接掩码-距离远的车辆对，注意力分数被压制到几乎为 0
        score = torch.exp(score).sum(-1).unsqueeze(-1)
        # 聚合注意力分数
        score = self.scoreatt(score)
        # 重新分配注意力权重-self.scoreatt 是一个 nn.Linear(1, 32)。它将每个节点的聚合信息量（1个标量）映射到32个节点的权重
        
        score = self.LN(score) # LayerNorm(32): 形状不变 对32个节点维度做归一化
        score = F.softmax(score, dim=-1) # 在最后一个维度（源节点j）做softmax
        # 归一化

        score = torch.cat((score, v), dim=-1)
        score = self.att(score)
        # 融合注意力权重和值
        scoreFF = self.FF(score) # FFN
        score = score + self.Rezero * scoreFF # 残差连接
        
        if self.is_concat: # concat=True：4 个头的 64 维输出拼接成 256 维
            return score.flatten(-2) # 从倒数第2维开始展平
        else:
            return score.mean(-2)
        

class TemporalConv(nn.Module): # 时序卷积层
    def __init__(self, hidden_size, sl):
        super(TemporalConv, self).__init__()
        c1 = sl//2 +1
        c2 = sl//3 +1

        self.conv1 = nn.Conv1d(sl+2,  sl+2, 7, 2, 3)
        self.conv2 = nn.Conv1d(c1,  sl+2, 7, 2, 3)
        self.conv3 = nn.Conv1d(c2,  sl+2, 7, 2, 3)
        # 三个并行卷积，每个处理不同时间粒度的信息
        # conv1 输入：所有时间步，conv2 输入：每隔一个时间步，conv3 输入：每隔两个时间步
        # 三个结果相加，实现多尺度时序特征融合

        self.out = nn.Linear(hidden_size//2, hidden_size//2)
        self.LN = nn.LayerNorm(hidden_size)
        self.Rezero = nn.Parameter(torch.zeros(hidden_size//2))
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, h):
        x0 = F.relu(self.conv1(h)) # 全分辨率
        x1 = F.relu(self.conv2(h[:,1::2])) # 2倍下采样
        x2 = F.relu(self.conv3(h[:,1::3])) # 3倍下采样
        x = self.dropout(x0 + x1 + x2) # 三路相加 + Dropout
        x = self.out(x)*self.Rezero + x # Rezero 残差
        return x
    
def positional_encoding(x, d_model): # 正余弦位置编码 纯函数，没有可学习参数

        # result.shape = (seq_len, d_model)
        result = torch.zeros(
            (x.size(1), d_model),
            dtype=torch.float,
            requires_grad=False
        )

        # pos.shape = (seq_len, 1)
        pos = torch.arange(0, x.size(1)).unsqueeze(1)

        # dim.shape = (d_model)
        dim = torch.arange(0, d_model, step=2)

        # Sine for even positions, cosine for odd dimensions
        result[:, 0::2] = torch.sin(pos / (10_000 ** (dim / d_model)))
        result[:, 1::2] = torch.cos(pos / (10_000 ** (dim / d_model)))
        return result.to(x.device)
        # 作用：让模型知道"第 0 帧"和"第 14 帧"在时间上的距离。不同频率的正余弦波构成时间位置指纹。



def target_mask(trgt, num_head = 0, device="cuda:3"):
    # 自回归解码器需要的因果掩码，确保预测时不会看到未来的信息
    B, SL, Nnodes, _= trgt.size()
    mask = 1- torch.triu(torch.ones(SL, SL, device=trgt.device), diagonal=1)# Upper triangular matrix
    return mask == 0

def target_mask0(trgt, num_head, device="cuda:3"): # 比 target_mask 多了一个功能：结合数据有效性
    B, SL, Nnodes, _= trgt.size()
    mask = 1- torch.triu(torch.ones(SL, SL, device=trgt.device), diagonal=1).unsqueeze(0).unsqueeze(0)  # Upper triangular matrix
    mask = mask.unsqueeze(4) * (trgt[:,:,:,1] != 0).unsqueeze(1).unsqueeze(3)
    if num_head > 1:
        mask = mask.repeat_interleave(num_head,dim=0)
    return mask == 0

def create_src_mask(src): # src => [B, SL, Nnodes, Features] 源序列掩码
    mask = src[:,:,:, 1] == 0
    return mask.permute(0,2,1).reshape(-1, src.size(1)) 
    # out => [B* Nnodes, SL]

def attach_sos_eos(src, sos, eos): # src => [B, SL0, Nnodes, Features] 序列起止标记
    sos = sos.to(src.device)  # 确保与 src 同设备
    eos = eos.to(src.device)  # 确保与 src 同设备
    return torch.cat((sos.repeat(src.size(0),1,1,1), src, eos.repeat(src.size(0),1,1,1)), dim=1) # out => [B, SL0+2, Nnodes, Features]

