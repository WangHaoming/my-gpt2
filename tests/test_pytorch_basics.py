"""
PyTorch 基础用法学习文件。

运行方式：pytest tests/test_pytorch_basics.py -s -v
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tensor 基础
# ─────────────────────────────────────────────────────────────────────────────

def test_tensor_creation() -> None:
    """创建 Tensor 的常见方式。"""

    # 从 Python 列表创建
    a = torch.tensor([1.0, 2.0, 3.0])
    print(f"从列表创建: {a}, shape={a.shape}, dtype={a.dtype}")

    # 全零 / 全一
    zeros = torch.zeros(2, 3)   # 2行3列，全为 0
    ones = torch.ones(2, 3)     # 2行3列，全为 1
    print(f"zeros:\n{zeros}")
    print(f"ones:\n{ones}")

    # 正态分布随机数（均值 0，标准差 1）
    rand_normal = torch.randn(2, 3)
    print(f"randn:\n{rand_normal}")

    # 均匀分布随机整数：[low, high) 范围内的随机整数
    rand_int = torch.randint(0, 10, (2, 3))
    print(f"randint(0,10):\n{rand_int}")

    assert a.shape == (3,)
    assert zeros.shape == (2, 3)


def test_tensor_shape_ops() -> None:
    """Tensor 的形状操作：view / reshape / transpose / squeeze / unsqueeze。"""

    x = torch.arange(24, dtype=torch.float)  # [0, 1, 2, ..., 23]，形状 (24,)
    print(f"原始: {x.shape}")

    # view：重新解释形状，不复制数据（要求内存连续）
    x_3d = x.view(2, 3, 4)     # 2 batch, 3 行, 4 列
    print(f"view(2,3,4): {x_3d.shape}")

    # transpose：交换两个维度
    x_t = x_3d.transpose(1, 2)   # (2, 4, 3)
    print(f"transpose(1,2): {x_t.shape}")

    # contiguous：transpose 后内存可能不连续，view 之前需要先调用
    x_back = x_t.contiguous().view(2, -1)   # -1 表示自动推断该维度大小
    print(f"contiguous().view(2,-1): {x_back.shape}")

    # unsqueeze：在指定位置插入大小为 1 的维度
    y = torch.tensor([1.0, 2.0, 3.0])   # shape (3,)
    y_col = y.unsqueeze(0)               # shape (1, 3)
    y_row = y.unsqueeze(1)               # shape (3, 1)
    print(f"unsqueeze(0): {y_col.shape}, unsqueeze(1): {y_row.shape}")

    # squeeze：删除大小为 1 的维度
    y_back = y_col.squeeze(0)            # shape (3,)
    print(f"squeeze(0): {y_back.shape}")

    assert x_3d.shape == (2, 3, 4)
    assert x_t.shape == (2, 4, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 矩阵运算
# ─────────────────────────────────────────────────────────────────────────────

def test_matmul() -> None:
    """矩阵乘法：@ 运算符 / torch.matmul。"""

    # 2D 矩阵乘法：(M, K) @ (K, N) → (M, N)
    a = torch.ones(3, 4)
    b = torch.ones(4, 5)
    c = a @ b              # 等价于 torch.matmul(a, b)
    print(f"(3,4) @ (4,5) = {c.shape}")   # (3, 5)

    # 批量矩阵乘法：(B, M, K) @ (B, K, N) → (B, M, N)
    a_batch = torch.ones(2, 3, 4)
    b_batch = torch.ones(2, 4, 5)
    c_batch = a_batch @ b_batch
    print(f"(2,3,4) @ (2,4,5) = {c_batch.shape}")   # (2, 3, 5)

    # 转置最后两个维度：(B, M, N) → (B, N, M)
    c_t = c_batch.transpose(-2, -1)
    print(f"transpose(-2,-1): {c_t.shape}")   # (2, 5, 3)

    assert c.shape == (3, 5)
    assert c_batch.shape == (2, 3, 5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Autograd：自动求导
# ─────────────────────────────────────────────────────────────────────────────

def test_autograd() -> None:
    """PyTorch 自动微分：requires_grad + backward。"""

    # requires_grad=True：告诉 PyTorch 需要对这个 tensor 求导
    x = torch.tensor(3.0, requires_grad=True)

    # 定义一个计算图：y = x^2 + 2x + 1
    y = x ** 2 + 2 * x + 1
    print(f"x={x.item()}, y={y.item()}")

    # 反向传播：计算 dy/dx
    y.backward()

    # x.grad 就是 dy/dx 在 x=3 处的值
    # dy/dx = 2x + 2，当 x=3 时 = 8
    print(f"dy/dx at x=3: {x.grad.item()}")   # 应该是 8.0
    assert abs(x.grad.item() - 8.0) < 1e-5

    # torch.no_grad()：推理时不需要梯度，节省内存
    with torch.no_grad():
        z = x ** 2
        print(f"no_grad 下 z.requires_grad = {z.requires_grad}")   # False


# ─────────────────────────────────────────────────────────────────────────────
# 4. 常用 nn 层
# ─────────────────────────────────────────────────────────────────────────────

def test_embedding() -> None:
    """nn.Embedding：把整数 token ID 映射成向量。"""

    vocab_size = 10   # 词表大小
    embd_dim = 4      # 每个 token 的向量维度

    embed = nn.Embedding(vocab_size, embd_dim)
    print(f"Embedding 权重形状: {embed.weight.shape}")   # (10, 4)

    # 输入：token ID 张量
    idx = torch.tensor([0, 3, 7])          # 3 个 token
    out = embed(idx)
    print(f"输入 shape: {idx.shape}, 输出 shape: {out.shape}")   # (3, 4)

    # 批量输入
    idx_batch = torch.tensor([[0, 1, 2], [3, 4, 5]])   # (batch=2, seq=3)
    out_batch = embed(idx_batch)
    print(f"批量输出 shape: {out_batch.shape}")   # (2, 3, 4)

    assert out.shape == (3, 4)
    assert out_batch.shape == (2, 3, 4)


def test_linear() -> None:
    """nn.Linear：全连接层，y = xW^T + b。"""

    # in_features=8, out_features=16
    linear = nn.Linear(8, 16)
    print(f"weight shape: {linear.weight.shape}")   # (16, 8)，注意是转置关系
    print(f"bias shape:   {linear.bias.shape}")     # (16,)

    x = torch.randn(4, 8)     # (batch=4, features=8)
    y = linear(x)
    print(f"输入 {x.shape} → 输出 {y.shape}")   # (4, 16)

    assert y.shape == (4, 16)


def test_layer_norm() -> None:
    """nn.LayerNorm：对最后一个维度做归一化，让每个 token 的特征分布稳定。"""

    # normalized_shape=8：对最后一维（大小 8）做归一化
    ln = nn.LayerNorm(8)

    x = torch.randn(4, 6, 8)   # (batch, seq_len, embd)
    y = ln(x)
    print(f"LayerNorm 输入 {x.shape} → 输出 {y.shape}")   # 形状不变

    # 验证归一化效果：每个 token 的 8 个特征，均值≈0，方差≈1
    mean = y[0, 0].mean().item()
    std  = y[0, 0].std().item()
    print(f"归一化后均值={mean:.4f}, 标准差={std:.4f}")

    assert y.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# 5. 激活函数与 softmax
# ─────────────────────────────────────────────────────────────────────────────

def test_activations() -> None:
    """常用激活函数：ReLU / GELU / Softmax。"""

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # ReLU：负数变 0，正数不变
    relu_out = F.relu(x)
    print(f"ReLU:    {relu_out}")

    # GELU：比 ReLU 更平滑，GPT 系列默认使用
    gelu_out = F.gelu(x)
    print(f"GELU:    {gelu_out}")

    # Softmax：把任意实数向量变成概率分布（和为 1），dim=-1 表示在最后一维做
    logits = torch.tensor([1.0, 2.0, 3.0])
    probs = F.softmax(logits, dim=-1)
    print(f"Softmax: {probs}，和={probs.sum().item():.4f}")   # 和为 1

    assert abs(probs.sum().item() - 1.0) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# 6. 损失函数
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_entropy() -> None:
    """交叉熵损失：语言模型训练中衡量预测分布与真实 token 之间的差距。"""

    # logits：模型输出的原始分数，形状 (batch * seq_len, vocab_size)
    # 不需要先做 softmax，F.cross_entropy 内部会处理
    logits = torch.tensor([
        [2.0, 1.0, 0.1],   # 第 0 个位置的预测分数
        [0.5, 2.5, 0.3],   # 第 1 个位置的预测分数
    ])

    # targets：真实 token ID，形状 (batch * seq_len,)
    targets = torch.tensor([0, 1])   # 第 0 位置目标是 token 0，第 1 位置目标是 token 1

    loss = F.cross_entropy(logits, targets)
    print(f"cross_entropy loss = {loss.item():.4f}")

    # 随机预测时（均匀分布），loss ≈ ln(vocab_size) = ln(3) ≈ 1.099
    # 当预测正确且置信度高时，loss 接近 0
    print(f"完美预测期望 loss ≈ 0，随机预测 loss ≈ {math.log(3):.4f}")

    assert loss.ndim == 0   # loss 是标量


# ─────────────────────────────────────────────────────────────────────────────
# 7. masked_fill：因果注意力掩码的核心操作
# ─────────────────────────────────────────────────────────────────────────────

def test_masked_fill() -> None:
    """masked_fill：把满足条件的位置替换成指定值，用于实现因果掩码。"""

    seq_len = 4
    # 下三角矩阵：1 表示可见，0 表示被遮住（未来位置）
    mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"因果掩码:\n{mask}")

    # 模拟注意力分数矩阵
    att = torch.zeros(seq_len, seq_len)

    # 把 mask==0 的位置（未来位置）填成 -inf，softmax 后变成 0
    att_masked = att.masked_fill(mask == 0, float("-inf"))
    print(f"mask 后的注意力分数:\n{att_masked}")

    # softmax 后，-inf 的位置变成 0，其余位置归一化
    att_prob = F.softmax(att_masked, dim=-1)
    print(f"softmax 后的注意力权重:\n{att_prob}")

    # 第 0 行（位置 0）只能看到自己，概率全给自己
    assert abs(att_prob[0, 0].item() - 1.0) < 1e-5
    # 第 3 行（位置 3）能看到所有 4 个位置，每个概率为 0.25
    assert abs(att_prob[3, 0].item() - 0.25) < 1e-5
