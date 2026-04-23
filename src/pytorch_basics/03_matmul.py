"""
03 - 矩阵乘法

@ 运算符 / torch.matmul / 批量矩阵乘法

运行：python src/pytorch_basics/03_matmul.py
"""

import torch

# ── 2D 矩阵乘法：(M, K) @ (K, N) → (M, N) ──────────────────────────────────
print("── 2D 矩阵乘法 ──")
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # (2, 2)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])   # (2, 2)
c = a @ b
print(f"a:\n{a}")
print(f"b:\n{b}")
print(f"a @ b:\n{c}")   # [[19, 22], [43, 50]]

# ── Linear 层的本质：xW^T + b ─────────────────────────────────────────────────
print("\n── Linear 层的本质 ──")
# nn.Linear(in=4, out=8) 等价于 x @ W.T + b
x = torch.randn(3, 4)        # (batch=3, in=4)
W = torch.randn(8, 4)        # weight shape: (out=8, in=4)
b = torch.zeros(8)

y_manual = x @ W.T + b      # 手动计算
print(f"x:      {x.shape}")
print(f"W:      {W.shape}")
print(f"x@W.T:  {(x @ W.T).shape}  →  等价于 nn.Linear(4, 8)(x)")

# ── 批量矩阵乘法：(B, M, K) @ (B, K, N) → (B, M, N) ────────────────────────
print("\n── 批量矩阵乘法 ──")
a_batch = torch.ones(2, 3, 4)
b_batch = torch.ones(2, 4, 5)
c_batch = a_batch @ b_batch
print(f"(2,3,4) @ (2,4,5) = {c_batch.shape}")   # (2, 3, 5)

# ── 注意力分数计算：Q @ K^T / sqrt(head_dim) ─────────────────────────────────
print("\n── 注意力分数（Q @ K^T）──")
batch, n_head, seq_len, head_dim = 2, 4, 8, 16

q = torch.randn(batch, n_head, seq_len, head_dim)
k = torch.randn(batch, n_head, seq_len, head_dim)

# k.transpose(-2, -1)：把 (batch, n_head, seq, head_dim) 转成 (batch, n_head, head_dim, seq)
att = q @ k.transpose(-2, -1)
print(f"Q:        {q.shape}")
print(f"K^T:      {k.transpose(-2, -1).shape}")
print(f"Q @ K^T:  {att.shape}")   # (batch, n_head, seq, seq)

import math
att_scaled = att * (1.0 / math.sqrt(head_dim))
print(f"缩放后 att: {att_scaled.shape}  （除以 sqrt({head_dim})={math.sqrt(head_dim):.2f}，防止梯度消失）")
