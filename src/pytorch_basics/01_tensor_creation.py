"""
01 - Tensor 的创建方式

运行：python src/pytorch_basics/01_tensor_creation.py
"""

import torch

# ── 从 Python 数据创建 ──────────────────────────────────────────────────────
a = torch.tensor([1.0, 2.0, 3.0])
print(f"从列表创建:  {a}")
print(f"  shape={a.shape}, dtype={a.dtype}")

b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\n从嵌套列表创建:\n{b}")
print(f"  shape={b.shape}, dtype={b.dtype}")

# ── 常用初始化方式 ────────────────────────────────────────────────────────────
print("\n── 全零 / 全一 ──")
zeros = torch.zeros(2, 3)       # 2行3列，全为 0.0
ones  = torch.ones(2, 3)        # 2行3列，全为 1.0
print(f"zeros:\n{zeros}")
print(f"ones:\n{ones}")

print("\n── 随机数 ──")
rand_uniform = torch.rand(2, 3)             # 均匀分布 [0, 1)
rand_normal  = torch.randn(2, 3)            # 正态分布，均值 0，标准差 1
rand_int     = torch.randint(0, 10, (2, 3)) # 整数，[0, 10) 范围
print(f"rand (uniform):\n{rand_uniform}")
print(f"randn (normal):\n{rand_normal}")
print(f"randint(0,10):\n{rand_int}")

print("\n── 等差序列 ──")
seq = torch.arange(0, 10, 2)   # 从 0 开始，步长 2，到 10（不含）→ [0,2,4,6,8]
print(f"arange(0,10,2): {seq}")

print("\n── 数据类型 dtype ──")
x_float32 = torch.tensor([1.0])        # 默认 float32
x_float64 = torch.tensor([1.0], dtype=torch.float64)
x_int64   = torch.tensor([1], dtype=torch.long)   # long = int64，Embedding 层要求
print(f"float32: {x_float32.dtype}")
print(f"float64: {x_float64.dtype}")
print(f"long:    {x_int64.dtype}")
