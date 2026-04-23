"""
04 - 自动求导（Autograd）

requires_grad / backward / grad / no_grad

运行：python src/pytorch_basics/04_autograd.py
"""

import torch

# ── 基本用法：标量求导 ────────────────────────────────────────────────────────
print("── 标量求导 ──")
x = torch.tensor(3.0, requires_grad=True)   # requires_grad=True：跟踪这个 tensor 的计算

y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1，构建计算图
print(f"x = {x.item()}")
print(f"y = x² + 2x + 1 = {y.item()}")

y.backward()   # 反向传播：计算 dy/dx

# dy/dx = 2x + 2，在 x=3 时 = 8
print(f"dy/dx at x=3 = {x.grad.item()}  （理论值: 2×3+2 = 8）")

# ── 向量求导：grad 是同形状的梯度 tensor ──────────────────────────────────────
print("\n── 向量求导 ──")
x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y2 = (x2 ** 2).sum()   # y = x₀² + x₁² + x₂²，先 sum 变成标量再 backward
y2.backward()
print(f"x  = {x2.data}")
print(f"y  = sum(x²) = {y2.item()}")
print(f"dy/dx = 2x = {x2.grad}  （每个位置的梯度）")

# ── 多次 backward 之前要清空梯度 ─────────────────────────────────────────────
print("\n── 梯度累积问题 ──")
x3 = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y3 = x3 ** 2
    y3.backward()
    print(f"第 {i+1} 次 backward 后，x3.grad = {x3.grad.item()}  （梯度在累积！）")
    # 如果不清空，梯度会一直叠加

# 正确做法：每次 backward 前清空
x4 = torch.tensor(2.0, requires_grad=True)
for i in range(3):
    if x4.grad is not None:
        x4.grad.zero_()   # 手动清零（训练循环中用 optimizer.zero_grad()）
    y4 = x4 ** 2
    y4.backward()
    print(f"清零后第 {i+1} 次，x4.grad = {x4.grad.item()}  （始终为 4.0）")

# ── torch.no_grad()：推理时关闭梯度追踪 ──────────────────────────────────────
print("\n── torch.no_grad() ──")
x5 = torch.tensor(3.0, requires_grad=True)

with torch.no_grad():
    z = x5 ** 2
    print(f"no_grad 下，z.requires_grad = {z.requires_grad}")   # False
    print(f"z = {z.item()}  （能计算，但不会构建计算图，节省内存）")

# ── 计算图可视化（打印 grad_fn）────────────────────────────────────────────────
print("\n── 计算图 grad_fn ──")
a = torch.tensor(2.0, requires_grad=True)
b = a * 3        # MulBackward
c = b + 1        # AddBackward
d = c ** 2       # PowBackward
print(f"a.grad_fn: {a.grad_fn}  （叶子节点，无 grad_fn）")
print(f"b.grad_fn: {b.grad_fn}")
print(f"c.grad_fn: {c.grad_fn}")
print(f"d.grad_fn: {d.grad_fn}")
