"""
05 - 常用 nn 层

Embedding / Linear / LayerNorm / Dropout

运行：python src/pytorch_basics/05_nn_layers.py
"""

import torch
import torch.nn as nn

# ── nn.Embedding：把 token ID 映射成向量 ─────────────────────────────────────
print("── nn.Embedding ──")
vocab_size = 10
embd_dim   = 4
embed = nn.Embedding(vocab_size, embd_dim)
print(f"权重矩阵 shape: {embed.weight.shape}  （vocab_size × embd_dim）")

idx_single = torch.tensor([0, 3, 7])              # 3 个 token ID
idx_batch  = torch.tensor([[0, 1, 2], [3, 4, 5]]) # (batch=2, seq=3)

out_single = embed(idx_single)
out_batch  = embed(idx_batch)
print(f"输入 {idx_single.shape} → 输出 {out_single.shape}  （每个 ID 变成一个 {embd_dim}维向量）")
print(f"输入 {idx_batch.shape} → 输出 {out_batch.shape}")
print(f"token 0 的向量: {embed.weight[0].data}")

# ── nn.Linear：全连接层 y = xW^T + b ─────────────────────────────────────────
print("\n── nn.Linear ──")
linear = nn.Linear(8, 16)
print(f"weight shape: {linear.weight.shape}  （out_features × in_features）")
print(f"bias shape:   {linear.bias.shape}")

x = torch.randn(4, 8)   # (batch=4, in=8)
y = linear(x)
print(f"输入 {x.shape} → 输出 {y.shape}")

# bias=False：不使用偏置项（lm_head 中常见）
linear_no_bias = nn.Linear(8, 16, bias=False)
print(f"bias=False 时，bias 为: {linear_no_bias.bias}")

# ── nn.LayerNorm：对最后一维做归一化 ──────────────────────────────────────────
print("\n── nn.LayerNorm ──")
ln = nn.LayerNorm(16)
x_ln = torch.randn(4, 8, 16)   # (batch, seq, embd)
y_ln = ln(x_ln)
print(f"输入 {x_ln.shape} → 输出 {y_ln.shape}  （形状不变）")

# 验证归一化效果：对 embd 维度，均值≈0，方差≈1
token0 = y_ln[0, 0]   # 取第一个 token 的 16 维向量
print(f"归一化后均值={token0.mean().item():.6f}  （应接近 0）")
print(f"归一化后标准差={token0.std().item():.4f}  （应接近 1）")

# ── nn.Dropout：随机丢弃神经元，防止过拟合 ────────────────────────────────────
print("\n── nn.Dropout ──")
dropout = nn.Dropout(p=0.5)   # 每个元素有 50% 概率被置为 0

x_drop = torch.ones(4, 8)

# 训练模式：dropout 生效
dropout.train()
y_drop_train = dropout(x_drop)
zero_ratio = (y_drop_train == 0).float().mean().item()
print(f"训练模式 - 置零比例: {zero_ratio:.2f}  （期望约 0.5）")
print(f"非零值: {y_drop_train[0]}  （非零的值会被放大到 1/0.5=2.0，保持期望不变）")

# 推理模式：dropout 关闭，原样输出
dropout.eval()
y_drop_eval = dropout(x_drop)
print(f"推理模式 - 输出全为: {y_drop_eval[0]}  （不丢弃）")
