"""
07 - 因果注意力掩码

masked_fill / tril / 完整的注意力权重计算过程

运行：python src/pytorch_basics/07_causal_mask.py
"""

import math
import torch
import torch.nn.functional as F

seq_len = 5

# ── 下三角掩码 ────────────────────────────────────────────────────────────────
print("── 下三角掩码（tril）──")
mask = torch.tril(torch.ones(seq_len, seq_len))
print(f"mask（1=可见，0=遮住未来）:\n{mask.int()}")
print("位置 i 只能看到 0..i，不能看到 i+1 以后")

# ── masked_fill：把掩码为 0 的位置填 -inf ────────────────────────────────────
print("\n── masked_fill ──")
att = torch.zeros(seq_len, seq_len)   # 模拟全为 0 的注意力分数
att_masked = att.masked_fill(mask == 0, float("-inf"))
print(f"masked_fill 后:\n{att_masked}")

# ── softmax 后，-inf 变成 0 ───────────────────────────────────────────────────
print("\n── softmax 后的注意力权重 ──")
att_prob = F.softmax(att_masked, dim=-1)
print(f"注意力权重:\n{att_prob.round(decimals=3)}")
print(f"位置 0 只能看自己，权重为: {att_prob[0].tolist()}")
print(f"位置 4 能看到所有，权重为: {[round(v,3) for v in att_prob[4].tolist()]}")

# ── 完整的多头注意力计算流程 ──────────────────────────────────────────────────
print("\n── 完整注意力计算流程 ──")
batch_size = 2
n_head     = 2
head_dim   = 8

# 随机初始化 Q、K、V（实际中由线性层生成）
q = torch.randn(batch_size, n_head, seq_len, head_dim)
k = torch.randn(batch_size, n_head, seq_len, head_dim)
v = torch.randn(batch_size, n_head, seq_len, head_dim)

# 步骤 1：计算注意力分数
att = q @ k.transpose(-2, -1)                      # (batch, n_head, seq, seq)
att = att * (1.0 / math.sqrt(head_dim))            # 缩放，防止点积过大
print(f"步骤1 - Q @ K^T 缩放后: {att.shape}")

# 步骤 2：应用因果掩码（截取当前 seq_len）
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
causal_mask = causal_mask.view(1, 1, seq_len, seq_len)   # 广播到 batch 和 head 维度
att = att.masked_fill(causal_mask == 0, float("-inf"))
print(f"步骤2 - masked_fill 后: {att.shape}  （未来位置为 -inf）")

# 步骤 3：softmax 归一化
att = F.softmax(att, dim=-1)
print(f"步骤3 - softmax 后: {att.shape}  （每行和为 1）")

# 步骤 4：加权求和 V
y = att @ v                                        # (batch, n_head, seq, head_dim)
print(f"步骤4 - att @ V: {y.shape}")

# 步骤 5：合并多头，还原形状
y = y.transpose(1, 2).contiguous()                # (batch, seq, n_head, head_dim)
y = y.view(batch_size, seq_len, n_head * head_dim)# (batch, seq, n_embd)
print(f"步骤5 - 合并多头后: {y.shape}  （n_head × head_dim = {n_head}×{head_dim}={n_head*head_dim}）")
