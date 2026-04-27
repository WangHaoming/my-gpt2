"""
02 - Tensor 的形状操作

view / reshape / transpose / squeeze / unsqueeze

运行：python src/pytorch_basics/02_tensor_shape_ops.py
"""

import torch

x = torch.arange(24, dtype=torch.float)   # [0, 1, ..., 23]，shape (24,)
print(f"原始 shape: {x.shape}  →  {x}")

# ── view：重新解释形状，不复制数据（内存必须连续）────────────────────────────
print("\n── view ──")
x_2d = x.view(4, 6)       # 4行6列
x_3d = x.view(2, 3, 4)    # 2个block，每个3行4列
print(f"view(4,6):   {x_2d.shape}")
print(f"view(2,3,4): {x_3d.shape}")
print(f"view(2,-1):  {x.view(2, -1).shape}  （-1 表示自动推断）")
print(f"view(x): {x.view(1,8,3).shape} -> {x.view(1,8,3)}")
print(f"view(x): {x.view(2,4,3).shape} -> {x.view(2,4,3)}")
print(f"view(x): {x.view(3,1,8).shape} -> {x.view(3,1,8)}")

# ── transpose：交换两个维度 ──────────────────────────────────────────────────
print("\n── transpose ──")
# 在注意力机制中：(batch, seq, n_head, head_dim) → (batch, n_head, seq, head_dim)
t = torch.zeros(2, 8, 4, 16)      # (batch, seq, n_head, head_dim)
t2 = t.transpose(1, 2)            # 交换 dim1 和 dim2
print(f"原始:         {t.shape}")
print(f"transpose(1,2): {t2.shape}")

# transpose 后内存不连续，view 之前需要 contiguous()
# 2个block，每个4行，-1表示自动推断该维度大小
t3 = t2.contiguous().view(2, 4, -1) 

print(f"contiguous().view(2,4,-1): {t3.shape}")

# 对最后两维转置：在计算 Q @ K^T 时常用
q = torch.zeros(2, 4, 8, 16)   # (batch, n_head, seq, head_dim)
k = torch.zeros(2, 4, 8, 16)
att = q @ k.transpose(-2, -1)  # -> (batch, n_head, head_dim, seq)
print(f"\nQ @ K^T: {q.shape} @ {k.transpose(-2,-1).shape} = {att.shape}")

# ── unsqueeze / squeeze：增减大小为 1 的维度 ──────────────────────────────────
print("\n── unsqueeze / squeeze ──")
y = torch.tensor([1.0, 2.0, 3.0])   # shape (3,)
print(f"原始:          {y.shape}")
print(f"unsqueeze(0):  {y.unsqueeze(0).shape}  → 增加 batch 维度")
print(f"unsqueeze(1):  {y.unsqueeze(1).shape}  → 增加列维度")
print(f"squeeze后:     {y.unsqueeze(0).squeeze(0).shape}  → 还原")

# ── 实际场景：位置嵌入广播 ────────────────────────────────────────────────────
print("\n── 广播示例（位置嵌入 + 词嵌入）──")
tok_emb = torch.zeros(4, 8, 16)   # (batch=4, seq=8, n_embd=16)
pos_emb = torch.zeros(8, 16)      # (seq=8, n_embd=16)，没有 batch 维度
# PyTorch 自动广播：pos_emb 被视作 (1, 8, 16)，与 tok_emb 相加
x = tok_emb + pos_emb
print(f"tok_emb {tok_emb.shape} + pos_emb {pos_emb.shape} = {x.shape}")
