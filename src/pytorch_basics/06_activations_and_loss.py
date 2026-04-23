"""
06 - 激活函数与损失函数

ReLU / GELU / Softmax / CrossEntropy

运行：python src/pytorch_basics/06_activations_and_loss.py
"""

import math
import torch
import torch.nn.functional as F

# ── 激活函数 ──────────────────────────────────────────────────────────────────
print("── 激活函数对比 ──")
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"输入:  {x.tolist()}")
print(f"ReLU:  {F.relu(x).tolist()}  （负数截断为 0，正数不变）")
print(f"GELU:  {[round(v, 3) for v in F.gelu(x).tolist()]}  （比 ReLU 平滑，GPT 系列默认）")
print(f"Tanh:  {[round(v, 3) for v in torch.tanh(x).tolist()]}  （输出范围 (-1, 1)）")

# ── Softmax：把 logits 转成概率分布 ──────────────────────────────────────────
print("\n── Softmax ──")
logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
probs  = F.softmax(logits, dim=-1)
print(f"logits: {logits.tolist()}")
print(f"probs:  {[round(v, 4) for v in probs.tolist()]}")
print(f"概率之和: {probs.sum().item():.6f}  （始终为 1）")
print(f"最大 logit 对应最高概率: logit={logits.max().item()}, prob={probs.max().item():.4f}")

# temperature 对分布的影响
print("\n温度（temperature）对 softmax 的影响：")
for temp in [0.5, 1.0, 2.0]:
    p = F.softmax(logits / temp, dim=-1)
    print(f"  temperature={temp}: {[round(v, 4) for v in p.tolist()]}  "
          f"（{'更集中' if temp < 1 else '更均匀' if temp > 1 else '原始'}）")

# ── 交叉熵损失 ────────────────────────────────────────────────────────────────
print("\n── 交叉熵损失（CrossEntropy）──")

# 场景：vocab_size=5，预测两个位置的下一个 token
logits_ce = torch.tensor([
    [3.0, 1.0, 0.5, 0.2, 0.1],   # 位置 0 的预测分数
    [0.1, 0.2, 0.5, 1.0, 3.0],   # 位置 1 的预测分数
])
targets = torch.tensor([0, 4])   # 位置 0 目标是 token 0，位置 1 目标是 token 4

loss = F.cross_entropy(logits_ce, targets)
print(f"logits shape: {logits_ce.shape}")
print(f"targets:      {targets.tolist()}")
print(f"loss:         {loss.item():.4f}  （预测正确且置信度高，loss 很小）")

# 对比：预测完全错误时的 loss
targets_wrong = torch.tensor([4, 0])   # 全部预测错了
loss_wrong = F.cross_entropy(logits_ce, targets_wrong)
print(f"预测错误时 loss: {loss_wrong.item():.4f}  （更大）")

# 随机初始化模型时的期望 loss
vocab_size = 32
print(f"\n随机初始化时期望 loss ≈ ln({vocab_size}) = {math.log(vocab_size):.4f}")
print(f"  → 这就是为什么测试中 loss ≈ 3.3（接近 ln(32)={math.log(32):.4f}）")

# ── Top-k 采样：generate 中的关键步骤 ────────────────────────────────────────
print("\n── Top-k 采样 ──")
logits_gen = torch.tensor([0.1, 2.5, 0.3, 1.8, 0.9])   # 5 个 token 的分数
k = 3

values, indices = torch.topk(logits_gen, k)
print(f"原始 logits: {logits_gen.tolist()}")
print(f"Top-{k} 值:   {values.tolist()}")
print(f"Top-{k} 索引: {indices.tolist()}")

# 过滤掉非 top-k 的 token（设为 -inf）
filtered = logits_gen.clone()
filtered[logits_gen < values[-1]] = float("-inf")
probs_topk = F.softmax(filtered, dim=-1)
print(f"过滤后 probs: {[round(v, 4) for v in probs_topk.tolist()]}  （非 top-k 概率为 0）")

# 按概率采样一个 token
sampled = torch.multinomial(probs_topk, num_samples=1)
print(f"采样结果 token ID: {sampled.item()}")
