import math

import torch
from torch import nn
from torch.nn import functional as F

from my_gpt2.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """因果自注意力模块：每个 token 只能看到它自己和它之前的 token（不能看未来）"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head    # 注意力头数
        self.n_embd = config.n_embd    # 嵌入维度
        self.dropout = config.dropout  # dropout 概率

        # 一个线性层同时生成 Q、K、V 三个矩阵，输出维度是输入的 3 倍
        # 输入: (batch, seq_len, n_embd) → 输出: (batch, seq_len, 3 * n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # 把多头注意力的结果投影回原始维度
        # 输入: (batch, seq_len, n_embd) → 输出: (batch, seq_len, n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)   # 作用在注意力权重上
        self.resid_dropout = nn.Dropout(config.dropout)  # 作用在输出残差上

        # 因果掩码（下三角矩阵）：位置 i 只能看到 0..i，看不到 i+1 以后的位置
        # tril 生成下三角矩阵，1 表示可以看到，0 表示被遮住
        # 形状: (1, 1, block_size, block_size)，前两个维度用于广播到 (batch, head, ...)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (batch_size, seq_len, n_embd)
        batch_size, seq_len, channels = x.size()

        # 用一个线性层同时算出 Q、K、V，然后沿最后一维切成三份
        # 每份形状: (batch_size, seq_len, n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 每个注意力头负责的维度大小
        head_dim = channels // self.n_head  # 例如 16 / 2 = 8

        # 把 n_embd 维度拆分成 (n_head, head_dim)，再转置让 head 维度在前
        # 变换: (batch, seq_len, n_embd) → (batch, seq_len, n_head, head_dim) → (batch, n_head, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        # 计算注意力分数：Q @ K^T / sqrt(head_dim)
        # Q: (batch, n_head, seq_len, head_dim)
        # K^T: (batch, n_head, head_dim, seq_len)
        # 结果 att: (batch, n_head, seq_len, seq_len)，每行是该位置对所有位置的注意力分数
        # 除以 sqrt(head_dim) 是为了防止点积值过大导致 softmax 梯度消失
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

        # 应用因果掩码：把未来位置的分数设为 -inf，softmax 后变成 0
        # self.bias[:, :, :seq_len, :seq_len] 截取当前序列长度的掩码
        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        # softmax 把分数归一化成概率，dim=-1 表示对最后一维（所有位置）做归一化
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)  # 随机丢弃部分注意力权重，防止过拟合

        # 用注意力权重对 V 加权求和，得到每个位置的输出
        # att: (batch, n_head, seq_len, seq_len)
        # v:   (batch, n_head, seq_len, head_dim)
        # y:   (batch, n_head, seq_len, head_dim)
        y = att @ v

        # 把多头结果拼回原始形状: (batch, n_head, seq_len, head_dim) → (batch, seq_len, n_embd)
        # contiguous() 保证内存连续，view() 才能正常工作
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)

        # 最后通过投影层，并加 dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """前馈神经网络（Feed-Forward Network）：对每个 token 独立做非线性变换"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # 先升维到 4 倍（GPT-2 的经典设计），再降回原始维度
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # 升维
        self.gelu = nn.GELU()                                                        # 非线性激活函数
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias) # 降维
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)       # 升维: n_embd → 4 * n_embd
        x = self.gelu(x)       # 非线性激活
        x = self.c_proj(x)     # 降维: 4 * n_embd → n_embd
        return self.dropout(x) # dropout 防止过拟合


class Block(nn.Module):
    """一个完整的 Transformer Block = LayerNorm + 自注意力 + LayerNorm + MLP，每步都有残差连接"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)  # 注意力前的归一化
        self.attn = CausalSelfAttention(config)                      # 因果自注意力
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)  # MLP 前的归一化
        self.mlp = MLP(config)                                       # 前馈网络

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm 结构：先归一化再做操作，然后加回输入（残差连接）
        # 残差连接让梯度可以直接流过，解决深层网络梯度消失问题
        x = x + self.attn(self.ln_1(x))  # 注意力子层：归一化 → 注意力 → 残差相加
        x = x + self.mlp(self.ln_2(x))   # MLP 子层：归一化 → MLP → 残差相加
        return x


class GPT2(nn.Module):
    """完整的 GPT-2 语言模型"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # transformer 包含所有核心子模块
        self.transformer = nn.ModuleDict(
            {
                # wte: Word Token Embedding，把 token ID 映射成向量
                # 输入: token ID (整数) → 输出: (n_embd,) 的向量
                "wte": nn.Embedding(config.vocab_size, config.n_embd),

                # wpe: Word Position Embedding，把位置索引映射成向量
                # 让模型知道每个 token 在序列中的位置
                "wpe": nn.Embedding(config.block_size, config.n_embd),

                # 对词嵌入 + 位置嵌入之和做 dropout
                "drop": nn.Dropout(config.dropout),

                # n_layer 个 Transformer Block，堆叠起来形成深层网络
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

                # 最后一层归一化，作用在所有 Block 之后
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )

        # 语言模型头：把最终隐藏状态映射到词表大小，输出每个 token 的预测分数（logits）
        # 输入: (batch, seq_len, n_embd) → 输出: (batch, seq_len, vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：让词嵌入矩阵和输出投影矩阵共享同一组参数
        # 这是 GPT-2 论文的重要设计，减少参数量，并让两者语义对齐
        self.transformer.wte.weight = self.lm_head.weight

        # 对所有子模块做权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """权重初始化：线性层和嵌入层用均值 0、标准差 0.02 的正态分布初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # bias 初始化为 0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx: 输入 token ID，形状 (batch_size, seq_len)
        # targets: 目标 token ID，形状同 idx，用于计算 loss；推理时可以为 None
        _, seq_len = idx.size()

        # 检查序列长度不超过模型支持的最大上下文长度
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")

        # 生成位置索引 [0, 1, 2, ..., seq_len-1]，设备与输入一致
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)

        # 词嵌入：token ID → 向量，形状 (batch, seq_len, n_embd)
        tok_emb = self.transformer.wte(idx)

        # 位置嵌入：位置索引 → 向量，形状 (seq_len, n_embd)，会自动广播到 batch 维度
        pos_emb = self.transformer.wpe(pos)

        # 词嵌入 + 位置嵌入，让模型同时感知"是什么 token"和"在哪个位置"
        x = self.transformer.drop(tok_emb + pos_emb)

        # 依次通过 n_layer 个 Transformer Block
        for block in self.transformer.h:
            x = block(x)

        # 最终归一化
        x = self.transformer.ln_f(x)

        # 映射到词表，得到每个位置、每个 token 的预测分数
        # 形状: (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 计算交叉熵损失
            # logits.view(-1, vocab_size): 把 (batch, seq_len) 展平成 (batch*seq_len, vocab_size)
            # targets.view(-1): 把目标也展平成 (batch*seq_len,)
            # 结果是一个标量，表示平均预测损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()  # 推理时不需要计算梯度，节省显存和计算
    def generate(
        self,
        idx: torch.Tensor,      # 已有的 token 序列，形状 (batch, seq_len)
        max_new_tokens: int,    # 最多再生成多少个 token
        temperature: float = 1.0,  # 温度：>1 更随机，<1 更保守，=1 不变
        top_k: int | None = None,  # 只从概率最高的 top_k 个 token 中采样
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # 如果序列超过 block_size，截取最后 block_size 个 token 作为上下文
            idx_cond = idx[:, -self.config.block_size :]

            # 前向传播，只取最后一个位置的 logits（预测下一个 token）
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 形状: (batch, vocab_size)

            if top_k is not None:
                # 找出 top_k 个最大值，把其余位置设为 -inf（softmax 后为 0）
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            # softmax 转成概率分布
            probs = F.softmax(logits, dim=-1)

            # 按概率分布随机采样 1 个 token，形状: (batch, 1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 把新生成的 token 拼接到序列末尾，继续下一轮生成
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
