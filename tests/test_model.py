import torch

from my_gpt2 import GPT2, GPTConfig


def test_gpt2_forward_shapes() -> None:
    # GPTConfig 是模型的超参数配置（frozen dataclass）
    config = GPTConfig(
        vocab_size=32,   # 词表大小：模型能认识的不同 token 数量（真实 GPT-2 是 50257）
        block_size=8,    # 上下文长度：每次输入的最大 token 数量（真实 GPT-2 是 1024）
        n_layer=2,       # Transformer 层数：堆叠多少个 Block（真实 GPT-2 是 12）
        n_head=2,        # 注意力头数：Multi-Head Attention 的头数（真实 GPT-2 是 12）
        n_embd=16,       # 嵌入维度：每个 token 用多少维向量表示（真实 GPT-2 是 768）
                         # 注意：n_embd 必须能被 n_head 整除，每个头的维度 = 16 / 2 = 8
    )
    # 用上面的配置实例化模型
    model = GPT2(config)

    # 生成随机 token ID 张量，模拟一批输入序列
    # 形状：(batch_size=4, seq_len=8)，即 4 条序列，每条 8 个随机 token ID
    idx = torch.randint(0, config.vocab_size, (4, config.block_size))

    # 前向传播：第一个 idx 是输入，第二个 idx 是 targets（用于计算交叉熵 loss）
    logits, loss = model(idx, idx)

    # logits 形状：(batch=4, seq_len=8, vocab=32)
    # 含义：每个位置上，对词表中所有 32 个 token 的预测得分
    assert logits.shape == (4, config.block_size, config.vocab_size)
    # 传入了 targets，loss 应该被计算出来
    assert loss is not None
    # loss 是 0 维标量张量（单个数值），不是向量或矩阵
    
    # 模型被随机初始化，直接如何一个单词，loss 就是在随机参数情况下预测的下一个单词和实际下一个单词的交叉熵损失
    print(loss)
    assert loss.ndim == 0


def test_gpt2_generate_extends_sequence() -> None:
    config = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT2(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    out = model.generate(idx, max_new_tokens=5, top_k=5)

    assert out.shape == (1, 8)
