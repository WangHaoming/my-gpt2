from dataclasses import dataclass


# frozen=True：创建后字段不可修改，像常量一样安全，防止训练过程中意外改动配置
@dataclass(frozen=True)
class GPTConfig:
    """GPT-2 模型的结构超参数，所有字段在实例化后不可更改。"""

    vocab_size: int          # 词表大小：模型能认识的不同 token 数量（字符级约几十，BPE 级约 5 万）

    block_size: int = 128    # 上下文长度：模型一次最多看多少个 token（等于序列的最大长度）
    n_layer: int = 4         # Transformer Block 层数：层数越多，模型越深、表达能力越强
    n_head: int = 4          # 注意力头数：每层把 n_embd 拆成 n_head 份并行计算注意力
    n_embd: int = 128        # 嵌入维度：每个 token 用多少维向量表示，也是所有隐藏层的宽度
    dropout: float = 0.1     # Dropout 概率：训练时随机丢弃 10% 的神经元，防止过拟合
    bias: bool = True        # 线性层和 LayerNorm 是否使用偏置项（bias=False 可略微减少参数量）

    def __post_init__(self) -> None:
        # dataclass 实例化后自动调用，用于做参数合法性校验
        # n_embd 必须能被 n_head 整除，因为每个头的维度 = n_embd / n_head，必须是整数
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
