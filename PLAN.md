# 手写 GPT-2 实现计划

这个项目建议按“先跑通，再对齐，再扩展”的顺序推进。每个阶段都保留可运行状态，这样你不会在一大坨数学和工程细节里迷路。

## 1. 项目基础

- 明确依赖：Python、PyTorch、pytest、ruff。
- 建立 `src/my_gpt2` 包结构。
- 建立最小测试，确保模型 forward 和 generate 能跑。
- 建立 `README.md`，记录运行方式。

## 2. 数据与 tokenizer

- 先实现字符级 tokenizer，用来快速验证训练闭环。
- 实现文本数据集，把长文本切成固定长度 token block。
- 后续再实现 GPT-2 BPE tokenizer，目标是兼容 OpenAI GPT-2 的 byte-level BPE 思路。
- 需要重点理解：vocab、token id、encode、decode、padding 是否需要、block size。

## 3. GPT-2 核心模块

- `GPTConfig`：集中管理 vocab size、block size、层数、头数、hidden size、dropout。
- Token embedding：把 token id 映射成向量。
- Position embedding：给每个位置加上可学习的位置向量。
- Causal self-attention：实现 Q/K/V、scaled dot-product attention、causal mask、多头拆分与合并。
- MLP：GPT-2 block 里的前馈网络，通常是 `4 * n_embd` hidden size。
- LayerNorm：使用 pre-norm 结构，即 attention 和 MLP 前先归一化。
- Residual connection：每个子层输出加回输入。
- LM head：把 hidden state 投影回 vocab logits。

## 4. 训练闭环

- 实现 next-token prediction：输入 `x[:, :-1]`，预测 `x[:, 1:]`。
- 使用 cross entropy loss。
- 实现 AdamW optimizer。
- 加入 batch sampling、训练 loop、loss logging。
- 保存 checkpoint：模型权重、配置、tokenizer、训练 step。

## 5. 生成与采样

- 实现 greedy decoding。
- 实现 temperature。
- 实现 top-k sampling。
- 可选实现 top-p sampling。
- 支持 prompt 编码、逐 token 生成、decode 回文本。

## 6. 对齐 GPT-2 细节

- 权重初始化：按 GPT-2 常见初始化方式调整标准差。
- tied embeddings：让 token embedding 和 lm head 共享权重。
- attention / residual dropout。
- 加载 Hugging Face GPT-2 权重做形状和输出 sanity check。
- 验证同配置下参数量是否接近预期。

## 7. 训练效率

- 支持 GPU / MPS / CPU 自动选择。
- 支持 mixed precision。
- 支持 gradient clipping。
- 支持 gradient accumulation。
- 后续可加入 flash attention 或 `torch.compile`。

## 8. 推荐实现顺序

1. 跑通当前字符级版本的测试。
2. 阅读并重写 `CausalSelfAttention`，确认 tensor shape。
3. 阅读并重写 `Block` 和 `GPT2`。
4. 用 `data/tiny.txt` 训练一个 overfit 小模型。
5. 加入更好的日志和 checkpoint。
6. 实现 byte-level BPE tokenizer。
7. 尝试加载官方 GPT-2 small 权重。

## 9. 你最终会拥有的模块

- `config.py`：模型超参数配置。
- `model.py`：GPT-2 网络本体。
- `tokenizer.py`：tokenizer 接口和实现。
- `data.py`：数据读取与 batch 构造。
- `train.py`：训练入口。
- `generate.py`：文本生成入口。
- `checkpoint.py`：后续可拆出的保存/加载逻辑。
- `tests/`：模块级 smoke test 和行为测试。
