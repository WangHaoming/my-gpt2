# GPT-2 模型详解

本文档详细说明 `src/my_gpt2/model.py` 中各个类的职责、相互关系，以及训练和推理过程中的完整函数调用链。

---

## 一、类层次关系

```
GPTConfig                   ← 所有超参数的来源（vocab_size, n_layer, n_head, n_embd, …）
    │
    ▼
CausalSelfAttention         ← 最底层：多头因果自注意力
MLP                         ← 最底层：前馈网络
    │
    ▼ 组合
Block                       ← 中间层：一个完整的 Transformer 层 = Attention + MLP
    │
    ▼ 堆叠 n_layer 个
GPT2                        ← 顶层：完整语言模型 = Embedding + N×Block + LM Head
```

每一层都只依赖比它更底层的模块，没有循环依赖。`GPTConfig` 是唯一的配置入口，所有模块在 `__init__` 时接收同一个 `config` 对象。

---

## 二、各类详解

### `GPTConfig`（`config.py`）

超参数数据类，`frozen=True` 保证训练过程中不会被意外修改。


| 字段           | 含义                              |
| ------------ | ------------------------------- |
| `vocab_size` | 词表大小，决定 Embedding 和 LM Head 的宽度 |
| `block_size` | 最大序列长度（上下文窗口）                   |
| `n_layer`    | Transformer Block 的层数           |
| `n_head`     | 每层的注意力头数                        |
| `n_embd`     | 嵌入维度，也是所有隐藏层的统一宽度               |
| `dropout`    | 训练时 Dropout 概率                  |
| `bias`       | 线性层/LayerNorm 是否带偏置             |


---

### `CausalSelfAttention`

**职责**：让每个 token 与序列中**它之前**的所有 token 做注意力交互，禁止看到未来位置。

**核心子模块**：


| 子模块             | 作用                                          |
| --------------- | ------------------------------------------- |
| `c_attn`        | 一个线性层同时生成 Q、K、V（输出维度 = 3×n_embd）            |
| `c_proj`        | 把多头结果投影回原始维度                                |
| `attn_dropout`  | 作用于注意力权重矩阵                                  |
| `resid_dropout` | 作用于输出残差                                     |
| `bias`（buffer）  | 下三角因果掩码，形状 `(1, 1, block_size, block_size)` |


`**forward` 内部步骤**：

```
输入 x: (B, T, C)
    │
    ├─ c_attn(x) → split → Q, K, V  各 (B, T, C)
    │
    ├─ reshape + transpose → Q, K, V 各 (B, n_head, T, head_dim)
    │
    ├─ att = Q @ K^T / sqrt(head_dim)     → (B, n_head, T, T)
    │
    ├─ masked_fill(causal_mask == 0, -inf) → 遮住未来位置
    │
    ├─ softmax(att) + attn_dropout        → 注意力概率
    │
    ├─ y = att @ V                        → (B, n_head, T, head_dim)
    │
    ├─ transpose + view → (B, T, C)       → 拼回多头结果
    │
    └─ c_proj + resid_dropout             → 输出 (B, T, C)
```

---

### `MLP`

**职责**：对每个 token 的表示做独立的非线性变换（token 之间不交互）。

**核心子模块**：


| 子模块       | 作用                        |
| --------- | ------------------------- |
| `c_fc`    | 升维线性层：`n_embd → 4×n_embd` |
| `gelu`    | 非线性激活函数（平滑版 ReLU）         |
| `c_proj`  | 降维线性层：`4×n_embd → n_embd` |
| `dropout` | 防过拟合                      |


`**forward` 内部步骤**：

```
输入 x: (B, T, C)
    │
    ├─ c_fc   → (B, T, 4C)   升维
    ├─ gelu   → (B, T, 4C)   非线性
    ├─ c_proj → (B, T, C)    降维
    └─ dropout               输出 (B, T, C)
```

---

### `Block`

**职责**：一层完整的 Transformer，将 `CausalSelfAttention` 和 `MLP` 用 **Pre-Norm + 残差连接** 串联起来。

**核心子模块**：


| 子模块    | 作用                       |
| ------ | ------------------------ |
| `ln_1` | 注意力子层前的 LayerNorm        |
| `attn` | `CausalSelfAttention` 实例 |
| `ln_2` | MLP 子层前的 LayerNorm       |
| `mlp`  | `MLP` 实例                 |


`**forward` 内部步骤**：

```
输入 x: (B, T, C)
    │
    ├─ x = x + attn(ln_1(x))   ← 注意力子层（Pre-Norm + 残差）
    └─ x = x + mlp(ln_2(x))    ← FFN 子层（Pre-Norm + 残差）

输出 x: (B, T, C)   形状不变
```

> **Pre-Norm vs Post-Norm**：这里先做 LayerNorm 再做操作（Pre-Norm），训练更稳定，梯度直接从残差路径流过，缓解深层网络梯度消失。

---

### `GPT2`

**职责**：完整的语言模型，负责将 token ID 序列转换为 logits（训练）或生成新 token（推理）。

**核心子模块**（存放在 `self.transformer` 这个 `ModuleDict` 中）：


| 名称     | 类型                              | 作用                            |
| ------ | ------------------------------- | ----------------------------- |
| `wte`  | `Embedding(vocab_size, n_embd)` | Token Embedding：token ID → 向量 |
| `wpe`  | `Embedding(block_size, n_embd)` | Position Embedding：位置索引 → 向量  |
| `drop` | `Dropout`                       | 作用于 `tok_emb + pos_emb` 之和    |
| `h`    | `ModuleList[Block × n_layer]`   | 堆叠的 Transformer Block         |
| `ln_f` | `LayerNorm`                     | 最后一个 Block 后的归一化              |


以及：


| 名称        | 类型                                       | 作用             |
| --------- | ---------------------------------------- | -------------- |
| `lm_head` | `Linear(n_embd, vocab_size, bias=False)` | 将隐藏状态映射到词表预测分数 |


**重要设计：权重共享**

```python
self.transformer.wte.weight = self.lm_head.weight
```

`wte`（输入嵌入）和 `lm_head`（输出投影）共享同一个权重矩阵（形状 `vocab_size × n_embd`）。好处：减少参数量，并让"输入空间"和"输出空间"的语义对齐。

---

## 三、训练时的函数调用链

训练时调用 `GPT2.forward(idx, targets)`，返回 `(logits, loss)`。

```
GPT2.forward(idx: (B,T), targets: (B,T))
    │
    ├─ [1] 位置索引
    │      pos = arange(0, T)                      → (T,)
    │
    ├─ [2] 嵌入层
    │      tok_emb = wte(idx)                      → (B, T, C)
    │      pos_emb = wpe(pos)                      → (T, C)  广播到 (B, T, C)
    │      x = drop(tok_emb + pos_emb)             → (B, T, C)
    │
    ├─ [3] N 个 Transformer Block（串行）
    │      for block in h:                         → 循环 n_layer 次
    │          x = block.forward(x)
    │              ├─ x = x + attn(ln_1(x))
    │              │       └─ CausalSelfAttention.forward(ln_1(x))
    │              │              ├─ c_attn → Q,K,V
    │              │              ├─ 多头 reshape
    │              │              ├─ att = QK^T/√d → mask → softmax
    │              │              └─ att@V → c_proj
    │              └─ x = x + mlp(ln_2(x))
    │                      └─ MLP.forward(ln_2(x))
    │                             ├─ c_fc → gelu → c_proj
    │                             └─ dropout
    │
    ├─ [4] 最终归一化
    │      x = ln_f(x)                             → (B, T, C)
    │
    ├─ [5] LM Head
    │      logits = lm_head(x)                     → (B, T, vocab_size)
    │
    └─ [6] 损失计算（训练时）
           logits_flat = logits.view(B*T, vocab_size)
           targets_flat = targets.view(B*T)
           loss = cross_entropy(logits_flat, targets_flat)  → 标量
```

**张量形状变化一览**：

```
idx           (B, T)
tok_emb       (B, T, C)           C = n_embd
pos_emb       (T, C)  → 广播
x after drop  (B, T, C)
x after h[0]  (B, T, C)           每层形状不变
...
x after ln_f  (B, T, C)
logits        (B, T, vocab_size)
loss          scalar
```

---

## 四、推理时的函数调用链

推理时调用 `GPT2.generate(idx, max_new_tokens)`，自回归地生成新 token。

```
GPT2.generate(idx: (B, T), max_new_tokens: int)
    │
    └─ 循环 max_new_tokens 次：
        │
        ├─ [1] 截断上下文（防止超过 block_size）
        │      idx_cond = idx[:, -block_size:]         → (B, T')   T'≤block_size
        │
        ├─ [2] 前向传播（复用训练时的 forward，targets=None）
        │      logits, _ = GPT2.forward(idx_cond)      → logits: (B, T', vocab_size)
        │      logits = logits[:, -1, :] / temperature → (B, vocab_size)  只取最后位置
        │
        ├─ [3] Top-K 截断（可选）
        │      找出 top-k 最大值，其余位置设为 -inf
        │      → 只在最可能的 k 个 token 中采样
        │
        ├─ [4] 采样
        │      probs = softmax(logits)                  → (B, vocab_size)
        │      idx_next = multinomial(probs, 1)         → (B, 1)
        │
        └─ [5] 拼接
               idx = cat([idx, idx_next], dim=1)        → (B, T+1)
                                                          下一轮继续
```

**推理与训练的关键区别**：


|     | 训练 (`forward`)            | 推理 (`generate`)         |
| --- | ------------------------- | ----------------------- |
| 输入  | `(B, T)`                  | `(B, T')` 每轮递增          |
| 输出  | `logits (B,T,V)` + `loss` | 只用 `logits[:,-1,:]`     |
| 梯度  | 需要（反向传播）                  | 不需要（`@torch.no_grad()`） |
| 目标  | 所有位置同时计算 loss             | 每次只预测下一个 token          |


---

## 五、初始化调用链（`GPT2.__init__`）

```
GPT2.__init__(config)
    │
    ├─ [1] 构建 transformer（ModuleDict）
    │      ├─ wte  = Embedding(vocab_size, n_embd)
    │      ├─ wpe  = Embedding(block_size, n_embd)
    │      ├─ drop = Dropout(dropout)
    │      ├─ h    = ModuleList([Block(config) × n_layer])
    │      │         └─ 每个 Block.__init__ 会创建：
    │      │              ├─ ln_1, ln_2 = LayerNorm
    │      │              ├─ attn = CausalSelfAttention(config)
    │      │              │         └─ 创建 c_attn, c_proj, dropout, causal mask
    │      │              └─ mlp  = MLP(config)
    │      │                        └─ 创建 c_fc, gelu, c_proj, dropout
    │      └─ ln_f = LayerNorm(n_embd)
    │
    ├─ [2] 构建 lm_head = Linear(n_embd, vocab_size, bias=False)
    │
    ├─ [3] 权重共享
    │      transformer.wte.weight = lm_head.weight
    │
    └─ [4] 权重初始化
           self.apply(_init_weights)
               ├─ Linear → Normal(0, 0.02) ; bias → zeros
               └─ Embedding → Normal(0, 0.02)
```

---

## 六、设计要点速查


| 设计                  | 位置                             | 作用                     |
| ------------------- | ------------------------------ | ---------------------- |
| 因果掩码（下三角）           | `CausalSelfAttention.__init__` | 禁止 token 看未来，保证自回归合法性  |
| 多头注意力               | `CausalSelfAttention.forward`  | 并行捕捉不同子空间的依赖关系         |
| `/ sqrt(head_dim)`  | `CausalSelfAttention.forward`  | 防止点积过大，softmax 梯度消失    |
| Pre-Norm            | `Block.forward`                | 训练更稳定，梯度可以直接走残差路径      |
| 残差连接                | `Block.forward`                | 解决深层网络梯度消失，允许信息直通      |
| FFN 4× 升维           | `MLP.__init__`                 | GPT-2 经典设计，增加非线性表达能力   |
| 权重共享（wte = lm_head） | `GPT2.__init_`_                | 减少参数，输入/输出空间语义对齐       |
| `@torch.no_grad()`  | `GPT2.generate`                | 推理时不建计算图，节省显存和计算       |
| temperature 采样      | `GPT2.generate`                | 控制生成多样性（>1 更随机，<1 更保守） |
| top-k 采样            | `GPT2.generate`                | 截断长尾概率，避免采到极低概率 token  |


