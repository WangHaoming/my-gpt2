# trace.py — 推理过程数值追踪

`trace.py` 是一个终端工具，把 GPT-2 单次前向传播中每一个矩阵乘法的**输入激活、输出激活、统计量、实际数值样本**打印到终端。

---

## 一、显示的是什么？

### 激活值（activation），不是权重参数（weight）

每个线性层的计算公式是：

```
输出 = 输入 @ 权重^T + 偏置
  y  =   x  @   W^T  +  b
```

trace 捕获的是 **x（输入激活）** 和 **y（输出激活）**，权重矩阵 W 本身不显示。

| 概念 | 含义 | 随输入变化？ | trace 显示？ |
|---|---|---|---|
| **权重参数** | 模型训练后固定的矩阵，代表"学到了什么" | 不变 | 否 |
| **激活值** | 本次输入数据流过网络时产生的中间结果 | 每次输入不同结果就不同 | **是** |

### 如果想看权重参数

```python
import torch
ck = torch.load("checkpoints/latest.pt", map_location="cpu")
state = ck["model"]

# Layer 0 的 c_attn 权重（形状 384×128，即 3C×C）
print(state["transformer.h.0.attn.c_attn.weight"])

# Layer 0 的 MLP c_fc 权重（形状 512×128，即 4C×C）
print(state["transformer.h.0.mlp.c_fc.weight"])
```

---

## 二、捕获的操作列表

每个 Transformer Block 有 6 个矩阵操作，加上全局 1 个 lm_head，共 **6×N + 1** 个。

| op_name | 位置 | input_mat（捕获的输入） | output_mat（捕获的输出） |
|---|---|---|---|
| `c_attn` | Attention | LayerNorm 后的 token 表示 (T×C) | QKV 联合投影 (T×3C) |
| `qk` | Attention | Q 向量 head 0 (T×d) | 原始注意力分数 Q@K^T (T×T) |
| `av` | Attention | softmax 注意力权重 (T×T) | 加权求和 att@V (T×d) |
| `c_proj_attn` | Attention | 多头拼接后 (T×C) | 输出投影 (T×C) |
| `c_fc` | MLP | LayerNorm 后的 token 表示 (T×C) | 升维激活 GELU 前 (T×4C) |
| `c_proj_mlp` | MLP | GELU 后的激活 (T×4C) | 降维输出 (T×C) |
| `lm_head` | 全局 | 最后一层 Block 输出 (T×C) | 词表 logits (T×vocab) |

> T = prompt 长度，C = n_embd（嵌入维度），d = C / n_head（每个注意力头的维度）

---

## 三、每个操作的展示格式

### 通用格式（c_attn、c_proj、c_fc）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Layer 0  ·  c_attn
  QKV 联合投影  (T×C → T×3C)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  输入  shape=(5, 128)  |  mean=+0.0001  std=0.9966  min=-2.5789  max=+3.1676
  输出  shape=(5, 384)  |  mean=-0.0051  std=0.3447  min=-1.2409  max=+1.1554

  输入矩阵（前 8 行 × 前 8 列）：
              dim0      dim1      dim2  ...
       h  +0.9664  +1.5468  -1.7818  ...
       e  -0.5201  -1.0517  -0.2245  ...
       ...
```

- 每行对应一个 token（用 prompt 中的字符标注）
- 列数超过 8 时显示 `...` 表示已截断（嵌入维度通常为 128/384/512）

### qk — Q@K^T 原始注意力分数

```
  Q 向量（head 0，每个 token 的查询向量，前 8 维）：
  ┌ 每行是一个 token 的 Q 向量，代表"我想找什么信息" ┐

  原始注意力分数 Q@K^T（softmax 之前）：
  （正值=更关注，负值=更忽略）
              dim0      dim1      dim2      dim3      dim4
       h  +0.5225  -0.5093  -0.0706  +0.1429  +0.3614
       e  -0.9703  +4.3519  -0.2059  -1.1606  -1.4206
       ...
```

这是 softmax 之前的原始得分，数值大小决定了注意力权重的分布。

### av — softmax(att)@V 注意力权重概率表

这是最直观的展示，直接看到每个 token "关注"了哪些位置：

```
  【注意力权重 softmax(QK^T/√d)】— 显示每个 token 对其他 token 的关注概率：
  （行=当前token 看 列=哪个token，因果掩码保证只能看过去）

           'h'      'e'      'l'      'l'      'o'
  'h'   1.0000      ----      ----      ----      ----
  'e'   0.2807    0.7193      ----      ----      ----
  'l'   0.2704    0.3722    0.3574      ----      ----
  'l'   0.1622    0.2057    0.3217    0.3103      ----
  'o'   0.1972    0.1389    0.2066    0.2402    0.2170
```

- `----` 表示被因果掩码屏蔽的未来位置（不能看）
- 每行之和 = 1.0（softmax 保证）
- 读法示例："处理第 3 个字符 `l` 时，它把 37% 的注意力放在 `e` 上，36% 放在自己身上，27% 放在 `h` 上"

### lm_head — 词表预测

```
  【最后一个 token 位置的 Top-10 预测】：
    # 1  token_id=   1  char=' '   logit= +5.6303
    # 2  token_id=  11  char='p'   logit= +0.1602
    # 3  token_id=  10  char='o'   logit= -0.1433
    ...
```

- 取序列**最后一个 token 位置**（即"接下来预测哪个字符"）
- logit 越大代表越可能是下一个字符
- 第 1 名和其他名次的 logit 差距越大，模型越"确定"

---

## 四、用法

```bash
# 只看第 0 层（推荐起点，输出最简洁）
python -m my_gpt2.trace --checkpoint checkpoints/latest.pt --prompt "hello" --layer 0

# 只看某一层（0-based）
python -m my_gpt2.trace --checkpoint checkpoints/latest.pt --prompt "hello" --layer 2

# 看全部层 + lm_head（输出较长，建议配合 --out）
python -m my_gpt2.trace --checkpoint checkpoints/latest.pt --prompt "hello"

# 保存到文件再阅读
python -m my_gpt2.trace --checkpoint checkpoints/latest.pt --prompt "hello" --out trace.txt
```

### 参数说明

| 参数 | 必填 | 说明 |
|---|---|---|
| `--checkpoint` | 是 | 训练好的 `.pt` 文件路径 |
| `--prompt` | 是 | 输入文本，每个字符作为一个 token |
| `--layer` | 否 | 只展示指定层（0-based），不填则展示全部层 + lm_head |
| `--out` | 否 | 把输出同时保存到文本文件 |

---

## 五、与热力图（visualize.py）的对比

| | `trace.py` | `visualize.py` |
|---|---|---|
| 输出形式 | 终端文字 | PNG 图片 |
| 展示内容 | 真实数值、统计量、注意力概率表、Top-10 预测 | 颜色深浅代表数值大小的热力图 |
| 适合场景 | 想看具体数字，验证计算逻辑 | 想看数值分布的整体形态 |
| 层过滤 | 支持 `--layer N` | 输出全部层 |

两者捕获的数据来源相同（均复用 `hooks.py`），展示方式不同。
