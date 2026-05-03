# my-gpt2

这是一个用于学习的 GPT-2 手写项目。目标不是一开始就复刻完整工业训练流程，而是把 Transformer Decoder 的关键零件逐个写清楚、跑通、测住。

---

## 项目结构

```
my-gpt2/
├── src/my_gpt2/
│   ├── config.py        # GPTConfig：所有超参数
│   ├── model.py         # 核心模型：CausalSelfAttention / MLP / Block / GPT2
│   ├── tokenizer.py     # 字符级 tokenizer
│   ├── data.py          # TinyTextDataset：滑动窗口数据集
│   ├── train.py         # 训练脚本（含 TensorBoard 日志）
│   ├── generate.py      # 推理/生成脚本
│   ├── hooks.py         # 矩阵乘法捕获 hook
│   ├── plotter.py       # 热力图生成器
│   ├── visualize.py     # 推理过程可视化 CLI（热力图 PNG）
│   ├── trace.py         # 推理过程数值追踪 CLI（终端文字输出）
│   └── inspect.py       # 模型结构与参数检查器
├── tests/               # pytest 测试
├── data/                # 训练文本文件（自行放入）
├── checkpoints/         # 训练后的模型权重（自动生成）
├── runs/                # TensorBoard 日志（自动生成）
├── viz_output/          # 矩阵乘法可视化图（自动生成）
├── MODEL.md             # 模型架构详解文档
└── README.md            # 本文件
```

---

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

依赖：`torch`、`tqdm`、`tensorboard`、`matplotlib`（均在 `pyproject.toml` 中声明）。

---

## 核心模块说明

### model.py — 模型架构

四个类，从底层到顶层：

| 类 | 职责 |
|---|---|
| `CausalSelfAttention` | 多头因果自注意力，每个 token 只能看自己和之前的 token |
| `MLP` | 前馈网络，先升维到 4×n_embd，GELU 激活，再降回 |
| `Block` | 一层 Transformer = LayerNorm + Attention + LayerNorm + MLP，各带残差连接 |
| `GPT2` | 完整模型：Embedding → N×Block → LayerNorm → LM Head |

> 详细的类关系、张量形状变化、训练/推理调用链见 [MODEL.md](MODEL.md)。

### train.py — 训练

训练时自动写入 TensorBoard 日志（`runs/checkpoints/`），训练完后保存 `checkpoints/latest.pt`。

### hooks.py + plotter.py + visualize.py — 矩阵乘法热力图可视化

在**不修改 model.py** 的前提下，捕获单次推理中所有矩阵乘法的输入/输出张量，并输出热力图 PNG。

每个 Transformer Block 生成一张 3×3 子图，包含：

| 位置 | 操作 | 形状变化 |
|---|---|---|
| Attention | c_attn（QKV 投影） | T×C → T×3C |
| Attention | Q @ K^T（head 0） | T×d @ d×T → T×T |
| Attention | softmax(att) @ V（head 0） | T×T @ T×d → T×d |
| Attention | c_proj（输出投影） | T×C → T×C |
| MLP | c_fc（升维） | T×C → T×4C |
| MLP | c_proj（降维） | T×4C → T×C |

另生成 `lm_head.png`（词表预测分数）。

### inspect.py — 模型结构与参数检查器

读取 checkpoint，打印模型超参数、每层线性层的形状与权重统计量、参数量汇总。`--values` 模式下展示具体数值（c_attn 权重拆分为 W_Q / W_K / W_V）。

### trace.py — 推理过程数值追踪

比热力图更直接：把每一步矩阵乘法的**实际数字**打印到终端，能看到真实数值、形状、统计量（mean/std/min/max）。

每个操作都有针对性的展示方式：

| 操作 | 展示内容 |
|---|---|
| `c_attn`、`c_fc`、`c_proj` | 输入/输出矩阵前 N 行 × 前 8 列（带 token 行标签） |
| `qk`（Q@K^T） | Q 向量 + 原始注意力分数矩阵（含因果掩码位置） |
| `av`（softmax(att)@V） | **注意力概率表**（行=当前 token 看哪些 token，----=被掩码）+ 加权输出 |
| `lm_head` | **Top-10 预测字符**（含 token_id、字符、logit 值）+ 全部位置输出矩阵 |

示例输出（`av` 操作的注意力概率表）：

```
  （行=当前token 看 列=哪个token，因果掩码保证只能看过去）
           'h'      'e'      'l'      'l'      'o'
  'h'   1.0000      ----      ----      ----      ----
  'e'   0.2807    0.7193      ----      ----      ----
  'l'   0.2704    0.3722    0.3574      ----      ----
  'l'   0.1622    0.2057    0.3217    0.3103      ----
  'o'   0.1972    0.1389    0.2066    0.2402    0.2170
```

示例输出（`lm_head` 的 Top-10 预测）：

```
  【最后一个 token 位置的 Top-10 预测】：
    # 1  token_id=   1  char=' '   logit= +5.6303
    # 2  token_id=  11  char='p'   logit= +0.1602
    ...
```

---

## 常用命令

### 准备数据

```bash
mkdir -p data
printf "hello gpt\nhello transformer\n" > data/tiny.txt
```

### 训练

```bash
python -m my_gpt2.train --input data/tiny.txt --steps 200
```

### 查看训练曲线（TensorBoard）

```bash
# 另开终端
tensorboard --logdir runs
# 浏览器打开 http://localhost:6006
```

训练过程中可实时看到 `train/loss` 收敛曲线，多次训练自动叠加对比。

### 生成文本

```bash
python -m my_gpt2.generate --checkpoint checkpoints/latest.pt --prompt "hello"
```

### 可视化推理中的矩阵乘法（热力图）

```bash
python -m my_gpt2.visualize \
    --checkpoint checkpoints/latest.pt \
    --prompt "hello" \
    --output-dir viz_output
```

输出到 `viz_output/prompt_hello/`：
- `layer_00.png` ~ `layer_NN.png`：每层的矩阵乘法热力图
- `lm_head.png`：最终词表预测分数

### 数值追踪推理中的矩阵乘法（终端文字）

```bash
# 只看第 0 层（推荐先从这里开始，输出最简洁）
python -m my_gpt2.trace --checkpoint checkpoints/latest.pt --prompt "hello" --layer 0

# 看全部层
python -m my_gpt2.trace --checkpoint checkpoints/latest.pt --prompt "hello"

# 保存到文件慢慢阅读
python -m my_gpt2.trace --checkpoint checkpoints/latest.pt --prompt "hello" --out trace.txt
```

输出内容包括每个矩阵操作的：形状、统计量（mean/std/min/max）、实际数值样本。
注意力权重以概率表形式展示，lm_head 直接显示 Top-10 预测字符。

> 详细说明（捕获的是激活值还是权重、每个操作的展示格式、与热力图的对比）见 [TRACE.md](TRACE.md)。

### 检查模型结构与参数权重

```bash
# 打印模型结构、各层形状、参数统计量
python -m my_gpt2.inspect --checkpoint checkpoints/latest.pt

# 同时展示具体参数数值（W_Q / W_K / W_V 分开显示）
python -m my_gpt2.inspect --checkpoint checkpoints/latest.pt --values
```

输出内容：
- **超参数**：n_layer / n_head / n_embd / head_dim 等
- **Embedding**：wte（词元嵌入）和 wpe（位置嵌入）的形状与数值
- **每个 Block**：操作流程说明 + c_attn（W_Q/W_K/W_V）/ c_proj / c_fc 的形状和统计量
- **参数量汇总**：按模块分组，并说明 wte 与 lm_head 共享权重

> 注意：inspect.py 显示的是**权重参数**（训练后固定），而 trace.py 显示的是**推理时的激活值**（随输入变化）。

### 运行测试

```bash
pytest       # 全部测试
pytest -s    # 显示 print 输出
```

---

## VS Code 调试配置

`.vscode/launch.json` 中内置两个配置，在 Run & Debug 面板（F5）直接选用：

| 配置名 | 执行命令 |
|---|---|
| Train tiny GPT-2 | `python -m my_gpt2.train --input data/tiny.txt --steps 200` |
| Generate from checkpoint | `python -m my_gpt2.generate --checkpoint checkpoints/latest.pt --prompt "hello"` |

---

## 学习路线

详细计划见 [PLAN.md](PLAN.md)。
