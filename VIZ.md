# viz_server — 神经网络参数可视化工具

`viz_server` 是一个本地 Web 应用，在浏览器中实时展示 GPT-2 模型的**权重矩阵**、**训练时参数如何变化**，以及**推理时激活值如何在各层流动**。

---

## 一、快速启动

```bash
# 启动服务器，加载已有 checkpoint
python -m my_gpt2.viz_server --checkpoint checkpoints/latest.pt

# 可选：指定端口（默认 5000）
python -m my_gpt2.viz_server --checkpoint checkpoints/latest.pt --port 5001
```

浏览器打开 **http://127.0.0.1:5000**，无需其他依赖。

---

## 二、三个 Tab 的功能

### Tab 1 · 权重矩阵

启动后自动加载，展示模型中所有 **2D 权重矩阵**（共约 36 个）。

按模型结构分组显示：

| 分组 | 包含的矩阵 |
|------|-----------|
| 嵌入层 | `wte`（token embedding）、`wpe`（position embedding） |
| Layer 0–3 | 每层的 `c_attn`、`c_proj_attn`、`c_fc`、`c_proj_mlp`，以及 LayerNorm 参数 |
| 输出层 | `ln_f`、`lm_head` |

**操作：** 单击任意矩阵卡片，弹出放大视图，可查看完整矩阵和详细统计量（mean / std / min / max）。

**颜色规则：**

```
蓝色（负值） ←─── 白色（零） ───→ 红色（正值）
```

使用 RdBu 发散色标，与 matplotlib 的 `seismic` 配色一致，绝对值越大颜色越深。

---

### Tab 2 · 训练过程

填写左侧侧边栏的参数后，点击 **▶ 开始训练**：

| 参数 | 说明 |
|------|------|
| 训练文本文件 | 训练用的纯文本文件路径（需与原模型使用相同字符集） |
| 训练步数 | 执行反向传播的次数 |
| 学习率 | AdamW 优化器的学习率 |
| 每隔多少步更新权重图 | 每 N 步向浏览器推送一次权重快照（越小越流畅但传输量越大） |

训练开始后页面实时显示：

- **Loss 折线图** — 每步更新一次，X 轴是 step，Y 轴是 cross-entropy loss
- **权重热力图** — 每隔 N 步刷新，默认显示**变化量（Δ）热力图**：橙色越深表示该权重在过去 N 步里变化越大
- **Δmax 徽章** — 每个矩阵卡片右下角显示该矩阵的最大变化量，用于快速判断哪一层学到了最多东西

点击 **■ 停止** 可随时中断训练，权重保持当前状态。

---

### Tab 3 · 推理过程

在左侧侧边栏输入 prompt，点击 **▶ 运行推理**：

页面展示完整的前向传播过程，分层显示每个矩阵操作的**输入激活**和**输出激活**：

| 操作名 | 位置 | 输入形状 | 输出形状 |
|--------|------|----------|----------|
| `c_attn` | Attention | T × C | T × 3C |
| `qk` | Attention | T × d | T × T（注意力分数） |
| `av` | Attention | T × T（注意力权重） | T × d |
| `c_proj_attn` | Attention | T × C | T × C |
| `c_fc` | MLP | T × C | T × 4C |
| `c_proj_mlp` | MLP | T × 4C | T × C |
| `lm_head` | 输出 | T × C | T × vocab |

> T = prompt 长度，C = n_embd，d = C / n_head

页面底部显示**下一个 token 的 Top-10 预测**，包含字符、概率和 logit 值，用横向进度条直观展示概率分布。

---

## 三、实现原理

### 架构总览

```
浏览器 (index.html)
    │
    │  HTTP REST / SSE 流
    │
Flask 服务器 (viz_server.py)
    │
    ├── /api/weights   ─── 读取 model.state_dict()，返回 JSON
    ├── /api/train     ─── SSE 流，边训练边推送 loss + 权重快照
    └── /api/infer     ─── 调用 hooks.py 捕获激活，返回 JSON
```

### 权重数据传输

`/api/weights` 把 `model.named_parameters()` 中所有 2D 张量序列化为 JSON：

```python
{
  "transformer.h.0.attn.c_attn.weight": {
    "shape": [384, 128],
    "data": [[...], ...],        # 二维浮点数组，保留 4 位小数
    "stats": {"mean": ..., "std": ..., "min": ..., "max": ...}
  },
  ...
}
```

对于这个小模型（n_embd=128, n_layer=4），全部权重数据约 **2–3 MB**，可以直接传输，无需降采样。

### SSE 流式训练

训练进度通过 [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) 推送，每个事件是一行 JSON：

```
data: {"type": "step", "step": 10, "loss": 1.8234}

data: {"type": "step", "step": 20, "loss": 1.6102, "weights": {...}}

data: {"type": "done", "total_steps": 200}
```

- **每步**发送 `type: "step"`，包含 step 和 loss
- **每隔 N 步**在同一个事件里附加 `weights` 字段，包含当前权重值和与上一次快照的差值（`diff`、`diff_max`）
- 训练在 Flask 的生成器函数内直接执行，天然支持流式输出

前端用 `fetch` + `ReadableStream` 逐块解析 SSE 事件，避免了 `EventSource` 不支持 POST body 的限制。

### Canvas 热力图渲染

权重矩阵用 HTML5 `<canvas>` 的 `ImageData` API 直接写像素，避免了 DOM 操作的开销：

```javascript
// 核心渲染逻辑（简化）
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    const norm = data2d[r][c] / absMax;  // 归一化到 [-1, 1]
    const [R, G, B] = valToRGB(norm);    // RdBu 色标插值
    // 写入 ImageData，每个值对应 scale×scale 个像素
  }
}
ctx.putImageData(img, 0, 0);
```

像素缩放 `scale = max(1, floor(maxPx / max(rows, cols)))`，保证小矩阵可见、大矩阵不溢出。

**变化量热力图**使用橙色单色渐变（白→橙），`absMax = diff_max`，颜色越深表示该位置变化越大。

### 激活值捕获

推理时复用了项目已有的 `hooks.py`：

```python
collector = MatmulCollector()
install_hooks(model, collector)   # 给每层装 forward hook
with torch.no_grad():
    logits, _ = model(idx)
uninstall_hooks(model)            # 还原模型，不污染后续调用
# collector.records 包含所有层的 input/output 矩阵快照
```

每条记录包含 `layer_idx`、`op_name`、输入矩阵、输出矩阵，服务器将其序列化后返回给前端渲染。

---

## 四、文件说明

```
src/my_gpt2/
├── viz_server.py          # Flask 服务器，所有 API 接口
└── viz_static/
    └── index.html         # 前端页面（单文件，无外部依赖，仅引用 CDN 无需联网）
```

| 文件 | 职责 |
|------|------|
| `viz_server.py` | 加载 checkpoint、提供 REST API、驱动训练循环和 SSE 推送 |
| `viz_static/index.html` | 三 Tab 页面、Canvas 热力图渲染、Loss 折线图、SSE 客户端 |

---

## 五、命令行参数

```
python -m my_gpt2.viz_server --help

  --checkpoint PATH   训练好的 .pt checkpoint 文件（必填）
  --host HOST         监听地址，默认 127.0.0.1（仅本机访问）
  --port PORT         监听端口，默认 5000
```

---

## 六、与其他可视化工具的对比

| 工具 | 展示对象 | 输出形式 | 交互性 |
|------|---------|----------|--------|
| `inspect.py` | 权重参数 | 终端文字 | 无 |
| `trace.py` | 激活值 | 终端文字 | 无 |
| `visualize.py` | 激活值 | PNG 热力图 | 无 |
| **`viz_server`** | 权重 + 激活 + 训练动态 | 交互式网页 | 实时更新、点击放大 |

`viz_server` 的数据来源与 `inspect.py`（权重）和 `trace.py` / `visualize.py`（激活，均复用 `hooks.py`）完全一致，只是以交互式网页的形式呈现。
