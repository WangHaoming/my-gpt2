# viz_server 可视化工具 — 内部实现详解

本文档详细说明可视化工具涉及的每个源文件、每个函数的职责，以及它们与模型代码的交互方式。

---

## 一、涉及的源文件总览

```
src/my_gpt2/
├── viz_server.py          ← 新增：Flask HTTP 服务器，全部后端逻辑
├── viz_static/
│   └── index.html         ← 新增：前端页面，全部 HTML/CSS/JS
│
├── hooks.py               ← 已有：被 viz_server 的推理接口复用
├── model.py               ← 已有：GPT2 模型类，被直接持有和调用
├── config.py              ← 已有：GPTConfig 数据类，被序列化后传给前端
├── tokenizer.py           ← 已有：CharTokenizer，用于编码 prompt 和训练文本
└── data.py                ← 已有：TinyTextDataset，训练接口用它构建数据集
```

---

## 二、`viz_server.py` — Flask 后端

### 职责

- 在进程启动时加载 checkpoint，把模型、tokenizer、config 保存为模块级全局变量
- 通过 Flask 提供四个 HTTP 接口，供前端 `index.html` 调用
- 训练接口以**生成器**的方式边训练边 yield SSE 事件，实现流式推送
- 推理接口调用 `hooks.py` 的 hook 机制捕获每层激活，整理成 JSON 返回

### 模块级全局变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `_model` | `GPT2 \| None` | 当前加载的模型实例，所有接口共享 |
| `_tokenizer` | `CharTokenizer \| None` | 与模型配套的字符级 tokenizer |
| `_config` | `GPTConfig \| None` | 模型结构超参数，会被序列化后发给前端 |
| `_device` | `str` | 运行设备，`"mps"` / `"cuda"` / `"cpu"` |
| `_train_stop` | `threading.Event` | 训练取消信号；`/api/stop` 设置它，训练循环每步检查它 |

这些变量在 `main()` 里赋值，之后各接口函数以闭包方式访问，无需参数传递。

---

### 函数详解

#### `_stats(t: Tensor) → dict`

```python
def _stats(t):
    f = t.float()
    return {"mean": ..., "std": ..., "min": ..., "max": ...}
```

把一个任意形状的张量转成统计摘要字典。所有数值 `round` 到 5 位小数，避免 JSON 体积膨胀。前端的 `fmtStats()` 函数直接消费这个格式。

---

#### `_tensor_to_list(t: Tensor) → list[list[float]]`

```python
def _tensor_to_list(t):
    return [[round(float(v), 4) for v in row] for row in t.cpu().float()]
```

把一个 **2D** 张量序列化为嵌套列表，方便 `json.dumps`。精度保留 4 位小数，对于 float32 的训练权重来说足够，同时大幅压缩 JSON 体积（float32 原始 24 位 → 最多 7 个字符）。

只接受 2D 张量；调用方负责确保维度正确。

---

#### `_all_weights() → dict`

```python
def _all_weights():
    for name, param in _model.named_parameters():
        if t.dim() == 2:
            result[name] = {"shape": ..., "data": ..., "stats": ...}
        elif t.dim() == 1 and ("ln_" in name or "ln_f" in name):
            # 1D LayerNorm 参数包装成 1×N 矩阵
            result[name] = {"shape": [1, N], "data": [[...]], "stats": ...}
    return result
```

遍历 `model.named_parameters()`，筛选出所有可用于可视化的参数：

- **2D 参数**：所有 `nn.Linear` 的 weight 矩阵（`c_attn.weight`、`c_proj.weight`、`c_fc.weight`、`c_proj_mlp.weight`、`wte.weight`、`wpe.weight`、`lm_head.weight`）
- **1D 参数**：`LayerNorm` 的 weight 和 bias（包含 `ln_` 或 `ln_f` 的名字），被人工包装成 `[1, N]` 形状，方便热力图渲染

返回的 dict 键是参数的完整点分路径（如 `transformer.h.0.attn.c_attn.weight`），前端用正则匹配这个路径来分组显示。

注意 `wte.weight` 和 `lm_head.weight` 是**同一个 tensor**（GPT-2 的权重共享设计），`named_parameters()` 会把它们各自列出，前端会在两个不同的格子里展示（内容相同）。

---

#### `_sse(data: dict) → str`

```python
def _sse(data):
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
```

把一个 Python dict 格式化成一条 SSE（Server-Sent Events）消息。SSE 协议规定每条消息以 `data: ` 开头、两个换行结尾。`ensure_ascii=False` 保留中文字符。

---

#### `index()` — `GET /`

返回静态文件 `viz_static/index.html`，浏览器加载后自动发起后续 API 请求。

---

#### `api_config()` — `GET /api/config`

把 `_config`（`GPTConfig` dataclass）用 `asdict` 转成 dict，额外追加 `device` 字段后返回 JSON。前端 `init()` 函数启动时调用此接口，在侧边栏显示模型超参数卡片。

---

#### `api_weights()` — `GET /api/weights`

直接调用 `_all_weights()` 并返回 JSON。前端 `init()` 在获得 config 后立即调用此接口，渲染初始的权重热力图网格。

对于默认的小模型（n_layer=4, n_embd=128），响应体约 2–3 MB，一次性传输不会有性能问题。

---

#### `api_train()` — `POST /api/train`

这是最复杂的接口。请求体 JSON 包含 `input_file`、`steps`、`lr`、`weight_every` 四个字段。

**返回值是一个 SSE 流**，通过 `Response(stream(), mimetype="text/event-stream")` 实现。

内部定义了一个生成器函数 `stream()`，执行完整的训练循环：

```
stream() 流程：
  1. 读取训练文本文件
  2. 用 _tokenizer 编码（不重建新 tokenizer，保证词表对齐）
  3. 构建 TinyTextDataset + DataLoader
  4. 创建 AdamW 优化器，调用 _model.train()
  5. 快照所有 2D 参数的初始值（用于计算每步的变化量）
  6. 训练循环：
     每步：
       - 前向传播 + 反向传播 + optimizer.step()
       - yield {"type":"step", "step":N, "loss":X}
     每隔 weight_every 步，额外附加 weights 字段：
       - 计算当前参数与上一次快照的 diff（绝对值）
       - 把当前值和 diff 都序列化进事件
       - 更新快照
  7. 训练结束：yield {"type":"done", "total_steps":N}
  8. 调用 _model.eval() 恢复推理模式
```

每一步的 `yield` 语句是关键：Flask 的 `Response(stream(), ...)` 会在每次 `yield` 时把这行数据立即刷写给浏览器，无需等待整个函数执行完毕。

`_train_stop` 事件让 `/api/stop` 接口可以从外部中断这个正在运行的生成器——生成器每步开头检查 `_train_stop.is_set()`，若已设置则 `break`。

---

#### `api_stop()` — `POST /api/stop`

调用 `_train_stop.set()`，设置停止信号。训练生成器在下一步迭代时会检测到并退出。

---

#### `api_infer()` — `POST /api/infer`

推理接口，请求体包含 `prompt` 字符串。

执行流程：

```
1. 用 _tokenizer.encode(prompt) 把字符串转为 token ID 列表
2. 包装成 (1, T) 形状的 tensor，发送到 _device
3. 创建 MatmulCollector，调用 install_hooks(_model, collector)
   → 给模型每层装上 forward hook，捕获矩阵乘法的输入输出
4. model.eval() + torch.no_grad() 执行一次前向传播
5. uninstall_hooks(_model) 卸载所有 hook，还原模型
6. 遍历 collector.records，把每条记录序列化为 JSON
7. 取最后一个位置的 logits，softmax 后取 Top-10
8. 返回 tokens、token_ids、layers（每层激活）、predictions（Top-10）
```

这是 viz_server.py 与 `hooks.py` 的唯一交互点。`install_hooks` / `uninstall_hooks` 是幂等的，每次推理都重新安装和卸载，不会在调用之间累积状态。

---

#### `main()`

命令行入口：

```
1. 解析 --checkpoint / --host / --port 参数
2. 自动选择设备（cuda → mps → cpu）
3. torch.load(checkpoint) 恢复 tokenizer、config、model
4. 调用 model.load_state_dict() 加载权重
5. model.eval() 设置推理模式
6. app.run(threaded=True) 启动 Flask，开始监听 HTTP 请求
```

`threaded=True` 让 Flask 对每个请求创建独立线程，这样训练 SSE 流和其他短请求可以并发处理（例如训练中途点击"停止"）。

---

## 三、`viz_static/index.html` — 前端页面

### 整体结构

```
HTML 结构：
  .layout
    .sidebar         ← 固定左侧控制面板（260px）
      #cfg-grid      ← 模型配置卡片（由 init() 动态填充）
      训练控制区      ← 文件路径 / 步数 / 学习率 / 开始按钮
      推理控制区      ← prompt 输入 / 运行按钮
      训练状态区      ← 实时步数 + loss 显示
    .content         ← 可滚动主区域
      #status-bar    ← 顶部状态栏（绿/红/灰文字）
      .tabs          ← 三个 Tab 按钮
      #pane-weights  ← 权重矩阵 Tab
        #weights-root
      #pane-training ← 训练过程 Tab
        #loss-chart  ← canvas 折线图
        #training-weights-root
      #pane-inference← 推理过程 Tab
        #inference-root
  #modal             ← 全局弹窗（点击权重矩阵时出现）
```

页面是**纯静态单文件**，不依赖任何第三方 JS 库，所有逻辑都在 `<script>` 块中。

---

### CSS 变量

```css
:root {
  --bg:      #0d1117;   /* 深黑背景 */
  --panel:   #161b22;   /* 侧边栏/卡片背景 */
  --border:  #30363d;   /* 边框颜色 */
  --accent:  #58a6ff;   /* 蓝色强调色（标题、高亮） */
  --yellow:  #d29922;   /* 训练时权重变化提示 */
  --green / --red      /* 状态栏成功/错误 */
}
```

配色模仿 GitHub Dark 主题，代码可读性高，和终端工具风格统一。

---

### JavaScript 函数详解

#### 颜色映射

##### `COLOR_STOPS`（常量数组）

定义 RdBu 发散色标的 7 个控制点，从深蓝（−1）经白色（0）到深红（+1）：

```js
[-1.0, [49, 54, 149]]    // 深蓝
[-0.5, [69, 117, 180]]   // 蓝
[-0.25,[145,191,219]]    // 浅蓝
[0.0,  [247,247,247]]    // 白
[0.25, [253,174, 97]]    // 浅橙
[0.5,  [215, 48, 39]]    // 红
[1.0,  [165,  0, 38]]    // 深红
```

这个配色与 matplotlib 的 `RdBu_r` 完全一致，熟悉 Python 可视化的用户看起来没有歧义。

##### `valToRGB(v) → [R, G, B]`

输入 `v ∈ [-1, 1]`，在 `COLOR_STOPS` 中找到 `v` 所在的区间，做**线性插值**得到 RGB 三元组。

这是热力图渲染的颜色核心函数，被 `renderHeatmap` 在像素循环中逐值调用。

##### `diffToRGB(v, maxV) → [R, G, B]`

专门用于**训练变化量热力图**的颜色函数。颜色从白（变化量 = 0）渐变到橙（变化量 = maxV）。单色渐变比双色更直观——只需关注"哪里变化了多少"，无需区分正负。

---

#### Canvas 热力图渲染

##### `renderHeatmap(canvas, data2d, opts)`

这是整个可视化工具最核心的渲染函数，被权重格子、激活展示、弹窗放大三处复用。

```
参数：
  canvas   — HTMLCanvasElement，函数内部会重新设置 width/height
  data2d   — 二维数字数组，data2d[row][col] 是矩阵的一个值
  opts：
    maxPx     — 显示尺寸上限（像素），默认 160
    absMax    — 归一化分母，不传则自动计算矩阵的绝对值最大值
    diffMode  — 若为 true，改用 diffToRGB 渲染（橙色变化量色标）
```

**核心流程：**

```
1. 计算像素缩放比 scale = floor(maxPx / max(rows, cols))
   （小矩阵会被放大，使每个值至少占 scale×scale 个像素）
   scale 上限为 8，避免单个值放得太大

2. 设置 canvas.width = cols*scale，canvas.height = rows*scale

3. 用 ctx.createImageData() 创建像素缓冲区
   （比逐个 fillRect 快得多，一次写完所有像素再 putImageData）

4. 双重循环 r, c：
   - 计算归一化值 norm = data2d[r][c] / absMax
   - 调用 valToRGB(norm) 或 diffToRGB(...) 得到 [R, G, B]
   - 用内层双重循环 dr, dc 把 scale×scale 个像素全部写为同一颜色

5. ctx.putImageData(img, 0, 0) 一次性刷新到 canvas
```

使用 `ImageData` 而非 `fillRect` 的原因：对于 384×128 的矩阵（49152 个值），`fillRect` 需要 49152 次 Canvas API 调用，`ImageData` 只需 1 次，性能差距明显。

---

#### Tab 切换

##### `showTab(name)`

通过给 `.tab` 和 `.pane` 元素切换 `active` CSS 类来控制显示/隐藏。`active` 在 CSS 中对应 `display: block`，非 active 对应 `display: none`。

---

#### 工具函数

##### `setStatus(msg, cls)`

更新顶部状态栏的文字和颜色（`ok`=绿、`err`=红、`info`=灰）。所有 API 调用前后都会调用此函数反馈进度。

##### `fmtStats(s)`

把 `{mean, std, min, max}` 格式化为带 `<b>` 加粗的 HTML 字符串，插入到每个权重卡片的统计区域。

##### `shortName(name)`

把 PyTorch 的完整参数名（如 `transformer.h.0.attn.c_attn.weight`）转换为简洁的显示名（如 `L0 · c_attn (QKV).weight`），通过一系列正则 `.replace()` 实现。

---

#### 权重网格系统

权重网格由三层组成：格子（`w-cell`）→ 行（`wg-row`）→ 节（`wg-section`），每层都按需动态创建。

两个 Tab（"权重矩阵"和"训练过程"）各自有一个独立的网格容器（`weights-root` 和 `training-weights-root`），用 `_rootCells` 对象分开管理状态，互不干扰。

##### `_rootCells`（对象）

```js
_rootCells = {
  "weights-root": {
    "transformer.h.0.attn.c_attn.weight": { cell, canvas, statsDiv, diffBadge, wInfo },
    ...
  },
  "training-weights-root": { ... }
}
```

以 rootId 为一级 key、参数名为二级 key，存储每个格子的 DOM 引用和当前数据。这样同一个参数在两个 Tab 里是完全独立的 DOM 节点，更新其中一个不影响另一个。

##### `_cells(rootId) → object`

`_rootCells` 的惰性访问器，若该 rootId 尚无记录则自动初始化为空对象。

##### `buildWeightCell(name, wInfo, row, rootId, showDiff)`

在指定 `row`（DOM 元素）中为参数 `name` 创建一个权重格子，或调用 `updateWeightCell` 更新已有格子。

创建时构建的 DOM 结构：

```
.w-cell
  .w-name       ← shortName(name)
  .w-shape      ← "384 × 128"
  canvas        ← 热力图
  .w-stats      ← mean/std/min/max
  .diff-badge   ← "Δmax 1.23e-4"（训练时才显示）
```

点击事件绑定为 `() => openModal(name, _cells(rootId)[name].wInfo)`，注意这里不捕获闭包里的 `wInfo`，而是每次点击时实时从 `_rootCells` 里读取**最新的** wInfo，确保弹窗展示的是更新后的数据。

##### `updateWeightCell(name, wInfo, rootId, showDiff)`

在格子已存在时更新其内容：

- 若 `showDiff && wInfo.diff` 为真：用 `diffToRGB` 渲染变化量热力图，显示橙色 Δmax 徽章，给格子加 `updated` CSS 动画类（黄色闪烁边框，0.8 秒后自动移除）
- 否则：用 `valToRGB` 渲染当前权重值，隐藏 Δmax 徽章
- 无论如何都更新统计数字

##### `buildWeightsGrid(weights, rootId, showDiff)`

接收从 `/api/weights` 或训练 SSE 事件里拿到的完整权重字典，按参数名的前缀把参数分到对应的节（section）：

```
transformer.wte.* / transformer.wpe.*   → "── 嵌入层"
transformer.h.N.*                       → "── Layer N"
transformer.ln_f.* / lm_head.*         → "── 输出层"
```

节按顺序排列（嵌入 → Layer 0 → 1 → ... → 输出），对每个参数调用 `buildWeightCell`。节的 DOM 节点用 `data-sec` 属性缓存，避免重复创建。

---

#### 弹窗

##### `openModal(name, wInfo)`

打开全局弹窗，以更大尺寸（最大 600px 或屏幕宽度的 80%）渲染同一个矩阵。调用 `renderHeatmap` 时传入更大的 `maxPx`，让每个值有更多像素，细节更清晰。

##### `closeModal(e)`

关闭弹窗。若传入了点击事件 `e`，只有点击弹窗背景层（`e.target === #modal`）才关闭，点击内部 `.modal` 区域不关闭（防止误触）。

---

#### Loss 折线图

##### `appendLoss(step, loss)`

把新的 `(step, loss)` 对 push 进 `_losses` 数组，然后立即调用 `drawLossChart()` 重绘。由 `onTrainEvent` 在每个训练步事件里调用。

##### `drawLossChart()`

纯 Canvas 2D API 实现的折线图，无任何外部图表库依赖。

```
绘制流程：
  1. 清除画布，填充深色背景
  2. 计算 X/Y 轴的数据范围（Y 轴留 5% 边距）
  3. 定义坐标变换函数 toX(step) 和 toY(loss)
  4. 绘制 6 条水平网格线，在左侧标注 loss 刻度值
  5. 在底部标注起止 step 编号
  6. 绘制蓝色折线（逐点 lineTo）
  7. 在最后一个点画实心圆，旁边标注当前 loss 数值
```

每次训练步事件都会调用此函数完整重绘一遍画布，不做增量更新（数据量小，全量重绘比差量更新简单可靠）。

---

#### 训练控制

##### `startTraining()`

读取侧边栏的四个输入框参数，发起 `POST /api/train` 请求，然后用 `fetch` + `ReadableStream` 解析响应流：

```
内部 read() 递归函数：
  1. reader.read() 获取下一块二进制数据
  2. 追加到 buf 字符串缓冲区
  3. 按 "\n\n" 分割 buf，得到完整的 SSE 消息
  4. 对每条消息去掉 "data: " 前缀，JSON.parse 后调用 onTrainEvent()
  5. 递归调用 read() 等待下一块
```

使用 `fetch` + `ReadableStream` 而非 `EventSource` 的原因：`EventSource` 只支持 GET 请求，无法携带 JSON body 传递训练参数。

每次调用 `startTraining()` 都会先清空 `_losses`、清空 training-weights-root 的 DOM 和缓存，再重新开始。

##### `onTrainEvent(ev)`

处理一条解析好的训练事件：

- `type === "error"` → 调用 `setStatus` 显示红色错误
- `type === "step"` → 调用 `appendLoss` 更新折线图，更新步数/loss 文字；若事件含 `weights` 字段，调用 `buildWeightsGrid` 更新两个网格（训练 Tab 显示 diff，权重 Tab 显示当前值）
- `type === "done"` → 调用 `setStatus` 显示绿色完成提示，调用 `onTrainDone` 恢复按钮状态

##### `onTrainDone()`

恢复"开始训练"按钮可见、隐藏"停止"按钮。由流结束和 `stopTraining` 都会调用。

##### `stopTraining()`

向 `/api/stop` 发 POST 请求（设置服务端的 `_train_stop` 事件），然后本地立即调用 `onTrainDone()` 更新 UI。前端不等待流真正结束。

---

#### 推理可视化

##### `runInference()`

读取 prompt，发起 `POST /api/infer`，拿到 JSON 数据后调用 `renderInference(data)`。

##### `renderInference(data)`

接收推理 API 的完整响应，渲染三个部分：

**① Token strip**

```html
<div class="token-strip">
  <div class="token prompt">h</div>
  <div class="token prompt">e</div>
  ...
</div>
```

每个 token 是一个带蓝色边框的小方块，hover title 显示字符和 token ID。空格字符显示为 `·`。

**② 逐层激活展示**

先把 `data.layers`（`MatmulRecord` 列表）按 `layer_idx` 分组，同一层的多个操作（c_attn、qk、av、c_proj_attn、c_fc、c_proj_mlp）归为一组：

```
Layer 0 block：
  [c_attn 输入 T×C] [c_attn 输出 T×3C]
  [qk 输入 T×d]     [qk 输出 T×T]
  [av 输入 T×T]     [av 输出 T×d]
  ...

lm_head block：
  [输入 T×C] [输出 T×vocab]
```

每个 block 的标题栏可点击折叠/展开（通过切换 `body.style.display`）。

**③ Top-10 预测**

每行一个候选 token，包含字符方块、横向进度条（宽度按概率比例）、百分比文字、logit 数值。

##### `makeActCell(label, shape, data2d, stats, isAttention)`

创建一个激活值展示格子（`act-cell`），包含标签、形状、canvas 热力图、统计数字。

`isAttention` 控制 canvas 的 `maxPx`：注意力矩阵（T×T）使用 120px（通常 T 较小），其他激活使用 180px（T×C 较宽）。

---

#### 初始化

##### `init()`（async，页面加载时自动调用）

```
1. GET /api/config
   → 填充侧边栏的模型配置卡片
   → 将 config 保存到全局 _config

2. GET /api/weights
   → 调用 buildWeightsGrid(weights, 'weights-root', false)
   → 在"权重矩阵" Tab 渲染初始热力图网格
```

两步之间有状态栏文字更新，出错时显示红色提示。

---

## 四、与模型代码的交互方式

### 权重读取（只读）

`viz_server.py` 通过 `_model.named_parameters()` 遍历模型参数：

```
model.py 定义：
  GPT2
    transformer.wte.weight      (Embedding)
    transformer.wpe.weight      (Embedding)
    transformer.h.0.ln_1.weight (LayerNorm)
    transformer.h.0.attn.c_attn.weight (Linear)
    ...（共约 36 个参数张量）

viz_server._all_weights() 读取：
  调用 model.named_parameters()，过滤 dim==2 或 LayerNorm
  detach().cpu().float() 复制到 CPU，不影响训练图
```

整个过程**不修改模型**，只读取参数值。

### 训练（原地更新 `_model`）

`api_train()` 的 `stream()` 生成器直接在 `_model` 上运行：

```
_model.train()              → 开启 dropout
optimizer = AdamW(_model.parameters())
loss.backward()             → 计算梯度到 _model 的 .grad 字段
optimizer.step()            → 原地更新 _model 的参数
_model.eval()               → 训练结束，关闭 dropout
```

训练完成后 `_model` 的参数已被修改，后续的 `/api/weights` 和 `/api/infer` 请求都会看到新的参数值。

### 推理激活捕获（通过 hooks.py）

`api_infer()` 与模型的交互分三步：

```
① install_hooks(_model, collector)
   hooks.py 把 CausalSelfAttention.forward 替换为 patched_forward，
   同时给 MLP 的 c_fc / c_proj 和 lm_head 注册 register_forward_hook。
   _model 的外部行为不变，只是每次矩阵乘法后额外调用 collector.record()。

② model(idx)
   正常前向传播，patched_forward 和 forward_hook 在内部自动触发，
   把每层的 input_mat / output_mat（batch=0，已移到 CPU）写入 collector.records。

③ uninstall_hooks(_model)
   还原所有 forward 方法，移除所有 hook handle。
   _model 恢复到干净状态，对后续调用无影响。
```

`hooks.py` 捕获的数据类型是 `MatmulRecord`，字段与 `api_infer()` 序列化的 JSON 字段一一对应：

| MatmulRecord 字段 | JSON 字段 |
|---|---|
| `layer_idx` | `layer_idx` |
| `op_name` | `op_name` |
| `input_mat` | `in_shape` / `in_stats` / `in_data` |
| `output_mat` | `out_shape` / `out_stats` / `out_data` |

---

## 五、数据流全景图

```
浏览器 index.html                          服务器 viz_server.py
─────────────────                          ───────────────────

页面加载
  init()
  ──GET /api/config──────────────────────→ api_config()
  ←──{n_layer,n_embd,…}──────────────────    return asdict(_config)
  显示配置卡片

  ──GET /api/weights────────────────────→ api_weights()
  ←──{name:{shape,data,stats},…}────────    _all_weights()
  渲染权重热力图网格                           ↑ model.named_parameters()

用户点击"开始训练"
  startTraining()
  ──POST /api/train {file,steps,lr,…}──→ api_train()
                                            stream() 生成器：
  ←──data:{"type":"step","loss":2.3}────      训练一步
  appendLoss(), 更新折线图                     yield _sse({"step":1,"loss":2.3})

  ←──data:{"type":"step","weights":{…}}─     每 N 步
  buildWeightsGrid(…, showDiff=true)           yield _sse({"weights":{cur+diff}})
  更新热力图（橙色 diff 叠加）

  ←──data:{"type":"done"}───────────────     训练结束
  onTrainDone()                               model.eval()

用户点击"运行推理"
  runInference()
  ──POST /api/infer {prompt:"hello"}───→ api_infer()
                                            install_hooks(model, collector)
                                            model(idx)      ← 触发 hooks
                                            uninstall_hooks(model)
  ←──{tokens,layers:[…],predictions}────  return jsonify({…})
  renderInference()
  逐层展示激活热力图 + Top-10 预测
```
