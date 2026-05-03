"""
trace.py — 推理过程数值追踪 CLI

把每一步矩阵乘法的实际数值打印到终端（或保存为文本文件），
比热力图更直接：能看到真实的数字、形状、统计量。

用法：
  python -m my_gpt2.trace --checkpoint checkpoints/latest.pt --prompt "hello"
  python -m my_gpt2.trace --checkpoint checkpoints/latest.pt --prompt "hello" --out trace.txt
"""

import argparse
from pathlib import Path

import torch

from my_gpt2.config import GPTConfig
from my_gpt2.hooks import MatmulCollector, MatmulRecord, install_hooks, uninstall_hooks
from my_gpt2.model import GPT2
from my_gpt2.tokenizer import CharTokenizer

_tokenizer: CharTokenizer | None = None

# 每个 op 的中文说明
_OP_DESC = {
    "c_attn":      "QKV 联合投影  (T×C → T×3C)",
    "qk":          "Q @ K^T  原始注意力分数  (T×d → T×T，head 0)",
    "av":          "softmax(att) @ V  加权求和  (T×T @ T×d → T×d，head 0)",
    "c_proj_attn": "c_proj  输出投影  (T×C → T×C)",
    "c_fc":        "c_fc  MLP 升维  (T×C → T×4C)",
    "c_proj_mlp":  "c_proj  MLP 降维  (T×4C → T×C)",
    "lm_head":     "lm_head  词表投影  (T×C → T×vocab)",
}

# 打印时每行最多展示的列数
_MAX_COLS = 8
# 打印时最多展示的行数（token 数）
_MAX_ROWS = 8


# ── 格式化工具 ─────────────────────────────────────────────────────────────────

def _stats(mat: torch.Tensor) -> str:
    """返回张量的统计摘要字符串"""
    return (
        f"mean={mat.mean():+.4f}  "
        f"std={mat.std():.4f}  "
        f"min={mat.min():+.4f}  "
        f"max={mat.max():+.4f}"
    )


def _fmt_val(v: float) -> str:
    """把单个浮点数格式化为固定宽度字符串"""
    return f"{v:+7.4f}"


def _print_matrix(
    mat: torch.Tensor,
    row_labels: list[str] | None,
    col_header: str,
    max_rows: int = _MAX_ROWS,
    max_cols: int = _MAX_COLS,
    lines: list[str] | None = None,
) -> None:
    """
    打印矩阵的前 max_rows 行、前 max_cols 列。
    lines 不为 None 时把输出追加到 lines 列表而不是 print。
    """
    def out(s: str = ""):
        if lines is not None:
            lines.append(s)
        else:
            print(s)

    rows = min(mat.shape[0], max_rows)
    cols = min(mat.shape[1], max_cols)
    truncated_r = mat.shape[0] > max_rows
    truncated_c = mat.shape[1] > max_cols

    # 列标题
    col_label_w = max(len(col_header), 6)
    header = f"  {'':>{col_label_w}}  " + "  ".join(f"{'dim'+str(j):>8}" for j in range(cols))
    if truncated_c:
        header += "  ..."
    out(header)

    # 数据行
    for i in range(rows):
        label = (row_labels[i] if row_labels and i < len(row_labels) else str(i))
        row_str = f"  {label:>{col_label_w}}  " + "  ".join(
            _fmt_val(mat[i, j].item()) for j in range(cols)
        )
        if truncated_c:
            row_str += "  ..."
        out(row_str)

    if truncated_r:
        out(f"  {'...':>{col_label_w}}")


def _print_attention_weights(
    softmax_weights: torch.Tensor,   # (T, T)，已经是 softmax 后的概率
    token_labels: list[str],
    lines: list[str] | None = None,
) -> None:
    """
    打印注意力权重矩阵（行=当前 token，列=被关注的 token）。
    因果掩码让上三角为 0，softmax 后下三角是概率。
    """
    def out(s: str = ""):
        if lines is not None:
            lines.append(s)
        else:
            print(s)

    T = softmax_weights.shape[0]
    max_t = min(T, _MAX_ROWS)
    labels = [f"'{c}'" for c in token_labels[:max_t]]
    max_lw = max(len(l) for l in labels)

    out("  （行=当前token 看 列=哪个token，因果掩码保证只能看过去）")
    # 列头
    header = f"  {'':>{max_lw}}  " + "  ".join(f"{l:>8}" for l in labels)
    if T > max_t:
        header += "  ..."
    out(header)

    for i in range(max_t):
        row_str = f"  {labels[i]:>{max_lw}}  "
        for j in range(max_t):
            v = softmax_weights[i, j].item()
            # 未来位置（因果掩码）显示为 ----
            if v < 1e-9 and j > i:
                row_str += f"  {'----':>8}"
            else:
                row_str += f"  {v:>8.4f}"
        if T > max_t:
            row_str += "  ..."
        out(row_str)


# ── 单条记录的打印逻辑 ─────────────────────────────────────────────────────────

def _render_record(
    rec: MatmulRecord,
    token_labels: list[str],
    lines: list[str],
) -> None:
    """把一条 MatmulRecord 渲染成可读文本，追加到 lines"""

    def out(s: str = ""):
        lines.append(s)

    desc = _OP_DESC.get(rec.op_name, rec.op_name)
    layer_str = f"Layer {rec.layer_idx}" if rec.layer_idx >= 0 else "Final"

    out("━" * 72)
    out(f"  {layer_str}  ·  {rec.op_name}")
    out(f"  {desc}")
    out("━" * 72)

    inp = rec.input_mat
    out = rec.output_mat  # 注意：这里 out 被重新赋值了，改用 lines.append

    def p(s: str = ""):
        lines.append(s)

    p(f"  输入  shape={tuple(inp.shape)}  |  {_stats(inp)}")
    p(f"  输出  shape={tuple(rec.output_mat.shape)}  |  {_stats(rec.output_mat)}")
    p()

    # ── 注意力权重：特殊展示 ──────────────────────────────────────────────────
    if rec.op_name == "av":
        # input_mat 是 softmax 后的注意力权重 (T, T)
        p("  【注意力权重 softmax(QK^T/√d)】— 显示每个 token 对其他 token 的关注概率：")
        _print_attention_weights(inp, token_labels, lines)
        p()
        p(f"  att @ V 输出（加权求和后，每个 token 的新表示，前{_MAX_COLS}维）：")
        _print_matrix(rec.output_mat, token_labels, "token", lines=lines)

    # ── lm_head：展示每个 token 位置最可能的下一个 token ─────────────────────
    elif rec.op_name == "lm_head":
        logits = rec.output_mat  # (T, vocab)
        p("  【最后一个 token 位置的 Top-10 预测】：")
        last_logits = logits[-1]   # 取序列最后一位
        top_v, top_i = last_logits.topk(10)
        for rank, (v, idx) in enumerate(zip(top_v.tolist(), top_i.tolist()), 1):
            char = _tokenizer.itos.get(idx, "?") if _tokenizer else "?"
            p(f"    #{rank:2d}  token_id={idx:4d}  char={char!r:4s}  logit={v:+8.4f}")
        p()
        p(f"  所有 token 位置的输出统计（输入 {_MAX_ROWS} 行 × 前 {_MAX_COLS} 列）：")
        _print_matrix(logits, token_labels, "token", lines=lines)

    # ── qk：展示原始分数（softmax 前）────────────────────────────────────────
    elif rec.op_name == "qk":
        p(f"  Q 向量（head 0，每个 token 的查询向量，前 {_MAX_COLS} 维）：")
        _print_matrix(inp, token_labels, "token", lines=lines)
        p()
        p("  原始注意力分数 Q@K^T（softmax 之前，负无穷=被掩码）：")
        # 把 -inf 替换为 -9999 方便打印
        raw_scores = rec.output_mat.clone()
        p("  （正值=更关注，负值=更忽略，-inf=因果掩码）")
        _print_matrix(raw_scores, token_labels, "token", lines=lines)

    # ── 普通线性层：展示输入和输出 ────────────────────────────────────────────
    else:
        p(f"  输入矩阵（前 {_MAX_ROWS} 行 × 前 {_MAX_COLS} 列）：")
        _print_matrix(inp, token_labels, "token", lines=lines)
        p()
        p(f"  输出矩阵（前 {_MAX_ROWS} 行 × 前 {_MAX_COLS} 列）：")
        _print_matrix(rec.output_mat, token_labels, "token", lines=lines)

    p()


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a human-readable numerical trace of GPT-2 inference."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="可选：把 trace 保存到文本文件（同时也会打印到终端）",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="只展示指定层（0-based），不指定则展示全部层",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    global _tokenizer
    checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    _tokenizer = tokenizer
    config = GPTConfig(**checkpoint["config"])
    model = GPT2(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    token_labels = list(args.prompt)

    collector = MatmulCollector()
    install_hooks(model, collector)
    with torch.no_grad():
        _ = model(idx)
    uninstall_hooks(model)

    # ── 构建输出文本 ──────────────────────────────────────────────────────────
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append(f"  GPT-2 推理数值追踪")
    lines.append(f"  prompt = {args.prompt!r}  (tokens: {list(args.prompt)})")
    lines.append(f"  model  = {config.n_layer} layers, {config.n_head} heads, n_embd={config.n_embd}")
    lines.append(f"  共捕获 {len(collector.records)} 个矩阵操作")
    lines.append("=" * 72)
    lines.append("")

    for rec in collector.records:
        # 过滤层（可选）
        if args.layer is not None and rec.layer_idx != args.layer and rec.layer_idx != -1:
            continue
        _render_record(rec, token_labels, lines)

    text = "\n".join(lines)

    # 打印到终端
    print(text)

    # 可选：保存到文件
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\n已保存到 {args.out}")


if __name__ == "__main__":
    main()
