"""
plotter.py — 热力图生成器

职责：接收 MatmulCollector 的数据，用 matplotlib 输出 PNG。

每个 Transformer Block 生成一张图（layer_NN.png），包含 3×3 子图：
  行0: c_attn 输出(T×3C) | Q@K^T att分数(T×T) | att@V 输出(T×d)
  行1: c_proj 输入(T×C)  | c_proj 输出(T×C)   | c_fc 输出(T×4C)
  行2: c_proj_mlp 输入   | c_proj_mlp 输出    | (空)

另生成 lm_head.png（只展示 top-K 词表列，避免过宽）。
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

from my_gpt2.hooks import MatmulCollector, MatmulRecord

# 每个 op 的展示名称和色图
_OP_META: dict[str, tuple[str, str]] = {
    "c_attn":      ("c_attn  input→output\n(T×C → T×3C)",  "RdBu_r"),
    "qk":          ("Q @ K^T  att score\n(T×d @ d×T → T×T, head 0)", "viridis"),
    "av":          ("softmax(att) @ V\n(T×T @ T×d → T×d, head 0)",   "plasma"),
    "c_proj_attn": ("c_proj_attn  input\n(T×C)",           "RdBu_r"),
    "c_proj_attn_out": ("c_proj_attn  output\n(T×C)",      "RdBu_r"),
    "c_fc":        ("c_fc  output\n(T×C → T×4C)",          "RdBu_r"),
    "c_proj_mlp":  ("c_proj_mlp  input\n(T×4C)",           "RdBu_r"),
    "c_proj_mlp_out": ("c_proj_mlp  output\n(T×C)",        "RdBu_r"),
    "lm_head":     ("lm_head  output\n(T×vocab, top-K cols)", "coolwarm"),
}


def _draw_heatmap(
    ax: plt.Axes,
    mat: torch.Tensor,
    title: str,
    cmap: str,
) -> None:
    """在指定 Axes 上绘制热力图，附色条和标题"""
    data = mat.numpy()
    # RdBu_r 色图以 0 为中心（正负对称），其余用 min-max
    norm = mcolors.CenteredNorm() if "RdBu" in cmap else None
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=7, pad=3)
    ax.set_xlabel(f"dim  ({data.shape[1]})", fontsize=6)
    ax.set_ylabel(f"token  ({data.shape[0]})", fontsize=6)
    ax.tick_params(labelsize=5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _get_mat(
    by_op: dict[str, MatmulRecord],
    op_name: str,
    attr: str,
    max_tokens: int = 64,
) -> torch.Tensor | None:
    """从 by_op 字典中取矩阵，截断到 max_tokens 行，找不到返回 None"""
    rec = by_op.get(op_name)
    if rec is None:
        return None
    mat = getattr(rec, attr)
    return mat[:max_tokens]


def plot_layer(
    layer_idx: int,
    records: list[MatmulRecord],
    output_dir: Path,
    token_labels: list[str] | None = None,
) -> Path:
    """
    为单个 Transformer Block 生成一张 3×3 热力图 PNG。
    返回输出文件路径。
    """
    by_op: dict[str, MatmulRecord] = {r.op_name: r for r in records}
    MAX_T = 64  # 最多展示 token 数，避免图过大

    # 3 列 × 3 行子图，定义每格要展示的内容
    # 格式：(op_name, attr, title_key)，None 表示留空
    layout = [
        # 行 0：Attention 核心操作
        ("c_attn",      "output_mat", "c_attn"),
        ("qk",          "output_mat", "qk"),
        ("av",          "output_mat", "av"),
        # 行 1：输出投影 + MLP 升维
        ("c_proj_attn", "input_mat",  "c_proj_attn"),
        ("c_proj_attn", "output_mat", "c_proj_attn_out"),
        ("c_fc",        "output_mat", "c_fc"),
        # 行 2：MLP 降维
        ("c_proj_mlp",  "input_mat",  "c_proj_mlp"),
        ("c_proj_mlp",  "output_mat", "c_proj_mlp_out"),
        None,   # 留空
    ]

    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    fig.suptitle(
        f"Layer {layer_idx}  —  Matrix Multiplication Heatmaps",
        fontsize=13, y=0.99,
    )

    for idx, cell in enumerate(layout):
        row, col = divmod(idx, 3)
        ax = axes[row][col]

        if cell is None:
            ax.axis("off")
            continue

        op_name, attr, title_key = cell
        mat = _get_mat(by_op, op_name, attr, MAX_T)

        if mat is None:
            ax.axis("off")
            ax.set_title(f"{op_name}\n(no data)", fontsize=7)
            continue

        title, cmap = _OP_META[title_key]
        title_with_shape = f"{title}\nshape={tuple(mat.shape)}"
        _draw_heatmap(ax, mat, title_with_shape, cmap)

        # 短序列时在 y 轴显示 token 字符标签
        if token_labels is not None and mat.shape[0] <= 24:
            labels = token_labels[:mat.shape[0]]
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = output_dir / f"layer_{layer_idx:02d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lm_head(
    record: MatmulRecord,
    output_dir: Path,
    top_k_vocab: int = 50,
    token_labels: list[str] | None = None,
) -> Path:
    """
    lm_head 输出形状 (T, vocab_size)，vocab_size 可能很大。
    只展示概率最高的 top_k_vocab 列（取每行 top-k 的并集）。
    """
    mat = record.output_mat  # (T, vocab)

    k = min(top_k_vocab, mat.shape[1])
    top_idx = mat.topk(k, dim=-1).indices          # (T, k)
    col_set = torch.unique(top_idx.flatten()).sort().values
    mat_trimmed = mat[:, col_set]                   # (T, ≤top_k_vocab)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        f"lm_head  —  Logits Heatmap\n"
        f"full shape={tuple(mat.shape)},  shown={tuple(mat_trimmed.shape)}  (top-{k} vocab cols)",
        fontsize=11,
    )
    _draw_heatmap(ax, mat_trimmed, "", "coolwarm")

    if token_labels is not None and mat_trimmed.shape[0] <= 24:
        labels = token_labels[:mat_trimmed.shape[0]]
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)

    plt.tight_layout()
    out_path = output_dir / "lm_head.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_plots(
    collector: MatmulCollector,
    output_dir: Path,
    n_layers: int,
    token_labels: list[str] | None = None,
) -> list[Path]:
    """
    对所有层批量生成 PNG，返回所有输出文件路径。

    调用方式：
        paths = generate_all_plots(collector, Path("viz_output"), config.n_layer, token_labels)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for layer_idx in range(n_layers):
        records = collector.by_layer(layer_idx)
        if records:
            p = plot_layer(layer_idx, records, output_dir, token_labels)
            paths.append(p)
            print(f"  ✓ {p}")

    # lm_head 用 layer_idx=-1
    lm_records = [r for r in collector.records if r.op_name == "lm_head"]
    if lm_records:
        p = plot_lm_head(lm_records[-1], output_dir, token_labels=token_labels)
        paths.append(p)
        print(f"  ✓ {p}")

    return paths
