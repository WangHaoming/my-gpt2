"""
inspect.py — 模型结构与参数检查器

读取 checkpoint，打印：
  - 模型超参数
  - Embedding 矩阵形状与样本值
  - 每个 Block 的 Attention / MLP 层参数形状与样本值
  - lm_head 参数
  - 全局参数量统计

用法：
  python -m my_gpt2.inspect --checkpoint checkpoints/latest.pt
  python -m my_gpt2.inspect --checkpoint checkpoints/latest.pt --values   # 展示具体数值
"""

import argparse
from pathlib import Path

import torch


# ── 格式化工具 ─────────────────────────────────────────────────────────────────

def _stats(t: torch.Tensor) -> str:
    return (
        f"mean={t.float().mean():+.4f}  "
        f"std={t.float().std():.4f}  "
        f"min={t.float().min():+.4f}  "
        f"max={t.float().max():+.4f}"
    )


def _fmt_row(t: torch.Tensor, max_cols: int = 8) -> str:
    """把 1D 张量格式化为一行，超出 max_cols 时显示 ..."""
    vals = [f"{v:+.4f}" for v in t[:max_cols].tolist()]
    suffix = "  ..." if t.shape[0] > max_cols else ""
    return "  ".join(vals) + suffix


def _print_matrix(t: torch.Tensor, max_rows: int = 6, max_cols: int = 8,
                  indent: str = "    ") -> None:
    """打印矩阵前 max_rows 行 × 前 max_cols 列"""
    for i in range(min(t.shape[0], max_rows)):
        print(f"{indent}[{i:3d}]  {_fmt_row(t[i], max_cols)}")
    if t.shape[0] > max_rows:
        print(f"{indent} ...")


def _section(title: str) -> None:
    print()
    print("━" * 72)
    print(f"  {title}")
    print("━" * 72)


def _param(name: str, t: torch.Tensor, show_values: bool) -> None:
    """打印一个参数的名称、形状、统计量，可选展示具体数值"""
    numel = t.numel()
    print(f"  {name}")
    print(f"    形状: {tuple(t.shape)}    参数量: {numel:,}")
    print(f"    统计: {_stats(t)}")
    if show_values:
        if t.dim() == 1:
            print(f"    数值: {_fmt_row(t)}")
        else:
            print(f"    数值（前 6 行 × 前 8 列）：")
            _print_matrix(t)
    print()


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print GPT-2 model structure and parameter info from a checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help=".pt checkpoint 文件路径")
    parser.add_argument("--values", action="store_true",
                        help="同时打印参数的具体数值（默认只打印统计量）")
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = ck["config"]
    state = ck["model"]
    show = args.values

    n_layer  = cfg["n_layer"]
    n_head   = cfg["n_head"]
    n_embd   = cfg["n_embd"]
    vocab_size = cfg["vocab_size"]
    block_size = cfg["block_size"]
    head_dim = n_embd // n_head

    # ── 1. 超参数 ──────────────────────────────────────────────────────────────
    _section("模型超参数")
    print(f"  vocab_size  = {vocab_size}   （词表大小，字符种数）")
    print(f"  block_size  = {block_size}   （最大序列长度 / 上下文窗口）")
    print(f"  n_layer     = {n_layer}    （Transformer Block 层数）")
    print(f"  n_head      = {n_head}    （每层注意力头数）")
    print(f"  n_embd      = {n_embd}  （嵌入维度 C）")
    print(f"  head_dim    = {head_dim}   （每个头的维度 d = C / n_head）")
    print(f"  bias        = {cfg['bias']}")
    print(f"  dropout     = {cfg['dropout']}")

    # ── 2. Embedding ──────────────────────────────────────────────────────────
    _section("Embedding 层")

    wte = state["transformer.wte.weight"]   # (vocab_size, n_embd)
    wpe = state["transformer.wpe.weight"]   # (block_size, n_embd)

    print("  【wte】Token Embedding  —  每个词元 ID → 一个 n_embd 维向量")
    print(f"    形状: {tuple(wte.shape)}  =  vocab_size × n_embd  =  {vocab_size} × {n_embd}")
    print(f"    统计: {_stats(wte)}")
    if show:
        print(f"    数值（前 6 个词元 × 前 8 维）：")
        _print_matrix(wte)
    print()

    print("  【wpe】Position Embedding  —  每个位置索引 → 一个 n_embd 维向量")
    print(f"    形状: {tuple(wpe.shape)}  =  block_size × n_embd  =  {block_size} × {n_embd}")
    print(f"    统计: {_stats(wpe)}")
    if show:
        print(f"    数值（前 6 个位置 × 前 8 维）：")
        _print_matrix(wpe)
    print()

    print("  ※ wte 与 lm_head.weight 共享同一个矩阵（权重绑定）")

    # ── 3. 每个 Block ──────────────────────────────────────────────────────────
    _section(f"Transformer Block（共 {n_layer} 层，每层结构相同）")

    print(f"  每个 Block 内的操作流程：")
    print(f"    LayerNorm (ln_1)")
    print(f"    └─ CausalSelfAttention")
    print(f"         1 次线性层  c_attn  ({n_embd} → {3*n_embd})  生成 Q、K、V")
    print(f"         {n_head} 个注意力头，每头独立计算：")
    print(f"           Q @ K^T  ({head_dim} × {head_dim} → T×T)  注意力分数")
    print(f"           softmax(·/√{head_dim}) + 因果掩码")
    print(f"           att @ V  (T×T @ T×{head_dim} → T×{head_dim})")
    print(f"         拼接 {n_head} 个头后 1 次线性层  c_proj  ({n_embd} → {n_embd})")
    print(f"    LayerNorm (ln_2)")
    print(f"    └─ MLP")
    print(f"         1 次线性层  c_fc    ({n_embd} → {4*n_embd})")
    print(f"         GELU 激活")
    print(f"         1 次线性层  c_proj  ({4*n_embd} → {n_embd})")
    print()
    print(f"  每层线性层数量：c_attn(1) + c_proj_attn(1) + c_fc(1) + c_proj_mlp(1) = 4 个")
    print(f"  每层注意力矩阵运算：Q@K^T × {n_head}头 + att@V × {n_head}头 = {2*n_head} 次")

    for li in range(n_layer):
        prefix = f"transformer.h.{li}"
        print()
        print(f"  ┌─ Layer {li} " + "─" * 55)

        # LayerNorm 1
        ln1_w = state[f"{prefix}.ln_1.weight"]
        ln1_b = state[f"{prefix}.ln_1.bias"]
        print(f"  │  [LayerNorm ln_1]  weight {tuple(ln1_w.shape)}  bias {tuple(ln1_b.shape)}")
        if show:
            print(f"  │    weight: {_fmt_row(ln1_w)}")
            print(f"  │    bias:   {_fmt_row(ln1_b)}")

        # c_attn
        ca_w = state[f"{prefix}.attn.c_attn.weight"]   # (3C, C)
        ca_b = state[f"{prefix}.attn.c_attn.bias"]     # (3C,)
        print(f"  │  [c_attn]  weight {tuple(ca_w.shape)} = 3×n_embd × n_embd")
        print(f"  │            bias   {tuple(ca_b.shape)}")
        print(f"  │            统计(weight): {_stats(ca_w)}")
        if show:
            # 分别展示 Q/K/V 部分
            q_w = ca_w[:n_embd]
            k_w = ca_w[n_embd:2*n_embd]
            v_w = ca_w[2*n_embd:]
            print(f"  │    W_Q ({tuple(q_w.shape)}) 前 4 行×前 8 列：")
            _print_matrix(q_w, max_rows=4, indent="  │      ")
            print(f"  │    W_K ({tuple(k_w.shape)}) 前 4 行×前 8 列：")
            _print_matrix(k_w, max_rows=4, indent="  │      ")
            print(f"  │    W_V ({tuple(v_w.shape)}) 前 4 行×前 8 列：")
            _print_matrix(v_w, max_rows=4, indent="  │      ")

        # c_proj (attention)
        cp_w = state[f"{prefix}.attn.c_proj.weight"]   # (C, C)
        cp_b = state[f"{prefix}.attn.c_proj.bias"]
        print(f"  │  [c_proj attn]  weight {tuple(cp_w.shape)}  bias {tuple(cp_b.shape)}")
        print(f"  │            统计(weight): {_stats(cp_w)}")
        if show:
            print(f"  │    前 4 行×前 8 列：")
            _print_matrix(cp_w, max_rows=4, indent="  │      ")

        # LayerNorm 2
        ln2_w = state[f"{prefix}.ln_2.weight"]
        ln2_b = state[f"{prefix}.ln_2.bias"]
        print(f"  │  [LayerNorm ln_2]  weight {tuple(ln2_w.shape)}  bias {tuple(ln2_b.shape)}")

        # c_fc
        fc_w = state[f"{prefix}.mlp.c_fc.weight"]      # (4C, C)
        fc_b = state[f"{prefix}.mlp.c_fc.bias"]
        print(f"  │  [c_fc  MLP升维]  weight {tuple(fc_w.shape)} = 4×n_embd × n_embd")
        print(f"  │            bias   {tuple(fc_b.shape)}")
        print(f"  │            统计(weight): {_stats(fc_w)}")
        if show:
            print(f"  │    前 4 行×前 8 列：")
            _print_matrix(fc_w, max_rows=4, indent="  │      ")

        # c_proj (mlp)
        mp_w = state[f"{prefix}.mlp.c_proj.weight"]    # (C, 4C)
        mp_b = state[f"{prefix}.mlp.c_proj.bias"]
        print(f"  │  [c_proj MLP降维]  weight {tuple(mp_w.shape)} = n_embd × 4×n_embd")
        print(f"  │            bias   {tuple(mp_b.shape)}")
        print(f"  │            统计(weight): {_stats(mp_w)}")
        if show:
            print(f"  │    前 4 行×前 8 列：")
            _print_matrix(mp_w, max_rows=4, indent="  │      ")

        print(f"  └─" + "─" * 60)

    # ── 4. 最终 LayerNorm + lm_head ────────────────────────────────────────────
    _section("最终 LayerNorm + lm_head")

    lnf_w = state["transformer.ln_f.weight"]
    lnf_b = state["transformer.ln_f.bias"]
    print(f"  [LayerNorm ln_f]  weight {tuple(lnf_w.shape)}  bias {tuple(lnf_b.shape)}")
    if show:
        print(f"    weight: {_fmt_row(lnf_w)}")
    print()

    lmh_w = state["lm_head.weight"]   # (vocab_size, n_embd)
    print(f"  [lm_head]  将 n_embd 维隐藏状态投影到词表")
    print(f"    形状: {tuple(lmh_w.shape)}  =  vocab_size × n_embd  =  {vocab_size} × {n_embd}")
    print(f"    统计: {_stats(lmh_w)}")
    print(f"    ※ lm_head.weight 与 wte.weight 是同一个矩阵（指针相同）")
    if show:
        print(f"    数值（前 6 行 × 前 8 列）：")
        _print_matrix(lmh_w)

    # ── 5. 参数量汇总 ──────────────────────────────────────────────────────────
    _section("参数量汇总")

    total = 0
    groups = {
        "Embedding (wte + wpe)": 0,
        "Attention (c_attn + c_proj) × 所有层": 0,
        "MLP (c_fc + c_proj) × 所有层": 0,
        "LayerNorm × 所有层": 0,
        "lm_head": 0,
    }
    for name, tensor in state.items():
        n = tensor.numel()
        total += n
        if "wte" in name or "wpe" in name:
            groups["Embedding (wte + wpe)"] += n
        elif "attn" in name and ("c_attn" in name or "c_proj" in name):
            groups["Attention (c_attn + c_proj) × 所有层"] += n
        elif "mlp" in name:
            groups["MLP (c_fc + c_proj) × 所有层"] += n
        elif "ln" in name:
            groups["LayerNorm × 所有层"] += n
        elif "lm_head" in name:
            groups["lm_head"] += n

    for gname, gn in groups.items():
        pct = gn / total * 100
        print(f"  {gname:<40s}  {gn:>8,}  ({pct:5.1f}%)")
    print(f"  {'─'*60}")
    print(f"  {'总参数量':<40s}  {total:>8,}")
    print()
    print(f"  ※ wte 和 lm_head 共享权重，实际不重复存储。")
    print(f"     去重后实际参数量 = {total - lmh_w.numel():,}  "
          f"（减去 lm_head {lmh_w.numel():,} 个）")


if __name__ == "__main__":
    main()
