"""
visualize.py — 推理过程矩阵乘法可视化 CLI

用法：
  python -m my_gpt2.visualize \\
      --checkpoint checkpoints/latest.pt \\
      --prompt "hello" \\
      --output-dir viz_output

输出：
  viz_output/prompt_hello/
    layer_00.png   ← Block 0 的所有矩阵乘法热力图
    layer_01.png
    ...
    lm_head.png    ← 最终词表预测分数

执行流程：
  1. 加载 checkpoint，重建 GPT2 模型
  2. install_hooks(model, collector)
       - MLP c_fc / c_proj 用 register_forward_hook
       - CausalSelfAttention 的 Q@K^T 和 att@V 用替换 forward 方法
       - lm_head 用 register_forward_hook
  3. model(idx) 单次前向传播 → hooks 自动填充 collector
  4. uninstall_hooks(model) 精确恢复原始状态
  5. generate_all_plots() 输出 PNG
"""

import argparse
from pathlib import Path

import torch

from my_gpt2.config import GPTConfig
from my_gpt2.hooks import MatmulCollector, install_hooks, uninstall_hooks
from my_gpt2.model import GPT2
from my_gpt2.plotter import generate_all_plots
from my_gpt2.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize matrix multiplications during GPT-2 inference."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="训练好的 .pt checkpoint 路径",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="输入 prompt 文本，可视化该 prompt 的单次前向传播",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("viz_output"),
        help="PNG 输出根目录（默认 viz_output/）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 设备选择（与 generate.py 保持一致）────────────────────────────────────
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ── 加载 checkpoint ───────────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    config = GPTConfig(**checkpoint["config"])
    model = GPT2(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Model: {config.n_layer} layers, {config.n_head} heads, n_embd={config.n_embd}")
    print(f"Prompt: {args.prompt!r}  (len={len(args.prompt)})")

    # ── 编码 prompt ───────────────────────────────────────────────────────────
    token_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    # token_labels 用于热力图 y 轴标注（每个字符作为一个 token）
    token_labels = list(args.prompt)

    # ── 安装捕获 hook，执行单次前向传播 ──────────────────────────────────────
    collector = MatmulCollector()
    install_hooks(model, collector)

    with torch.no_grad():
        _ = model(idx)   # 触发所有 hook，填充 collector

    uninstall_hooks(model)   # 精确恢复模型，不留副作用

    print(f"Captured {len(collector.records)} matrix operations across {config.n_layer} layers\n")

    # ── 输出目录：viz_output/prompt_<前20字符>/ ──────────────────────────────
    safe_name = args.prompt[:20].replace(" ", "_").replace("/", "_")
    out_dir = args.output_dir / f"prompt_{safe_name}"

    print(f"Generating plots → {out_dir}/")
    paths = generate_all_plots(
        collector=collector,
        output_dir=out_dir,
        n_layers=config.n_layer,
        token_labels=token_labels,
    )

    print(f"\nDone. {len(paths)} PNG files saved.")


if __name__ == "__main__":
    main()
