import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_gpt2.config import GPTConfig
from my_gpt2.data import TinyTextDataset
from my_gpt2.manual_model import ManualGPT2
from my_gpt2.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training the hand-written GPT model."""
    parser = argparse.ArgumentParser(description="Train a tiny hand-written GPT-2 style model.")
    parser.add_argument("--input", type=Path, required=True, help="训练文本文件路径，会被读取为 UTF-8 文本。")
    parser.add_argument("--steps", type=int, default=500, help="训练更新步数。")
    parser.add_argument("--batch-size", type=int, default=16, help="每个 batch 中并行训练的文本片段数量。")
    parser.add_argument("--block-size", type=int, default=12, help="每个训练样本的上下文长度。")
    parser.add_argument("--n-layer", type=int, default=4, help="Transformer block 的层数。")
    parser.add_argument("--n-head", type=int, default=4, help="每层多头注意力的 head 数量。")
    parser.add_argument("--n-embd", type=int, default=128, help="token 嵌入和隐藏状态的向量维度。")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW 优化器使用的学习率。")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints/manual"),
        help="保存 ManualGPT2 checkpoint 的输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    """Train ManualGPT2 with the same data flow as train.py."""
    args = parse_args()

    text = args.input.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    dataset = TinyTextDataset(tokenizer.encode(text), block_size=args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = ManualGPT2(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    progress = tqdm(total=args.steps, desc="manual training")
    while step < args.steps:
        for x, y in loader:
            # x 和 y 都是 LongTensor，形状 (batch_size, block_size), y是 x 右移一位
            x = x.to(device)
            y = y.to(device)

            # 前向传播：传入 targets=y 时模型内部计算交叉熵 loss
            # 等价于 model.forward(x, y)
            _, loss = model(x, y)
            assert loss is not None
            

            # 从损失（即与目标值的误差）计算梯度
            optimizer.zero_grad(set_to_none=True)  # 清空上一步的梯度（set_to_none 更省内存）
            # 反向传播计算梯度
            loss.backward()
            optimizer.step()

            step += 1

            # 更新进度条
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step >= args.steps:
                break
    progress.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "config": asdict(config),
        "tokenizer": tokenizer.to_dict(),
        "model_type": "ManualGPT2",
    }
    torch.save(checkpoint, args.out_dir / "latest.pt")


if __name__ == "__main__":
    main()
