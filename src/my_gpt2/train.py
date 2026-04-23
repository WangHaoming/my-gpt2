import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_gpt2.config import GPTConfig
from my_gpt2.data import TinyTextDataset
from my_gpt2.model import GPT2
from my_gpt2.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回带属性访问能力的 Namespace 对象。"""
    parser = argparse.ArgumentParser(description="Train a tiny GPT-2 style language model.")

    # --input：训练文本文件路径，必填
    parser.add_argument("--input", type=Path, required=True, help="训练文本文件路径，会被读取为 UTF-8 文本。")

    # --steps：总训练步数，每步执行一次前向 + 反向传播 + 参数更新
    parser.add_argument("--steps", type=int, default=500, help="训练更新步数，每一步执行一次反向传播和参数更新。")

    # --batch-size：每个 batch 并行处理多少条样本，越大训练越稳定但显存占用越多
    parser.add_argument("--batch-size", type=int, default=16, help="每个 batch 中并行训练的文本片段数量。")

    # --block-size：每条样本的 token 数（上下文窗口长度），等于 GPTConfig.block_size
    parser.add_argument("--block-size", type=int, default=12, help="每个训练样本的上下文长度，也就是模型一次看到的 token 数。")

    # 以下三个参数直接对应 GPTConfig 的同名字段
    parser.add_argument("--n-layer", type=int, default=4, help="Transformer block 的层数，层数越多模型容量越大。")
    parser.add_argument("--n-head", type=int, default=4, help="每层多头注意力的 head 数量。")
    parser.add_argument("--n-embd", type=int, default=128, help="token 嵌入和隐藏状态的向量维度。")

    # --lr：AdamW 学习率，3e-4 是 GPT 类模型常用的初始值
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW 优化器使用的学习率。")

    # --out-dir：checkpoint 保存目录，不存在时会自动创建
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"), help="保存模型 checkpoint 的输出目录。")
    return parser.parse_args()


def main() -> None:
    """端到端训练流程：读数据 → 构建 tokenizer/dataset/model → 训练循环 → 保存 checkpoint。"""
    args = parse_args()

    # ── 数据准备 ──────────────────────────────────────────────────────────────
    text = args.input.read_text(encoding="utf-8")

    # 从训练文本构建字符级 tokenizer，统计所有字符并分配 ID
    # tokenizer初始化之后，在内部有两个字典stoi 和 itos，分别是字符到token id 和 token id 到字符的映射
    tokenizer = CharTokenizer.from_text(text)

    # 把全文编码成 token ID 列表，再切成 (x, y) 对的滑动窗口数据集
    dataset = TinyTextDataset(tokenizer.encode(text), block_size=args.block_size)

    # DataLoader 负责批量采样：shuffle=True 打乱顺序，drop_last=True 丢弃不足一个 batch 的尾部
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ── 设备选择 ──────────────────────────────────────────────────────────────
    # 优先 NVIDIA GPU（cuda），其次 Apple Silicon GPU（mps），最后 CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # ── 模型构建 ──────────────────────────────────────────────────────────────
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,  # 词表大小由训练文本中的不同字符数决定
        block_size=args.block_size,       # 上下文长度，必须小于训练文本长度
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT2(config).to(device)  # 把模型参数移动到选定设备

    # AdamW 是 Adam 的改进版，加入了权重衰减（L2 正则化），适合训练 Transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    model.train()  # 切换到训练模式，启用 dropout 等训练时行为
    step = 0
    progress = tqdm(total=args.steps, desc="training")  # 终端进度条

    # 外层 while 控制总步数；内层 for 遍历每个 batch
    while step < args.steps:
        for x, y in loader:
            x = x.to(device)  # 输入 token ID，形状 (batch_size, block_size)
            y = y.to(device)  # 目标 token ID，形状 (batch_size, block_size)，是 x 右移一位

            # 前向传播：传入 targets=y 时模型内部计算交叉熵 loss
            _, loss = model(x, y)
            assert loss is not None  # 训练时必须有 loss，这里做运行时断言防御

            optimizer.zero_grad(set_to_none=True)  # 清空上一步的梯度（set_to_none 更省内存）
            loss.backward()    # 反向传播：计算 loss 对所有参数的梯度
            optimizer.step()   # 用梯度更新参数

            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")  # 在进度条右侧实时显示 loss

            if step >= args.steps:
                break  # 达到目标步数后提前退出当前 epoch

    progress.close()

    # ── 保存 checkpoint ───────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)  # 递归创建目录，已存在不报错

    checkpoint = {
        "model": model.state_dict(),    # 模型权重，推理时用 load_state_dict 恢复
        "config": asdict(config),       # 模型结构配置，恢复时重建同形状的 GPT2
        "tokenizer": tokenizer.to_dict(), # 字符表映射，推理时编码/解码文本必须用同一套
    }
    # 保存到 out_dir/latest.pt，下次推理直接加载这个文件
    torch.save(checkpoint, args.out_dir / "latest.pt")


if __name__ == "__main__":
    main()
