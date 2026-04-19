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
    """解析训练脚本的命令行参数，并返回带属性访问能力的 argparse.Namespace。"""
    parser = argparse.ArgumentParser(description="Train a tiny GPT-2 style language model.")
    parser.add_argument("--input", type=Path, required=True, help="训练文本文件路径，会被读取为 UTF-8 文本。")
    parser.add_argument("--steps", type=int, default=500, help="训练更新步数，每一步执行一次反向传播和参数更新。")
    parser.add_argument("--batch-size", type=int, default=16, help="每个 batch 中并行训练的文本片段数量。")
    parser.add_argument("--block-size", type=int, default=64, help="每个训练样本的上下文长度，也就是模型一次看到的 token 数。")
    parser.add_argument("--n-layer", type=int, default=4, help="Transformer block 的层数，层数越多模型容量越大。")
    parser.add_argument("--n-head", type=int, default=4, help="每层多头注意力的 head 数量。")
    parser.add_argument("--n-embd", type=int, default=128, help="token 嵌入和隐藏状态的向量维度。")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW 优化器使用的学习率。")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"), help="保存模型 checkpoint 的输出目录。")
    return parser.parse_args()


def main() -> None:
    """完成一次端到端训练：读数据、建 tokenizer/dataset/model、训练并保存 checkpoint。"""
    args = parse_args()  # 读取命令行参数；返回值包含 input、steps、batch_size 等训练配置。
    text = args.input.read_text(encoding="utf-8")  # 从 --input 指定的路径读取原始训练文本。
    tokenizer = CharTokenizer.from_text(text)  # 根据训练文本构建字符级 tokenizer，确定字符到 token id 的映射。
    dataset = TinyTextDataset(tokenizer.encode(text), block_size=args.block_size)  # 把全文编码成 token id，并切成长度为 block_size 的训练样本。
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)  # 按 batch_size 组 batch，打乱样本，丢弃不足一个 batch 的尾部数据。

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # 优先使用 NVIDIA GPU，其次 Apple MPS，最后退回 CPU。
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,  # 词表大小；字符级 tokenizer 中等于训练文本里的不同字符数。
        block_size=args.block_size,  # 最大上下文长度；必须和 dataset 生成的样本长度一致。
        n_layer=args.n_layer,  # Transformer block 层数，来自 --n-layer。
        n_head=args.n_head,  # 每层注意力头数量，来自 --n-head。
        n_embd=args.n_embd,  # token embedding 和内部隐藏向量维度，来自 --n-embd。
    )  # 汇总模型结构参数，供 GPT2 初始化使用。
    model = GPT2(config).to(device)  # 创建 GPT2 模型，并把参数移动到选中的计算设备。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 创建 AdamW 优化器；lr 是 --lr 指定的学习率。

    model.train()  # 切换到训练模式，启用训练时行为，比如 dropout。
    step = 0  # 记录已经完成的优化步数。
    progress = tqdm(total=args.steps, desc="training")  # 创建进度条，总长度等于 --steps。
    while step < args.steps:  # 只要还没达到目标训练步数，就继续遍历数据。
        for x, y in loader:  # 从 DataLoader 中取一个 batch；x 是输入 token，y 是预测目标 token。
            x = x.to(device)  # 把输入 batch 移动到和模型相同的设备。
            y = y.to(device)  # 把目标 batch 移动到和模型相同的设备。
            _, loss = model(x, y)  # 前向传播；传入 y 时模型会计算语言模型训练 loss。
            assert loss is not None  # 训练时必须得到 loss；这个断言帮助类型检查和运行时防御。
            optimizer.zero_grad(set_to_none=True)  # 清空上一轮梯度；set_to_none=True 可以减少内存写入。
            loss.backward()  # 反向传播，根据 loss 计算每个参数的梯度。
            optimizer.step()  # 使用 AdamW 根据当前梯度更新模型参数。

            step += 1  # 完成一次参数更新后，训练步数加一。
            progress.update(1)  # 进度条前进一格。
            progress.set_postfix(loss=f"{loss.item():.4f}")  # 在进度条右侧显示当前 loss，保留四位小数。
            if step >= args.steps:  # 如果已经达到目标训练步数，停止当前 epoch 的剩余 batch。
                break  # 跳出 for 循环，随后结束 while 循环。
    progress.close()  # 关闭进度条，避免终端显示残留状态。

    args.out_dir.mkdir(parents=True, exist_ok=True)  # 创建 checkpoint 输出目录；父目录不存在时一并创建。
    checkpoint = {
        "model": model.state_dict(),  # 保存模型权重张量，之后可用于恢复模型参数。
        "config": asdict(config),  # 保存模型结构配置，加载时可重建同样形状的 GPT2。
        "tokenizer": tokenizer.to_dict(),  # 保存 tokenizer 的字符表映射，推理时需要用同一套编码。
    }  # checkpoint 把权重、模型配置和 tokenizer 信息放在同一个文件中。
    torch.save(checkpoint, args.out_dir / "latest.pt")  # 把 checkpoint 写入 out_dir/latest.pt。


if __name__ == "__main__":
    main()
