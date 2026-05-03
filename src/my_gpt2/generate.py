import argparse
from pathlib import Path

import torch

from my_gpt2.config import GPTConfig
from my_gpt2.model import GPT2
from my_gpt2.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    """解析推理脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")

    # --checkpoint：训练好的 .pt 文件路径，必填
    parser.add_argument("--checkpoint", type=Path, required=True)

    # --prompt：起始文本（提示词），模型会续写这段文字，必填
    parser.add_argument("--prompt", type=str, required=True)

    # --max-new-tokens：最多续写多少个字符（token）
    parser.add_argument("--max-new-tokens", type=int, default=20)

    # --temperature：采样温度，控制生成的随机程度
    #   < 1.0：更保守，倾向高概率 token，输出更确定
    #   > 1.0：更随机，概率分布更平坦，输出更多样
    #   = 1.0：保持模型原始概率分布不变
    parser.add_argument("--temperature", type=float, default=0.8)

    # --top-k：只从概率最高的 top_k 个 token 中采样，过滤掉长尾低概率 token
    #   值越小输出越保守，值越大输出越多样
    parser.add_argument("--top-k", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    """推理流程：加载 checkpoint → 恢复模型 → 编码 prompt → 生成 → 解码打印。"""
    args = parse_args()

    # ── 设备选择 ──────────────────────────────────────────────────────────────
    # 优先 NVIDIA GPU，其次 Apple Silicon GPU，最后 CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # ── 加载 checkpoint ───────────────────────────────────────────────────────
    # map_location=device：把 checkpoint 中的 tensor 直接加载到目标设备，避免先加载到 CPU 再转移
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 从 checkpoint 恢复 tokenizer（字符表映射）
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])

    # 从 checkpoint 恢复模型配置（结构超参数），**解包 dict 传给 GPTConfig
    config = GPTConfig(**checkpoint["config"])

    # 用相同配置重建空模型，再加载保存的权重
    model = GPT2(config).to(device)
    model.load_state_dict(checkpoint["model"])  # 恢复训练好的参数

    # 切换到推理模式：关闭 dropout，节省计算
    model.eval()

    # ── 编码 prompt ───────────────────────────────────────────────────────────
    # tokenizer.encode 把字符串转成 token ID 列表
    # torch.tensor([...]) 增加 batch 维度，形状变为 (1, prompt_len)
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    # ── 生成 ──────────────────────────────────────────────────────────────────
    # model.generate 自回归地逐步追加新 token，返回完整序列（包含原始 prompt）
    # 形状: (1, prompt_len + max_new_tokens)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # ── 解码并打印 ────────────────────────────────────────────────────────────
    # out[0]：取第 0 条（batch 中唯一的一条），形状 (total_len,)
    # .tolist()：把 tensor 转成普通 Python 列表，供 tokenizer.decode 使用
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
