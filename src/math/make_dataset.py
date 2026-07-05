"""生成 mnist_data.js —— 手写数字演示页面的数据文件。

一次性脚本：
1. 下载 MNIST 的 4 个 IDX 文件（缓存到 src/math/.cache/，二次运行跳过下载）
2. 纯 gzip + struct 解析 IDX 格式（不依赖 torchvision / numpy）
3. 28×28 → 14×14（每 2×2 块取均值）
4. 类别均衡抽样：训练 3000 张（每类 300）、测试 500 张（每类 50），固定种子打乱
5. base64 编码后写出 mnist_data.js，供 index.html 用 <script src> 加载

用法：
    python src/math/make_dataset.py            # 生成 mnist_data.js
    python src/math/make_dataset.py --check    # 额外用 torch 训一遍同构网络，验证超参可达标
"""

import base64
import gzip
import random
import struct
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / ".cache"

# 主镜像 + 备用镜像（原 yann.lecun.com 源经常 403）
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

SIZE = 14          # 降采样后的边长
N_TRAIN_PER = 1200  # 训练集每类张数 → 共 12000
# 实测（torch 对照）：3000 张 ~91%，6000 张 ~94%，12000 张 + 动量/衰减/增强 ~97%。
# 页面「改进实验室」关闭"教材翻倍"开关时只用前 6000 张。
N_TEST_PER = 50    # 测试集每类张数 → 共 500
SEED = 42


# ── 下载与解析 ────────────────────────────────────────────────────────────────

def download(name: str) -> Path:
    """下载一个 IDX 文件到缓存目录，已存在则跳过。"""
    CACHE.mkdir(exist_ok=True)
    dest = CACHE / name
    if dest.exists():
        print(f"  已缓存: {name}")
        return dest
    last_err = None
    for mirror in MIRRORS:
        url = mirror + name
        try:
            print(f"  下载中: {url}")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as e:  # noqa: BLE001 - 换镜像重试
            last_err = e
            print(f"  失败（{e}），尝试下一个镜像…")
    raise RuntimeError(f"所有镜像都下载失败: {name}") from last_err


def parse_images(path: Path) -> tuple[int, int, int, bytes]:
    """解析 IDX3 图像文件，返回 (数量, 行, 列, 逐像素 uint8 字节)。"""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x803, f"图像文件魔数错误: {magic:#x}"
        return n, rows, cols, f.read(n * rows * cols)


def parse_labels(path: Path) -> bytes:
    """解析 IDX1 标签文件，返回逐字节标签。"""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 0x801, f"标签文件魔数错误: {magic:#x}"
        return f.read(n)


# ── 降采样与抽样 ──────────────────────────────────────────────────────────────

def downsample(img28: bytes) -> bytes:
    """28×28 → 14×14：每 2×2 块取均值。"""
    out = bytearray(SIZE * SIZE)
    for r in range(SIZE):
        for c in range(SIZE):
            i = (r * 2) * 28 + c * 2
            out[r * SIZE + c] = (img28[i] + img28[i + 1] + img28[i + 28] + img28[i + 29]) // 4
    return bytes(out)


def balanced_subset(images: bytes, labels: bytes, rows: int, cols: int,
                    per_class: int, rng: random.Random) -> tuple[bytes, bytes]:
    """每类取前 per_class 张，降采样后整体打乱，返回 (图像字节, 标签字节)。"""
    picked: list[tuple[bytes, int]] = []
    counts = [0] * 10
    px = rows * cols
    for i, lab in enumerate(labels):
        if counts[lab] >= per_class:
            continue
        counts[lab] += 1
        picked.append((downsample(images[i * px:(i + 1) * px]), lab))
        if len(picked) == per_class * 10:
            break
    assert all(c == per_class for c in counts), f"类别不均衡: {counts}"
    rng.shuffle(picked)
    return b"".join(p[0] for p in picked), bytes(p[1] for p in picked)


# ── 输出 ─────────────────────────────────────────────────────────────────────

def ascii_art(img: bytes, label: int) -> str:
    """十级灰度字符画，用于人工校验方向/字节序。"""
    ramp = " .:-=+*#%@"
    lines = [f"标签 = {label}"]
    for r in range(SIZE):
        lines.append("".join(ramp[min(9, img[r * SIZE + c] * 10 // 256)] * 2 for c in range(SIZE)))
    return "\n".join(lines)


def write_js(train_i: bytes, train_l: bytes, test_i: bytes, test_l: bytes) -> Path:
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    out = HERE / "mnist_data.js"
    out.write_text(
        "// 由 make_dataset.py 自动生成，请勿手改。\n"
        "// 图像为 14x14 uint8 灰度（0-255），base64 编码，按行主序平铺。\n"
        "const MNIST = {\n"
        f"  size: {SIZE},\n"
        f"  nTrain: {len(train_l)},\n"
        f"  nTest: {len(test_l)},\n"
        f'  trainImages: "{b64(train_i)}",\n'
        f'  trainLabels: "{b64(train_l)}",\n'
        f'  testImages: "{b64(test_i)}",\n'
        f'  testLabels: "{b64(test_l)}",\n'
        "};\n",
        encoding="utf-8",
    )
    return out


# ── torch 对照训练（--check）─────────────────────────────────────────────────

def check_with_torch(train_i: bytes, train_l: bytes, test_i: bytes, test_l: bytes) -> None:
    """用 torch 按 JS 侧「改进全开」配置训一遍，打印前几步 loss 和最终正确率。

    配置与页面步骤 3 的改进实验室默认值一致：
    batch=32、lr=0.05、动量 0.9、学习率线性衰减到 10%、±1 像素随机平移增强、20 epoch。
    目的：1) 确认此配置能到 ~97%；2) 给 JS 手写实现提供数值对照。
    """
    import torch
    import torch.nn.functional as F

    def to_tensor(imgs: bytes, labs: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.frombuffer(bytearray(imgs), dtype=torch.uint8).view(-1, SIZE * SIZE).float() / 255
        y = torch.frombuffer(bytearray(labs), dtype=torch.uint8).long()
        return x, y

    xtr, ytr = to_tensor(train_i, train_l)
    xte, yte = to_tensor(test_i, test_l)

    def shift_batch(x: "torch.Tensor", g: "torch.Generator") -> "torch.Tensor":
        """±1 像素随机平移（出界补 0），与 JS 侧的数据增强一致。"""
        n = x.shape[0]
        img = x.view(n, SIZE, SIZE)
        out = torch.zeros_like(img)
        dxs = torch.randint(-1, 2, (n,), generator=g)
        dys = torch.randint(-1, 2, (n,), generator=g)
        for b in range(n):
            sx, sy = dxs[b].item(), dys[b].item()
            out[b, max(0, sy):SIZE + min(0, sy), max(0, sx):SIZE + min(0, sx)] = \
                img[b, max(0, -sy):SIZE + min(0, -sy), max(0, -sx):SIZE + min(0, -sx)]
        return out.view(n, SIZE * SIZE)

    g = torch.Generator().manual_seed(SEED)
    w1 = torch.randn(SIZE * SIZE, 64, generator=g) * (2 / (SIZE * SIZE)) ** 0.5
    b1 = torch.zeros(64)
    w2 = torch.randn(64, 10, generator=g) * (2 / 64) ** 0.5
    b2 = torch.zeros(10)
    params = (w1, b1, w2, b2)
    for p in params:
        p.requires_grad_(True)
    vels = [torch.zeros_like(p) for p in params]

    batch, lr, momentum, epochs = 32, 0.05, 0.9, 20
    n = len(ytr)
    total = epochs * (n // batch)
    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n - batch + 1, batch):
            cur_lr = lr * (1 - 0.9 * step / total)   # 线性衰减到 10%
            idx = perm[s:s + batch]
            h = (shift_batch(xtr[idx], g) @ w1 + b1).relu()
            loss = F.cross_entropy(h @ w2 + b2, ytr[idx])
            for p in params:
                p.grad = None
            loss.backward()
            with torch.no_grad():
                for p, v in zip(params, vels):
                    v.mul_(momentum).add_(p.grad)
                    p -= cur_lr * v
            if step < 5 or step % 1000 == 0:
                print(f"  step {step:5d}  loss {loss.item():.4f}")
            step += 1

    with torch.no_grad():
        acc = ((xte @ w1 + b1).relu() @ w2 + b2).argmax(1).eq(yte).float().mean().item()
    print(f"  torch 对照结果: {step} 步后测试正确率 = {acc:.1%}（JS 侧改进全开应接近此值）")


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("1) 下载 MNIST …")
    paths = {k: download(v) for k, v in FILES.items()}

    print("2) 解析 IDX …")
    n, rows, cols, train_images = parse_images(paths["train_images"])
    train_labels = parse_labels(paths["train_labels"])
    _, _, _, test_images = parse_images(paths["test_images"])
    test_labels = parse_labels(paths["test_labels"])
    print(f"  训练集 {n} 张 {rows}x{cols}")

    print(f"3) 降采样到 {SIZE}x{SIZE} 并均衡抽样 …")
    rng = random.Random(SEED)
    tr_i, tr_l = balanced_subset(train_images, train_labels, rows, cols, N_TRAIN_PER, rng)
    te_i, te_l = balanced_subset(test_images, test_labels, rows, cols, N_TEST_PER, rng)
    print(f"  训练 {len(tr_l)} 张（每类 {N_TRAIN_PER}），测试 {len(te_l)} 张（每类 {N_TEST_PER}）")

    print("4) 写出 mnist_data.js …")
    out = write_js(tr_i, tr_l, te_i, te_l)
    print(f"  {out}（{out.stat().st_size / 1024:.0f} KB）")

    print("5) 自检（第一张训练图的字符画，肉眼确认是个数字且方向正常）：")
    print(ascii_art(tr_i[:SIZE * SIZE], tr_l[0]))

    if "--check" in sys.argv:
        print("6) torch 对照训练 …")
        check_with_torch(tr_i, tr_l, te_i, te_l)


if __name__ == "__main__":
    main()
