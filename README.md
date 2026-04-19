# my-gpt2

这是一个用于学习的 GPT-2 手写项目。目标不是一开始就复刻完整工业训练流程，而是把 Transformer Decoder 的关键零件逐个写清楚、跑通、测住。

## 当前包含

- GPT-2 配置对象：`GPTConfig`
- Token embedding + position embedding
- Causal self-attention
- MLP / FFN
- Transformer block
- GPT-2 language model forward loss
- 自回归 `generate`
- 简单字符级 tokenizer
- 最小训练脚本和生成脚本
- smoke tests

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
# 安装 dev 依赖（测试、格式化等工具）， 从 pyproject.toml中获得依赖库信息
pip install -e ".[dev]"
```

## 运行测试

```bash
pytest

#如果加-s，可以打印在测试用例源码中写的 print 函数结果
pytest -s
```

## 训练一个玩具模型

先准备一个文本文件，例如：

```bash
mkdir -p data
printf "hello gpt\nhello transformer\n" > data/tiny.txt
```

训练：

```bash
python -m my_gpt2.train --input data/tiny.txt --steps 200
```

生成：

```bash
python -m my_gpt2.generate --checkpoint checkpoints/latest.pt --prompt "hello"
```

## 学习路线

详细计划见 [PLAN.md](PLAN.md)。