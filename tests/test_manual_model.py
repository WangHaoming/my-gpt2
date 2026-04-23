import torch

from my_gpt2 import GPTConfig, ManualGPT2


def test_manual_gpt2_forward_shapes() -> None:
    config = GPTConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = ManualGPT2(config)
    idx = torch.randint(0, config.vocab_size, (4, config.block_size))

    logits, loss = model(idx, idx)

    assert logits.shape == (4, config.block_size, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_manual_gpt2_supports_bias_free_config() -> None:
    config = GPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, bias=False)
    model = ManualGPT2(config)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))

    logits, loss = model(idx, idx)

    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None
    assert not any(name.endswith(".bias") for name in model.params)


def test_manual_gpt2_backward_populates_grads() -> None:
    config = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = ManualGPT2(config)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))

    _, loss = model(idx, idx)
    assert loss is not None
    loss.backward()

    assert all(param.grad is not None for param in model.parameters())


def test_manual_gpt2_generate_extends_sequence() -> None:
    config = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = ManualGPT2(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    out = model.generate(idx, max_new_tokens=5, top_k=5)

    assert out.shape == (1, 8)
