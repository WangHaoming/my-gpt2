import math
from collections.abc import Iterator

import torch

from my_gpt2.config import GPTConfig


class ManualGPT2:
    """A GPT-2 style model without torch.nn.Module or torch.nn layers.

    This implementation still uses torch tensors and autograd, but all parameters
    are created and managed by hand. The forward pass uses plain tensor ops for
    embedding lookup, linear layers, layer norm, dropout, attention, MLP, and loss.
    """

    def __init__(self, config: GPTConfig, device: str | torch.device | None = None) -> None:
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.training = True
        self.params: dict[str, torch.Tensor] = {}
        self._init_params()

        mask = torch.tril(torch.ones(config.block_size, config.block_size, device=self.device))
        self.causal_mask = mask.view(1, 1, config.block_size, config.block_size)

    def _init_params(self) -> None:
        cfg = self.config
        self.params["wte.weight"] = self._normal((cfg.vocab_size, cfg.n_embd))
        self.params["wpe.weight"] = self._normal((cfg.block_size, cfg.n_embd))

        for layer_idx in range(cfg.n_layer):
            prefix = f"h.{layer_idx}"
            self.params[f"{prefix}.ln_1.weight"] = torch.ones(
                cfg.n_embd, device=self.device, requires_grad=True
            )
            if cfg.bias:
                self.params[f"{prefix}.ln_1.bias"] = self._zeros(cfg.n_embd)
            self.params[f"{prefix}.attn.c_attn.weight"] = self._normal(
                (3 * cfg.n_embd, cfg.n_embd)
            )
            if cfg.bias:
                self.params[f"{prefix}.attn.c_attn.bias"] = self._zeros(3 * cfg.n_embd)
            self.params[f"{prefix}.attn.c_proj.weight"] = self._normal((cfg.n_embd, cfg.n_embd))
            if cfg.bias:
                self.params[f"{prefix}.attn.c_proj.bias"] = self._zeros(cfg.n_embd)
            self.params[f"{prefix}.ln_2.weight"] = torch.ones(
                cfg.n_embd, device=self.device, requires_grad=True
            )
            if cfg.bias:
                self.params[f"{prefix}.ln_2.bias"] = self._zeros(cfg.n_embd)
            self.params[f"{prefix}.mlp.c_fc.weight"] = self._normal((4 * cfg.n_embd, cfg.n_embd))
            if cfg.bias:
                self.params[f"{prefix}.mlp.c_fc.bias"] = self._zeros(4 * cfg.n_embd)
            self.params[f"{prefix}.mlp.c_proj.weight"] = self._normal(
                (cfg.n_embd, 4 * cfg.n_embd)
            )
            if cfg.bias:
                self.params[f"{prefix}.mlp.c_proj.bias"] = self._zeros(cfg.n_embd)

        self.params["ln_f.weight"] = torch.ones(cfg.n_embd, device=self.device, requires_grad=True)
        if cfg.bias:
            self.params["ln_f.bias"] = self._zeros(cfg.n_embd)

    def _normal(self, shape: tuple[int, ...]) -> torch.Tensor:
        return (torch.randn(shape, device=self.device) * 0.02).requires_grad_(True)

    def _zeros(self, size: int) -> torch.Tensor:
        return torch.zeros(size, device=self.device, requires_grad=True)

    def parameters(self) -> Iterator[torch.Tensor]:
        yield from self.params.values()

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = None

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def to(self, device: str | torch.device) -> "ManualGPT2":
        self.device = torch.device(device)
        self.params = {
            name: param.detach().to(self.device).requires_grad_(True)
            for name, param in self.params.items()
        }
        self.causal_mask = self.causal_mask.to(self.device)
        return self

    # 保存模型参数
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in self.params.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing = set(self.params) - set(state_dict)
        unexpected = set(state_dict) - set(self.params)
        if missing or unexpected:
            raise ValueError(f"missing keys: {sorted(missing)}, unexpected keys: {sorted(unexpected)}")

        self.params = {
            name: state_dict[name].detach().to(self.device).clone().requires_grad_(True)
            for name in self.params
        }

    def _linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        y = x @ weight.t()
        if bias is not None:
            y = y + bias
        return y

    def _layer_norm(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float = 1e-5
    ) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) * torch.rsqrt(var + eps)
        y = x_hat * weight
        if bias is not None:
            y = y + bias
        return y

    def _gelu(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

    # 丢弃法，用于训练时随机丢弃一些神经元，防止过拟合
    # 在测试时，不进行丢弃，返回原始输入
    # 在训练时，按照丢弃概率 p 随机丢弃一些神经元，返回丢弃后的输入
    # 丢弃后的输入是原始输入乘以丢弃概率的倒数
    # 丢弃后的输入是原始输入乘以丢弃概率的倒数
    def _dropout(self, x: torch.Tensor) -> torch.Tensor:
        p = self.config.dropout
        if not self.training or p == 0:
            return x
        keep_prob = 1.0 - p
        mask = torch.rand_like(x) < keep_prob
        return x * mask / keep_prob

    def _attention(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch_size, seq_len, channels = x.size()
        prefix = f"h.{layer_idx}.attn"

        qkv = self._linear(
            x,
            self.params[f"{prefix}.c_attn.weight"],
            self.params.get(f"{prefix}.c_attn.bias"),
        )
        q, k, v = qkv.split(cfg.n_embd, dim=2)

        head_dim = channels // cfg.n_head
        q = q.view(batch_size, seq_len, cfg.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, cfg.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, cfg.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self._dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self._linear(
            y,
            self.params[f"{prefix}.c_proj.weight"],
            self.params.get(f"{prefix}.c_proj.bias"),
        )
        return self._dropout(y)

    def _mlp(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        prefix = f"h.{layer_idx}.mlp"
        x = self._linear(
            x,
            self.params[f"{prefix}.c_fc.weight"],
            self.params.get(f"{prefix}.c_fc.bias"),
        )
        x = self._gelu(x)
        x = self._linear(
            x,
            self.params[f"{prefix}.c_proj.weight"],
            self.params.get(f"{prefix}.c_proj.bias"),
        )
        return self._dropout(x)

    def _block(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        prefix = f"h.{layer_idx}"
        attn_input = self._layer_norm(
            x,
            self.params[f"{prefix}.ln_1.weight"],
            self.params.get(f"{prefix}.ln_1.bias"),
        )
        x = x + self._attention(layer_idx, attn_input)

        mlp_input = self._layer_norm(
            x,
            self.params[f"{prefix}.ln_2.weight"],
            self.params.get(f"{prefix}.ln_2.bias"),
        )
        x = x + self._mlp(layer_idx, mlp_input)
        return x

    # 重载 __call__ 方法，使得实例可以直接作为函数调用
    # ManualGPT2(idx, targets) 等价于 model.forward(idx, targets)
    def __call__(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward(idx, targets)


    # 前向传播：计算模型的输出 logits 和 loss
    # 即模型推理，计算预测结果的损失
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")

        idx = idx.to(self.device)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=self.device)
        tok_emb = self.params["wte.weight"][idx]
        pos_emb = self.params["wpe.weight"][pos]
        x = self._dropout(tok_emb + pos_emb)

        for layer_idx in range(self.config.n_layer):
            x = self._block(layer_idx, x)

        x = self._layer_norm(x, self.params["ln_f.weight"], self.params.get("ln_f.bias"))

        # Weight tying: the token embedding matrix is also the language-model head.
        logits = x @ self.params["wte.weight"].t()

        loss = None
        if targets is not None:
            targets = targets.to(self.device)
            loss = self._cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def _cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return -log_probs[torch.arange(targets.numel(), device=targets.device), targets].mean()

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        idx = idx.to(self.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        if was_training:
            self.train()
        return idx
