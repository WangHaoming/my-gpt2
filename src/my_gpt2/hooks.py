"""
hooks.py — 矩阵乘法数据捕获器

职责：
  1. MatmulRecord / MatmulCollector：存储每次矩阵乘法的输入/输出快照
  2. install_hooks(model, collector)：
       - nn.Linear 层（MLP 的 c_fc / c_proj，lm_head）用 register_forward_hook
       - CausalSelfAttention 的两个纯 Tensor @ 运算，用运行时替换实例 forward 方法
  3. uninstall_hooks(model)：移除 hook，精确恢复 CausalSelfAttention.forward

不修改 model.py 中的任何代码。
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class MatmulRecord:
    """单次矩阵乘法的快照（batch=0，已移到 CPU float32）"""
    layer_idx: int         # 所在 Block 层编号（lm_head 用 -1）
    op_name: str           # 操作名，见下方 OP_NAMES
    input_mat: torch.Tensor   # 左矩阵 / 主输入（取 batch=0）
    output_mat: torch.Tensor  # 输出矩阵（取 batch=0）


# op_name 约定：
#   "c_attn"       — QKV 联合投影（nn.Linear）
#   "qk"           — Q @ K^T（纯 Tensor @，head 0）
#   "av"           — softmax(att) @ V（纯 Tensor @，head 0）
#   "c_proj_attn"  — 注意力输出投影（nn.Linear）
#   "c_fc"         — MLP 升维（nn.Linear）
#   "c_proj_mlp"   — MLP 降维（nn.Linear）
#   "lm_head"      — 词表投影（nn.Linear）


class MatmulCollector:
    """收集所有层矩阵乘法记录"""

    def __init__(self) -> None:
        self.records: list[MatmulRecord] = []

    def record(
        self,
        op_name: str,
        input_mat: torch.Tensor,
        output_mat: torch.Tensor,
        layer_idx: int,
    ) -> None:
        self.records.append(MatmulRecord(
            layer_idx=layer_idx,
            op_name=op_name,
            input_mat=input_mat.detach().cpu().float(),
            output_mat=output_mat.detach().cpu().float(),
        ))

    def clear(self) -> None:
        self.records.clear()

    def by_layer(self, layer_idx: int) -> list[MatmulRecord]:
        return [r for r in self.records if r.layer_idx == layer_idx]


# ── 全局状态（供 uninstall_hooks 使用）────────────────────────────────────────

_hook_handles: list[Any] = []
_original_forwards: dict[int, Any] = {}   # id(attn_module) → original forward


# ── patched CausalSelfAttention.forward ──────────────────────────────────────

def _make_patched_forward(layer_idx: int, attn_self: nn.Module):
    """
    返回一个与 CausalSelfAttention.forward 逻辑完全一致的替代函数，
    但在 Q@K^T 和 att@V 前后插入 collector.record()。

    使用闭包捕获 layer_idx 和 attn_self（模块实例），
    绑定到实例后 self 参数由 Python 自动传入，这里用 attn_self 访问模块属性。
    """
    def patched_forward(x: torch.Tensor) -> torch.Tensor:
        # 与 model.py CausalSelfAttention.forward 逻辑完全对应
        batch_size, seq_len, channels = x.size()
        head_dim = channels // attn_self.n_head

        # ── c_attn：QKV 联合投影 ──────────────────────────────────────────────
        # 记录：输入 (T, C)，输出 (T, 3C)
        qkv = attn_self.c_attn(x)
        _collector.record("c_attn", x[0], qkv[0], layer_idx)

        q, k, v = qkv.split(attn_self.n_embd, dim=2)

        # reshape: (B, T, C) → (B, n_head, T, head_dim)
        k = k.view(batch_size, seq_len, attn_self.n_head, head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, attn_self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, attn_self.n_head, head_dim).transpose(1, 2)

        # ── Q @ K^T：注意力分数 ───────────────────────────────────────────────
        # 只记录 head 0：q[0,0] (T, d)，raw_att[0,0] (T, T)
        raw_att = q @ k.transpose(-2, -1)
        _collector.record("qk", q[0, 0], raw_att[0, 0], layer_idx)

        att = raw_att * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(
            attn_self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = attn_self.attn_dropout(att)

        # ── att @ V：加权求和 ─────────────────────────────────────────────────
        # 只记录 head 0：att[0,0] (T, T)，y[0,0] (T, d)
        y = att @ v
        _collector.record("av", att[0, 0], y[0, 0], layer_idx)

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)

        # ── c_proj：输出投影 ──────────────────────────────────────────────────
        # 记录：输入 (T, C)，输出 (T, C)
        proj_out = attn_self.resid_dropout(attn_self.c_proj(y))
        _collector.record("c_proj_attn", y[0], proj_out[0], layer_idx)

        return proj_out

    return patched_forward


# ── 公开 API ──────────────────────────────────────────────────────────────────

# 全局 collector 引用，install_hooks 设置后 patched_forward 可访问
_collector: MatmulCollector | None = None


def install_hooks(model: nn.Module, collector: MatmulCollector) -> None:
    """
    在模型上安装所有捕获 hook。

    调用方式：
        collector = MatmulCollector()
        install_hooks(model, collector)
        with torch.no_grad():
            model(idx)
        uninstall_hooks(model)
    """
    global _collector
    _collector = collector

    from my_gpt2.model import CausalSelfAttention  # 避免循环导入

    for layer_idx, block in enumerate(model.transformer.h):
        # ── 1. CausalSelfAttention：替换实例 forward 捕获纯 Tensor @ ──────────
        attn: CausalSelfAttention = block.attn
        _original_forwards[id(attn)] = attn.forward          # 保存原始方法
        attn.forward = _make_patched_forward(layer_idx, attn) # 绑定新方法

        # ── 2. MLP 线性层：用标准 forward hook ──────────────────────────────────
        def _mlp_hook(op_name: str, lidx: int):
            def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
                _collector.record(op_name, inp[0][0], out[0], lidx)
            return hook

        h1 = block.mlp.c_fc.register_forward_hook(_mlp_hook("c_fc", layer_idx))
        h2 = block.mlp.c_proj.register_forward_hook(_mlp_hook("c_proj_mlp", layer_idx))
        _hook_handles.extend([h1, h2])

    # ── 3. lm_head：用标准 forward hook ─────────────────────────────────────
    def _lm_head_hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        _collector.record("lm_head", inp[0][0], out[0], layer_idx=-1)

    h3 = model.lm_head.register_forward_hook(_lm_head_hook)
    _hook_handles.append(h3)


def uninstall_hooks(model: nn.Module) -> None:
    """移除所有 hook，精确恢复 CausalSelfAttention.forward，保证模型状态干净"""
    global _collector

    for handle in _hook_handles:
        handle.remove()
    _hook_handles.clear()

    for block in model.transformer.h:
        attn = block.attn
        orig = _original_forwards.pop(id(attn), None)
        if orig is not None:
            attn.forward = orig

    _original_forwards.clear()
    _collector = None
