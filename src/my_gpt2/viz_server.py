"""
网络可视化服务器 — 实时展示权重矩阵和训练/推理过程

用法:
  python -m my_gpt2.viz_server --checkpoint checkpoints/latest.pt

然后在浏览器中打开 http://localhost:5000
"""

import argparse
import json
import threading
from dataclasses import asdict
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request, send_from_directory
from torch.utils.data import DataLoader

from my_gpt2.config import GPTConfig
from my_gpt2.data import TinyTextDataset
from my_gpt2.hooks import MatmulCollector, install_hooks, uninstall_hooks
from my_gpt2.model import GPT2
from my_gpt2.tokenizer import CharTokenizer


# ── Flask app ─────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "viz_static"
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# ── Global model state ────────────────────────────────────────────────────────

_model: GPT2 | None = None
_tokenizer: CharTokenizer | None = None
_config: GPTConfig | None = None
_device: str = "cpu"
_train_stop = threading.Event()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stats(t: torch.Tensor) -> dict:
    f = t.float()
    return {
        "mean": round(float(f.mean()), 5),
        "std":  round(float(f.std()),  5),
        "min":  round(float(f.min()),  5),
        "max":  round(float(f.max()),  5),
    }


def _tensor_to_list(t: torch.Tensor) -> list:
    """Convert 2D tensor to nested list, rounding to 4 decimal places."""
    return [[round(float(v), 4) for v in row] for row in t.cpu().float()]


def _all_weights() -> dict:
    """Return all weight matrices + 1D LayerNorm params as JSON-ready dicts."""
    if _model is None:
        return {}
    result = {}
    for name, param in _model.named_parameters():
        t = param.detach().cpu().float()
        if t.dim() == 2:
            result[name] = {
                "shape": list(t.shape),
                "data": _tensor_to_list(t),
                "stats": _stats(t),
            }
        elif t.dim() == 1 and ("ln_" in name or "ln_f" in name):
            # LayerNorm weights/biases as 1-row matrices
            result[name] = {
                "shape": [1, t.shape[0]],
                "data": [[round(float(v), 4) for v in t]],
                "stats": _stats(t),
            }
    return result


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/config")
def api_config():
    if _config is None:
        return jsonify({"error": "no model loaded"}), 400
    cfg = asdict(_config)
    cfg["device"] = _device
    return jsonify(cfg)


@app.route("/api/weights")
def api_weights():
    return jsonify(_all_weights())


@app.route("/api/train", methods=["POST"])
def api_train():
    """Stream training progress via Server-Sent Events."""
    body = request.get_json(force=True)
    input_file = body.get("input_file", "")
    steps = int(body.get("steps", 100))
    lr = float(body.get("lr", 3e-4))
    weight_every = int(body.get("weight_every", 10))

    def stream():
        if _model is None:
            yield _sse({"type": "error", "message": "no model loaded"})
            return

        try:
            text = Path(input_file).read_text(encoding="utf-8")
        except Exception as e:
            yield _sse({"type": "error", "message": f"cannot read file: {e}"})
            return

        # Use the loaded tokenizer so vocab stays aligned with the model.
        # Characters not in the original vocab are silently skipped.
        try:
            ids = _tokenizer.encode(text)
        except KeyError:
            ids = [_tokenizer.stoi[ch] for ch in text if ch in _tokenizer.stoi]
        if len(ids) < _config.block_size + 1:
            yield _sse({"type": "error", "message": "训练文本太短，请提供更长的文本"})
            return
        dataset = TinyTextDataset(ids, block_size=_config.block_size)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(_model.parameters(), lr=lr)
        _model.train()
        _train_stop.clear()

        # Snapshot weights before first step for diff computation
        prev = {
            n: p.detach().cpu().float().clone()
            for n, p in _model.named_parameters()
            if p.dim() == 2
        }

        step = 0
        while step < steps:
            for x, y in loader:
                if _train_stop.is_set() or step >= steps:
                    break
                x, y = x.to(_device), y.to(_device)
                _, loss = _model(x, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1

                event: dict = {"type": "step", "step": step, "loss": round(float(loss), 5)}

                # Every weight_every steps send updated weights + diffs
                if step % weight_every == 0 or step == steps:
                    wdata = {}
                    for n, p in _model.named_parameters():
                        if p.dim() != 2:
                            continue
                        cur = p.detach().cpu().float()
                        d = (cur - prev[n]).abs() if n in prev else None
                        wdata[n] = {
                            "shape": list(cur.shape),
                            "data": _tensor_to_list(cur),
                            "stats": _stats(cur),
                            "diff_max": round(float(d.max()), 6) if d is not None else 0,
                            "diff": _tensor_to_list(d) if d is not None else None,
                        }
                        prev[n] = cur.clone()
                    event["weights"] = wdata

                yield _sse(event)

        _model.eval()
        yield _sse({"type": "done", "total_steps": step})

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/stop", methods=["POST"])
def api_stop():
    _train_stop.set()
    return jsonify({"ok": True})


@app.route("/api/infer", methods=["POST"])
def api_infer():
    """Run forward pass and return per-layer activations."""
    body = request.get_json(force=True)
    prompt = body.get("prompt", "")

    if _model is None or _tokenizer is None:
        return jsonify({"error": "no model loaded"}), 400
    if not prompt:
        return jsonify({"error": "empty prompt"}), 400

    try:
        token_ids = _tokenizer.encode(prompt)
    except KeyError as e:
        return jsonify({"error": f"unknown character: {e}"}), 400

    idx = torch.tensor([token_ids], dtype=torch.long, device=_device)

    collector = MatmulCollector()
    install_hooks(_model, collector)
    _model.eval()
    with torch.no_grad():
        logits, _ = _model(idx)
    uninstall_hooks(_model)

    layers = []
    for rec in collector.records:
        layers.append({
            "layer_idx": rec.layer_idx,
            "op_name":   rec.op_name,
            "in_shape":  list(rec.input_mat.shape),
            "out_shape": list(rec.output_mat.shape),
            "in_stats":  _stats(rec.input_mat),
            "out_stats": _stats(rec.output_mat),
            "in_data":   _tensor_to_list(rec.input_mat),
            "out_data":  _tensor_to_list(rec.output_mat),
        })

    last_logits = logits[0, -1, :].float()
    probs = torch.softmax(last_logits, dim=-1)
    top_vals, top_ids = torch.topk(probs, min(10, probs.size(0)))
    predictions = [
        {
            "char":   _tokenizer.itos.get(int(top_ids[i]), "?"),
            "prob":   round(float(top_vals[i]), 5),
            "logit":  round(float(last_logits[top_ids[i]]), 4),
        }
        for i in range(len(top_vals))
    ]

    return jsonify({
        "tokens":      list(prompt),
        "token_ids":   token_ids,
        "layers":      layers,
        "predictions": predictions,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _model, _tokenizer, _config, _device

    parser = argparse.ArgumentParser(description="GPT-2 可视化服务器")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    _device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ck = torch.load(args.checkpoint, map_location=_device, weights_only=False)
    _tokenizer = CharTokenizer.from_dict(ck["tokenizer"])
    _config = GPTConfig(**ck["config"])
    _model = GPT2(_config).to(_device)
    _model.load_state_dict(ck["model"])
    _model.eval()

    STATIC_DIR.mkdir(exist_ok=True)

    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {_device}")
    print(f"n_layer={_config.n_layer}  n_head={_config.n_head}  n_embd={_config.n_embd}")
    print(f"vocab_size={_config.vocab_size}  block_size={_config.block_size}")
    print(f"\n>>> 浏览器访问: http://{args.host}:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
