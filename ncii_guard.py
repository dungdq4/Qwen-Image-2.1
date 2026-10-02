from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import threading

GUARD_ID = "hfmlsoc/ncii-guard-v02"

_lock = threading.Lock()
_process: subprocess.Popen | None = None


def _read(timeout: float) -> dict:
    readable, _, _ = select.select([_process.stdout], [], [], timeout)
    if not readable:
        raise TimeoutError(f"the guard did not answer within {timeout}s")
    line = _process.stdout.readline()
    if not line:
        raise EOFError("the guard process died")
    return json.loads(line)


def _spawn() -> None:
    global _process
    _process = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ready = _read(600.0)
    if ready.get("status") != "ready":
        raise RuntimeError(f"guard failed to start: {ready}")


def start() -> None:
    with _lock:
        _spawn()


def score(prompt: str, timeout: float = 60.0) -> float:
    with _lock:
        for attempt in (0, 1):
            try:
                if _process is None or _process.poll() is not None:
                    _spawn()
                _process.stdin.write(json.dumps({"prompt": prompt or ""}) + "\n")
                _process.stdin.flush()
                return float(_read(timeout)["score"])
            except Exception:
                if attempt:
                    raise
                if _process is not None and _process.poll() is None:
                    _process.kill()


def _snapshot() -> str:
    from huggingface_hub import snapshot_download

    src = snapshot_download(
        GUARD_ID,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"],
    )
    dst = os.path.join(tempfile.gettempdir(), "ncii-guard-v02-normalised")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        link = os.path.join(dst, name)
        if not os.path.exists(link):
            os.symlink(os.path.realpath(os.path.join(src, name)), link)

    config = json.load(open(os.path.join(src, "config.json")))
    rope = config.get("rope_parameters")
    if isinstance(rope, dict):
        per_layer = {k: v for k, v in rope.items() if isinstance(v, dict)}
        if per_layer:
            config["rope_parameters"] = per_layer
    config_path = os.path.join(dst, "config.json")
    if os.path.islink(config_path):
        os.remove(config_path)
    with open(config_path, "w") as fh:
        json.dump(config, fh)
    return dst


def _serve() -> None:
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    guard_dir = _snapshot()
    tokenizer = AutoTokenizer.from_pretrained(guard_dir)
    model = AutoModelForSequenceClassification.from_pretrained(guard_dir, dtype=torch.float32).eval()
    protocol.write(json.dumps({"status": "ready", "id2label": model.config.id2label}) + "\n")
    for line in sys.stdin:
        text = json.loads(line)["prompt"]
        batch = tokenizer(text, truncation=True, max_length=256, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**batch).logits.float()
        protocol.write(json.dumps({"score": torch.softmax(logits, dim=-1)[0, 1].item()}) + "\n")


if __name__ == "__main__":
    _serve()
