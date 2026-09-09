#!/usr/bin/env python3
"""Fetches Microsoft's official Phi-3-mini-4k-instruct GGUF (4-bit
quantized), verified by checksum.

Not committed to git (a ~2.4GB binary doesn't belong in version
control) — fetched here instead, at Docker build time (see the
Dockerfile) or once locally before running tests without Docker.

Source: microsoft/Phi-3-mini-4k-instruct-gguf on HuggingFace — confirmed
ungated (`gated: false`) and MIT licensed (`license: "mit"`) via the HF
API before adopting, same standing discipline as every other model in
this project. This is Microsoft's OWN official quantization, not a
third-party requantization — see
app/adapters/semantic_risk/local_llm_semantic_classifier.py's own
docstring for the full verification note.

Usage:
    python fetch_llm_model.py
Idempotent: does nothing if the file already exists and matches the
expected checksum; re-downloads if it's missing or corrupted.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/"
    "resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
)
EXPECTED_SHA256 = "8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef"

MODEL_DIR = Path(__file__).resolve().parent / ".cache"
MODEL_PATH = MODEL_DIR / "Phi-3-mini-4k-instruct-q4.gguf"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model() -> Path:
    """Returns the model path, downloading it first if necessary.
    Raises if a download happens but the checksum doesn't match — never
    silently uses a corrupted or unexpected file."""
    if MODEL_PATH.exists() and _sha256(MODEL_PATH) == EXPECTED_SHA256:
        return MODEL_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Phi-3-mini-4k-instruct GGUF from {URL} ...", file=sys.stderr)
    urllib.request.urlretrieve(URL, MODEL_PATH)

    actual = _sha256(MODEL_PATH)
    if actual != EXPECTED_SHA256:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded Phi-3-mini GGUF checksum mismatch: "
            f"expected {EXPECTED_SHA256}, got {actual}. Refusing to use it."
        )
    print(f"Verified checksum, saved to {MODEL_PATH}", file=sys.stderr)
    return MODEL_PATH


if __name__ == "__main__":
    ensure_model()
