#!/usr/bin/env python3
"""Fetches the pretrained AASIST checkpoint, verified by checksum.

Not committed to git (binary weights don't belong in version control) —
fetched here instead, at Docker build time (see the Dockerfile) or once
locally before running tests without Docker.

Source, pinned to a specific commit for reproducibility:
    https://github.com/clovaai/aasist @ a04c9863f63d44471dde8a6abcb3b082b07cd1d1
    (MIT licensed, NAVER Corp. — see AASIST_LICENSE in this directory)

Usage:
    python fetch_checkpoint.py
Idempotent: does nothing if the file already exists and matches the
expected checksum; re-downloads if it's missing or corrupted.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
URL = f"https://raw.githubusercontent.com/clovaai/aasist/{COMMIT}/models/weights/AASIST.pth"
EXPECTED_SHA256 = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "AASIST.pth"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_checkpoint() -> Path:
    """Returns the checkpoint path, downloading it first if necessary.
    Raises if a download happens but the checksum doesn't match — never
    silently uses a corrupted or unexpected file."""
    if CHECKPOINT_PATH.exists() and _sha256(CHECKPOINT_PATH) == EXPECTED_SHA256:
        return CHECKPOINT_PATH

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AASIST checkpoint from {URL} ...", file=sys.stderr)
    urllib.request.urlretrieve(URL, CHECKPOINT_PATH)

    actual = _sha256(CHECKPOINT_PATH)
    if actual != EXPECTED_SHA256:
        CHECKPOINT_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded AASIST checkpoint checksum mismatch: "
            f"expected {EXPECTED_SHA256}, got {actual}. Refusing to use it."
        )
    print(f"Verified checksum, saved to {CHECKPOINT_PATH}", file=sys.stderr)
    return CHECKPOINT_PATH


if __name__ == "__main__":
    ensure_checkpoint()
