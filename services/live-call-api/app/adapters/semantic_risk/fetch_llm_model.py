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

DOWNLOAD MECHANISM — changed 2026-09-11, discussed (not fixed) first
when the user noticed a cold download "takes a good amount of time":
plain `urllib.request.urlretrieve()` used to fetch this over ONE
connection with no resume — any interruption of this ~2.4GB transfer
restarted it from zero. Switched to `huggingface_hub.hf_hub_download()`,
which gets resumable downloads (HTTP Range-based retry) for free, no
extra flag needed — the same download machinery
fetch_intent_model.py's `transformers.pipeline(...)` and
fetch_ecapa_model.py's `EncoderClassifier.from_hparams(...)` already use
under the hood.

CACHE LOCATION — deliberately NOT changed by default: every other
fetch_*.py script in this project keeps its cache inside the repo tree
specifically so `docker build` bakes the model straight into the image
(see fetch_whisper_model.py's and fetch_intent_model.py's own
docstrings for that same stated rationale) — silently switching this
one script to huggingface_hub's machine-wide default
(~/.cache/huggingface/hub) would break that property for Docker/CI and
be inconsistent with the other four fetch_*.py scripts for no reason
tied to what was actually asked. Instead: if the developer has
explicitly opted in via huggingface_hub's OWN standard env vars
(HF_HUB_CACHE, HF_HOME, or the older HUGGINGFACE_HUB_CACHE alias), THAT
is honoured instead of the repo-tree default. Real, repo-specific
motivation for offering this at all: this machine can have multiple
`git worktree`s of this exact repo checked out at once (this project's
own session history — a `.claude/worktrees/data-pipeline` worktree
existed alongside the main checkout), each with its own separate
app/adapters/semantic_risk/.cache/ under the default (worktree-local)
path — independently re-downloading the same ~2.4GB file. Setting
HF_HOME once (e.g. in ~/.zshrc) lets every worktree/checkout on this
machine share one download instead. Docker builds and CI runners never
have these env vars set, so their behaviour is 100% unchanged.

Usage:
    python fetch_llm_model.py
Idempotent: does nothing (no network call beyond a cheap metadata
check) if the file already exists and matches the expected checksum;
re-downloads if it's missing or corrupted. Checksum is re-verified on
every call, cached-or-not, same "never silently use a corrupted or
unexpected file" discipline as before — a mismatch now raises without
attempting to delete anything (unlike the old version): huggingface_hub
owns its own cache's internal blob/symlink layout, and this script must
not risk corrupting it (which could now be SHARED with other worktrees)
by reaching in and deleting files itself.

MIGRATION NOTE: the OLD version of this script (urlretrieve-based) left
the file at a flat path, MODEL_DIR / FILENAME — huggingface_hub's own
cache layout nests it instead (models--.../snapshots/<rev>/<filename>).
Checked first, before calling into huggingface_hub at all: any
already-downloaded flat-layout file from before this change is reused
as-is (checksum-verified, same as always) rather than silently
re-downloaded — confirmed against this exact repo's own working copy,
which already had one at the old flat path when this was written.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
EXPECTED_SHA256 = "8a83c7fb9049a9b2e92266fa7ad04933bb53aa1e85136b7b30f1b8000ff2edef"

# Repo-tree-local default — see this module's own CACHE LOCATION note
# above for why this stays the default rather than huggingface_hub's own
# machine-wide one.
MODEL_DIR = Path(__file__).resolve().parent / ".cache"

# Where the OLD urlretrieve-based version of this script used to leave
# the file — see this module's own MIGRATION NOTE above.
_LEGACY_MODEL_PATH = MODEL_DIR / FILENAME

# huggingface_hub's own standard env vars, in the same precedence order
# it itself uses (HF_HUB_CACHE wins, then HF_HOME/hub, then the older
# HUGGINGFACE_HUB_CACHE alias) — checked only to decide whether to force
# our repo-tree default via an explicit `cache_dir=`, or step aside and
# let huggingface_hub resolve its own default from these same variables.
_HF_CACHE_ENV_VARS = ("HF_HUB_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model() -> Path:
    """Returns the model path, downloading it first if necessary (or
    resuming an interrupted prior download — see this module's own
    DOWNLOAD MECHANISM note). Raises if the checksum doesn't match —
    never silently uses a corrupted or unexpected file."""
    # See this module's own MIGRATION NOTE — reuse an already-downloaded
    # old-layout file as-is instead of paying for a redundant download
    # just because the cache layout changed underneath it.
    if _LEGACY_MODEL_PATH.exists() and _sha256(_LEGACY_MODEL_PATH) == EXPECTED_SHA256:
        return _LEGACY_MODEL_PATH

    cache_dir = None if any(os.environ.get(v) for v in _HF_CACHE_ENV_VARS) else str(MODEL_DIR)
    print(f"Fetching {FILENAME} from {REPO_ID} (cache_dir={cache_dir or 'huggingface_hub default'}) ...", file=sys.stderr)
    downloaded = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, cache_dir=cache_dir))

    actual = _sha256(downloaded)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(
            f"Phi-3-mini GGUF checksum mismatch at {downloaded}: "
            f"expected {EXPECTED_SHA256}, got {actual}. Refusing to use it. "
            "Not deleting it automatically — this cache may be shared with "
            "other checkouts; remove it by hand if you're sure it's corrupt."
        )
    print(f"Verified checksum: {downloaded}", file=sys.stderr)
    return downloaded


if __name__ == "__main__":
    ensure_model()
