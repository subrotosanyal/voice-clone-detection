#!/usr/bin/env python3
"""Fine-tunes AASIST's final classification layer on the Indian-language
corpus (genuine + synthetic manifests) — see ../README.md's "Status" for
exactly what data this needs (both classes; requires
generate_synthetic_corpus.py's output to exist first).

Must be run with services/live-call-api's own venv active, same reason as
run_held_out_eval.py:

    source ../../services/live-call-api/.venv/bin/activate
    python scripts/fine_tune_aasist.py

WHY ONLY THE FINAL LAYER, NOT THE WHOLE MODEL: with a few dozen examples
total (20-40 genuine + whatever generate_synthetic_corpus.py produced —
currently 20 synthetic Hindi utterances from 2 cloned speakers), fully
fine-tuning AASIST's ~300k-parameter graph-attention network would badly
overfit — it would memorise this tiny set rather than learn anything that
generalises to a real caller's voice. Freezing every layer except
AasistNet's final `out_layer` (a single `nn.Linear(5*gat_dims[1], 2)`) is
closer to recalibrating the existing English-trained features for Hindi
than retraining the model — a defensible thing to do with this little
data. Full fine-tuning needs hundreds-to-thousands of examples per class
to be trustworthy; that's exactly why real dataset collection (more
volunteers, more languages, Malvi still entirely missing) matters more
than this script's cleverness does.
"""
from __future__ import annotations

import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

_EVAL_DIR = Path(__file__).resolve().parent.parent
_LIVE_CALL_API_DIR = _EVAL_DIR.parent.parent / "services" / "live-call-api"
sys.path.insert(0, str(_LIVE_CALL_API_DIR))

from app.adapters.detectors.acoustic_aasist import _DEFAULT_MODEL_CONFIG, _deterministic_pad  # noqa: E402
from app.adapters.detectors.vendor.aasist_model import AasistNet  # noqa: E402

CHECKPOINT_PATH = _LIVE_CALL_API_DIR / "app/adapters/detectors/vendor/checkpoints/AASIST.pth"
MANIFEST_BASE_DIR = _EVAL_DIR
RESULTS_DIR = _EVAL_DIR / "results"
NB_SAMP = _DEFAULT_MODEL_CONFIG["nb_samp"]
MODEL_SAMPLE_RATE = 16_000


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _load_and_pad(wav_path: Path) -> np.ndarray:
    samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    if sr != MODEL_SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = gcd(sr, MODEL_SAMPLE_RATE)
        samples = resample_poly(samples, MODEL_SAMPLE_RATE // g, sr // g).astype(np.float32)
    return _deterministic_pad(samples, NB_SAMP)


def build_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    genuine = _load_manifest(MANIFEST_BASE_DIR / "data" / "genuine" / "manifest.json")
    synthetic = _load_manifest(MANIFEST_BASE_DIR / "data" / "synthetic" / "manifest.json")

    if not synthetic:
        raise SystemExit(
            "No synthetic corpus found at data/synthetic/manifest.json — "
            "run scripts/generate_synthetic_corpus.py first. Fine-tuning "
            "needs both classes; see README.md's Status section."
        )

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for e in genuine:
        xs.append(_load_and_pad(MANIFEST_BASE_DIR / e["wav_path"]))
        ys.append(1)  # bonafide — matches AASIST's own training label convention
    for e in synthetic:
        xs.append(_load_and_pad(MANIFEST_BASE_DIR / e["wav_path"]))
        ys.append(0)  # spoof

    return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def train_val_split(n: int, val_fraction: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    return idx[n_val:], idx[:n_val]


def main() -> None:
    print("Building dataset from genuine + synthetic manifests...")
    x, y = build_dataset()
    n_bonafide = int((y == 1).sum())
    n_spoof = int((y == 0).sum())
    print(f"Total examples: {len(y)} (bonafide={n_bonafide}, spoof={n_spoof})")

    train_idx, val_idx = train_val_split(len(y))
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    print(f"Loading AASIST checkpoint from {CHECKPOINT_PATH} ...")
    model = AasistNet(_DEFAULT_MODEL_CONFIG)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"), strict=True)

    # Freeze everything except the final classification layer — see the
    # module docstring for why, given how little data exists right now.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.out_layer.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.out_layer.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    n_epochs = 30
    best_val_acc = -1.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        total_loss = 0.0
        for i in perm:
            optimizer.zero_grad()
            _, logits = model(x[i : i + 1])
            loss = loss_fn(logits, y[i : i + 1])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            _, val_logits = model(x[val_idx])
            val_preds = val_logits.argmax(dim=-1)
            val_acc = float((val_preds == y[val_idx]).float().mean())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.out_layer.state_dict().items()}

        print(f"epoch {epoch + 1:02d}/{n_epochs}  train_loss={total_loss / len(train_idx):.4f}  val_acc={val_acc:.3f}")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    fine_tuned_path = RESULTS_DIR / "aasist_out_layer_finetuned.pth"
    torch.save(best_state, fine_tuned_path)
    print(f"Saved fine-tuned output layer to {fine_tuned_path}")
    print(
        f"\nHonesty note: this is a calibration-scale result on {len(y)} total "
        f"examples ({len(val_idx)} held out for validation) — informative "
        "about whether the fine-tuning mechanism works end to end, not a "
        "claim this generalises to real-world Hindi calls. Re-run "
        "run_held_out_eval.py's mechanism against this fine-tuned layer "
        "once a genuinely held-out second synthesis system exists, for the "
        "number that actually matters."
    )


if __name__ == "__main__":
    main()
