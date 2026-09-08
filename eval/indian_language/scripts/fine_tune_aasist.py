#!/usr/bin/env python3
"""Fine-tunes AASIST's final classification layer on the Indian-language
corpus (genuine + synthetic manifests) — see ../README.md's "Status" for
exactly what data this needs (both classes; requires
generate_synthetic_corpus.py's output to exist first).

Must be run with services/live-call-api's own venv active, same reason as
run_held_out_eval.py:

    source ../../services/live-call-api/.venv/bin/activate
    python scripts/fine_tune_aasist.py

WHY ONLY THE FINAL LAYER, NOT THE WHOLE MODEL: with a few hundred examples
total (200 genuine Hindi + whatever generate_synthetic_corpus.py produced
— currently 40 synthetic Hindi utterances from 4 cloned speakers, all
one gender code per fetch_genuine_corpus.py's own honest finding that
the source corpus's gender field wasn't actually mixed in the fetched
range), fully fine-tuning AASIST's ~300k-parameter graph-attention
network would still badly overfit — it would memorise this comparatively
tiny set rather than learn anything that generalises to a real caller's
voice. Freezing every layer except AasistNet's final `out_layer` (a
single `nn.Linear(5*gat_dims[1], 2)`) is closer to recalibrating the
existing English-trained features for Hindi than retraining the model —
a defensible thing to do with this little data. Full fine-tuning needs
hundreds-to-thousands of examples PER CLASS to be trustworthy — the
genuine side is there, the spoof side (40) very much isn't yet; that's
exactly why real dataset collection (more volunteers, more languages,
Malvi still entirely missing) matters more than this script's
cleverness does.

CLASS IMBALANCE (fixed 2026-09-08): scaling genuine 10x while only
scaling synthetic 2x left the training set 5.5:1 imbalanced, which a
plain unweighted loss + raw-accuracy model selection badly mishandled —
see the REAL BUG comment above the training loop in main() for the
measured before/after numbers and the two-part fix (class-weighted loss,
balanced-accuracy model selection).
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


def train_val_split(y: torch.Tensor, val_fraction: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """STRATIFIED by class, not a plain random split — real bug found and
    fixed 2026-09-08: with an imbalanced dataset (220 bonafide vs. 40
    spoof after scaling up the genuine corpus), a bare random split can by
    chance under-represent the minority class in validation, making any
    per-class metric computed on it noisy. Splitting each class
    separately guarantees exactly val_fraction of EACH class ends up in
    validation, regardless of the overall imbalance."""
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    for cls in torch.unique(y).tolist():
        cls_idx = rng.permutation(np.where(y.numpy() == cls)[0])
        n_val = max(1, int(len(cls_idx) * val_fraction))
        val_idx.append(cls_idx[:n_val])
        train_idx.append(cls_idx[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


def _per_class_accuracy(preds: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Returns bonafide/spoof/balanced accuracy — balanced (the mean of
    the two per-class accuracies) is what model selection uses below,
    NOT raw overall accuracy, precisely because raw accuracy on an
    imbalanced set rewards a degenerate "always predict the majority
    class" model — see the REAL BUG note in main()'s docstring-length
    comment before the training loop."""
    bonafide_mask = y == 1
    spoof_mask = y == 0
    bonafide_acc = float((preds[bonafide_mask] == 1).float().mean()) if bonafide_mask.any() else float("nan")
    spoof_acc = float((preds[spoof_mask] == 0).float().mean()) if spoof_mask.any() else float("nan")
    overall_acc = float((preds == y).float().mean())
    balanced_acc = (bonafide_acc + spoof_acc) / 2
    return {
        "overall_acc": overall_acc,
        "bonafide_acc": bonafide_acc,
        "spoof_acc": spoof_acc,
        "balanced_acc": balanced_acc,
    }


def main() -> None:
    print("Building dataset from genuine + synthetic manifests...")
    x, y = build_dataset()
    n_bonafide = int((y == 1).sum())
    n_spoof = int((y == 0).sum())
    print(f"Total examples: {len(y)} (bonafide={n_bonafide}, spoof={n_spoof})")

    train_idx, val_idx = train_val_split(y)
    n_train_bonafide = int((y[train_idx] == 1).sum())
    n_train_spoof = int((y[train_idx] == 0).sum())
    n_val_bonafide = int((y[val_idx] == 1).sum())
    n_val_spoof = int((y[val_idx] == 0).sum())
    print(
        f"Train: {len(train_idx)} (bonafide={n_train_bonafide}, spoof={n_train_spoof})  "
        f"Val: {len(val_idx)} (bonafide={n_val_bonafide}, spoof={n_val_spoof})"
    )
    if n_val_spoof < 10:
        print(
            f"  NOTE: only {n_val_spoof} spoof examples in validation — spoof_acc/balanced_acc "
            "below will be noisy (each single example is worth "
            f"{100 / n_val_spoof:.0f} percentage points). Real fix is more spoof data, not "
            "a bigger validation split."
        )

    print(f"Loading AASIST checkpoint from {CHECKPOINT_PATH} ...")
    model = AasistNet(_DEFAULT_MODEL_CONFIG)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"), strict=True)

    # Freeze everything except the final classification layer — see the
    # module docstring for why, given how little data exists right now.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.out_layer.parameters():
        param.requires_grad = True

    # REAL BUG, found and fixed 2026-09-08 (independent of the class-
    # imbalance bug below, and MORE fundamental): AasistNet has 18
    # BatchNorm1d/2d layers (track_running_stats=True) plus Dropout
    # layers in the "frozen" backbone. Setting requires_grad=False stops
    # their WEIGHTS from updating, but does nothing to stop BatchNorm's
    # running_mean/running_var buffers from drifting (they update via a
    # momentum rule on every forward pass made in .train() mode,
    # independent of gradients) or Dropout from injecting fresh random
    # noise into every forward pass, again only in .train() mode. Verified
    # by hand: the exact same out_layer weights, reloaded fresh (base
    # checkpoint's original BatchNorm stats, no dropout noise) vs.
    # evaluated live mid-training, gave WILDLY different predictions on
    # the same audio — one held-out spoof example read
    # spoof_probability~1.0 live during training but 0.39 after reloading
    # the saved checkpoint. Every earlier fine-tuning run's "AFTER
    # fine-tuning" number (including both the un-fixed and imbalance-
    # fixed runs earlier this session) was measuring this drifted, never-
    # saved, never-reproducible live state — not the weights actually
    # persisted to disk. Fixed the simplest correct way: out_layer is a
    # bare nn.Linear with no train/eval-mode-dependent behaviour of its
    # own, so the WHOLE model can stay in .eval() mode throughout
    # training — .eval() only changes BatchNorm/Dropout's forward
    # behaviour, it does not disable autograd, so out_layer's gradients
    # still flow normally.
    model.eval()

    optimizer = torch.optim.Adam(model.out_layer.parameters(), lr=1e-3, weight_decay=1e-2)

    # REAL BUG, fixed 2026-09-08: scaling the genuine corpus 10x (20->200)
    # while only scaling synthetic 2x (20->40) made the training set
    # heavily imbalanced (5.5:1). An unweighted loss plus selecting the
    # "best" checkpoint by raw validation accuracy let the model collapse
    # toward always predicting the majority class ("bonafide") — measured
    # directly: bonafide accuracy hit 100% but spoof accuracy fell to 25%
    # (from 82.5% before fine-tuning), and the EER (the number that
    # actually measures separability) got slightly WORSE, not better, not
    # an improvement at all, just a threshold shift disguised as one by
    # an accuracy metric the imbalance was gaming. Fixed two ways: (1) a
    # class-weighted loss (inverse frequency) so a spoof example counts as
    # much as ~5.5 bonafide examples during training, and (2) selecting
    # the best checkpoint by BALANCED accuracy (mean of the two per-class
    # accuracies), which a majority-class collapse cannot game the way
    # raw accuracy could.
    class_counts = torch.tensor([n_spoof, n_bonafide], dtype=torch.float32)  # index 0=spoof, 1=bonafide
    class_weights = class_counts.sum() / (2 * class_counts)
    print(f"Class weights (inverse frequency): spoof={class_weights[0]:.3f}  bonafide={class_weights[1]:.3f}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    n_epochs = 30
    best_balanced_acc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(n_epochs):
        # Deliberately NOT model.train() — see the REAL BUG comment above
        # the optimizer/loss setup for why the whole model stays in
        # .eval() mode even during training.
        perm = np.random.permutation(train_idx)
        total_loss = 0.0
        for step, i in enumerate(perm, start=1):
            optimizer.zero_grad()
            _, logits = model(x[i : i + 1])
            loss = loss_fn(logits, y[i : i + 1])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            if step % 50 == 0 or step == len(perm):
                print(
                    f"  epoch {epoch + 1:02d}/{n_epochs}  step {step:3d}/{len(perm)}  "
                    f"running_avg_loss={total_loss / step:.4f}"
                )

        model.eval()
        with torch.no_grad():
            _, val_logits = model(x[val_idx])
            val_preds = val_logits.argmax(dim=-1)
        val_metrics = _per_class_accuracy(val_preds, y[val_idx])

        is_best = val_metrics["balanced_acc"] > best_balanced_acc
        if is_best:
            best_balanced_acc = val_metrics["balanced_acc"]
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.out_layer.state_dict().items()}

        print(
            f"epoch {epoch + 1:02d}/{n_epochs}  train_loss={total_loss / len(train_idx):.4f}  "
            f"val_overall_acc={val_metrics['overall_acc']:.3f}  "
            f"val_bonafide_acc={val_metrics['bonafide_acc']:.3f}  "
            f"val_spoof_acc={val_metrics['spoof_acc']:.3f}  "
            f"val_balanced_acc={val_metrics['balanced_acc']:.3f}"
            f"{'  <-- BEST' if is_best else ''}"
        )

    print(f"\nBest epoch: {best_epoch}  Best balanced accuracy: {best_balanced_acc:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    fine_tuned_path = RESULTS_DIR / "aasist_out_layer_finetuned.pth"
    torch.save(best_state, fine_tuned_path)
    print(f"Saved fine-tuned output layer to {fine_tuned_path}")
    print(
        f"\nHonesty note: this is a calibration-scale result on {len(y)} total "
        f"examples ({len(val_idx)} held out for validation, stratified by "
        "class) — informative about whether the fine-tuning mechanism "
        "works end to end, not a claim this generalises to real-world "
        "Hindi calls. Model selection now uses balanced (per-class "
        "averaged) accuracy, not raw accuracy, specifically because this "
        "dataset is imbalanced (see the REAL BUG comment above the "
        "training loop for what raw-accuracy selection did before this "
        f"fix) — but with only {n_val_spoof} spoof examples in "
        "validation, val_spoof_acc/val_balanced_acc are still noisy "
        "single-digit-example measurements, not statistically solid ones. "
        "Re-run run_held_out_eval.py's mechanism against this fine-tuned "
        "layer once a genuinely held-out second synthesis system exists, "
        "for the number that actually matters."
    )


if __name__ == "__main__":
    main()
