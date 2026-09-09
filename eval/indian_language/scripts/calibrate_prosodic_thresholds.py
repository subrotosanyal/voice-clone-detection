#!/usr/bin/env python3
"""Calibrates the prosodic detector's jitter/shimmer/HNR thresholds
against real labeled data — see ../TODO.md's "Phase 3" and
app/adapters/detectors/prosody_parselmouth.py's own HONESTY NOTE: the
currently-deployed `_JITTER_FLOOR`/`_SHIMMER_FLOOR`/`_HNR_CEILING`
constants are a placeholder hypothesis ("clinical voice-quality
literature commonly cites..."), never validated against a labeled
genuine-vs-synthetic corpus. This script is that validation.

Must be run with services/live-call-api's own venv active, same reason
as fine_tune_aasist.py/run_held_out_eval.py (imports Parselmouth's own
`_measure_voice_quality` directly rather than re-implementing it):

    source ../../services/live-call-api/.venv/bin/activate
    python scripts/calibrate_prosodic_thresholds.py

TRAIN vs HELD-OUT split, same discipline as run_held_out_eval.py/
fine_tune_aasist.py: Chatterbox-cloned spoof (`data/synthetic_chatterbox/`)
and the local speakers' own held-out test utterances
(`data/local_speakers/test/`) are NEVER used for calibration, only for
reporting a genuinely held-out number afterward — a threshold that only
looks good on the same data it was tuned against isn't a real result
(see README.md's "Held-out generalisation check" for why this project
already treats that distinction as load-bearing).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_EVAL_DIR = Path(__file__).resolve().parent.parent
_LIVE_CALL_API_DIR = _EVAL_DIR.parent.parent / "services" / "live-call-api"
sys.path.insert(0, str(_LIVE_CALL_API_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.adapters.detectors.prosody_parselmouth import (  # noqa: E402
    _HNR_CEILING,
    _JITTER_FLOOR,
    _MIN_DURATION_S,
    _SHIMMER_FLOOR,
    _measure_voice_quality,
)
from _corpus_utils import local_speaker_entries  # noqa: E402

RESULTS_DIR = _EVAL_DIR / "results"


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _collect_features(entries: list[dict], label: str) -> tuple[np.ndarray, list[dict]]:
    """Runs Praat's real jitter/shimmer/HNR extraction (via the exact same
    function the deployed detector calls) over every file, printing
    verbose per-file progress. Returns an (N, 3) feature array plus the
    detail dicts for files that succeeded."""
    features = []
    kept_entries = []
    n_abstained = 0
    start = time.monotonic()

    for i, entry in enumerate(entries):
        wav_path = _EVAL_DIR / entry["wav_path"]
        samples, sr = sf.read(wav_path, dtype="float64", always_2d=False)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        duration_s = samples.size / sr if sr else 0.0

        if duration_s < _MIN_DURATION_S:
            n_abstained += 1
            print(f"  [{label}] [{i + 1}/{len(entries)}] {entry['utterance_id']}: SKIP (too short, {duration_s * 1000:.0f}ms)")
            continue
        try:
            jitter, shimmer, hnr_db = _measure_voice_quality(samples, sr)
        except Exception as exc:
            n_abstained += 1
            print(f"  [{label}] [{i + 1}/{len(entries)}] {entry['utterance_id']}: SKIP (Praat error: {exc})")
            continue
        if any(np.isnan(v) for v in (jitter, shimmer, hnr_db)):
            n_abstained += 1
            print(f"  [{label}] [{i + 1}/{len(entries)}] {entry['utterance_id']}: SKIP (no voiced pitch periods)")
            continue

        features.append([jitter, shimmer, hnr_db])
        kept_entries.append(entry)
        if (i + 1) % 25 == 0 or (i + 1) == len(entries):
            elapsed = time.monotonic() - start
            print(
                f"  [{label}] [{i + 1}/{len(entries)}] {entry['utterance_id']}: "
                f"jitter={jitter:.4f} shimmer={shimmer:.4f} hnr={hnr_db:.1f}dB "
                f"({elapsed:.0f}s elapsed, {n_abstained} abstained so far)"
            )

    print(f"  [{label}] done: {len(kept_entries)}/{len(entries)} usable ({n_abstained} abstained)")
    return np.array(features), kept_entries


def _distribution_summary(features: np.ndarray) -> dict:
    names = ["jitter_local", "shimmer_local", "hnr_db"]
    return {
        name: {
            "mean": float(np.mean(features[:, i])),
            "std": float(np.std(features[:, i])),
            "p10": float(np.percentile(features[:, i], 10)),
            "p50": float(np.percentile(features[:, i], 50)),
            "p90": float(np.percentile(features[:, i], 90)),
        }
        for i, name in enumerate(names)
    }


def _current_formula_scores(features: np.ndarray, jitter_floor: float, shimmer_floor: float, hnr_ceiling: float) -> np.ndarray:
    """Mirrors ParselmouthProsodyDetector.score()'s exact math — the point
    is to measure how the CURRENTLY DEPLOYED thresholds actually perform
    against real data, not a reimplementation that could silently drift
    from production."""
    jitter, shimmer, hnr = features[:, 0], features[:, 1], features[:, 2]
    jitter_risk = np.clip((jitter_floor - jitter) / jitter_floor, 0.0, 1.0)
    shimmer_risk = np.clip((shimmer_floor - shimmer) / shimmer_floor, 0.0, 1.0)
    hnr_span = 30.0 - hnr_ceiling
    hnr_risk = np.clip((hnr - hnr_ceiling) / hnr_span, 0.0, 1.0) if hnr_span > 0 else np.zeros_like(hnr)
    return np.clip((jitter_risk + shimmer_risk + hnr_risk) / 3.0, 0.0, 1.0)


def _balanced_accuracy_at_threshold(genuine_scores: np.ndarray, spoof_scores: np.ndarray, threshold: float = 0.5) -> dict:
    genuine_correct = float(np.mean(genuine_scores < threshold))  # correctly NOT flagged
    spoof_correct = float(np.mean(spoof_scores >= threshold))  # correctly flagged
    return {
        "genuine_accuracy": genuine_correct,
        "spoof_accuracy": spoof_correct,
        "balanced_accuracy": (genuine_correct + spoof_correct) / 2.0,
    }


def main() -> None:
    print("=" * 70)
    print("Step 1/4: collecting genuine (bonafide) TRAIN entries")
    print("=" * 70)
    genuine_train_entries = (
        _load_manifest(_EVAL_DIR / "data" / "genuine" / "manifest.json")
        + _load_manifest(_EVAL_DIR / "data" / "genuine" / "hi_kathbath" / "manifest.json")
        + _load_manifest(_EVAL_DIR / "data" / "genuine" / "hi_movie_musnomix" / "manifest.json")
        + local_speaker_entries(_EVAL_DIR / "data" / "local_speakers" / "train", _EVAL_DIR)
    )
    print(f"Total genuine TRAIN candidates: {len(genuine_train_entries)}")
    genuine_train_features, genuine_train_kept = _collect_features(genuine_train_entries, "genuine/train")

    print("\n" + "=" * 70)
    print("Step 2/4: collecting spoof TRAIN entries (XTTS-v2 only — Chatterbox held out)")
    print("=" * 70)
    spoof_train_entries = _load_manifest(_EVAL_DIR / "data" / "synthetic" / "manifest.json") + _load_manifest(
        _EVAL_DIR / "data" / "synthetic_local_speakers" / "manifest.json"
    )
    print(f"Total spoof TRAIN candidates: {len(spoof_train_entries)}")
    spoof_train_features, spoof_train_kept = _collect_features(spoof_train_entries, "spoof/train")

    print("\n" + "=" * 70)
    print("Step 3/4: collecting HELD-OUT entries (never used for calibration)")
    print("=" * 70)
    genuine_heldout_entries = local_speaker_entries(_EVAL_DIR / "data" / "local_speakers" / "test", _EVAL_DIR)
    spoof_heldout_entries = _load_manifest(_EVAL_DIR / "data" / "synthetic_chatterbox" / "manifest.json")
    print(f"Held-out genuine (local speakers' own test utterances): {len(genuine_heldout_entries)}")
    print(f"Held-out spoof (Chatterbox — never seen by anything above): {len(spoof_heldout_entries)}")
    genuine_heldout_features, _ = _collect_features(genuine_heldout_entries, "genuine/held-out")
    spoof_heldout_features, _ = _collect_features(spoof_heldout_entries, "spoof/held-out")

    print("\n" + "=" * 70)
    print("Step 4/4: distributions, current-threshold performance, and recalibration")
    print("=" * 70)
    print("\nGenuine (train) feature distribution:")
    genuine_dist = _distribution_summary(genuine_train_features)
    print(json.dumps(genuine_dist, indent=2))
    print("\nSpoof (train) feature distribution:")
    spoof_dist = _distribution_summary(spoof_train_features)
    print(json.dumps(spoof_dist, indent=2))

    current_thresholds = {"jitter_floor": _JITTER_FLOOR, "shimmer_floor": _SHIMMER_FLOOR, "hnr_ceiling": _HNR_CEILING}
    print(f"\nCurrently deployed thresholds (unvalidated placeholders): {current_thresholds}")

    genuine_train_scores = _current_formula_scores(genuine_train_features, **current_thresholds)
    spoof_train_scores = _current_formula_scores(spoof_train_features, **current_thresholds)
    current_perf_train = _balanced_accuracy_at_threshold(genuine_train_scores, spoof_train_scores)
    print(f"Current thresholds on TRAIN data: {current_perf_train}")

    genuine_heldout_scores = _current_formula_scores(genuine_heldout_features, **current_thresholds)
    spoof_heldout_scores = _current_formula_scores(spoof_heldout_features, **current_thresholds)
    current_perf_heldout = _balanced_accuracy_at_threshold(genuine_heldout_scores, spoof_heldout_scores)
    print(f"Current thresholds on HELD-OUT data: {current_perf_heldout}")

    # Data-driven recalibration: set each floor/ceiling to the genuine
    # distribution's own 10th/90th percentile (i.e. "unnaturally low/high
    # relative to what real speech in THIS corpus actually looks like",
    # replacing the generic clinical-literature guess with this project's
    # own measured reference range).
    recommended_thresholds = {
        "jitter_floor": genuine_dist["jitter_local"]["p10"],
        "shimmer_floor": genuine_dist["shimmer_local"]["p10"],
        "hnr_ceiling": genuine_dist["hnr_db"]["p90"],
    }
    print(f"\nRecommended thresholds (genuine-distribution-derived): {recommended_thresholds}")

    genuine_train_scores_new = _current_formula_scores(genuine_train_features, **recommended_thresholds)
    spoof_train_scores_new = _current_formula_scores(spoof_train_features, **recommended_thresholds)
    new_perf_train = _balanced_accuracy_at_threshold(genuine_train_scores_new, spoof_train_scores_new)
    print(f"Recommended thresholds on TRAIN data: {new_perf_train}")

    genuine_heldout_scores_new = _current_formula_scores(genuine_heldout_features, **recommended_thresholds)
    spoof_heldout_scores_new = _current_formula_scores(spoof_heldout_features, **recommended_thresholds)
    new_perf_heldout = _balanced_accuracy_at_threshold(genuine_heldout_scores_new, spoof_heldout_scores_new)
    print(f"Recommended thresholds on HELD-OUT data: {new_perf_heldout}")

    # Alternative: a trained classifier (logistic regression on the raw
    # jitter/shimmer/HNR triple) — reported honestly ALONGSIDE the
    # threshold-tuning approach above, not instead of it, so the actual
    # gain (if any) from a learned model over a hand-tuned formula is
    # visible rather than assumed.
    #
    # REAL BUG found and fixed 2026-09-09: fitting LogisticRegression
    # directly on raw (jitter, shimmer, hnr_db) triggered `RuntimeWarning:
    # divide by zero / overflow encountered in matmul` during the first
    # run — jitter lives on a ~0.01-0.03 scale, HNR on a ~10-20 scale, so
    # an unstandardized fit lets HNR dominate the gradient and pushes the
    # optimizer toward numerically unstable weights. Standardizing each
    # feature (zero mean, unit variance) before fitting is the standard
    # fix and the numerically honest way to report this number.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = np.vstack([genuine_train_features, spoof_train_features])
    y_train = np.concatenate([np.zeros(len(genuine_train_features)), np.ones(len(spoof_train_features))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    clf.fit(X_train, y_train)

    X_heldout = np.vstack([genuine_heldout_features, spoof_heldout_features])
    heldout_pred = clf.predict(X_heldout)
    logreg_genuine_acc = float(np.mean(heldout_pred[: len(genuine_heldout_features)] == 0))
    logreg_spoof_acc = float(np.mean(heldout_pred[len(genuine_heldout_features) :] == 1))
    logreg_perf_heldout = {
        "genuine_accuracy": logreg_genuine_acc,
        "spoof_accuracy": logreg_spoof_acc,
        "balanced_accuracy": (logreg_genuine_acc + logreg_spoof_acc) / 2.0,
    }
    logreg = clf.named_steps["logisticregression"]
    print(f"\nLogistic regression (jitter/shimmer/HNR -> bonafide/spoof) on HELD-OUT data: {logreg_perf_heldout}")
    print(
        f"Logistic regression coefficients on STANDARDIZED features (jitter, shimmer, hnr_db): "
        f"{logreg.coef_[0].tolist()}, intercept={logreg.intercept_[0]:.4f}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "train_set_sizes": {"genuine": len(genuine_train_kept), "spoof": len(spoof_train_kept)},
        "held_out_set_sizes": {"genuine": len(genuine_heldout_features), "spoof": len(spoof_heldout_features)},
        "feature_distributions": {"genuine_train": genuine_dist, "spoof_train": spoof_dist},
        "current_thresholds": current_thresholds,
        "current_thresholds_performance": {"train": current_perf_train, "held_out": current_perf_heldout},
        "recommended_thresholds": recommended_thresholds,
        "recommended_thresholds_performance": {"train": new_perf_train, "held_out": new_perf_heldout},
        "logistic_regression": {
            "note": "coefficients are on STANDARDIZED (zero-mean, unit-variance) features — see StandardScaler means_/scale_ below to convert back to raw units",
            "scaler_mean_jitter_shimmer_hnr": clf.named_steps["standardscaler"].mean_.tolist(),
            "scaler_scale_jitter_shimmer_hnr": clf.named_steps["standardscaler"].scale_.tolist(),
            "coefficients_jitter_shimmer_hnr": logreg.coef_[0].tolist(),
            "intercept": float(logreg.intercept_[0]),
            "held_out_performance": logreg_perf_heldout,
        },
    }
    results_path = RESULTS_DIR / "prosodic_calibration.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nWrote full calibration results to {results_path}")


if __name__ == "__main__":
    main()
