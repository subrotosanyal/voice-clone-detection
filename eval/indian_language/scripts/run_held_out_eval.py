#!/usr/bin/env python3
"""Runs the SAME AasistAcousticDetector the live service uses over the
genuine (and, once generated, synthetic) Indian-language corpora, and
reports accuracy/EER per language — this is what produces the "Indian-
language gap" number referenced throughout docs/blueprint.html.

Must be run with services/live-call-api's own venv active (it imports
AasistAcousticDetector directly from there, rather than duplicating a
second torch install just for this offline script):

    source ../../services/live-call-api/.venv/bin/activate
    python scripts/run_held_out_eval.py

Honesty note: as of the first real run of this script, only the genuine
(bonafide) side of the corpus exists — see ../README.md's "Status". That
means this currently reports "does AASIST, trained only on English,
correctly recognise genuine Hindi/Marathi speech as NOT synthetic" (a
false-positive-rate check), not yet the full bonafide-vs-spoof EER the
blueprint's §04/§08 ultimately wants — that needs
generate_synthetic_corpus.py's output too. The script computes EER
automatically the moment synthetic data is present; nothing here needs to
change when that happens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

_EVAL_DIR = Path(__file__).resolve().parent.parent
_LIVE_CALL_API_DIR = _EVAL_DIR.parent.parent / "services" / "live-call-api"
sys.path.insert(0, str(_LIVE_CALL_API_DIR))

from app.adapters.detectors.acoustic_aasist import AasistAcousticDetector  # noqa: E402
from app.domain.models import AudioWindow  # noqa: E402

CHECKPOINT_PATH = _LIVE_CALL_API_DIR / "app/adapters/detectors/vendor/checkpoints/AASIST.pth"
# Manifests store wav_path relative to eval/indian_language/ (see
# fetch_genuine_corpus.py's DATA_DIR.parent.parent, which IS this
# directory) — not the repo root two levels further up.
MANIFEST_BASE_DIR = _EVAL_DIR


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _score_utterance(detector: AasistAcousticDetector, entry: dict) -> float | None:
    wav_path = MANIFEST_BASE_DIR / entry["wav_path"]
    samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    window = AudioWindow(session_id="eval", seq=0, sample_rate=sample_rate, samples=samples, window_start_ms=0)
    result = detector.score(window, context={})
    if result.score is None:
        print(f"    {entry['utterance_id']}: abstained ({result.abstain_reason})")
        return None
    return result.detail["spoof_probability"]


def _compute_eer(bonafide_scores: list[float], spoof_scores: list[float]) -> float | None:
    """Equal Error Rate via a simple threshold sweep — no external metric
    library, so every number here is traceable to plain arithmetic, same
    explainability standard as the rest of this project."""
    if not bonafide_scores or not spoof_scores:
        return None
    thresholds = np.linspace(0.0, 1.0, 1001)
    bonafide = np.array(bonafide_scores)
    spoof = np.array(spoof_scores)
    best_gap = None
    eer = None
    for t in thresholds:
        far = float(np.mean(spoof < t))  # spoof wrongly accepted as bonafide
        frr = float(np.mean(bonafide >= t))  # bonafide wrongly rejected as spoof
        gap = abs(far - frr)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            eer = (far + frr) / 2
    return eer


def _score_all_languages(detector: AasistAcousticDetector, genuine: list[dict], synthetic: list[dict]) -> dict:
    languages = sorted({e["language"] for e in genuine})
    result: dict = {}
    for lang in languages:
        print(f"\n=== {lang} ===")
        genuine_entries = [e for e in genuine if e["language"] == lang]
        spoof_entries = [e for e in synthetic if e["language"] == lang]

        print(f"  scoring {len(genuine_entries)} genuine utterances...")
        bonafide_scores = [s for e in genuine_entries if (s := _score_utterance(detector, e)) is not None]

        spoof_scores = []
        if spoof_entries:
            print(f"  scoring {len(spoof_entries)} synthetic utterances...")
            spoof_scores = [s for e in spoof_entries if (s := _score_utterance(detector, e)) is not None]

        bonafide_accuracy = float(np.mean([s < 0.5 for s in bonafide_scores])) if bonafide_scores else None
        spoof_accuracy = float(np.mean([s >= 0.5 for s in spoof_scores])) if spoof_scores else None
        eer = _compute_eer(bonafide_scores, spoof_scores)

        result[lang] = {
            "n_genuine": len(bonafide_scores),
            "n_synthetic": len(spoof_scores),
            "bonafide_accuracy": bonafide_accuracy,  # fraction of genuine speech correctly NOT flagged as spoof
            "spoof_accuracy": spoof_accuracy,  # fraction of synthetic speech correctly flagged as spoof
            "eer": eer,
            "mean_spoof_probability_genuine": float(np.mean(bonafide_scores)) if bonafide_scores else None,
        }
        print(f"  bonafide_accuracy={bonafide_accuracy}  spoof_accuracy={spoof_accuracy}  eer={eer}")
    return result


def main() -> None:
    genuine = _load_manifest(_EVAL_DIR / "data" / "genuine" / "manifest.json")
    synthetic = _load_manifest(_EVAL_DIR / "data" / "synthetic" / "manifest.json")

    if not genuine:
        print("No genuine corpus found — run scripts/fetch_genuine_corpus.py first.")
        return

    summary: dict = {"notes": []}
    if not synthetic:
        summary["notes"].append(
            "No synthetic corpus found — this run only measures AASIST's "
            "false-positive rate on genuine speech, not full bonafide-vs-"
            "spoof EER. Run scripts/generate_synthetic_corpus.py to add the "
            "spoof side."
        )

    print(f"Loading AASIST (pretrained checkpoint) from {CHECKPOINT_PATH} ...")
    detector = AasistAcousticDetector(checkpoint_path=str(CHECKPOINT_PATH))
    print("\n--- BEFORE fine-tuning ---")
    summary["before_finetune"] = _score_all_languages(detector, genuine, synthetic)
    # kept for backward compatibility with anything reading the old shape
    summary["languages"] = summary["before_finetune"]

    finetuned_path = _EVAL_DIR / "results" / "aasist_out_layer_finetuned.pth"
    if finetuned_path.exists():
        print(f"\nLoading fine-tuned output layer from {finetuned_path} ...")
        # Reaches into AasistAcousticDetector's private _model to swap just
        # the out_layer weights — a pragmatic shortcut acceptable in this
        # standalone eval script (not production code, see fine_tune_aasist.py
        # for why only this one layer is fine-tuned at all).
        finetuned_detector = AasistAcousticDetector(checkpoint_path=str(CHECKPOINT_PATH))
        finetuned_state = torch.load(finetuned_path, map_location="cpu")
        finetuned_detector._model.out_layer.load_state_dict(finetuned_state)
        finetuned_detector._model.eval()
        print("\n--- AFTER fine-tuning ---")
        summary["after_finetune"] = _score_all_languages(finetuned_detector, genuine, synthetic)
        summary["notes"].append(
            "after_finetune is evaluated on the SAME data the fine-tune ran "
            "on (no genuinely held-out second synthesis system exists yet) "
            "— read this as 'did the mechanism improve fit', not a "
            "generalisation claim. See fine_tune_aasist.py's docstring."
        )
    else:
        summary["notes"].append(
            "No fine-tuned checkpoint found — run scripts/fine_tune_aasist.py "
            "to add a before/after comparison."
        )

    results_dir = _EVAL_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
