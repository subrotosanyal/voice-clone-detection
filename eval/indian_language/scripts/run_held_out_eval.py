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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.adapters.detectors.acoustic_aasist import AasistAcousticDetector  # noqa: E402
from app.domain.models import AudioWindow  # noqa: E402
from _corpus_utils import local_speaker_entries  # noqa: E402

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


def _local_speaker_genuine(split_dir: Path) -> list[dict]:
    """local_speaker_entries() returns bare {utterance_id, wav_path} —
    _score_all_languages groups by "language", so tag each entry "hi"
    (confirmed via a real Whisper transcription check, see
    generate_local_speaker_clones.py's own docstring)."""
    return [{**e, "language": "hi"} for e in local_speaker_entries(split_dir, _EVAL_DIR)]


def main() -> None:
    # Expanded 2026-09-09 to match fine_tune_aasist.py's build_dataset():
    # same four genuine + two spoof TRAIN sources, so "before/after
    # fine-tuning" here measures the same data the fine-tune actually ran
    # on. data/local_speakers/test/ is NEVER included here — see
    # held_out_local_speakers below, the whole point of keeping it separate.
    genuine = (
        _load_manifest(_EVAL_DIR / "data" / "genuine" / "manifest.json")
        + _load_manifest(_EVAL_DIR / "data" / "genuine" / "hi_kathbath" / "manifest.json")
        + _load_manifest(_EVAL_DIR / "data" / "genuine" / "hi_movie_musnomix" / "manifest.json")
        + _local_speaker_genuine(_EVAL_DIR / "data" / "local_speakers" / "train")
    )
    synthetic = _load_manifest(_EVAL_DIR / "data" / "synthetic" / "manifest.json") + _load_manifest(
        _EVAL_DIR / "data" / "synthetic_local_speakers" / "manifest.json"
    )
    synthetic_chatterbox = _load_manifest(_EVAL_DIR / "data" / "synthetic_chatterbox" / "manifest.json")
    genuine_heldout_local_speakers = _local_speaker_genuine(_EVAL_DIR / "data" / "local_speakers" / "test")

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
        print("\n--- AFTER fine-tuning (same synthesis system it was fine-tuned on) ---")
        summary["after_finetune"] = _score_all_languages(finetuned_detector, genuine, synthetic)
        summary["notes"].append(
            "after_finetune is evaluated on the SAME data (XTTS-v2) the "
            "fine-tune ran on — read this as 'did the mechanism improve "
            "fit', not a generalisation claim. See held_out_chatterbox "
            "below for the genuine generalisation check."
        )

        if synthetic_chatterbox:
            print("\n--- AFTER fine-tuning, HELD-OUT second synthesis system (Chatterbox) ---")
            summary["held_out_chatterbox"] = _score_all_languages(
                finetuned_detector, genuine, synthetic_chatterbox
            )
            summary["notes"].append(
                "held_out_chatterbox is the genuine generalisation check the "
                "blueprint's held-out-generator design always needed: "
                "Chatterbox (Resemble AI) is a completely different model "
                "family/training data from XTTS-v2 (Coqui), and its output "
                "was never used in fine-tuning — see "
                "generate_synthetic_corpus_chatterbox.py's own docstring for "
                "how it was generated and verified."
            )

            if genuine_heldout_local_speakers:
                print(
                    "\n--- AFTER fine-tuning, HELD-OUT on BOTH dimensions "
                    "(local speakers' own reserved test clips x Chatterbox) ---"
                )
                summary["held_out_local_speakers"] = _score_all_languages(
                    finetuned_detector, genuine_heldout_local_speakers, synthetic_chatterbox
                )
                summary["notes"].append(
                    "held_out_local_speakers is the MOST rigorous held-out "
                    "check available: genuine side is each local speaker's "
                    "own RESERVED test utterance (never in build_dataset()'s "
                    "training data — only that speaker's other 5 clips were), "
                    "spoof side is Chatterbox (never in training either). "
                    "Neither the specific utterance nor the synthesis system "
                    "was seen during fine-tuning."
                )
        else:
            summary["notes"].append(
                "No held-out Chatterbox corpus found — run "
                "scripts/generate_synthetic_corpus_chatterbox.py (its own "
                "dedicated venv, see that script's docstring) for the "
                "genuine generalisation check."
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
