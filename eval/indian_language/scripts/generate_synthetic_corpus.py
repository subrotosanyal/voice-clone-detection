#!/usr/bin/env python3
"""Generates synthetic (spoof) Hindi speech via Coqui XTTS-v2, cloning each
reference speaker's own genuine voice to read this project's fixed
sentence set — see ../README.md's "Rejected options" for why XTTS-v2
(not IndicF5/Indic Parler TTS, both gated) is the only viable option
found, and why it covers Hindi only (XTTS-v2 doesn't support Marathi).

Licence note: the `coqui-tts` library itself is MPL-2.0 (permissive); the
XTTS-v2 *model weights* are CPML (Coqui Public Model License) —
non-commercial without a paid tier, already judged "fine for a hackathon
eval" in the original blueprint's own component table. COQUI_TOS_AGREED=1
accepts that model licence non-interactively; it is set here, not hidden
from you — read https://coqui.ai/cpml before using this for anything
beyond evaluation.

Honesty note: this produces spoof examples from Coqui XTTS-v2 — see
generate_synthetic_corpus_chatterbox.py for the SECOND, independent
synthesis system (Resemble Chatterbox) the blueprint's held-out-generator
design needs, held out entirely from anything the acoustic detector is
fine-tuned on. Run BOTH before drawing a generalisation conclusion —
either alone is "the trained-on system", not the honest held-out number.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

_EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpus_utils import parse_hindi_sentences, pick_reference_speakers  # noqa: E402

DATA_DIR = _EVAL_DIR / "data" / "synthetic"
GENUINE_MANIFEST = _EVAL_DIR / "data" / "genuine" / "manifest.json"
SENTENCES_MD = _EVAL_DIR / "prompts" / "sentences.md"

_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
_LICENSE_NOTE = (
    "Coqui XTTS-v2 (CPML — non-commercial, see https://coqui.ai/cpml); "
    "voice-cloned from a genuine speaker in the CC BY 4.0 IndicTTS-Hindi corpus"
)


def main() -> None:
    from TTS.api import TTS

    genuine = json.loads(GENUINE_MANIFEST.read_text())
    sentences = parse_hindi_sentences(SENTENCES_MD)
    print(f"Loaded {len(sentences)} Hindi sentences from {SENTENCES_MD.name}")

    reference_speakers = pick_reference_speakers(genuine, _EVAL_DIR)
    print(f"Reference speakers ({len(reference_speakers)} total): {list(reference_speakers.keys())}")
    for key, ref in reference_speakers.items():
        print(f"  {key} (gender={ref['gender']}): {ref['utterance_id']} ({ref['_duration']:.1f}s)")

    print(f"\nLoading {_MODEL_ID} ...")
    tts = TTS(_MODEL_ID).to("cpu")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    i = 0
    for key, ref in reference_speakers.items():
        reference_wav = _EVAL_DIR / ref["wav_path"]
        for text in sentences:
            utterance_id = f"hi_synth_{i:05d}"
            wav_path = DATA_DIR / f"{utterance_id}.wav"
            print(f"  synthesizing {utterance_id} (ref={key}): {text[:40]}...")
            tts.tts_to_file(
                text=text,
                speaker_wav=str(reference_wav),
                language="hi",
                file_path=str(wav_path),
            )
            manifest_entries.append(
                {
                    "utterance_id": utterance_id,
                    "language": "hi",
                    "text": text,
                    "source_dataset": "coqui/XTTS-v2",
                    "license": _LICENSE_NOTE,
                    "gender": ref["gender"],
                    "reference_utterance_id": ref["utterance_id"],
                    "wav_path": str(wav_path.relative_to(_EVAL_DIR)),
                    "sample_rate": 24000,  # XTTS-v2's native output rate
                }
            )
            i += 1

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(manifest_entries)} synthetic entries to {manifest_path}")


if __name__ == "__main__":
    main()
