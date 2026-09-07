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

Honesty note: this produces spoof examples from exactly ONE synthesis
system. The blueprint's held-out-generator design needs a SECOND system,
held out entirely from anything the acoustic detector is fine-tuned on —
that's still missing (see README.md's Status). Everything generated here
should be treated as "the trained-on system", not the honest held-out
number, until a second system is wired in.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

_EVAL_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _EVAL_DIR / "data" / "synthetic"
GENUINE_MANIFEST = _EVAL_DIR / "data" / "genuine" / "manifest.json"
SENTENCES_MD = _EVAL_DIR / "prompts" / "sentences.md"

_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
_LICENSE_NOTE = (
    "Coqui XTTS-v2 (CPML — non-commercial, see https://coqui.ai/cpml); "
    "voice-cloned from a genuine speaker in the CC BY 4.0 IndicTTS-Hindi corpus"
)


def _parse_hindi_sentences() -> list[str]:
    """Pulls the Hindi column straight out of sentences.md's table — one
    source of truth, not a duplicated hardcoded list that could drift out
    of sync with the file a human actually reads and edits."""
    rows = []
    for line in SENTENCES_MD.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("| #") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        rows.append(cells[2])  # column order: #, English, Hindi, Marathi
    return rows


def _pick_reference_speakers(genuine: list[dict]) -> dict[int, dict]:
    """One reference utterance per gender — the longest available recording
    for that gender, since XTTS-v2's cloning quality benefits from more
    reference audio (its own docs recommend ~6+ seconds)."""
    by_gender: dict[int, dict] = {}
    for entry in genuine:
        if entry["language"] != "hi":
            continue
        gender = entry["gender"]
        wav_path = _EVAL_DIR / entry["wav_path"]
        import soundfile as sf

        info = sf.info(wav_path)
        duration = info.frames / info.samplerate
        if gender not in by_gender or duration > by_gender[gender]["_duration"]:
            entry_with_duration = {**entry, "_duration": duration}
            by_gender[gender] = entry_with_duration
    return by_gender


def main() -> None:
    from TTS.api import TTS

    genuine = json.loads(GENUINE_MANIFEST.read_text())
    sentences = _parse_hindi_sentences()
    print(f"Loaded {len(sentences)} Hindi sentences from {SENTENCES_MD.name}")

    reference_speakers = _pick_reference_speakers(genuine)
    print(f"Reference speakers (by gender code): {list(reference_speakers.keys())}")
    for gender, ref in reference_speakers.items():
        print(f"  gender={gender}: {ref['utterance_id']} ({ref['_duration']:.1f}s)")

    print(f"\nLoading {_MODEL_ID} ...")
    tts = TTS(_MODEL_ID).to("cpu")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    i = 0
    for gender, ref in reference_speakers.items():
        reference_wav = _EVAL_DIR / ref["wav_path"]
        for text in sentences:
            utterance_id = f"hi_synth_{i:05d}"
            wav_path = DATA_DIR / f"{utterance_id}.wav"
            print(f"  synthesizing {utterance_id} (gender={gender}): {text[:40]}...")
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
                    "gender": gender,
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
