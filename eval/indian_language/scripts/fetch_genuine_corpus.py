#!/usr/bin/env python3
"""Fetches a small, reproducible subset of two CC BY 4.0 genuine-speech
corpora (Hindi, Marathi) — see ../README.md's "Datasets" table for the
license verification behind this choice (cross-checked against the
IndicTTS23 paper, not just the HF dataset card).

Uses `datasets`' streaming mode so it never downloads the full multi-GB
parquet files for either corpus — only the first N examples per language
are pulled and written out as individual WAV files plus one manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "genuine"
N_PER_LANGUAGE = 20

_CORPORA = {
    "hi": {"dataset": "SPRINGLab/IndicTTS-Hindi", "license": "CC BY 4.0"},
    "mr": {
        "dataset": "SPRINGLab/IndicTTS_Marathi",
        "license": "CC BY 4.0 (inherited from the same Indic TTS Database "
        "source project as the Hindi corpus — see README.md's caveat "
        "about this specific dataset card's copy-paste artifact)",
    },
}

# Crude but real telltale-word check — Marathi's own grammar particles
# ("आहे", "मला", "तुम्ही") vs Hindi's ("है", "मुझे", "आप") — used only to
# flag a suspiciously Hindi-looking "Marathi" corpus, per README.md's
# caveat about that dataset's own README containing a line that looks
# copy-pasted from the Hindi one ("this... contains the Hindi monolingual
# portion"). Not a substitute for a native speaker actually listening.
_MARATHI_MARKERS = ["आहे", "आहेत", "मला", "तुम्ही", "करा"]
_HINDI_MARKERS = ["है", "हैं", "मुझे", "आप", "करें"]


def _verify_language(texts: list[str], expected: str) -> None:
    marathi_hits = sum(1 for t in texts for m in _MARATHI_MARKERS if m in t)
    hindi_hits = sum(1 for t in texts for m in _HINDI_MARKERS if m in t)
    print(f"  language sanity check: marathi-marker hits={marathi_hits}, hindi-marker hits={hindi_hits}")
    if expected == "mr" and hindi_hits > marathi_hits and hindi_hits > 3:
        print(
            "  WARNING: this 'Marathi' corpus's transcripts look more Hindi than "
            "Marathi by a crude marker-word count — matches the README's caveat "
            "about a copy-paste artifact in its dataset card. Do not trust this "
            "as genuine Marathi without a native speaker actually listening to a sample."
        )


def fetch(language: str, n: int = N_PER_LANGUAGE) -> list[dict]:
    corpus = _CORPORA[language]
    print(f"Streaming {corpus['dataset']} ({language})...")
    ds = load_dataset(corpus["dataset"], split="train", streaming=True)

    out_dir = DATA_DIR / language
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    texts = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        audio = row["audio"]
        samples = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = audio["sampling_rate"]
        text = row.get("text") or row.get("sentence") or row.get("raw_text") or ""
        texts.append(text)

        utterance_id = f"{language}_{i:05d}"
        wav_path = out_dir / f"{utterance_id}.wav"
        sf.write(wav_path, samples, sample_rate)

        manifest_entries.append(
            {
                "utterance_id": utterance_id,
                "language": language,
                "text": text,
                "source_dataset": corpus["dataset"],
                "license": corpus["license"],
                "gender": row.get("gender"),
                "wav_path": str(wav_path.relative_to(DATA_DIR.parent.parent)),
                "sample_rate": sample_rate,
            }
        )
        print(f"  saved {utterance_id} ({len(samples) / sample_rate:.1f}s)")

    _verify_language(texts, language)
    return manifest_entries


def main() -> None:
    all_entries: list[dict] = []
    for lang in _CORPORA:
        all_entries.extend(fetch(lang))

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(all_entries)} entries to {manifest_path}")


if __name__ == "__main__":
    main()
