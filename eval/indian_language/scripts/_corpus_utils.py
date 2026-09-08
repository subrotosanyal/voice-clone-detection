"""Shared, venv-agnostic helpers for the synthetic-corpus generation
scripts. Deliberately has NO torch/TTS-library import — generate_synthetic_
corpus.py (Coqui XTTS-v2) and generate_synthetic_corpus_chatterbox.py
(Resemble Chatterbox) run under two SEPARATE, mutually incompatible venvs
(see ../README.md's "Datasets" section — chatterbox-tts hard-pins
torch==2.6.0/transformers==5.2.0, which conflicts with coqui-tts's
torch==2.11.0/transformers==4.57.6 in the same environment), so anything
shared between them must avoid importing either's heavy dependencies.
"""
from __future__ import annotations

from pathlib import Path


def parse_hindi_sentences(sentences_md_path: Path) -> list[str]:
    """Pulls the Hindi column straight out of sentences.md's table — one
    source of truth, not a duplicated hardcoded list that could drift out
    of sync with the file a human actually reads and edits."""
    rows = []
    for line in sentences_md_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("| #") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        rows.append(cells[2])  # column order: #, English, Hindi, Marathi
    return rows


def pick_reference_speakers(
    genuine: list[dict], eval_dir: Path, max_per_gender: int = 4
) -> dict[str, dict]:
    """Up to max_per_gender reference utterances per gender (the longest
    available recordings for that gender — more reference audio gives a
    voice-cloning model more to work with), not just one. See
    generate_synthetic_corpus.py's own docstring for why this was scaled
    up from a single speaker per gender."""
    import soundfile as sf

    by_gender: dict[int, list[dict]] = {}
    for entry in genuine:
        if entry["language"] != "hi":
            continue
        gender = entry["gender"]
        wav_path = eval_dir / entry["wav_path"]
        info = sf.info(wav_path)
        duration = info.frames / info.samplerate
        by_gender.setdefault(gender, []).append({**entry, "_duration": duration})

    picked: dict[str, dict] = {}
    for gender, entries in by_gender.items():
        entries.sort(key=lambda e: e["_duration"], reverse=True)
        for rank, entry in enumerate(entries[:max_per_gender]):
            picked[f"{gender}_{rank}"] = entry
    return picked
