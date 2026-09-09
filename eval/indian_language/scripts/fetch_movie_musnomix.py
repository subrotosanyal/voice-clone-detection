#!/usr/bin/env python3
"""Fetches the "speech" class from Movie-MUSNOMIX — real Hindi movie
dialogue, a genuinely different acoustic domain (dramatic delivery,
background music bleed, 1966-2011 film-audio recording chains) from both
IndicTTS-Hindi's studio TTS recordings and Kathbath's crowd-sourced read
speech. See README.md's "Datasets" table for the license verification.

REAL LICENSE NOTE (verified 2026-09-09): the repo's own LICENSE file is
CC0 1.0 — but that covers the AUTHORS' OWN contribution (the annotations:
class labels, segment boundaries), not the underlying movie audio itself,
which the repo's own README states explicitly: "this dataset ... is made
available as a public resource solely for academic purposes ... No one
is permitted to use this dataset commercially." Confirmed with the user
before fetching (2026-09-09) that this project is non-commercial, so
that restriction is satisfied — but this data must never be used for
anything that could be read as commercial (this is tracked here, not
enforced by the license itself, so don't lose track of it if this
project's status ever changes).

Only the "speech" class (203 of 1161 total annotated segments — the
other 960 are music/noise/mixed-class segments, not genuine speech) is
fetched: real mp3 files hosted directly in the GitHub repo (not just
timestamp annotations against an external video the user would need to
separately source), one GitHub raw-content download per segment,
converted to wav via ffmpeg (same as fetch_kathbath_hindi.py).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "genuine" / "hi_movie_musnomix"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache" / "movie_musnomix"

_REPO_RAW_BASE = "https://raw.githubusercontent.com/mrinmoy-iitg/Movie-MUSNOMIX/master"
_ANNOTATIONS_URL = f"{_REPO_RAW_BASE}/annotations_all_files.csv"
_LICENSE = (
    "CC0 1.0 (annotations only) — underlying movie audio restricted to "
    "non-commercial/academic use per the dataset's own README disclaimer; "
    "confirmed applicable to this (non-commercial) project by the user "
    "on 2026-09-09, see this file's module docstring"
)
_MOVIES = {
    "movie0": "Devar (1966)",
    "movie1": "Jaal (1967)",
    "movie2": "Adhikar (1986)",
    "movie3": "Utt Patang (2011)",
}


def _download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:
        print(f"    WARNING: failed to download {url}: {exc}")
        return False


def _mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_path), "-ar", "16000", "-ac", "1", str(wav_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"    WARNING: ffmpeg failed on {mp3_path.name}: {result.stderr.strip()[:200]}")
        return False
    return True


def fetch() -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    annotations_path = CACHE_DIR / "annotations_all_files.csv"
    print("Step 1/2: downloading annotations CSV")
    if not _download(_ANNOTATIONS_URL, annotations_path):
        print("FATAL: could not fetch annotations CSV — aborting")
        sys.exit(1)

    with annotations_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    speech_rows = [r for r in rows if r["Class"] == "speech"]
    print(f"  {len(rows)} total annotated segments, {len(speech_rows)} are pure 'speech' class")

    print("Step 2/2: downloading + converting each 'speech' segment")
    manifest_entries: list[dict] = []
    mp3_scratch = CACHE_DIR / "_mp3_scratch"
    mp3_scratch.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(speech_rows):
        filename = row["Filename"]  # e.g. "movie1_segment47_p0.mp3"
        utt_id = Path(filename).stem
        movie_key = filename.split("_", 1)[0]  # "movie1"
        duration_s = float(row["Duration (in secs)"]) if row["Duration (in secs)"] else None

        mp3_path = mp3_scratch / filename
        url = f"{_REPO_RAW_BASE}/speech/{filename}"
        if not _download(url, mp3_path):
            print(f"  [{i + 1}/{len(speech_rows)}] SKIP {utt_id} (download failed)")
            continue

        wav_path = DATA_DIR / f"{utt_id}.wav"
        ok = _mp3_to_wav(mp3_path, wav_path)
        mp3_path.unlink(missing_ok=True)
        if not ok:
            continue

        manifest_entries.append(
            {
                "utterance_id": utt_id,
                "language": "hi",
                "text": "",  # no transcript provided by this dataset — class-labeled audio only
                "source_dataset": "mrinmoy-iitg/Movie-MUSNOMIX (speech class)",
                "license": _LICENSE,
                "movie": _MOVIES.get(movie_key, movie_key),
                "duration_s": duration_s,
                "wav_path": str(wav_path.relative_to(DATA_DIR.parent.parent.parent)),
                "sample_rate": 16000,
            }
        )
        print(f"  [{i + 1}/{len(speech_rows)}] saved {utt_id} ({_MOVIES.get(movie_key, movie_key)}, {duration_s:.1f}s)")

    mp3_scratch.rmdir()
    return manifest_entries


def main() -> None:
    entries = fetch()
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(entries)} Movie-MUSNOMIX Hindi speech entries to {manifest_path}")
    if not entries:
        print("WARNING: zero entries fetched — check the errors above before relying on this corpus.")
        sys.exit(1)


if __name__ == "__main__":
    main()
