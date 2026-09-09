#!/usr/bin/env python3
"""Fetches Hindi genuine speech from AI4Bharat's Kathbath corpus — see
README.md's "Datasets" table for the license verification behind this
choice.

REAL LICENSE NOTE (verified 2026-09-09, not assumed): `ai4bharat/Kathbath`
is CC BY 4.0 per its own HF dataset-card metadata (`license: cc-by-4.0`)
but is `gated: auto` on HuggingFace itself — same situation this project
has already hit with Mann-ki-Baat and Spoken-Tutorial, and the same rule
applies: gating is a hard blocker regardless of the underlying license,
UNLESS an ungated alternate distribution channel exists. One does here:
AI4Bharat's own `IndicSUPERB` GitHub repo (github.com/AI4Bharat/
indicSUPERB, Apache-2.0 *code* license — the dataset itself is the
CC BY 4.0 Kathbath data described above, hosted separately) links direct,
no-auth downloads of the exact same Kathbath audio via E2E Networks
object storage. That's the channel this script uses.

The upstream tar files bundle ALL 12 languages together (folder-per-
language inside one archive) — there is no separate "Hindi-only" tar. To
avoid a 3GB+ download turning into a 12x-oversized local cache, this
script streams the tar member-by-member and only extracts+converts the
`data/hindi/` entries, discarding the other 11 languages' bytes without
ever writing them to disk.

Uses the "clean/valid" split specifically (not "train", which is 85GB and
not needed here): real crowd-sourced read speech from a genuinely
different recording batch than IndicTTS-Hindi (studio TTS voice-talent
recordings) — a useful complement, not a duplicate of what
fetch_genuine_corpus.py already pulls. Audio ships as m4a; each file is
converted to wav via a real ffmpeg subprocess call (ffmpeg is already a
hard dependency elsewhere in this project — see
tests/unit/test_degraded_audio_robustness.py).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "genuine" / "hi_kathbath"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache" / "kathbath"

_VALID_AUDIO_URL = "https://objectstore.e2enetworks.net/indic-superb/kathbath/clean/valid_audio.tar"
_TRANSCRIPTS_URL = "https://objectstore.e2enetworks.net/indic-superb/kathbath/clean/transcripts_n2w.tar"
_LICENSE = "CC BY 4.0 (ai4bharat/Kathbath — gated on HF; fetched via the ungated IndicSUPERB/E2E-Networks mirror, see this file's module docstring)"

# Bounds the local corpus to a real, reproducible, disk-friendly size —
# same "not a token gesture, but not the whole 150 hours either" spirit
# as fetch_genuine_corpus.py's N_HINDI_GENUINE.
MAX_HINDI_UTTERANCES = 500


def _download_with_progress(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already downloaded: {dest} ({dest.stat().st_size / 1e6:.0f} MB) — skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    # REAL BUG found and fixed 2026-09-09: urlretrieve's reporthook fires
    # once per network block (default ~8KB), so an unconditional print
    # here means ~450,000 lines for a 3.6GB file — even with `\r`, that's
    # fine on an interactive terminal but explodes into a hundreds-of-MB
    # log the moment stdout is redirected to a file (no real terminal to
    # collapse the carriage returns), and the sheer volume of print()
    # calls measurably slows the download itself. Gated to print at most
    # once per ~1% of progress (or every 5000 blocks when size is
    # unknown) instead.
    _last_pct_reported = [-1.0]

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        done = block_num * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * done / total_size)
            if pct - _last_pct_reported[0] >= 1.0 or pct >= 100.0:
                _last_pct_reported[0] = pct
                print(f"    {done / 1e6:8.1f} MB / {total_size / 1e6:.1f} MB ({pct:5.1f}%)", flush=True)
        elif block_num % 5000 == 0:
            print(f"    {done / 1e6:8.1f} MB (total size unknown)", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=_report)
    tmp.rename(dest)
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def _load_hindi_transcripts(transcripts_tar_path: Path) -> dict[str, str]:
    print(f"  reading Hindi transcripts from {transcripts_tar_path.name}...")
    transcripts: dict[str, str] = {}
    with tarfile.open(transcripts_tar_path, "r") as tar:
        for member in tar:
            if "hindi" not in member.name or not member.name.endswith(".txt"):
                continue
            print(f"    reading {member.name}")
            fh = tar.extractfile(member)
            if fh is None:
                continue
            for line in fh.read().decode("utf-8", errors="replace").splitlines():
                # REAL BUG found and fixed 2026-09-09: each line is
                # "<utt_id>.m4a\t<text>" — the key here MUST keep the
                # ".m4a" suffix (or the audio side must add it back), or
                # every single lookup below silently misses. Verified by
                # reading the raw tar member directly: transcript keys
                # come out as "844424930703439-329-f.m4a", but the audio
                # side's own utt_id is the extension-stripped stem — the
                # two must be reconciled on ONE convention, not two.
                parts = line.strip().split("\t", maxsplit=1)
                if len(parts) == 2:
                    utt_id_with_ext, text = parts
                    utt_id = Path(utt_id_with_ext).stem  # strip ".m4a" to match the audio-side utt_id
                    transcripts[utt_id] = text
    print(f"  loaded {len(transcripts)} Hindi transcript entries")
    return transcripts


def _m4a_to_wav(m4a_path: Path, wav_path: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(m4a_path), "-ar", "16000", "-ac", "1", str(wav_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"    WARNING: ffmpeg failed on {m4a_path.name}: {result.stderr.strip()[:200]}")
        return False
    return True


def fetch() -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    audio_tar_path = CACHE_DIR / "valid_audio.tar"
    transcripts_tar_path = CACHE_DIR / "transcripts_n2w.tar"

    print("Step 1/3: downloading clean/valid audio tar (~3GB, all 12 languages bundled)")
    _download_with_progress(_VALID_AUDIO_URL, audio_tar_path)

    print("Step 2/3: downloading clean transcripts tar")
    _download_with_progress(_TRANSCRIPTS_URL, transcripts_tar_path)
    transcripts = _load_hindi_transcripts(transcripts_tar_path)

    print("Step 3/3: extracting Hindi-only audio members, converting m4a->wav")
    manifest_entries: list[dict] = []
    m4a_scratch = CACHE_DIR / "_m4a_scratch"
    m4a_scratch.mkdir(parents=True, exist_ok=True)

    with tarfile.open(audio_tar_path, "r") as tar:
        hindi_members = [m for m in tar if "/hindi/" in m.name and m.name.endswith(".m4a")]
        print(f"  found {len(hindi_members)} Hindi audio members in the tar (12-language bundle)")
        if len(hindi_members) > MAX_HINDI_UTTERANCES:
            print(f"  capping to first {MAX_HINDI_UTTERANCES} (of {len(hindi_members)}) for a bounded, reproducible corpus")
        hindi_members = hindi_members[:MAX_HINDI_UTTERANCES]

        for i, member in enumerate(hindi_members):
            utt_id = Path(member.name).stem
            fh = tar.extractfile(member)
            if fh is None:
                print(f"  [{i + 1}/{len(hindi_members)}] SKIP {utt_id} (empty tar member)")
                continue
            m4a_path = m4a_scratch / f"{utt_id}.m4a"
            m4a_path.write_bytes(fh.read())

            wav_path = DATA_DIR / f"{utt_id}.wav"
            ok = _m4a_to_wav(m4a_path, wav_path)
            m4a_path.unlink(missing_ok=True)
            if not ok:
                continue

            text = transcripts.get(utt_id, "")
            manifest_entries.append(
                {
                    "utterance_id": utt_id,
                    "language": "hi",
                    "text": text,
                    "source_dataset": "ai4bharat/Kathbath (clean/valid split, via IndicSUPERB/E2E-Networks)",
                    "license": _LICENSE,
                    "wav_path": str(wav_path.relative_to(DATA_DIR.parent.parent.parent)),
                    "sample_rate": 16000,
                }
            )
            print(f"  [{i + 1}/{len(hindi_members)}] saved {utt_id}" + (" (no transcript found)" if not text else ""))

    m4a_scratch.rmdir()
    return manifest_entries


def main() -> None:
    entries = fetch()
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(entries)} Kathbath Hindi entries to {manifest_path}")
    if not entries:
        print("WARNING: zero entries fetched — check the errors above before relying on this corpus.")
        sys.exit(1)


if __name__ == "__main__":
    main()
