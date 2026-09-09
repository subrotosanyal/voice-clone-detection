#!/usr/bin/env python3
"""Generates matched spoof/clone audio for the 100 local speakers in
`data/local_speakers/train/` — see ../README.md's "Datasets" table and
../TODO.md's "Phase 2" for why this exists: that set is genuine-only (no
spoof counterpart), so it can't be used for AASIST/prosodic training as
committed. This clones each speaker's OWN voice (their own training
clips as reference) via Coqui XTTS-v2, the same engine and licence terms
as `generate_synthetic_corpus.py` (CPML, non-commercial) — see that
script's own docstring for the full licence note, not repeated here.

Confirmed via a real Whisper transcription check (2026-09-09) that this
corpus is Hindi speech, matching the project's Hindi sentence set
(`prompts/sentences.md`) rather than needing a separate English pass.

Reference audio: the LONGEST of each speaker's 5 training clips (more
reference audio gives XTTS-v2 more to work with — same reasoning as
`_corpus_utils.pick_reference_speakers`, just applied per-speaker here
instead of picking a handful of representative speakers).

Scope, honestly: N_SENTENCES_PER_SPEAKER is deliberately small (not the
full 10-sentence set every speaker) — CPU-based XTTS-v2 synthesis is slow
enough (see the per-clip timing this script prints) that 100 speakers x
10 sentences would be a many-hour run. Diversity of SPEAKER identity
matters more than diversity of TEXT CONTENT for spoof-detection training
(acoustic artifacts of the cloning system, not what's being said, are
what the detector needs to learn) — so this favours breadth across all
100 speakers over depth per speaker. Bump N_SENTENCES_PER_SPEAKER and
re-run (it's resumable — already-synthesized files are skipped) if more
per-speaker diversity turns out to matter after a first fine-tuning pass.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

_EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpus_utils import parse_hindi_sentences  # noqa: E402

LOCAL_SPEAKERS_DIR = _EVAL_DIR / "data" / "local_speakers" / "train"
DATA_DIR = _EVAL_DIR / "data" / "synthetic_local_speakers"
SENTENCES_MD = _EVAL_DIR / "prompts" / "sentences.md"

_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
_LICENSE_NOTE = "Coqui XTTS-v2 (CPML — non-commercial, see https://coqui.ai/cpml); voice-cloned from a local speaker's own reference recording"

N_SENTENCES_PER_SPEAKER = 3  # see module docstring's "Scope, honestly" note


def _longest_reference_clip(speaker_dir: Path) -> Path:
    import soundfile as sf

    clips = sorted(speaker_dir.glob("*.wav"))
    durations = []
    for clip in clips:
        info = sf.info(str(clip))
        durations.append((info.duration, clip))
    durations.sort(reverse=True)
    return durations[0][1]


def main() -> None:
    from TTS.api import TTS

    speaker_dirs = sorted(p for p in LOCAL_SPEAKERS_DIR.iterdir() if p.is_dir())
    print(f"Found {len(speaker_dirs)} local speaker folders in {LOCAL_SPEAKERS_DIR}")

    sentences = parse_hindi_sentences(SENTENCES_MD)[:N_SENTENCES_PER_SPEAKER]
    print(f"Using {len(sentences)} Hindi sentences per speaker (of {len(parse_hindi_sentences(SENTENCES_MD))} available)")

    print(f"\nLoading {_MODEL_ID} ...")
    load_start = time.monotonic()
    tts = TTS(_MODEL_ID).to("cpu")
    print(f"  model loaded in {time.monotonic() - load_start:.1f}s")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    manifest_entries: list[dict] = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    already_done = {e["utterance_id"] for e in manifest_entries}
    if already_done:
        print(f"  resuming: {len(already_done)} clips already done in a previous run, will be skipped")

    total_clips = len(speaker_dirs) * len(sentences)
    done_count = len(already_done)
    run_start = time.monotonic()

    for speaker_idx, speaker_dir in enumerate(speaker_dirs):
        speaker_id = speaker_dir.name  # e.g. "Aadiksha-007"
        reference_wav = _longest_reference_clip(speaker_dir)

        for sent_idx, text in enumerate(sentences):
            utterance_id = f"local_spk_{speaker_id}_s{sent_idx:02d}"
            if utterance_id in already_done:
                continue

            wav_path = DATA_DIR / f"{utterance_id}.wav"
            clip_start = time.monotonic()
            print(
                f"  [{done_count + 1}/{total_clips}] speaker {speaker_idx + 1}/{len(speaker_dirs)} "
                f"({speaker_id}, ref={reference_wav.name}): \"{text[:40]}...\"",
                flush=True,
            )
            tts.tts_to_file(
                text=text,
                speaker_wav=str(reference_wav),
                language="hi",
                file_path=str(wav_path),
            )
            elapsed = time.monotonic() - clip_start
            done_count += 1
            avg_s_per_clip = (time.monotonic() - run_start) / max(1, done_count - len(already_done))
            remaining = total_clips - done_count
            print(
                f"    done in {elapsed:.1f}s (running avg {avg_s_per_clip:.1f}s/clip, "
                f"~{remaining * avg_s_per_clip / 60:.0f} min remaining for {remaining} clips)",
                flush=True,
            )

            manifest_entries.append(
                {
                    "utterance_id": utterance_id,
                    "language": "hi",
                    "text": text,
                    "source_dataset": f"coqui/XTTS-v2 (cloned from local speaker {speaker_id})",
                    "license": _LICENSE_NOTE,
                    "speaker_id": speaker_id,
                    "reference_wav": str(reference_wav.relative_to(_EVAL_DIR)),
                    "wav_path": str(wav_path.relative_to(_EVAL_DIR)),
                    "sample_rate": 24000,  # XTTS-v2's native output rate
                }
            )
            # Write after every clip, not just at the end — a multi-hour
            # run that gets interrupted (or is checked mid-run) should
            # never lose already-synthesized work.
            manifest_path.write_text(json.dumps(manifest_entries, ensure_ascii=False, indent=2))

    print(f"\nWrote {len(manifest_entries)} total synthetic entries to {manifest_path}")


if __name__ == "__main__":
    main()
