#!/usr/bin/env python3
"""Measures live-mic transcription accuracy (word error rate) against
labeled reference audio — and, for comparison, the file-upload path's
accuracy on the SAME audio — using the REAL production transcriber
objects and the REAL live-path windowing/cadence logic, not a
reimplementation of either.

WHY THIS EXISTS: every "honesty note" this project has documented about
live transcription (see docs/architecture.md, "Live transcription", and
app/pipeline/live_transcription.py's own docstring) is about LATENCY —
how long a transcript takes to arrive. Nothing anywhere in this codebase
actually measures transcription ACCURACY (word error rate) on the live
path specifically, or how it compares to the file-upload path on the
same audio. Before tuning decode parameters, VAD, model size, or
routing further, this answers a more basic question: where is accuracy
actually being lost, if anywhere — language-ID misroutes, chunking
cutting words at window boundaries, the "last moment of speech" gap
whisperlive_transcriber.py's own docstring documents, or something else?

NO DATASET SHIPS WITH THIS SCRIPT. Every audio corpus already in this
repo (eval/indian_language/data/) is for spoof-detection (genuine vs.
synthetic voice), not labeled with reference transcripts — using it here
would need transcribing it first, which is exactly what this script is
trying to independently measure the accuracy of. You need your own
labeled (audio, reference transcript) pairs — see MANIFEST FORMAT below.

MANIFEST FORMAT — a JSON file, a list of objects:
    [
      {"audio_path": "samples/call1.wav", "reference_text": "hello this is a test call"},
      {"audio_path": "samples/call2.wav", "reference_text": "..."}
    ]
Paths are resolved relative to the manifest file's own directory unless
absolute. `reference_text` is the ground-truth transcript — normalized
(lowercased, punctuation stripped) the same way as every hypothesis
before comparing, so exact formatting doesn't matter.

WHAT IT DOES, per manifest entry:
  1. Loads the audio at its native sample rate (no resampling here — the
     real transcriber adapters already handle that internally, same as
     they would for a real call).
  2. FILE-UPLOAD path: calls the configured `transcription:` transcriber
     once on the whole buffer — exactly what POST /v1/score/file does
     (see app/pipeline/engine.py's score_call()).
  3. LIVE path: drives the REAL app/pipeline/live_transcription.py
     LiveTranscriptionBuffer — not a reimplementation — feeding it
     windows shaped exactly like app/ui/app.js's mic capture does
     (WINDOW_MS/HOP_MS below, matching that file's own constants), using
     the configured `live_transcription:` transcriber wrapped through
     `.create_session()` when available (same as app/api/ws_router.py
     does for a real WS session) so the language-ID caching and
     persistent indic connection are exercised too, not bypassed.
     Reads back LiveTranscriptionBuffer's own `transcript` field — the
     latest cycle's rolling text, the exact thing a real call's
     detectors (contextual keywords, intent, semantic risk) actually
     score against. Deliberately NOT `full_session_transcript` — that
     field is display-only and its own docstring documents a real
     overlap-duplication caveat (repeats a phrase that straddled two
     cycles' boundaries); using it here inflated measured WER on real
     clips with visibly duplicated text (found and fixed 2026-09-11).
  4. Computes word error rate (Levenshtein edit distance over words,
     the standard ASR metric — no external dependency; this project has
     no jiwer/similar already installed) for both paths against the same
     reference, and reports them side by side.

USAGE (from services/live-call-api/, same cwd convention as `pytest`):
    .venv/bin/python eval/wer_eval.py --manifest path/to/manifest.json
    .venv/bin/python eval/wer_eval.py --manifest path/to/manifest.json --config config/risk_formula.yaml

CAVEAT: this measures whatever `config/risk_formula.yaml` currently
points `transcription:`/`live_transcription:` at (RoutingTranscriber,
WhisperLiveTranscriber, etc., each possibly hitting a real vexyl-stt/
whisper-live sidecar over the network) — this script does not fake or
mock either transcriber. Sidecars must actually be running (`docker
compose up whisper-live vexyl-stt`) for a realistic live-path number;
if they're unreachable, the configured transcribers degrade to their
own "abstain, don't crash" behavior (see each adapter's own docstring),
which will show up here as inflated WER, not a script bug.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.registry import build_pipeline  # noqa: E402
from app.pipeline.live_transcription import LiveTranscriptionBuffer  # noqa: E402

# Mirrors app/ui/app.js's own WINDOW_MS/HOP_MS constants (that file's own
# comment: "mirrors config/risk_formula.yaml defaults") — kept in sync by
# hand, same as that file already is; not read from config here since
# config's `windowing:` section governs the SCORING windows, a separate
# (if numerically identical, by convention) concern from what the browser
# actually sends.
WINDOW_MS = 2000
HOP_MS = 500


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — applied to
    both reference and hypothesis before comparing, so formatting
    differences (capitalization, a trailing period) don't inflate WER
    for reasons that have nothing to do with transcription accuracy."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard ASR word error rate: word-level Levenshtein edit distance
    divided by the reference word count. 0.0 = perfect match; can exceed
    1.0 if the hypothesis is much longer than the reference (heavily
    over-generating words counts against it, same as any real WER)."""
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    prev_row = list(range(len(hyp_words) + 1))
    for i, rw in enumerate(ref_words, start=1):
        curr_row = [i] + [0] * len(hyp_words)
        for j, hw in enumerate(hyp_words, start=1):
            if rw == hw:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(prev_row[j], curr_row[j - 1], prev_row[j - 1])
        prev_row = curr_row

    return prev_row[-1] / len(ref_words)


@dataclass
class ManifestEntry:
    audio_path: Path
    reference_text: str


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.resolve().parent
    entries = []
    for row in raw:
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = base_dir / audio_path
        entries.append(ManifestEntry(audio_path=audio_path, reference_text=row["reference_text"]))
    return entries


def transcribe_file_upload(transcriber: Any, samples: np.ndarray, sample_rate: int) -> str:
    if transcriber is None:
        return ""
    return transcriber.transcribe(samples, sample_rate).text


async def simulate_live_path(transcriber: Any, samples: np.ndarray, sample_rate: int) -> str:
    """Drives the REAL LiveTranscriptionBuffer with windows shaped like
    app/ui/app.js's mic capture (overlapping WINDOW_MS windows, sent
    every HOP_MS of new audio) — the exact same object app/api/
    ws_router.py uses for a real WebSocket session, not a stand-in.
    `window_start_ms` is fed exactly as the browser computes it
    (elapsed - WINDOW_MS), which is what lets LiveTranscriptionBuffer's
    own real hop-mismatch-safe elapsed-time derivation run for real
    here too, not the fallback fixed-hop path unit tests exercise."""
    if transcriber is None:
        return ""

    # Same session-scoping app/api/ws_router.py applies for a real WS
    # connection — see routing_transcriber.py's RoutingTranscriber.
    # create_session() for why this matters (caches the language
    # decision once, holds one persistent indic connection).
    create_session = getattr(transcriber, "create_session", None)
    session_transcriber = create_session() if create_session is not None else transcriber

    buffer = LiveTranscriptionBuffer(transcriber=session_transcriber, intent_classifier=None, hop_ms=HOP_MS)
    hop_samples = round(sample_rate * HOP_MS / 1000)
    window_samples = round(sample_rate * WINDOW_MS / 1000)

    pos = 0
    elapsed_ms = 0
    total = len(samples)
    while pos < total:
        pos = min(pos + hop_samples, total)
        elapsed_ms += HOP_MS
        window_start = max(0, pos - window_samples)
        window = samples[window_start:pos]
        buffer.ingest(window, sample_rate, max(0, elapsed_ms - WINDOW_MS))
        await asyncio.sleep(0)  # let any scheduled transcription task run

    # Same flush app/api/ws_router.py now runs when a real WS session
    # ends (added 2026-09-11) — without it, any call shorter than
    # LiveTranscriptionBuffer's own _TRANSCRIBE_EVERY_MS (4s), or any
    # trailing audio after the last completed cycle, would never be
    # transcribed at all. This IS the real production code path, not a
    # simulation-only step.
    await buffer.flush()

    session_close = getattr(session_transcriber, "close", None)
    if session_close is not None:
        session_close()

    # REAL BUG found and fixed 2026-09-11: this used to prefer
    # `full_session_transcript` — LiveTranscriptionBuffer's own naive
    # whole-session concatenation, which its own docstring already
    # documents as duplicative across overlapping cycle boundaries (never
    # read by a detector for exactly that reason). Using it here inflated
    # measured WER on several real clips with visibly repeated phrases,
    # e.g. "...टेक्नोलॉजी का सेल्फी फोकस फीचर्स होंगे... टेक्नोलॉजी का
    # इस्तेमाल करेंगे" instead of the clean text. `transcript` — the
    # latest cycle's own rolling text, what ContextualRulesDetector/
    # intent/semantic-risk actually score against on a real call — is the
    # right field to measure accuracy against.
    return buffer.latest_context.get("transcript", "")


async def run(manifest_path: Path, config_path: Path) -> int:
    entries = load_manifest(manifest_path)
    if not entries:
        print(f"No entries in {manifest_path} — nothing to evaluate.")
        return 1

    pipeline = build_pipeline(config_path)

    file_upload_wers: list[float] = []
    live_wers: list[float] = []

    header = f"{'file':40s}  {'file-upload WER':>16s}  {'live-path WER':>14s}"
    print(header)
    print("-" * len(header))

    for entry in entries:
        samples, sample_rate = sf.read(str(entry.audio_path), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)  # downmix to mono, same assumption every detector here makes

        file_hyp = transcribe_file_upload(pipeline.transcriber, samples, sample_rate)
        live_hyp = await simulate_live_path(pipeline.live_transcriber or pipeline.transcriber, samples, sample_rate)

        file_wer = word_error_rate(entry.reference_text, file_hyp)
        live_wer = word_error_rate(entry.reference_text, live_hyp)
        file_upload_wers.append(file_wer)
        live_wers.append(live_wer)

        print(f"{entry.audio_path.name:40s}  {file_wer:16.2%}  {live_wer:14.2%}")

    print("-" * len(header))
    avg_file = sum(file_upload_wers) / len(file_upload_wers)
    avg_live = sum(live_wers) / len(live_wers)
    print(f"{'AVERAGE':40s}  {avg_file:16.2%}  {avg_live:14.2%}")

    if avg_live > avg_file:
        gap = avg_live - avg_file
        print(
            f"\nLive-path WER is {gap:.1%} worse than file-upload on this set — the live-specific "
            "factors (chunking, the last-moment-of-speech gap, language-ID timing) are a real, "
            "measured cost here, not just a theoretical one."
        )
    else:
        print("\nLive-path WER is not worse than file-upload on this set.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="Path to a JSON manifest (see this script's own docstring for the format).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "risk_formula.yaml",
        help="Path to risk_formula.yaml (default: this repo's own config/risk_formula.yaml).",
    )
    args = parser.parse_args()
    return asyncio.run(run(args.manifest, args.config))


if __name__ == "__main__":
    raise SystemExit(main())
