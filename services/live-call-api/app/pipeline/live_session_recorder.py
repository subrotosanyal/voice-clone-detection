"""LiveSessionRecorder — reconstructs a live WebSocket session's whole,
non-overlapping audio so it can be diarized and scored once the call ends.

WHY THIS EXISTS: full per-speaker diarization (app/adapters/diarization/
embedding_cluster_diarizer.py) needs the WHOLE recording — agglomerative
clustering can't run incrementally on partial audio the way
app/pipeline/live_diarization.py's simpler nearest-centroid tracker does.
A live call has no "whole recording" until it ends — see
app/api/ws_router.py's own docstring for how this is used ("diarize on
hangup": once the WebSocket closes, this buffer is diarized and each
speaker rescored through Engine.diarize_and_score(), the exact same code
path POST /v1/score/file?diarize=true already uses).

HOW: incoming windows OVERLAP each other by construction (2s window, 0.5s
hop by default) — appending every window's full samples would duplicate
audio ~4x over. Each call to ingest() appends only the TRAILING new-audio
slice, using the same window_start_ms-delta approach as
app/pipeline/live_transcription.py's REAL BUG fix (2026-09-10) — derived
from data every chunk already carries, not assumed from config's hop_ms.

BOUNDED, honestly: `max_duration_ms` caps memory on a pathologically long
call by dropping the OLDEST audio once the cap is exceeded. A speaker who
ONLY spoke in the dropped portion will not appear in the final diarized
summary — a real, documented trade-off (this project's `_MAX_BUFFER_MS`
in live_transcription.py makes the same kind of bounded-memory trade-off
for a different purpose), not a silent one. Default cap is generous
(10 minutes) for the demo/eval call lengths this project targets.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_DEFAULT_MAX_DURATION_MS = 10 * 60 * 1000
# Same reasoning as live_transcription.py's _MAX_PLAUSIBLE_GAP_MULTIPLE —
# a client hiccup shouldn't be able to report an implausible single jump.
_MAX_PLAUSIBLE_GAP_MULTIPLE = 10


class LiveSessionRecorder:
    """One instance per WebSocket session — same "held by the caller, one
    per connection" shape as LiveTranscriptionBuffer/LiveSpeakerTracker."""

    def __init__(self, hop_ms: int, max_duration_ms: int = _DEFAULT_MAX_DURATION_MS) -> None:
        self._hop_ms = hop_ms
        self._max_duration_ms = max_duration_ms
        self._chunks: list[np.ndarray] = []
        self._duration_ms = 0
        self._last_window_start_ms: Optional[int] = None
        self.sample_rate: Optional[int] = None

    def ingest(self, samples: np.ndarray, sample_rate: int, window_start_ms: Optional[int] = None) -> None:
        """Call once per incoming window, same call site as
        LiveTranscriptionBuffer.ingest() and LiveSpeakerTracker.ingest()
        in app/api/ws_router.py.

        REAL BUG found and fixed while writing the integration test for
        this class: on the very FIRST call, falling back to `hop_ms` (the
        same fallback LiveTranscriptionBuffer's rolling buffer uses, where
        losing (window_ms - hop_ms) of the call's absolute start is an
        accepted, minor trade-off for a rolling window) would have kept
        only the TRAILING hop_ms of that first window here too — but this
        class's whole point is reconstructing the FULL session for
        diarization, so silently dropping the rest of window 1 defeats
        it. Fixed: with no prior window to diff against, the first call
        keeps the WHOLE window's samples, not just a hop-sized slice.
        """
        self.sample_rate = sample_rate  # assumed constant for one session, same assumption AudioWindow itself makes

        if self._last_window_start_ms is None:
            # First call ever: nothing to diff against, so the whole
            # window is new — see this method's own REAL BUG note above.
            elapsed_ms = round(samples.size / sample_rate * 1000)
        elif window_start_ms is not None:
            delta = window_start_ms - self._last_window_start_ms
            elapsed_ms = max(0, min(delta, self._hop_ms * _MAX_PLAUSIBLE_GAP_MULTIPLE))
        else:
            elapsed_ms = self._hop_ms
        if window_start_ms is not None:
            self._last_window_start_ms = window_start_ms

        if elapsed_ms <= 0:
            return

        new_samples = round(sample_rate * elapsed_ms / 1000)
        new_tail = samples[-new_samples:] if samples.size > new_samples else samples
        if new_tail.size == 0:
            return

        self._chunks.append(new_tail)
        self._duration_ms += elapsed_ms

        while self._duration_ms > self._max_duration_ms and len(self._chunks) > 1:
            dropped = self._chunks.pop(0)
            self._duration_ms -= round(dropped.size / sample_rate * 1000)

    def get_full_audio(self) -> Optional[np.ndarray]:
        """The reconstructed, non-overlapping session audio so far, or
        None if nothing was ever ingested."""
        if not self._chunks:
            return None
        return np.concatenate(self._chunks)
