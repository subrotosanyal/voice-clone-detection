"""Shared, process-wide lock around Praat calls.

Praat's C++ core predates any concept of being called from multiple
threads at once (it was originally a single-threaded desktop app), and
Parselmouth doesn't document it as thread-safe. Every caller into Praat —
currently prosody_parselmouth.py (jitter/shimmer/HNR) and
app/adapters/diarization/pause_segmentation.py (silence detection) — goes
through app/api/http_router.py's run_in_threadpool, so concurrent calls
into Praat are a real possibility whenever a scoring request and a
diarization request overlap. This lock must be the SAME object across
every caller — two separate per-module locks would not actually serialise
anything against each other. Cheap to hold: one Praat call here is tens of
milliseconds.
"""
from __future__ import annotations

import threading

PRAAT_LOCK = threading.Lock()
