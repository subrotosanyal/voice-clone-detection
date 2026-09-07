"""Slices a mono float32 audio buffer into overlapping AudioWindows.

Matches the blueprint's §02 stage 2: fixed window length, fixed hop, so the
score has visible momentum instead of jumping between unrelated frames.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from app.domain.models import AudioWindow


def make_windows(
    session_id: str,
    samples: np.ndarray,
    sample_rate: int,
    window_ms: int,
    hop_ms: int,
    start_seq: int = 0,
) -> Iterator[AudioWindow]:
    """Yield overlapping windows over `samples`.

    The final partial window (shorter than window_ms) is still yielded —
    detectors are expected to handle short windows gracefully (most will
    just have less signal to work with, or abstain).
    """
    window_len = int(sample_rate * window_ms / 1000)
    hop_len = int(sample_rate * hop_ms / 1000)
    if window_len <= 0 or hop_len <= 0:
        raise ValueError("window_ms and hop_ms must both be positive")

    total = samples.shape[0]
    seq = start_seq
    start = 0
    if total == 0:
        return
    while start < total:
        end = min(start + window_len, total)
        chunk = samples[start:end].astype(np.float32, copy=False)
        window_start_ms = int(start / sample_rate * 1000)
        yield AudioWindow(
            session_id=session_id,
            seq=seq,
            sample_rate=sample_rate,
            samples=chunk,
            window_start_ms=window_start_ms,
        )
        seq += 1
        start += hop_len
