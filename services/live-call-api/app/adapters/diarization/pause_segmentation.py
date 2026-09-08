"""Pause-aware speech segmentation — cuts segment boundaries at detected
silences instead of a fixed clock, so a segment doesn't straddle a real
speaker change mid-window.

WHY THIS MATTERS — measured, not assumed: a segment that spans a real
speaker change produces an ECAPA-TDNN embedding that resembles NEITHER
speaker (cosine similarity 0.04-0.34 against either pure voice, verified
by hand with synthetic two-speaker splices — far below even the normal
cross-speaker baseline of ~0.67-0.70), not a "blend" of the two as might
be expected. Under clustering, that reliably becomes an isolated,
spurious cluster — a phantom extra "speaker" — the same over-counting
failure class embedding_cluster_diarizer.py's CALIBRATION HISTORY already
documents, but via a completely different mechanism (temporal straddling,
not natural single-speaker variation) that survives that earlier fix
untouched.

HOW: Praat's own silence-detection (`To Intensity` + `To TextGrid
(silences)`) finds real acoustic pauses — no new dependency, Parselmouth
is already a hard requirement in this project (see
app/adapters/detectors/prosody_parselmouth.py).

HONESTY NOTE — this catches MOST real speaker turns, not all. Verified by
hand on synthetic two-voice clips: a 150ms gap between speakers IS
detected as a boundary; a 50ms micro-gap or a genuine zero-gap (immediate
back-to-back turn-taking, or overlapping speech) is NOT — there is no
acoustic silence for any pause-based method to find in those cases, no
matter how this is tuned. `max_segment_ms` bounds (does not eliminate) the
damage: any uninterrupted "sounding" stretch longer than that is
sub-sliced at a fixed length anyway, same as the old behaviour, so a
missed zero-gap turn inside a long stretch can corrupt at most one
max_segment_ms-sized chunk, not the whole stretch.
"""
from __future__ import annotations

import numpy as np

from app.adapters.praat_lock import PRAAT_LOCK


def detect_speech_segments(
    samples: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -25.0,
    min_silence_s: float = 0.1,
    min_sounding_s: float = 0.1,
    min_segment_ms: int = 500,
    max_segment_ms: int = 4000,
) -> list[tuple[int, int]]:
    """Returns (start_ms, end_ms) pairs for real speech regions,
    boundary-aligned to detected pauses rather than a fixed clock.

    Any region longer than max_segment_ms is sub-sliced at that length —
    a long uninterrupted stretch may still contain an undetectable
    zero-gap turn change; see this module's HONESTY NOTE. Regions shorter
    than min_segment_ms are dropped — too little audio for a reliable
    embedding, same reasoning as embedding_cluster_diarizer.py's own
    trailing-sliver check.

    Falls back to treating the whole buffer as one region (still
    sub-sliced below like any other) if Praat's silence detection itself
    raises — a real, if rare, possibility on unusual input (e.g. a clip
    with almost no loudness variation at all), same graceful-degradation
    spirit as prosody_parselmouth.py's own abstain-on-failure handling.
    """
    if samples.size == 0:
        return []

    import parselmouth
    from parselmouth.praat import call

    duration_s = samples.size / sample_rate

    with PRAAT_LOCK:
        sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sample_rate)
        try:
            intensity = call(sound, "To Intensity", 100, 0.0, True)
            textgrid = call(
                intensity,
                "To TextGrid (silences)",
                silence_threshold_db,
                min_silence_s,
                min_sounding_s,
                "silent",
                "sounding",
            )
            n_intervals = call(textgrid, "Get number of intervals", 1)
            intervals = [
                (
                    call(textgrid, "Get start time of interval", 1, i),
                    call(textgrid, "Get end time of interval", 1, i),
                    call(textgrid, "Get label of interval", 1, i),
                )
                for i in range(1, n_intervals + 1)
            ]
        except Exception:  # noqa: BLE001 — Praat can raise on unusual input; degrade, don't crash
            intervals = [(0.0, duration_s, "sounding")]

    segments: list[tuple[int, int]] = []
    for start_s, end_s, label in intervals:
        if label != "sounding":
            continue
        start_ms, end_ms = int(round(start_s * 1000)), int(round(end_s * 1000))
        # Sub-slice long stretches so an undetectable (zero-gap) turn
        # change inside one doesn't corrupt an arbitrarily long window.
        cursor = start_ms
        while cursor < end_ms:
            chunk_end = min(cursor + max_segment_ms, end_ms)
            if chunk_end - cursor >= min_segment_ms:
                segments.append((cursor, chunk_end))
            cursor = chunk_end

    return segments
