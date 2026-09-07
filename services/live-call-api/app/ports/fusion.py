"""The Fusion port.

The strategy that combines a window's DetectorResults into one FusedScore.
Swapping the risk formula — different weights, a different smoothing
method, a learned model instead of a weighted sum — means writing a new
class against this Protocol and pointing `fusion.class` at it in
config/risk_formula.yaml. Nothing in the API layer or the detectors needs
to change.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

from app.domain.models import DetectorResult, FusedScore


class FusionPort(Protocol):
    """Combines detector outputs into one explainable, banded score."""

    version: str

    def fuse(
        self,
        session_id: str,
        seq: int,
        window_start_ms: int,
        results: list[DetectorResult],
        previous_smoothed_score: Optional[float],
        config: dict[str, Any],
    ) -> FusedScore:
        """Produce a FusedScore for one window.

        `previous_smoothed_score` is this session's last smoothed score (or
        None for the first window) — how it's used (EMA, plain average, or
        ignored) is entirely up to the fusion implementation, but it must be
        the ONLY piece of cross-window state a fusion strategy is given, so
        that reproducing a score never depends on hidden mutable state.
        """
        ...
