"""Core domain types for the Satya-Vani live-call detection pipeline.

These are plain dataclasses / enums with no framework dependency (no FastAPI,
no pydantic) so that ports and adapters never need to import the web layer.
See docs/architecture.md for how this fits the ports-and-adapters layering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np


class Band(str, Enum):
    """Risk bands, matching the thresholds in config/risk_formula.yaml."""

    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


@dataclass(frozen=True)
class AudioWindow:
    """One overlapping window of a call's audio, ready to be scored.

    `samples` is mono float32 PCM in [-1, 1]. `window_start_ms` is the offset
    of this window's first sample from the start of the session's audio.
    """

    session_id: str
    seq: int
    sample_rate: int
    samples: np.ndarray
    window_start_ms: int


@dataclass(frozen=True)
class DetectorResult:
    """What one pluggable detector returns for one window.

    `score` is None when the detector abstains (e.g. the consistency
    detector with no enrolled voiceprint on file). `detail` carries the raw
    feature values that produced the score, so every number in a fused
    result can be traced back to something concrete and re-computed by
    hand — this is what makes a score "explainable" rather than a black box.
    """

    detector_name: str
    detector_version: str
    score: Optional[float]  # 0.0-1.0, or None if abstained
    detail: dict[str, Any] = field(default_factory=dict)
    abstain_reason: Optional[str] = None


@dataclass(frozen=True)
class ComponentContribution:
    """One detector's accounted-for contribution to a fused score."""

    name: str
    raw_score: Optional[float]
    weight_configured: float
    weight_effective: float  # after renormalising for abstained components
    contribution: float  # weight_effective * raw_score, or 0 if abstained
    abstained: bool
    detail: dict[str, Any]


@dataclass(frozen=True)
class FusedScore:
    """The full, explainable output of one window's fusion step.

    Every field here is either a direct input or something reproducibly
    derived from those inputs plus the config in risk_formula.yaml — nothing
    here is computed off any state that isn't itself in `components` or
    `formula_version`.
    """

    session_id: str
    seq: int
    window_start_ms: int
    raw_score_0_100: float  # this window alone, before smoothing
    smoothed_score_0_100: float  # after EMA across the session
    band: Band
    formula_version: str
    third_signal_mode: str
    components: list[ComponentContribution]
    recommended_action: str
