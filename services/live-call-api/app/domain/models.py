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
    """One detector's accounted-for contribution to a fused score.

    `raw_score` is THIS WINDOW alone — None if this specific window
    abstained. `smoothed_score` (added 2026-09-10) is the per-component
    EMA aggregate CARRIED FORWARD across abstains — the actual number
    `weight_effective`/`contribution` are computed from, i.e. what really
    fed the fusion formula. A component can show `abstained=True` (no
    fresh reading this window) while `smoothed_score` is still not None
    (it has history from earlier windows this session) — see
    WeightedSumFusion's own docstring for why this split exists.
    """

    name: str
    raw_score: Optional[float]
    smoothed_score: Optional[float]
    weight_configured: float
    weight_effective: float  # after renormalising for components with no usable value at all
    contribution: float  # weight_effective * smoothed_score, or 0 if never scored this session
    abstained: bool
    detail: dict[str, Any]


@dataclass(frozen=True)
class LiveSpeakerInfo:
    """Live, incremental "who's talking now" — WebSocket path only, built
    up one window at a time as the call progresses. NOT the same thing as
    SpeakerSegment below (that partitions a COMPLETE recording after the
    fact, file-upload only — see app/adapters/diarization/
    embedding_cluster_diarizer.py). See app/pipeline/live_diarization.py
    for how this is produced, including the honest calibration/scope caveats.

    NOT persisted: app/adapters/history/sqlite_store.py's save() doesn't
    reference this field, so replaying a past live session from History
    shows the score breakdown, not who was talking at each point — a
    deliberate scope cut, not an oversight.
    """

    speaker_label: str  # "speaker_1", "speaker_2", ... first-seen order, THIS call only
    is_new_speaker: bool  # True on the window this label was first created
    speaker_count: int  # distinct speakers seen so far in this call


@dataclass(frozen=True)
class FusedScore:
    """The full, explainable output of one window's fusion step.

    Every field here is either a direct input or something reproducibly
    derived from those inputs plus the config in risk_formula.yaml — nothing
    here is computed off any state that isn't itself in `components` or
    `formula_version`. The one exception is `live_speaker` (see its own
    docstring): it doesn't feed the formula at all, purely informational.
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
    live_speaker: Optional[LiveSpeakerInfo] = None


@dataclass(frozen=True)
class SessionSummary:
    """One row in the session history list — enough to pick a session to
    open, without pulling its whole trace."""

    session_id: str
    window_count: int
    started_at: str  # ISO 8601 UTC
    ended_at: str
    final_smoothed_score: float
    final_band: Band


@dataclass(frozen=True)
class EnrollmentSummary:
    """One enrolled voiceprint's metadata — never the raw embedding vector
    itself (that stays inside the enrollment store, only ever compared to,
    never returned over the API)."""

    identity: str
    embedding_model: str
    enrolled_at: str  # ISO 8601 UTC
    updated_at: str


@dataclass(frozen=True)
class SpeakerSegment:
    """One contiguous stretch of audio attributed to one speaker cluster
    by the diarizer. `speaker_label` is a stable-within-one-call id
    ("speaker_1", "speaker_2", ...) ordered by first appearance — it is
    NOT a claimed or verified identity, just "the same voice as before"."""

    speaker_label: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SpeakerCallResult:
    """One diarized speaker's own scoring breakdown for a single call —
    the return shape of Engine.diarize_and_score() (app/pipeline/
    engine.py), shared by both the file-upload path (POST /v1/score/
    file?diarize=true) and the live WebSocket path's "diarize on hangup"
    (see app/api/ws_router.py's own docstring on why a live call gets its
    per-speaker breakdown only once it ends, not window-by-window — see
    SpeakerCallSummary below and LiveSpeakerInfo above for the two OTHER,
    narrower ways a live call's speakers show up before that point)."""

    speaker_label: str
    session_id: str  # derived: "{base_session_id}::{speaker_label}"
    segment_count: int
    total_duration_ms: int
    fused_scores: list[FusedScore]


@dataclass(frozen=True)
class SpeakerCallSummary:
    """The small, persisted part of a SpeakerCallResult — everything
    EXCEPT the fused_scores trace itself, which is already recoverable
    via HistoryStorePort.get_session(session_id) since
    Engine.diarize_and_score() scores each speaker through the normal
    score_call() path (same auto-persist-every-window behaviour as any
    other session). This is the one bit that ISN'T re-derivable after the
    fact (segment_count/total_duration_ms come from the diarizer's own
    segments, not from FusedScore history) — see
    HistoryStorePort.save_speaker_summary()/list_speaker_summaries()."""

    base_session_id: str
    speaker_label: str
    speaker_session_id: str
    segment_count: int
    total_duration_ms: int


@dataclass(frozen=True)
class TranscriptResult:
    """What a transcriber returns for one call's audio — feeds the
    contextual third signal's urgency/financial-request keyword detection
    (see app/adapters/detectors/contextual_rules.py). `language` is the
    transcriber's own detected language code (e.g. "en", "hi", "mr"), not a
    claim from the caller. Empty `text` means no speech was found — not an
    error, just nothing to analyse (silence, non-speech audio)."""

    text: str
    language: Optional[str]
    detector_name: str
    detector_version: str


@dataclass(frozen=True)
class IntentClassificationResult:
    """What an IntentClassifierPort returns for one call's transcript —
    see app/ports/intent_classifier.py and app/adapters/intent/. `label_scores`
    carries every candidate label's own score (not just the winner), so a
    result stays fully explainable: which hypotheses the model considered,
    not only which one it picked. This signal IS wired into the active
    risk formula (config/risk_formula.yaml's `intent` detector, weight
    0.15) on both the file-upload and live WebSocket paths — see
    app/adapters/intent/zero_shot_intent_classifier.py's own honesty note
    for the known calibration caveat that motivated the low weight."""

    text: str
    top_label: str
    top_score: float
    label_scores: dict[str, float]
    detector_name: str
    detector_version: str


@dataclass(frozen=True)
class SemanticRiskAssessment:
    """What a SemanticRiskClassifierPort returns for one call's transcript
    — see app/ports/semantic_risk_classifier.py and
    app/adapters/semantic_risk/. Distinct from IntentClassificationResult
    above (a zero-shot NLI classifier scoring FIXED candidate labels):
    this is a generative LLM reading the transcript and producing a
    richer, more nuanced structured judgment — including
    `isolation_request` ("don't hang up", "don't tell anyone"), a signal
    neither the zero-shot classifier nor the keyword-based contextual
    rules (app/adapters/detectors/contextual_rules.py) currently capture.

    STATUS as of 2026-09-09: wired into the active risk formula
    (config/risk_formula.yaml's `semantic_risk` entry, weight 0.15,
    running alongside — not replacing — `intent`) after a real, measured
    comparison against the existing zero-shot classifier's known false-
    positive problem (see zero_shot_intent_classifier.py's HONESTY NOTE
    and app/adapters/semantic_risk/local_llm_semantic_classifier.py's
    own HONESTY NOTE for the exact evaluation). `reasoning` is a human-readable
    explanation ONLY — never itself trusted as a score override (see the
    HONESTY NOTE in local_llm_semantic_classifier.py for why: unlike
    every number in this dataclass, free-form reasoning text can't be
    traced back to a specific computation the way a raw feature can, and
    letting an LLM's own narrative override other signals would break
    this project's reproducibility guarantee)."""

    text: str
    urgency_level: float  # 0.0-1.0
    financial_solicitation: bool
    authority_claim: bool
    isolation_request: bool
    reasoning: str
    detector_name: str
    detector_version: str
