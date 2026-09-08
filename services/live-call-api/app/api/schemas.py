"""Request/response schemas for the live-call API.

Kept in one file, at the top of the api layer, per the project convention
(see the blueprint's reference-templates note) rather than scattered next
to each route.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.domain.models import Band, EnrollmentSummary, FusedScore, SessionSummary


class CallContext(BaseModel):
    """Session metadata — everything the contextual detector reads.
    All fields optional; omit what you don't have."""

    claimed_identity: Optional[str] = None
    known_number: Optional[bool] = None
    hour_of_day: Optional[int] = Field(default=None, ge=0, le=23)
    is_financial_request: Optional[bool] = None
    urgency_keywords: list[str] = Field(default_factory=list)
    authority_claim: Optional[bool] = None


class ComponentContributionOut(BaseModel):
    name: str
    raw_score: Optional[float]
    weight_configured: float
    weight_effective: float
    contribution: float
    abstained: bool
    detail: dict[str, Any]


class FusedScoreOut(BaseModel):
    session_id: str
    seq: int
    window_start_ms: int
    raw_score_0_100: float
    smoothed_score_0_100: float
    band: Band
    formula_version: str
    third_signal_mode: str
    components: list[ComponentContributionOut]
    recommended_action: str

    @classmethod
    def from_domain(cls, fused: FusedScore) -> "FusedScoreOut":
        return cls(
            session_id=fused.session_id,
            seq=fused.seq,
            window_start_ms=fused.window_start_ms,
            raw_score_0_100=fused.raw_score_0_100,
            smoothed_score_0_100=fused.smoothed_score_0_100,
            band=fused.band,
            formula_version=fused.formula_version,
            third_signal_mode=fused.third_signal_mode,
            components=[
                ComponentContributionOut(
                    name=c.name,
                    raw_score=c.raw_score,
                    weight_configured=c.weight_configured,
                    weight_effective=c.weight_effective,
                    contribution=c.contribution,
                    abstained=c.abstained,
                    detail=c.detail,
                )
                for c in fused.components
            ],
            recommended_action=fused.recommended_action,
        )


class SpeakerScoreOut(BaseModel):
    """One diarized speaker's own scoring breakdown — same shape as the
    whole-call result, just scoped to that speaker's audio only."""

    speaker_label: str
    session_id: str
    segment_count: int
    total_duration_ms: int
    final: FusedScoreOut
    trace: list[FusedScoreOut]


class ScoreFileResponse(BaseModel):
    session_id: str
    window_count: int
    final: FusedScoreOut
    trace: list[FusedScoreOut]
    speakers: Optional[list[SpeakerScoreOut]] = None


class SessionSummaryOut(BaseModel):
    session_id: str
    window_count: int
    started_at: str
    ended_at: str
    final_smoothed_score: float
    final_band: Band

    @classmethod
    def from_domain(cls, s: SessionSummary) -> "SessionSummaryOut":
        return cls(
            session_id=s.session_id,
            window_count=s.window_count,
            started_at=s.started_at,
            ended_at=s.ended_at,
            final_smoothed_score=s.final_smoothed_score,
            final_band=s.final_band,
        )


class SessionDetailResponse(BaseModel):
    session_id: str
    window_count: int
    final: FusedScoreOut
    trace: list[FusedScoreOut]


class EnrollmentSummaryOut(BaseModel):
    identity: str
    embedding_model: str
    enrolled_at: str
    updated_at: str

    @classmethod
    def from_domain(cls, e: EnrollmentSummary) -> "EnrollmentSummaryOut":
        return cls(
            identity=e.identity,
            embedding_model=e.embedding_model,
            enrolled_at=e.enrolled_at,
            updated_at=e.updated_at,
        )


class EnrollResponse(BaseModel):
    identity: str
    embedding_model: str
    message: str


class StreamChunkIn(BaseModel):
    """One WebSocket message from the client — matches the blueprint's
    audio-chunk contract (§05 appendix)."""

    seq: int
    sample_rate: int
    pcm_f32: list[float]
    window_start_ms: int
    context: Optional[CallContext] = None
