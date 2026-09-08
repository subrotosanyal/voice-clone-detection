"""Voiceprint consistency detector — the real "consistency" third signal.

Replaces consistency_stub.py as the active config choice (kept in the
codebase as a documented, dependency-light fallback for when no enrollment/
embedding stack is wanted — see config/risk_formula.yaml's comment on
swapping it back in).

WHAT THIS DOES: compares a live call window's speaker embedding (SpeechBrain
ECAPA-TDNN — see app/adapters/embeddings/ecapa_embedding.py for why this
model was chosen) against a voiceprint enrolled earlier for the identity the
caller is claiming (`context["claimed_identity"]`). A low cosine similarity
means "this voice doesn't match who they say they are" — the same kind of
embedding-comparison signal the SASV Challenge 2022 baseline (referenced in
the project blueprint's §03) uses.

HONESTY NOTE — the similarity->risk mapping below is NOT a calibrated
classifier. `_MATCH_SIMILARITY`/`_NO_MATCH_SIMILARITY` are placeholder
anchor points (same-speaker ECAPA-TDNN cosine similarities on clean audio
are typically well above 0.5; different-speaker pairs typically sit near
or below 0.1-0.2), not thresholds tuned against a labeled same/different-
speaker dataset run through this exact preprocessing pipeline. Proper
calibration — picking an operating point off an EER curve on held-out
labeled pairs — is the same kind of validation work docs/risk-model.md's
honesty section already flags as outstanding for the prosodic detector;
this detector inherits the same caveat. Until that calibration happens,
treat `raw_score` as directionally correct (higher = less voice-consistent)
but not a validated probability.

Enrollment is required before this can produce anything but an abstain —
see `.enroll()` below and app/api/enrollment_router.py for the HTTP surface.
Nothing here stores raw audio; only the derived embedding vector (see
app/adapters/enrollment/sqlite_enrollment_store.py's schema note).
"""
from __future__ import annotations

import numpy as np

from app.adapters.embeddings.ecapa_embedding import EcapaEmbeddingExtractor, cosine_similarity
from app.adapters.enrollment.sqlite_enrollment_store import SqliteEnrollmentStore
from app.domain.models import AudioWindow, DetectorResult, EnrollmentSummary

# Placeholder anchor points for the similarity -> risk mapping — see the
# HONESTY NOTE above. Configurable so a future calibration pass can update
# them via config/risk_formula.yaml without a code change.
_MATCH_SIMILARITY = 0.65
_NO_MATCH_SIMILARITY = 0.05


class VoiceprintConsistencyDetector:
    name = "voiceprint_consistency"
    version = "0.1.0-ecapa-tdnn"

    def __init__(
        self,
        enrollment_db_path: str,
        embedding_model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        match_similarity: float = _MATCH_SIMILARITY,
        no_match_similarity: float = _NO_MATCH_SIMILARITY,
        floor_rms: float = 1e-4,
    ) -> None:
        self.embedding_model_source = embedding_model_source
        self.match_similarity = match_similarity
        self.no_match_similarity = no_match_similarity
        self.floor_rms = floor_rms
        self._enrollment_store = SqliteEnrollmentStore(enrollment_db_path)
        self._embedding_extractor = EcapaEmbeddingExtractor(source=embedding_model_source)

    def score(self, window: AudioWindow, context: dict) -> DetectorResult:
        claimed_identity = context.get("claimed_identity")
        if not claimed_identity:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={},
                abstain_reason="no claimed_identity in context — nothing to compare the voice against",
            )

        enrolled_embedding = self._enrollment_store.get_embedding(claimed_identity)
        if enrolled_embedding is None:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"claimed_identity": claimed_identity},
                abstain_reason=(
                    f"no enrolled voiceprint for identity {claimed_identity!r} — "
                    "enroll one first via POST /v1/enroll"
                ),
            )

        samples = window.samples
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        if rms < self.floor_rms:
            return DetectorResult(
                detector_name=self.name,
                detector_version=self.version,
                score=None,
                detail={"claimed_identity": claimed_identity, "rms": rms},
                abstain_reason="window is near-silent — not enough voice to compare",
            )

        live_embedding = self._embedding_extractor.extract(samples, window.sample_rate)
        similarity = cosine_similarity(live_embedding, enrolled_embedding)

        span = self.match_similarity - self.no_match_similarity
        raw_score = 0.0 if span <= 0 else (self.match_similarity - similarity) / span
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        return DetectorResult(
            detector_name=self.name,
            detector_version=self.version,
            score=raw_score,
            detail={
                "claimed_identity": claimed_identity,
                "cosine_similarity": similarity,
                "match_similarity_anchor": self.match_similarity,
                "no_match_similarity_anchor": self.no_match_similarity,
                "embedding_model": self.embedding_model_source,
                "rms": rms,
                "explanation": _build_explanation(
                    similarity, self.match_similarity, self.no_match_similarity, claimed_identity, raw_score
                ),
            },
        )

    # --- Enrollment management, called from app/api/enrollment_router.py ---
    # (reuses this same detector instance's already-loaded embedding model,
    # rather than loading a second copy — see app/main.py's lifespan.)

    def enroll(self, identity: str, samples: np.ndarray, sample_rate: int) -> None:
        embedding = self._embedding_extractor.extract(samples, sample_rate)
        self._enrollment_store.enroll(identity, embedding, self.embedding_model_source)

    def list_enrollments(self) -> list[EnrollmentSummary]:
        return self._enrollment_store.list_enrollments()

    def delete_enrollment(self, identity: str) -> bool:
        return self._enrollment_store.delete(identity)


def _build_explanation(
    similarity: float, match_anchor: float, no_match_anchor: float, claimed_identity: str, raw_score: float
) -> str:
    """States the measured cosine similarity against both reference anchors
    already in `detail` — no inference beyond those three numbers."""
    similarity_pct = similarity * 100
    if raw_score < 0.15:
        return (
            f"The live voice closely matches the enrolled voiceprint for {claimed_identity!r} "
            f"({similarity_pct:.0f}% similarity, at/above the {match_anchor * 100:.0f}% match reference) "
            "— consistent with the claimed identity."
        )
    if raw_score > 0.6:
        return (
            f"The live voice does not match the enrolled voiceprint for {claimed_identity!r} well "
            f"({similarity_pct:.0f}% similarity, close to the {no_match_anchor * 100:.0f}% no-match reference) "
            "— inconsistent with the claimed identity."
        )
    return (
        f"The live voice partially matches the enrolled voiceprint for {claimed_identity!r} "
        f"({similarity_pct:.0f}% similarity, between the {no_match_anchor * 100:.0f}% no-match and "
        f"{match_anchor * 100:.0f}% match references) — not a confident match either way."
    )
