"""LiveSpeakerTracker — incremental, live "who's talking now" for the
WebSocket path.

WHY THIS EXISTS: the file-upload diarizer (EmbeddingClusterDiarizer, see
app/adapters/diarization/embedding_cluster_diarizer.py) partitions a
COMPLETE recording after the fact — agglomerative clustering needs every
segment's embedding before it can group them. A live call has no "after
the fact": speakers must be discovered incrementally, one window at a
time, while the call is still happening — see docs/architecture.md's own
"real-time diarization on the live WebSocket path" entry under "What's
not built yet" (now filled in, this file). Genuinely different problem,
genuinely different (simpler) algorithm: nearest-centroid-with-a-
threshold, not agglomerative clustering over a whole recording.

HOW IT WORKS (real, but intentionally simple, same "state the honest
limits" standard as the file-upload diarizer):
  1. Skip near-silent windows (floor_rms) — nothing to attribute.
  2. Extract an ECAPA-TDNN embedding per window (same shared-
     implementation, own-instance convention as
     app/adapters/embeddings/ecapa_embedding.py's own docstring
     describes — this module owns its OWN EcapaEmbeddingExtractor
     instance, separate from EmbeddingClusterDiarizer's, so both
     features can be enabled/disabled independently; the real cost of
     that choice is a second copy of the model in memory when both are
     configured — an honest trade-off, not an oversight).
  3. Compare against every speaker centroid seen so far this call
     (cosine similarity). Above `similarity_threshold`: same speaker —
     the matched centroid is nudged toward the new embedding via EMA
     (`ema_alpha`), so it can drift with natural voice variation across
     a call. Below threshold: a new speaker, up to `max_speakers` (past
     that cap, attribute to the closest existing centroid instead of
     minting speaker_9, speaker_10, ... indefinitely — same safety-net
     reasoning as embedding_cluster_diarizer.py's own max_speakers cap).

CALIBRATION HONESTY NOTE, same caveat class as
embedding_cluster_diarizer.py's own CALIBRATION HISTORY: `similarity_
threshold` (0.75 default) and `ema_alpha` (0.1 default) are reasonable-
looking, NOT validated against any real labeled multi-speaker LIVE-call
dataset — no such corpus exists in this repo (see
eval/indian_language/README.md's own note on why that data doesn't exist
either). The 0.75 default is chosen deliberately ABOVE (not equal to)
the file-upload diarizer's own effective threshold (distance_threshold
0.4 == similarity 0.6): that diarizer's own docstring documents a real
measured cross-speaker cosine-similarity baseline of ~0.67-0.70 for
ECAPA-TDNN embeddings (a known property of this embedding space — cross-
speaker similarities cluster well above 0, not near it), which sits
ABOVE that diarizer's own 0.6 effective cutoff — meaning 0.6 risks
merging two different real speakers into one live-tracked "speaker".
0.75 is chosen to sit above that measured baseline instead, but is still
an unvalidated starting point, not a tuned result.

SCOPE, honestly stated:
  - Per-window, not per-utterance: unlike the file-upload diarizer
    (which segments on detected pauses first, see
    pause_segmentation.py), this attributes EVERY window independently.
    Two consecutive windows from the same speaker across a normal pause
    are still correctly matched to the same centroid (that is what the
    threshold is for), but there is no pause-aware segment boundary the
    way the file-upload path has.
  - Not persisted: see LiveSpeakerInfo's own docstring in
    app/domain/models.py — session history does not record this.
  - Doesn't feed scoring: unlike transcript/intent (which merge into
    `context` and change a detector's score — see
    app/pipeline/live_transcription.py), this is purely informational.
    No detector's output changes based on who is currently speaking. A
    future improvement (per-speaker EMA score smoothing, so one caller's
    calm voice isn't dragged down by a co-present agitated one) would
    need this, but is not attempted here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.adapters.embeddings.ecapa_embedding import EcapaEmbeddingExtractor, cosine_similarity
from app.domain.models import LiveSpeakerInfo


class LiveSpeakerTracker:
    """One instance per WebSocket session — mirrors LiveTranscriptionBuffer's
    "one instance per session, held by the caller" shape in
    app/pipeline/live_transcription.py. `embedder` is shared (loaded once
    at startup, see app/adapters/registry.py), only the per-session
    speaker centroids below are per-instance state."""

    def __init__(
        self,
        embedder: Optional[EcapaEmbeddingExtractor],
        floor_rms: float = 1e-4,
        similarity_threshold: float = 0.75,
        ema_alpha: float = 0.1,
        max_speakers: int = 8,
    ) -> None:
        self._embedder = embedder
        self.floor_rms = floor_rms
        self.similarity_threshold = similarity_threshold
        self.ema_alpha = ema_alpha
        self.max_speakers = max_speakers
        self._centroids: list[np.ndarray] = []  # index 0 == "speaker_1", in first-seen order

    def ingest(self, samples: np.ndarray, sample_rate: int) -> Optional[LiveSpeakerInfo]:
        """Call once per incoming window. Returns None (nothing to
        attach to this window's response) when no embedder is configured
        or the window is near-silent — same "abstain quietly, caller just
        doesn't get this field" convention every detector in this project
        already follows for DetectorResult.abstain_reason."""
        if self._embedder is None or samples.size == 0:
            return None

        rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms < self.floor_rms:
            return None

        embedding = self._embedder.extract(samples, sample_rate)

        if not self._centroids:
            self._centroids.append(embedding)
            return LiveSpeakerInfo(speaker_label="speaker_1", is_new_speaker=True, speaker_count=1)

        similarities = [cosine_similarity(embedding, centroid) for centroid in self._centroids]
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        if best_score >= self.similarity_threshold or len(self._centroids) >= self.max_speakers:
            # Either a real match, or the max_speakers safety net forcing
            # attribution to the closest existing speaker instead of
            # minting an unbounded number of new ones — see this module's
            # own docstring for why that cap exists.
            self._centroids[best_idx] = (1 - self.ema_alpha) * self._centroids[best_idx] + self.ema_alpha * embedding
            return LiveSpeakerInfo(
                speaker_label=f"speaker_{best_idx + 1}",
                is_new_speaker=False,
                speaker_count=len(self._centroids),
            )

        self._centroids.append(embedding)
        return LiveSpeakerInfo(
            speaker_label=f"speaker_{len(self._centroids)}",
            is_new_speaker=True,
            speaker_count=len(self._centroids),
        )
