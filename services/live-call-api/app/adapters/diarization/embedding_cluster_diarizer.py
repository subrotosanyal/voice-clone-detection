"""EmbeddingClusterDiarizer — a lightweight, embedding-clustering diarizer.

Deliberately NOT using pyannote.audio's pretrained neural diarization
pipelines: every one of them is gated on HuggingFace (`gated: "auto"`,
confirmed via the HF API and a direct-download 401), which would require
every user to create an HF account and manage a personal access token just
to run `docker compose up --build` — inconsistent with this project's
"no auth, it just works" guarantee. See
app/adapters/embeddings/ecapa_embedding.py's docstring for the same
reasoning applied to the embedding model this diarizer reuses.

HOW IT WORKS (real, but intentionally simple):
  1. Slice the recording into fixed, non-overlapping segments.
  2. Skip near-silent segments (nothing to attribute to a speaker).
  3. Extract an ECAPA-TDNN embedding per remaining segment.
  4. Agglomerative clustering (scipy.cluster.hierarchy — average linkage,
     cosine distance, cut at `distance_threshold`) groups segments that
     sound like the same voice.
  5. Merge consecutive same-cluster segments into contiguous
     SpeakerSegments, labelled "speaker_1", "speaker_2", ... in order of
     first appearance (an arbitrary, per-call id — never a claimed or
     verified identity).

HONESTY NOTE: this is not a state-of-the-art diarizer. Fixed-length
segmentation (rather than a proper voice-activity/change-point front end)
means a speaker change mid-segment is missed.

CALIBRATION (2026-09-08 — was over-counting speakers on real calls):
`segment_ms` and `distance_threshold` were originally arbitrary
placeholders (1500ms / 0.35), and in production this over-segmented a
single speaker into several — natural moment-to-moment variation in a
person's own voice (breath, background noise, prosody drift) pushed
same-speaker embedding distance above the cutoff. Measured empirically
(see tests/unit/test_embedding_cluster_diarizer.py, using
tests/audio_fixtures.py's formant_voice() with per-take pitch/noise jitter
to stand in for one person's natural variation across a call, since no
real labeled multi-speaker corpus exists in this repo — see
eval/indian_language/README.md's own note on why that data doesn't exist
either): at the old 1500ms segment length, same-speaker distance ranged up
to 0.254 while a genuinely different speaker measured as low as 0.327 —
only a 0.073 margin for the 0.35 cutoff to sit in, and real speech is
noisier than this synthetic simulation. Lengthening segments to 2500ms
(more audio per embedding averages out more of that per-segment noise)
widened the margin to 0.168 (same-speaker max 0.204, cross-speaker min
0.372); `distance_threshold` was moved to 0.28 — comfortably inside that
gap, closer to the same-speaker side, since the real-world failure being
fixed was over- not under-counting. Still not validated against real
recorded speech — see voiceprint_consistency.py's similarity anchors for
the same class of caveat — but a real, reproducible, measured improvement
over the previous unexamined guess, not just a different guess.

Scope: file-upload path only (see app/ports/diarizer.py's scope note).
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from app.adapters.embeddings.ecapa_embedding import EcapaEmbeddingExtractor
from app.domain.models import SpeakerSegment


class EmbeddingClusterDiarizer:
    name = "embedding_cluster_diarizer"
    version = "0.1.0-ecapa-tdnn"

    def __init__(
        self,
        embedding_model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        segment_ms: int = 2500,
        floor_rms: float = 1e-4,
        distance_threshold: float = 0.28,
    ) -> None:
        self.segment_ms = segment_ms
        self.floor_rms = floor_rms
        self.distance_threshold = distance_threshold
        self._embedding_extractor = EcapaEmbeddingExtractor(source=embedding_model_source)

    def diarize(self, samples: np.ndarray, sample_rate: int) -> list[SpeakerSegment]:
        seg_len = int(sample_rate * self.segment_ms / 1000)
        if seg_len <= 0 or samples.size == 0:
            return []

        raw_segments: list[tuple[int, int]] = []
        embeddings: list[np.ndarray] = []
        for start in range(0, samples.size, seg_len):
            chunk = samples[start : start + seg_len]
            if chunk.size < seg_len // 2:  # trailing sliver too short to trust
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            if rms < self.floor_rms:
                continue  # silence — not attributed to any speaker
            embeddings.append(self._embedding_extractor.extract(chunk, sample_rate))
            start_ms = int(start / sample_rate * 1000)
            end_ms = int(min(start + seg_len, samples.size) / sample_rate * 1000)
            raw_segments.append((start_ms, end_ms))

        if not raw_segments:
            return []

        if len(raw_segments) == 1:
            cluster_labels = np.array([0])
        else:
            embedding_matrix = np.stack(embeddings)
            distances = pdist(embedding_matrix, metric="cosine")
            linkage_matrix = linkage(distances, method="average")
            cluster_labels = fcluster(linkage_matrix, t=self.distance_threshold, criterion="distance")

        speaker_labels = _stable_speaker_labels(cluster_labels)
        return _merge_consecutive(raw_segments, speaker_labels)


def _stable_speaker_labels(cluster_labels: np.ndarray) -> list[str]:
    """Maps scipy's arbitrary cluster ids to "speaker_1", "speaker_2", ...
    ordered by first appearance in time, so the same recording always
    produces the same labels regardless of scipy's internal id assignment."""
    first_seen: dict[int, int] = {}
    for cluster_id in cluster_labels:
        if int(cluster_id) not in first_seen:
            first_seen[int(cluster_id)] = len(first_seen)
    return [f"speaker_{first_seen[int(c)] + 1}" for c in cluster_labels]


def _merge_consecutive(
    segments: list[tuple[int, int]], speaker_labels: list[str]
) -> list[SpeakerSegment]:
    merged: list[SpeakerSegment] = []
    for (start_ms, end_ms), label in zip(segments, speaker_labels):
        if merged and merged[-1].speaker_label == label and merged[-1].end_ms == start_ms:
            merged[-1] = SpeakerSegment(speaker_label=label, start_ms=merged[-1].start_ms, end_ms=end_ms)
        else:
            merged.append(SpeakerSegment(speaker_label=label, start_ms=start_ms, end_ms=end_ms))
    return merged
