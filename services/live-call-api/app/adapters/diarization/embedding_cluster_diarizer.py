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

CALIBRATION HISTORY — read this before touching distance_threshold again:
2026-09-08 (first attempt, WRONG DIRECTION): a real over-counting bug was
reported (one speaker split into several). Measured same-/cross-speaker
embedding distance on synthetic fixtures and moved `distance_threshold`
from 0.35 DOWN to 0.28 — but a *lower* fcluster distance cutoff is
*stricter* (requires segments to be more similar to merge), which makes
over-counting WORSE, not better; this was a real reasoning error, not a
deliberate trade-off, and made it into a shipped commit. Confirmed wrong
by a real user report immediately after: a real call produced 100+
"speakers" (`speaker_124`).

2026-09-08 (correction, same day): `distance_threshold` raised to 0.4 —
*higher* than even the original 0.35 — because fixing over-counting means
making it *easier* to merge two segments into the same speaker, not
harder. `segment_ms` stays at 2500 (longer segments give a more stable
embedding regardless of threshold, independently useful). A hard
`max_speakers` cap was also added as a real safety net: no matter what
distance_threshold turns out to be wrong about on some future real
recording, this diarizer will never again report more than `max_speakers`
distinct speakers for one call — if the distance-based clustering produces
more than that, it re-clusters with a fixed cluster count instead. This is
the actual fix for the catastrophic failure mode; the threshold value
remains an unvalidated guess (no real labeled multi-speaker corpus exists
in this repo — see eval/indian_language/README.md's own note on why that
data doesn't exist either) and should not be trusted without real
calibration data — see voiceprint_consistency.py's similarity anchors for
the same class of caveat.

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
        distance_threshold: float = 0.4,
        max_speakers: int = 8,
    ) -> None:
        self.segment_ms = segment_ms
        self.floor_rms = floor_rms
        self.distance_threshold = distance_threshold
        self.max_speakers = max_speakers
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

            # Safety net, independent of whether distance_threshold happens
            # to be well-tuned for this recording: a real phone call is
            # never going to have more than a handful of participants, so
            # if the distance-based cut produced more than max_speakers
            # clusters (e.g. from noisy real audio the threshold wasn't
            # calibrated against), fall back to a fixed cluster COUNT
            # instead of a fixed distance — bounds the worst case no
            # matter how wrong distance_threshold turns out to be.
            if len(set(cluster_labels)) > self.max_speakers:
                cluster_labels = fcluster(linkage_matrix, t=self.max_speakers, criterion="maxclust")

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
