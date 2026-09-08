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
  1. Slice the recording into segments boundary-aligned to detected
     pauses (app/adapters/diarization/pause_segmentation.py), not a fixed
     clock — see that module's docstring for why: a segment straddling a
     real speaker change produces an embedding that resembles NEITHER
     speaker, reliably becoming a phantom extra "speaker" under
     clustering. Long uninterrupted stretches are still sub-sliced at
     `max_segment_ms`, same fallback as before.
  2. Skip near-silent segments (nothing to attribute to a speaker) — kept
     as an independent check even though pause_segmentation already
     excludes detected silence, since Praat's silence detector can
     mislabel a perfectly flat/silent clip as "sounding" (verified by
     hand on an all-zeros clip) — this RMS floor catches that case.
  3. Extract an ECAPA-TDNN embedding per remaining segment.
  4. Agglomerative clustering (scipy.cluster.hierarchy — average linkage,
     cosine distance, cut at `distance_threshold`) groups segments that
     sound like the same voice.
  5. Merge consecutive same-cluster segments into contiguous
     SpeakerSegments, labelled "speaker_1", "speaker_2", ... in order of
     first appearance (an arbitrary, per-call id — never a claimed or
     verified identity).

HONESTY NOTE: this is not a state-of-the-art diarizer. Pause-aware
segmentation (2026-09-08) catches most real speaker turns but not all —
a genuine zero-gap turn-take (immediate back-to-back speech, or
overlapping speech) leaves no acoustic silence for any pause-based
method to detect, no matter how it's tuned; see
pause_segmentation.py's own HONESTY NOTE for the measurements.

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

2026-09-08 (later same day): fixed-length segmentation replaced with
pause-aware segmentation (see pause_segmentation.py) after measuring that
a segment straddling a real speaker change produces an embedding
resembling neither speaker (cosine similarity 0.04-0.34 against either
pure voice — far below even the normal cross-speaker baseline of
~0.67-0.70) — not the moderately-confusable "blend" that might be
expected. That reliably forms an isolated phantom cluster under
clustering, the same over-counting failure class as above, but via a
completely different mechanism (temporal straddling, not natural
single-speaker variation) that survived the distance_threshold/
max_speakers fix untouched. `segment_ms` (fixed grid) became
`max_segment_ms` (a cap for sub-slicing only, now that segmentation is
otherwise pause-aligned) — kept at the same value the segment_ms
calibration above already measured to help embedding stability.
`distance_threshold`/`max_speakers` are unchanged; they still matter for
real speaker-clustering decisions and for whatever this new segmentation
still misses (a genuine zero-gap turn-take — see the HONESTY NOTE above).

Scope: file-upload path only (see app/ports/diarizer.py's scope note).
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from app.adapters.diarization.pause_segmentation import detect_speech_segments
from app.adapters.embeddings.ecapa_embedding import EcapaEmbeddingExtractor
from app.domain.models import SpeakerSegment


class EmbeddingClusterDiarizer:
    name = "embedding_cluster_diarizer"
    version = "0.1.0-ecapa-tdnn"

    def __init__(
        self,
        embedding_model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        max_segment_ms: int = 2500,
        min_segment_ms: int = 500,
        silence_threshold_db: float = -25.0,
        min_silence_s: float = 0.1,
        min_sounding_s: float = 0.1,
        floor_rms: float = 1e-4,
        distance_threshold: float = 0.4,
        max_speakers: int = 8,
    ) -> None:
        self.max_segment_ms = max_segment_ms
        self.min_segment_ms = min_segment_ms
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_s = min_silence_s
        self.min_sounding_s = min_sounding_s
        self.floor_rms = floor_rms
        self.distance_threshold = distance_threshold
        self.max_speakers = max_speakers
        self._embedding_extractor = EcapaEmbeddingExtractor(source=embedding_model_source)

    def diarize(self, samples: np.ndarray, sample_rate: int) -> list[SpeakerSegment]:
        if samples.size == 0:
            return []

        detected_segments = detect_speech_segments(
            samples,
            sample_rate,
            silence_threshold_db=self.silence_threshold_db,
            min_silence_s=self.min_silence_s,
            min_sounding_s=self.min_sounding_s,
            min_segment_ms=self.min_segment_ms,
            max_segment_ms=self.max_segment_ms,
        )

        raw_segments: list[tuple[int, int]] = []
        embeddings: list[np.ndarray] = []
        for start_ms, end_ms in detected_segments:
            start_sample = int(start_ms / 1000 * sample_rate)
            end_sample = min(int(end_ms / 1000 * sample_rate), samples.size)
            chunk = samples[start_sample:end_sample]
            if chunk.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            if rms < self.floor_rms:
                # silence — not attributed to any speaker. Kept as an
                # independent check even though pause_segmentation already
                # tries to exclude silence, since Praat's silence detector
                # can mislabel a perfectly flat/silent clip as "sounding"
                # (verified by hand on an all-zeros clip) — this RMS floor
                # catches that case.
                continue
            embeddings.append(self._embedding_extractor.extract(chunk, sample_rate))
            raw_segments.append((start_ms, min(end_ms, int(samples.size / sample_rate * 1000))))

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
