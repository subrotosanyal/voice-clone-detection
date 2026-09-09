"""Tests for LiveSpeakerTracker — a fast, deterministic fake embedder for
the clustering-logic tests (same pattern as test_live_transcription.py's
_FakeTranscriber), plus real formant_voice fixtures (see
tests/audio_fixtures.py) through the ACTUAL EcapaEmbeddingExtractor for
the one test that needs to prove real same/different-speaker separation,
not just that the arithmetic is right.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.adapters.embeddings.ecapa_embedding import EcapaEmbeddingExtractor
from app.pipeline.live_diarization import LiveSpeakerTracker
from tests.audio_fixtures import VOICE_A_FORMANTS, VOICE_B_FORMANTS, formant_voice

SR = 16_000


class _FakeEmbedder:
    """Returns whatever fixed vector `samples` maps to via `vectors` — lets
    clustering-logic tests control embeddings exactly, no real model load."""

    def __init__(self, vectors: dict[int, np.ndarray]) -> None:
        self.vectors = vectors
        self.calls = 0

    def extract(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        return self.vectors[int(samples[0])]


def _tagged_window(tag: int, n: int = 320) -> np.ndarray:
    """A loud (clears floor_rms), non-zero window whose first sample is
    `tag` — _FakeEmbedder reads that tag back out to pick a fixed vector."""
    samples = np.full(n, 0.1, dtype=np.float32)
    samples[0] = tag
    return samples


def test_returns_none_without_an_embedder():
    tracker = LiveSpeakerTracker(embedder=None)

    result = tracker.ingest(_tagged_window(1), SR)

    assert result is None


def test_abstains_on_near_silence():
    embedder = _FakeEmbedder({0: np.array([1.0, 0.0])})
    tracker = LiveSpeakerTracker(embedder=embedder, floor_rms=1e-3)
    silence = np.zeros(320, dtype=np.float32)

    result = tracker.ingest(silence, SR)

    assert result is None
    assert embedder.calls == 0  # never even reaches the model for silence


def test_first_window_becomes_speaker_1():
    embedder = _FakeEmbedder({1: np.array([1.0, 0.0])})
    tracker = LiveSpeakerTracker(embedder=embedder)

    result = tracker.ingest(_tagged_window(1), SR)

    assert result.speaker_label == "speaker_1"
    assert result.is_new_speaker is True
    assert result.speaker_count == 1


def test_a_similar_embedding_reuses_the_same_speaker_label():
    # [1.0, 0.0] and [0.9, 0.1] are highly cosine-similar (~0.99) — clearly
    # above any reasonable threshold, standing in for "same speaker, second
    # utterance" without needing a real model.
    embedder = _FakeEmbedder({1: np.array([1.0, 0.0]), 2: np.array([0.9, 0.1])})
    tracker = LiveSpeakerTracker(embedder=embedder, similarity_threshold=0.75)

    first = tracker.ingest(_tagged_window(1), SR)
    second = tracker.ingest(_tagged_window(2), SR)

    assert first.speaker_label == second.speaker_label == "speaker_1"
    assert second.is_new_speaker is False
    assert second.speaker_count == 1


def test_a_dissimilar_embedding_creates_a_new_speaker():
    # Orthogonal vectors -> cosine similarity 0.0 -> clearly a new speaker.
    embedder = _FakeEmbedder({1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])})
    tracker = LiveSpeakerTracker(embedder=embedder, similarity_threshold=0.75)

    first = tracker.ingest(_tagged_window(1), SR)
    second = tracker.ingest(_tagged_window(2), SR)

    assert first.speaker_label == "speaker_1"
    assert second.speaker_label == "speaker_2"
    assert second.is_new_speaker is True
    assert second.speaker_count == 2


def test_max_speakers_cap_forces_reattribution_instead_of_unbounded_growth():
    """Same safety-net reasoning as embedding_cluster_diarizer.py's own
    max_speakers cap: never claim more distinct speakers than the cap,
    no matter how dissimilar a new embedding looks."""
    vectors = {i: v for i, v in enumerate([np.eye(4)[i % 4] for i in range(5)])}
    embedder = _FakeEmbedder(vectors)
    tracker = LiveSpeakerTracker(embedder=embedder, similarity_threshold=0.99, max_speakers=2)

    labels = [tracker.ingest(_tagged_window(i), SR).speaker_label for i in range(5)]

    assert len(set(labels)) <= 2
    assert all(label in ("speaker_1", "speaker_2") for label in labels)


def test_ema_update_pulls_the_centroid_toward_new_samples():
    """The matched centroid should move toward the new embedding, not stay
    frozen at the first one seen — verified by checking the updated
    centroid sits exactly where ema_alpha says it should."""
    original = np.array([1.0, 0.0])
    drifted = np.array([0.0, 1.0])
    embedder = _FakeEmbedder({1: original, 2: drifted})
    # max_speakers=1 forces the second (orthogonal) sample to still match
    # the first centroid, regardless of similarity_threshold — this test
    # is about the EMA update itself, not the matching decision.
    tracker = LiveSpeakerTracker(embedder=embedder, similarity_threshold=0.75, ema_alpha=0.5, max_speakers=1)

    tracker.ingest(_tagged_window(1), SR)
    tracker.ingest(_tagged_window(2), SR)

    updated_centroid = tracker._centroids[0]
    # alpha=0.5: exactly halfway between [1,0] and [0,1].
    assert updated_centroid == pytest.approx([0.5, 0.5])


def _real_embedder() -> EcapaEmbeddingExtractor:
    try:
        return EcapaEmbeddingExtractor()
    except Exception as exc:  # noqa: BLE001 — model load can fail offline
        pytest.skip(f"ECAPA-TDNN model unavailable (no network?): {exc}")


def test_real_model_separates_two_different_voices_and_reunites_one():
    """The real correctness property, with the actual model: two takes of
    the SAME formant voice should be tracked as one speaker; a different
    formant voice should be tracked as a second one. Same fixtures
    test_voiceprint_consistency.py already trusts for "the model can tell
    these apart" (see tests/audio_fixtures.py's own measured ~0.87 vs ~0.55
    cosine-similarity margin)."""
    tracker = LiveSpeakerTracker(embedder=_real_embedder(), similarity_threshold=0.75)

    voice_a_take1 = tracker.ingest(formant_voice(VOICE_A_FORMANTS, seed=1, pitch_hz=120), SR)
    voice_b_take1 = tracker.ingest(formant_voice(VOICE_B_FORMANTS, seed=3, pitch_hz=180), SR)
    voice_a_take2 = tracker.ingest(formant_voice(VOICE_A_FORMANTS, seed=2, pitch_hz=122), SR)

    assert voice_a_take1.speaker_label == "speaker_1"
    assert voice_b_take1.speaker_label == "speaker_2"
    assert voice_a_take2.speaker_label == "speaker_1"  # reunited with the first voice, not a new speaker_3
    assert voice_a_take2.is_new_speaker is False
    assert voice_a_take2.speaker_count == 2
