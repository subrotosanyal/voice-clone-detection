"""Shared speaker-embedding extractor — SpeechBrain's ECAPA-TDNN.

This is NOT a DetectorPort. It's a plain utility wrapping one pretrained
model, used independently by two features that both need "a vector that
represents who is speaking": app/adapters/detectors/voiceprint_consistency.py
(compares a live embedding to an enrolled one) and
app/adapters/diarization/embedding_cluster_diarizer.py (clusters embeddings
from different time segments to tell speakers apart). Each of those owns
its own instance of this class (same self-contained-adapter pattern as
acoustic_aasist.py owning its own AASIST model) — this module just holds
the one implementation so it isn't duplicated.

WHY THIS MODEL (see docs/risk-model.md for the full comparison):
  - speechbrain/spkrec-ecapa-voxceleb — Apache-2.0, confirmed UNGATED on
    HuggingFace (`gated: false`, plain 302-redirect download, no auth
    token needed) — the only option consistent with this project's
    "docker compose up --build just works, no account/token setup"
    guarantee.
  - pyannote.audio's pretrained pipelines were rejected: every one of them
    is gated (`gated: "auto"`, 401 without an HF token).
  - Resemblyzer was rejected: its `webrtcvad` dependency is unmaintained
    (last touched ~2017) and breaks on modern setuptools (imports the
    now-removed `pkg_resources` at load time).

Trained on VoxCeleb1+2 (English-majority, but speaker embedding — unlike
spoof detection — is much less language-dependent: it's modelling vocal-
tract/prosody characteristics, not phonetic content, so this generalises
across languages far better than AASIST's spoof classification does).
"""
from __future__ import annotations

import threading

import numpy as np
import torch

_MODEL_SAMPLE_RATE = 16_000


class EcapaEmbeddingExtractor:
    """Loads speechbrain/spkrec-ecapa-voxceleb once; extracts a fixed-size
    (192-dim) speaker embedding from arbitrary-length mono audio.

    Thread-safe: SpeechBrain's EncoderClassifier isn't documented as safe
    for concurrent forward passes, so a lock serialises calls — an
    acceptable trade-off for this project's low request-rate demo scope
    (the same trade-off SqliteHistoryStore already makes with its single-
    lock shared connection).
    """

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "app/adapters/embeddings/.cache/spkrec-ecapa-voxceleb",
        device: str = "cpu",
    ) -> None:
        # Imported lazily so importing this module doesn't require
        # speechbrain to be installed unless something actually uses it.
        from speechbrain.inference.speaker import EncoderClassifier

        self.source = source
        self.device = device
        self._lock = threading.Lock()
        self._classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            run_opts={"device": device},
        )

    def extract(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Returns a (192,) float32 embedding. Resamples to 16kHz first if
        needed — the model was trained on 16kHz audio."""
        if sample_rate != _MODEL_SAMPLE_RATE:
            samples = _resample(samples, sample_rate, _MODEL_SAMPLE_RATE)

        tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        with self._lock, torch.no_grad():
            embedding = self._classifier.encode_batch(tensor)
        return embedding.squeeze().cpu().numpy().astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(samples, up, down).astype(np.float32)
