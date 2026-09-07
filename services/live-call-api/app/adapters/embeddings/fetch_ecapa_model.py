"""Pre-downloads speechbrain/spkrec-ecapa-voxceleb into EcapaEmbeddingExtractor's
on-disk cache, so `docker build` bakes the model into the image and the
container never needs network access at runtime for it — the same pattern
app/adapters/detectors/vendor/fetch_checkpoint.py already establishes for
the AASIST checkpoint (see docs/architecture.md, "Reproducibility").

Idempotent: EncoderClassifier.from_hparams() itself skips re-downloading
whenever its savedir already has the cached files.
"""
from __future__ import annotations

from app.adapters.embeddings.ecapa_embedding import EcapaEmbeddingExtractor


def ensure_ecapa_model() -> None:
    EcapaEmbeddingExtractor()


if __name__ == "__main__":
    ensure_ecapa_model()
    print("speechbrain/spkrec-ecapa-voxceleb cached.")
