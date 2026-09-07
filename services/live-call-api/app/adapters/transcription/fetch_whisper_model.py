"""Pre-downloads the Whisper "small" model into app/adapters/transcription/
.cache/ (not Whisper's default ~/.cache/whisper/, kept inside the repo
tree for consistency with every other model cache in this project), so
`docker build` bakes it into the image and the container never needs
network access at runtime for it — same pattern as
app/adapters/detectors/vendor/fetch_checkpoint.py (AASIST) and
app/adapters/embeddings/fetch_ecapa_model.py (ECAPA-TDNN).

Idempotent: whisper.load_model() itself skips re-downloading whenever its
cache dir already has the file, and verifies a checksum before reusing it.
"""
from __future__ import annotations

_DOWNLOAD_ROOT = "app/adapters/transcription/.cache"


def ensure_whisper_model(model_size: str = "small") -> None:
    import whisper

    whisper.load_model(model_size, download_root=_DOWNLOAD_ROOT)


if __name__ == "__main__":
    ensure_whisper_model()
    print("Whisper 'small' model cached.")
