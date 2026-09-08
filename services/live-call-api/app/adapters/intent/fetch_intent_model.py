"""Pre-downloads the zero-shot intent-classification model into
app/adapters/intent/.cache/ (kept inside the repo tree, same pattern as
every other model cache in this project — see fetch_checkpoint.py,
fetch_ecapa_model.py, fetch_whisper_model.py).

Called from the Dockerfile at build time: ZeroShotIntentClassifier is
enabled by default in config/risk_formula.yaml's active `detectors:` list
(with a known calibration caveat — see that module's own HONESTY NOTE).
Run this by hand (`python app/adapters/intent/fetch_intent_model.py`)
outside Docker, same as the other fetch_*.py scripts in this project.

Idempotent: huggingface_hub's download machinery skips re-downloading
whenever cache_dir already has the file.
"""
from __future__ import annotations

_CACHE_DIR = "app/adapters/intent/.cache"


def ensure_intent_model(model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli") -> None:
    from transformers import pipeline

    pipeline("zero-shot-classification", model=model_name, model_kwargs={"cache_dir": _CACHE_DIR})


if __name__ == "__main__":
    ensure_intent_model()
    print("Zero-shot intent-classification model cached.")
