"""Pre-downloads the zero-shot intent-classification model into
app/adapters/intent/.cache/ (kept inside the repo tree, same pattern as
every other model cache in this project — see fetch_checkpoint.py,
fetch_ecapa_model.py, fetch_whisper_model.py).

NOT called from the Dockerfile by default: ZeroShotIntentClassifier is
built and verified but not wired into config/risk_formula.yaml's active
`detectors:` list — see that module's own HONESTY NOTE on why. Run this
by hand (`python app/adapters/intent/fetch_intent_model.py`) only if
you're deliberately opting into the feature after addressing the
calibration issue described there.

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
