"""Pre-downloads ai4bharat/indic-conformer-600m-multilingual into this
container's HuggingFace cache, so `docker build` bakes it into the image
(~2.4GB) and the container never needs network access for it at runtime —
same pattern as app/adapters/transcription/fetch_whisper_model.py,
app/adapters/embeddings/fetch_ecapa_model.py, and
app/adapters/detectors/vendor/fetch_checkpoint.py in the live-call-api
service (all in this project's other Dockerfile).

GATED MODEL, real consequence (not the un-gated case those three fetch
scripts handle): this model requires visiting
https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual while
logged in, requesting access (a click-through "agree to share contact
info" gate — free, no payment, but a real one-time manual step per HF
account), and building with that account's HF_TOKEN passed as a build
secret (see this directory's own Dockerfile for how — the SAME
`--mount=type=secret,id=hf_token,env=HF_TOKEN` pattern live-call-api's
Dockerfile uses, not the upstream vexyl-stt repo's own `ARG HF_TOKEN`,
which would leak the token into `docker history`). Building this image
with no token, or a token whose account hasn't been granted access, fails
this step outright — there is no anonymous fallback the way there is for
Whisper/AASIST/ECAPA, because the model itself refuses the request
server-side, not because of anything this script chooses.

Idempotent: huggingface_hub's own snapshot_download() skips re-fetching
whatever this container's cache already has.
"""
from __future__ import annotations

_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


def ensure_model_cached() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(_MODEL_ID)


if __name__ == "__main__":
    ensure_model_cached()
    print(f"{_MODEL_ID} cached.")
