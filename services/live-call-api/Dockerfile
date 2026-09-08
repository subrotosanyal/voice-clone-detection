# praat-parselmouth ships prebuilt wheels on PyPI only for x86_64/i686
# Linux (and macOS/Windows) — NOT for linux/aarch64, which is what this
# Dockerfile builds on an Apple Silicon host (and on any arm64 server).
# Without a wheel, pip falls back to building Parselmouth's C++ source
# (it bundles the actual Praat codebase) from an sdist, which needs cmake
# + a build tool pip's own resolver can't provide — confirmed by hand: a
# plain `pip install praat-parselmouth` on this arch fails with "CMake
# Error: CMake was unable to find a build program". This builder stage
# compiles just that one wheel with the tools it needs; the final image
# below never installs a C/C++ toolchain at all, keeping it slim.
FROM python:3.12-slim AS parselmouth-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN pip wheel --no-cache-dir --wheel-dir /wheels praat-parselmouth==0.4.7


FROM python:3.12-slim

WORKDIR /app

# libsndfile1: the `soundfile` package's wheel doesn't reliably bundle this
# native library on every platform (notably arm64) — without it, importing
# soundfile raises OSError at startup. See docs/testing.md, "Common errors".
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# The wheel built above — installed first so the plain
# `praat-parselmouth==0.4.7` line in requirements.txt (needed for local dev
# on macOS/x86_64, where a prebuilt wheel exists and this problem doesn't
# happen) is already satisfied and pip skips it below. Same pattern as the
# torch/torchaudio pre-install right after this.
COPY --from=parselmouth-builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/praat_parselmouth-*.whl

# torch + torchaudio next, from PyTorch's own CPU-only wheel index — plain
# `pip install torch` on linux resolves the CUDA-bundled build, which drags
# in ~2GB of nvidia_cudnn/cublas/cusolver/etc. this CPU-only container
# never uses (and makes the build look hung on a slow connection). Pinning
# the exact +cpu builds here means the plain `torch==2.11.0`/
# `torchaudio==2.11.0` lines in requirements.txt (needed for local dev on
# macOS, where this problem doesn't exist) are already satisfied and pip
# skips them below. Both are pinned to 2.11.0 together because the CPU
# wheel index doesn't publish a torchaudio build for every torch version —
# see docs/testing.md if you ever need to bump either.
RUN pip install --no-cache-dir torch==2.11.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Everything else next, so `docker compose build` reuses this layer on
# every code-only change — the whole point of "easy to debug and iterate"
# is a fast rebuild loop.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY samples ./samples

# Fetch the pretrained AASIST checkpoint at build time (checksum-verified —
# see app/adapters/detectors/vendor/fetch_checkpoint.py). Not committed to
# git; baked into the image here so the container never needs network
# access at runtime.
RUN python app/adapters/detectors/vendor/fetch_checkpoint.py

# Same reasoning for the ECAPA-TDNN speaker-embedding model used by the
# voiceprint consistency detector and the diarizer — see
# app/adapters/embeddings/fetch_ecapa_model.py. Run as `-m`, not a plain
# script path: unlike fetch_checkpoint.py (stdlib-only, self-contained),
# this script imports from the `app` package, which needs /app (this
# WORKDIR) on sys.path — `python -m` adds the cwd there; `python path.py`
# only adds the script's own directory, which fails with
# "ModuleNotFoundError: No module named 'app'".
RUN python -m app.adapters.embeddings.fetch_ecapa_model

# Same reasoning for the Whisper "small" transcription model — see
# app/adapters/transcription/fetch_whisper_model.py. No `app.*` imports in
# that script itself, so (unlike the ECAPA fetch above) a plain script
# path works fine here.
RUN python app/adapters/transcription/fetch_whisper_model.py

# Same reasoning for the zero-shot intent-classification model — see
# app/adapters/intent/fetch_intent_model.py (no `app.*` imports, plain
# script path works). Active by default since config/risk_formula.yaml's
# `intent` detector is enabled — see that config's own comment for the
# known calibration caveat this signal carries.
RUN python app/adapters/intent/fetch_intent_model.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
