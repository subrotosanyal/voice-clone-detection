# syntax=docker/dockerfile:1
# ^ pinned explicitly (not left to the Docker Engine's own default
# frontend version) so the `--mount=type=secret,...,env=` shorthand used
# below is guaranteed available regardless of which Docker version
# builds this — see the HF_TOKEN secret-mount note further down for
# why. REAL BUG found 2026-09-11 verifying this: pinning the exact
# `1.7` frontend (an earlier version of this same line) actually
# FAILED with `unexpected key 'env' in 'env=HF_TOKEN'` — that shorthand
# needs a newer 1.x frontend than 1.7. Using the untagged `1` (always
# resolves to the latest 1.x release) rather than guessing/re-pinning
# the exact minor version that introduced it.

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

# llama-cpp-python (app/adapters/semantic_risk/local_llm_semantic_classifier.py)
# compiles llama.cpp's C++ core at install time — same "no prebuilt wheel
# for this arch" situation as Parselmouth above, same fix: a dedicated
# builder stage with the compiler toolchain, so the final image below
# still never installs one. Reuses build-essential/cmake/ninja-build
# already installed in the stage above.
FROM parselmouth-builder AS llm-builder
RUN pip wheel --no-cache-dir --wheel-dir /llm-wheels llama-cpp-python==0.3.35


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

# Same reasoning, for llama-cpp-python's wheel (see the llm-builder stage
# above).
COPY --from=llm-builder /llm-wheels /llm-wheels
RUN pip install --no-cache-dir /llm-wheels/llama_cpp_python-*.whl

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
#
# HF_TOKEN — added 2026-09-11: this and the two other HuggingFace-Hub-
# backed fetches below (intent classifier, Phi-3 GGUF; Whisper and
# AASIST above are NOT HF Hub, they don't benefit) were logging
# "You are sending unauthenticated requests to the HF Hub" during every
# build. `--mount=type=secret,...,env=HF_TOKEN` injects an optional
# token ONLY into this one RUN step's process environment — never
# written to a file in the image, never an ARG/ENV (which WOULD leak
# into the image's layer history, retrievable via `docker history` even
# after a later layer unsets it — a real anti-pattern this deliberately
# avoids). huggingface_hub reads HF_TOKEN from the environment
# automatically; no code change needed. Safe with no token at all: an
# unset or empty secret leaves HF_TOKEN empty, and huggingface_hub's own
# `_get_token_from_environment()` treats an empty string as falsy and
# falls through to today's exact anonymous-request behavior (confirmed
# against its actual source, not assumed) — so this is a pure
# opt-in speedup, not a new requirement. See docker-compose.yml's own
# `secrets:` block for where the value actually comes from.
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    python -m app.adapters.embeddings.fetch_ecapa_model

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
#
# HF_TOKEN — see the ECAPA fetch step's own note above for the full
# rationale (same secret, same opt-in, same safe-with-nothing-set
# behavior).
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    python app/adapters/intent/fetch_intent_model.py

# Same reasoning for the local semantic-risk LLM (Microsoft's Phi-3-mini
# GGUF) — see app/adapters/semantic_risk/fetch_llm_model.py. No
# `app.*` imports in that script, plain script path works.
#
# HF_TOKEN — see the ECAPA fetch step's own note above for the full
# rationale (same secret, same opt-in, same safe-with-nothing-set
# behavior). Worth the most here: this is the ~2.4GB fetch.
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    python app/adapters/semantic_risk/fetch_llm_model.py

# REAL BUG found 2026-09-09 (live-mic parity investigation): none of
# torch/OpenBLAS/MKL/numexpr had a thread-count limit, so each of
# AASIST, Parselmouth/Praat, ECAPA-TDNN, and llama.cpp's own forward
# pass defaulted to grabbing every visible core per call. On the live
# WS path, several of these run every ~500ms concurrently with each
# other AND with the periodic background transcription task — measured
# by hand: per-window latency inflated to 0.85-1.5s against the 500ms
# target, starving the transcriber of CPU behind that contention (see
# app/pipeline/live_transcription.py's own docstring for the resulting
# "small piece, then nothing" symptom). Bounding every BLAS/OpenMP
# thread pool process-wide fixes the oversubscription directly, without
# needing per-model code changes. 2 is a conservative starting point,
# not a measured-optimal one — this host has 14 cores and typically 3-4
# of these model calls can be in flight at once; rebalance only with
# real concurrency numbers in hand, same discipline as
# ws_router.py's own _HANGUP_DIARIZE_MAX_CONCURRENT note.
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
