FROM python:3.12-slim

WORKDIR /app

# libsndfile1: the `soundfile` package's wheel doesn't reliably bundle this
# native library on every platform (notably arm64) — without it, importing
# soundfile raises OSError at startup. See docs/testing.md, "Common errors".
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# torch FIRST, from PyTorch's own CPU-only wheel index — plain `pip
# install torch` on linux resolves the CUDA-bundled build, which drags in
# ~2GB of nvidia_cudnn/cublas/cusolver/etc. this CPU-only container never
# uses (and makes the build look hung on a slow connection). Pinning the
# exact +cpu build here means the plain `torch==2.14.0` line in
# requirements.txt (needed for local dev on macOS, where this problem
# doesn't exist) is already satisfied and pip skips it below.
RUN pip install --no-cache-dir torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu

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

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
