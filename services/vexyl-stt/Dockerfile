# syntax=docker/dockerfile:1
# ^ pinned explicitly so the `--mount=type=secret,...,env=` shorthand
# below is guaranteed available — see live-call-api/Dockerfile's own
# identical note for the exact frontend-version story (a REAL bug found
# there: an earlier pinned "1.7" failed with "unexpected key 'env' in
# 'env=HF_TOKEN'").
FROM python:3.11-slim

# System deps, unchanged from the upstream repo's own Dockerfile
# (services/vexyl-stt/vendor/vexyl_stt_server.py's own header names the
# source): libgomp1 for PyTorch OpenMP, libsndfile1 for soundfile, ffmpeg
# for MP3/M4A batch-API input.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU-only first (large, cacheable layer), matching every other
# service in this project.
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model fetch — REAL DEVIATION from the upstream repo's own Dockerfile,
# deliberate: upstream uses `ARG HF_TOKEN` baked into a `python -c "..."`
# RUN line, which writes the token in plain text into this layer's
# history (`docker history` reveals it even after a later layer "unsets"
# it — the exact anti-pattern live-call-api/Dockerfile's own HF_TOKEN
# note already documented and avoided for every other model in this
# project). Using the same BuildKit secret mount instead: the token is
# injected only into THIS RUN step's process environment, never written
# to any image layer. See fetch_model.py's own docstring for what
# building without a valid, access-granted token does (this one fails
# outright — no anonymous fallback, unlike Whisper/AASIST/ECAPA).
COPY fetch_model.py .
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    python fetch_model.py

COPY vendor/vexyl_stt_server.py .

# Cloud Run injects PORT; VEXYL_STT_PORT is this project's own default
# for local docker-compose use (see docker-compose.yml's own vexyl-stt
# service block).
ENV VEXYL_STT_HOST=0.0.0.0 \
    VEXYL_STT_PORT=8091 \
    VEXYL_STT_DECODE=ctc \
    VEXYL_STT_DEVICE=cpu

EXPOSE 8091

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import socket; socket.create_connection(('localhost', 8091), timeout=3).close()"

CMD ["python", "-u", "vexyl_stt_server.py"]
