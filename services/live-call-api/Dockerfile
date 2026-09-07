FROM python:3.12-slim

WORKDIR /app

# libsndfile1: the `soundfile` package's wheel doesn't reliably bundle this
# native library on every platform (notably arm64) — without it, importing
# soundfile raises OSError at startup. See docs/testing.md, "Common errors".
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Only requirements next, so `docker compose build` reuses the pip-install
# layer on every code-only change — the whole point of "easy to debug and
# iterate" is a fast rebuild loop.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY samples ./samples

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
