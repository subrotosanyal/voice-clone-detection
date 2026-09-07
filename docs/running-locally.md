# Running locally

## With Docker Compose (recommended)

```bash
docker compose up --build
```

This starts:

- **`live-call-api`** — the FastAPI service, on http://localhost:8020
  (mapped from container port 8000 — see the comment in `docker-compose.yml`
  if you need to change the host port).
- **`dozzle`** — a central log viewer, on http://localhost:8888.

`config/risk_formula.yaml` is mounted read-only into the container, so
editing it and running `docker compose restart live-call-api` is enough
to try a new formula — no rebuild needed. Editing `app/` code *does* need
a rebuild: `docker compose up --build`.

Check it's healthy:

```bash
curl http://localhost:8020/healthz
# {"status":"ok"}

curl http://localhost:8020/v1/config | python3 -m json.tool
# the exact formula this running instance loaded
```

**For anyone who isn't going to type curl commands:** open
http://localhost:8020/ in a browser. Upload a recording, or click the
microphone tab and speak — the risk meter, the per-signal breakdown, and
the score-over-time chart all update from the same two endpoints above.
Microphone access needs a secure context (`localhost` counts) and browser
permission on first use.

Score a file:

```bash
curl -F "file=@services/live-call-api/samples/genuine_tone.wav" \
     -F 'context={"known_number": true, "hour_of_day": 14}' \
     http://localhost:8020/v1/score/file | python3 -m json.tool
```

`context` is optional JSON — see `app/api/schemas.py::CallContext` for the
full field list (`known_number`, `hour_of_day`, `is_financial_request`,
`urgency_keywords`, `claimed_identity`).

Interactive API docs (try requests from the browser): http://localhost:8020/docs

Stop everything: `docker compose down`.

## Without Docker (faster iteration on the Python code)

```bash
cd services/live-call-api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python ../../scripts/gen_test_audio.py                          # writes samples/*.wav, run once
python app/adapters/detectors/vendor/fetch_checkpoint.py         # fetches AASIST.pth, run once
uvicorn app.main:app --reload --port 8010
```

Same curl commands as above, just against port 8010. The checkpoint fetch
needs internet access once; it's idempotent (safe to re-run, does nothing
if already present and valid) and `docker compose up --build` does the
same thing automatically at image-build time.

## Regenerating the synthetic test fixtures

```bash
python scripts/gen_test_audio.py
```

Writes `services/live-call-api/samples/genuine_tone.wav` and
`noisy_clip.wav`. These are synthetic sine/noise clips for exercising the
plumbing — **not** real voice or deepfake samples. See the script's
docstring and `docs/risk-model.md` for what would actually be needed to
evaluate detection accuracy.
