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
Every detector name and risk band has a small "?" icon — hover or focus it
for a plain-language explanation of what it means. Microphone access needs
a secure context (`localhost` counts) and browser permission on first use.
The **History** tab lists every session ever scored (persisted in
`services/live-call-api/data/sessions.db`, mounted as a docker-compose
volume — it survives `docker compose down` and rebuilds); click one to
replay its full breakdown, or delete it.

To reset history entirely: stop the stack and delete
`services/live-call-api/data/sessions.db` — it's recreated empty on next
startup.

Score a file:

```bash
curl -F "file=@services/live-call-api/samples/genuine_tone.wav" \
     -F 'context={"known_number": true, "hour_of_day": 14}' \
     http://localhost:8020/v1/score/file | python3 -m json.tool
```

`context` is optional JSON — see `app/api/schemas.py::CallContext` for the
full field list (`known_number`, `hour_of_day`, `is_financial_request`,
`urgency_keywords`, `claimed_identity`).

Split a recording by speaker and score each one separately (file-upload
only — see `docs/architecture.md`, "Diarization"):

```bash
curl -F "file=@services/live-call-api/samples/genuine_tone.wav" \
     -F "diarize=true" \
     http://localhost:8020/v1/score/file | python3 -m json.tool
```

The response's `speakers` array (`null` unless `diarize=true`) has one
entry per detected voice, each with its own `final`/`trace` — same shape
as the whole-call result, and independently browsable later from the
History tab (`session_id` suffixed `::speaker_1`, `::speaker_2`, ...). Also
available from the dashboard's Upload tab via the "Detect multiple
speakers" checkbox.

## Voiceprint enrollment

The third signal's `consistency` mode (see `docs/risk-model.md`) compares a
live call against an enrolled voiceprint, keyed by `claimed_identity`.
Enroll one:

```bash
curl -F "identity=alice" -F "file=@enroll_sample.wav" \
     http://localhost:8020/v1/enroll
```

Then include `"claimed_identity": "alice"` in `context` on any
`/v1/score/file` or `/v1/stream/{id}` call to have that signal compare
against it instead of abstaining. List or remove enrollments:

```bash
curl http://localhost:8020/v1/enrollments | python3 -m json.tool
curl -X DELETE http://localhost:8020/v1/enrollments/alice
```

Only the derived embedding vector is stored (never raw audio) in
`services/live-call-api/data/voiceprints.db` — same docker-compose volume
as session history. The dashboard's **Voiceprints** tab wraps all three
endpoints, plus a "Claimed caller identity" field in Call details on the
Upload/Microphone tabs.

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
