# Satya-Vani — Live Call Path

Real-time voice-spoof risk scoring: listens to a call, computes a
continuously-updating risk score from independent, pluggable signals, and
tells you exactly why the score is what it is. This repo is the working
implementation of the Satya-Vani solution blueprint (SIH26104) — right
now it implements just the **Live Call Path** (the runtime detection
pipeline). See `docs/architecture.md` for what's built vs. what's still on
the roadmap.

## Quickstart

```bash
docker compose up --build
```

Then open **http://localhost:8020/** in a browser — a dashboard to upload a
recording, speak into your microphone and watch the risk score move
live, or look back through every past session's full explainability
breakdown — no `curl` needed. If you'd rather script it:

```bash
curl -F "file=@services/live-call-api/samples/genuine_tone.wav" \
     -F 'context={"known_number": true, "hour_of_day": 14}' \
     http://localhost:8020/v1/score/file | python3 -m json.tool
```

- **Dashboard:** http://localhost:8020/
- **API docs (interactive):** http://localhost:8020/docs
- **Central log viewer (Dozzle):** http://localhost:8888
- **Current live formula:** http://localhost:8020/v1/config

Host port is **8020**, not 8000 — chosen to avoid clashing with other
projects on a shared dev machine. Change it in `docker-compose.yml` if
8020 is also taken on yours.

## What this is, in one paragraph

Audio is cut into overlapping 2-second windows. Each window is scored by
several independent, pluggable **detectors** (acoustic, prosodic, and a
config-selectable "third signal" — rule-based context, or voiceprint
consistency once that's built). A **fusion** step combines their scores
into one 0–100 risk number with a full, human-readable breakdown of
exactly which signal contributed what. Nothing here is a black box —
see `docs/risk-model.md`.

## Repo layout

```
docker-compose.yml              # docker compose up — the whole thing
services/live-call-api/         # the one service that exists so far
  app/
    domain/models.py            # framework-free core types
    ports/                      # the interfaces — detector.py, fusion.py, history_store.py
    adapters/                   # concrete detectors + fusion + history store + the plugin registry
    pipeline/                   # windowing + orchestration (engine.py)
    api/                        # FastAPI routes (REST + WebSocket + session history)
    ui/                         # the browser dashboard (no build step — plain HTML/JS)
  config/risk_formula.yaml      # the formula — weights, thresholds, third-signal mode
  data/sessions.db              # session history (SQLite, gitignored — fetched/created at runtime)
  tests/                        # unit + integration, see docs/testing.md
scripts/gen_test_audio.py       # synthetic (non-voice) smoke-test fixtures
docs/                           # architecture, risk model, running, testing
```

## Adding a new risk factor

1. Write a class implementing `DetectorPort` (`app/ports/detector.py`) —
   copy `app/adapters/detectors/contextual_rules.py`, it's the simplest one.
2. Add it to `detectors:` in `config/risk_formula.yaml` with a name and a
   weight.
3. Restart the container. Nothing else changes.

Full details: `docs/architecture.md`.

## Status

This implements the **Live Call Path** only (the runtime detection
pipeline). The acoustic detector now wraps a real pretrained
countermeasure — **AASIST** (clovaai/aasist, MIT licensed, ASVspoof2019
LA) — fetched and checksum-verified automatically (`docker compose
up --build` handles it; see `docs/running-locally.md` for running
without Docker). It's real and reproducible, but trained on English
speech only — see `docs/risk-model.md` for exactly what it does and
doesn't prove. The prosodic detector is still a heuristic v0 baseline
(documented in its own docstring); a lightweight, torch-free acoustic
fallback is still available via a one-line config swap.

A browser dashboard exists (upload, live microphone, and session
history — every past score kept and replayable, persisted across
restarts via a SQLite-backed history store). Not built yet: the
Indian-language dataset/eval pipeline, the mock banking approval flow,
voiceprint enrollment. See `docs/architecture.md`, "What's not built yet".
