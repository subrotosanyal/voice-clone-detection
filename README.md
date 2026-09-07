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

Then, in another terminal:

```bash
curl -F "file=@services/live-call-api/samples/genuine_tone.wav" \
     -F 'context={"known_number": true, "hour_of_day": 14}' \
     http://localhost:8020/v1/score/file | python3 -m json.tool
```

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
    ports/                      # the interfaces — detector.py, fusion.py
    adapters/                   # concrete detectors + fusion + the plugin registry
    pipeline/                   # windowing + orchestration (engine.py)
    api/                        # FastAPI routes (REST + WebSocket)
  config/risk_formula.yaml      # the formula — weights, thresholds, third-signal mode
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
pipeline). The detectors shipped are honest, deterministic **v0
baselines** — real DSP, not fabricated numbers, but explicitly not
validated spoof detectors yet (each one says so in its own docstring).
Swapping in a real pretrained countermeasure (AASIST, RawGAT-ST) is a
config + one adapter class change, not a rewrite — see
`docs/risk-model.md`, "What's a placeholder right now".

Not built yet: the Indian-language dataset/eval pipeline, the operator
dashboard UI, the mock banking approval flow, voiceprint enrollment. See
`docs/architecture.md`, "What's not built yet".
