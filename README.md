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
several independent, pluggable **detectors** (acoustic — AASIST; prosodic —
Parselmouth/Praat; and a config-selectable "third signal" — rule-based
context, fed by real **Whisper** transcription and transparent urgency/
financial-language keyword detection, or real ECAPA-TDNN voiceprint
consistency). A **fusion** step combines their scores into one 0–100 risk
number with a full, human-readable breakdown of exactly which signal
contributed what. On file uploads, an optional **diarizer** can also split
a multi-speaker recording and score each voice separately. Nothing here is
a black box — see `docs/risk-model.md`.

## Repo layout

```
docker-compose.yml              # docker compose up — the whole thing
services/live-call-api/         # the one service that exists so far
  app/
    domain/models.py            # framework-free core types
    ports/                      # the interfaces — detector.py, fusion.py, history_store.py,
                                 # enrollment_store.py, diarizer.py, transcriber.py
    adapters/                   # concrete detectors + fusion + history/enrollment stores +
                                 # diarizer + transcription/ (Whisper + urgency-keyword
                                 # detection) + the shared ECAPA-TDNN embedding extractor +
                                 # the plugin registry
    pipeline/                   # windowing + orchestration (engine.py)
    api/                        # FastAPI routes (REST + WebSocket + session history + enrollment)
    ui/                         # the browser dashboard (no build step — plain HTML/JS)
  config/risk_formula.yaml      # the formula — weights, thresholds, third-signal mode, diarizer
  data/sessions.db              # session history (SQLite, gitignored — fetched/created at runtime)
  data/voiceprints.db           # enrolled voiceprint embeddings (SQLite, gitignored)
  tests/                        # unit + integration, see docs/testing.md
scripts/gen_test_audio.py       # synthetic (non-voice) smoke-test fixtures
docs/                           # architecture, risk model, running, testing
docs/blueprint.html             # the original SIH26104 planning blueprint, annotated
                                 # with real build status + a progress Gantt chart (§00)
eval/indian_language/           # offline pipeline: Indian-language dataset + held-out
                                 # eval (blueprint §04) — see its own README for status
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
pipeline). All three scoring detectors now wrap real, published
implementations, not hand-rolled heuristics alone:

- **Acoustic** — **AASIST** (clovaai/aasist, MIT licensed, ASVspoof2019
  LA), fetched and checksum-verified automatically. Real and reproducible,
  but trained on English speech only.
- **Prosodic** — **Parselmouth/Praat**-derived jitter, shimmer, and HNR —
  real validated voice-quality features; the risk mapping on top is still
  an honestly-documented heuristic.
- **Third signal (consistency mode)** — real **ECAPA-TDNN**
  (`speechbrain/spkrec-ecapa-voxceleb`, Apache-2.0) voiceprint comparison
  against an enrollment (`POST /v1/enroll`); abstains gracefully until
  someone is enrolled.
- **Third signal (contextual mode)** — now fed by real transcription:
  **Whisper** (MIT) transcribes each uploaded call once, and a transparent
  keyword list flags urgency/financial-request language in English, Hindi,
  and Marathi — no manual keyword entry required, though you still can.

Every model/checkpoint is fetched and cached at `docker compose up --build`
time — see `docs/running-locally.md` for running without Docker. See
`docs/risk-model.md` for exactly what each does and doesn't prove.
Dependency-light, torch/Praat-free fallbacks remain available for the
acoustic and prosodic detectors via a one-line config swap (see the
comments in `config/risk_formula.yaml`).

**Diarization** (file-upload only): `POST /v1/score/file?diarize=true`
splits a multi-speaker recording by voice (embedding clustering, reusing
the same ECAPA-TDNN model) and scores each speaker separately, alongside
the whole-call score. Not attempted on the live microphone/WebSocket path
— see `docs/architecture.md`, "What's not built yet".

A browser dashboard exists (upload with optional diarization, live
microphone, voiceprint enrollment, and session history — every past score
kept and replayable, persisted across restarts via a SQLite-backed history
store). Every detector name and risk band has an inline "?" tooltip
explaining what it means.

The **Indian-language dataset/eval pipeline** (`eval/indian_language/`) has
genuinely started, not just been planned: real, license-verified (CC BY
4.0) Hindi and Marathi genuine speech, real XTTS-v2-cloned Hindi spoof
audio, and an actual fine-tune of AASIST's final layer on that data — first
real number: 10% Equal Error Rate for Hindi, unchanged before and after
fine-tuning (an honest "re-balanced, not improved" finding given only ~50
examples, not a fabricated improvement). Still missing: a second, genuinely
held-out synthesis system (needed to prove generalisation rather than
recalibration), a synthetic/spoof Marathi corpus, and any Malvi data at
all. See `eval/indian_language/README.md` for the full pipeline and
results.

Not built yet: the mock banking approval flow, and calibrated risk
thresholds for the newer detectors. See `docs/architecture.md`, "What's
not built yet".
