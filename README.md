# Satya-Vani — Live Call Path

**[CI status](https://github.com/subrotosanyal/voice-clone-detection/actions/workflows/ci.yml)** — the inline badge image doesn't render for private repos (GitHub only serves it to an authenticated viewer with access; confirmed by hand, not a bug in the workflow itself — both jobs are passing, see the link).

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
financial-language/authority-claim keyword detection — an authority claim
plus a financial request together fires its own explicit combined-pressure
rule — or real ECAPA-TDNN voiceprint consistency). A **fusion** step
combines their scores into one 0–100 risk number with a full,
human-readable breakdown of exactly which signal contributed what, plus a
one-sentence plain-language explanation of *why* under each component. On
file uploads, an optional **diarizer** can also split a multi-speaker
recording and score each voice separately. Nothing here is a black box —
see `docs/risk-model.md`.

A fifth, **zero-shot intent classifier** (real model, real MIT license,
real Hindi coverage) scores the transcript against fraud-relevant
candidate labels — enabled by default with a deliberately low weight and
full UI transparency (every candidate label's own score is shown, not
just the winner), because a real calibration check found it can
misjudge ordinary conversation. See `docs/risk-model.md`, "Intent
detection" for the honest account and why it shipped anyway.

## Repo layout

```
docker-compose.yml              # docker compose up — the whole thing
services/live-call-api/         # the one service that exists so far
  app/
    domain/models.py            # framework-free core types
    ports/                      # the interfaces — detector.py, fusion.py, history_store.py,
                                 # enrollment_store.py, diarizer.py, transcriber.py,
                                 # intent_classifier.py
    adapters/                   # concrete detectors + fusion + history/enrollment stores +
                                 # diarizer + transcription/ (Whisper + urgency/authority-
                                 # claim keyword detection) + intent/ (zero-shot classifier,
                                 # active by default, known calibration caveat) + the shared
                                 # ECAPA-TDNN embedding extractor + the plugin registry
    pipeline/                   # windowing + orchestration (engine.py) + live transcription
                                 # and live diarization buffers for the WebSocket path
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
  **Whisper** (MIT) transcribes each uploaded call once (and, on the live
  microphone path, periodically in the background as the call proceeds —
  see `docs/architecture.md`, "Live transcription"), and a transparent
  keyword list flags urgency/financial-request/authority-claim language in
  English, Hindi, and Marathi — no manual keyword entry required, though
  you still can. An authority claim ("this is your bank") paired with a
  financial request fires its own explicit combined-pressure rule.
- **Intent classification** — a zero-shot NLI classifier
  (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, MIT, real Hindi coverage)
  that scores the transcript against fraud-relevant candidate labels, on
  both call paths. Enabled by default with a deliberately low weight
  (0.15) — a real calibration check found it can misclassify ordinary
  conversation as fraud-relevant, and the dashboard shows every candidate
  label's own score for exactly that reason (see `docs/risk-model.md`).
- **Watermark check** — checks for Resemble AI's **Perth** neural
  watermark (MIT, `resemble-perth`), the fingerprint Chatterbox and other
  Perth-integrated cloning tools embed in every clip. A narrow,
  high-precision complement to Acoustic — verified by hand that genuine
  speech and a different, non-Perth TTS system both read near-zero, while
  Perth-watermarked audio reads 1.0 (see `docs/risk-model.md`).

Every model/checkpoint is fetched and cached at `docker compose up --build`
time — see `docs/running-locally.md` for running without Docker. See
`docs/risk-model.md` for exactly what each does and doesn't prove.
Dependency-light, torch/Praat-free fallbacks remain available for the
acoustic and prosodic detectors via a one-line config swap (see the
comments in `config/risk_formula.yaml`).

**Diarization**: `POST /v1/score/file?diarize=true` splits a
multi-speaker recording by voice (embedding clustering, ECAPA-TDNN) and
scores each speaker separately, alongside the whole-call score. The live
microphone/WebSocket path has its own, genuinely different **live
diarization** (added 2026-09-09): an incremental "who's talking now"
tracker (`app/pipeline/live_diarization.py`) that discovers speakers one
window at a time as the call progresses — same ECAPA-TDNN embeddings,
a simpler nearest-centroid-with-a-threshold algorithm instead of
clustering a complete recording. Purely informational (doesn't change
the risk score), and not yet calibrated against a real labeled
multi-speaker corpus — see `docs/architecture.md`, "Live diarization".

A browser dashboard exists (upload with optional diarization, live
microphone, voiceprint enrollment, and session history — every past score
kept and replayable, persisted across restarts via a SQLite-backed history
store). Every detector name and risk band has an inline "?" tooltip
explaining what it means.

The **Indian-language dataset/eval pipeline** (`eval/indian_language/`)
targets **English and Hindi only, for now** — Marathi and Malvi are not
required scope (a deliberate narrowing; see `eval/indian_language/README.md`
for the reasoning). It has genuinely started, not just been planned: real,
license-verified (CC BY 4.0) Hindi genuine speech, real XTTS-v2-cloned
Hindi spoof audio, and an actual fine-tune of AASIST's final layer on that
data, at real volume (200 genuine + 40 spoof examples) — **17.75% → 3.0%
EER** on the trained-on synthesis system (XTTS-v2), after finding and
fixing two real fine-tuning bugs (class imbalance, BatchNorm/Dropout
statistics drift — see that README for the full account). A second,
genuinely held-out synthesis system (**Resemble AI's Chatterbox**, never
used in training) confirms this generalises, not just memorises XTTS-v2:
**17.75% → 10.25% EER** held-out. This recalibration is the same one wired
into production above ("Acoustic"). Still missing: an
independently-measured English baseline EER against a public benchmark,
and more held-out spoof volume (currently 40 examples) to make that
held-out number statistically solid rather than calibration-scale. See
`eval/indian_language/README.md` for the full pipeline and results.

Not built yet: the mock banking approval flow, and calibrated risk
thresholds for the newer detectors. See `docs/architecture.md`, "What's
not built yet".
