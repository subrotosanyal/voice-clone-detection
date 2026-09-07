# Architecture

## Layering

```
domain/     framework-free types (AudioWindow, DetectorResult, FusedScore, Band)
ports/      interfaces — DetectorPort, FusionPort. Nothing outside adapters/
            ever imports a concrete detector or fusion class directly.
adapters/   concrete implementations of the ports, plus the registry that
            builds them from config/risk_formula.yaml by dotted path.
pipeline/   windowing (audio -> AudioWindow) and the engine that
            orchestrates detectors -> fusion -> logging for one window.
api/        FastAPI routes. Thin — every route just calls into pipeline/.
```

This is the same ports-and-adapters shape as the rest of the Satya-Vani
project's reference services. The rule that keeps it pluggable: **nothing
outside `app/adapters/registry.py` ever writes the name of a concrete
detector or fusion class.** Everything else talks to `DetectorPort` /
`FusionPort`.

## The live-call data flow

```
   audio bytes
        |
   [windowing.py]  -- 2s window, 0.5s hop -->  AudioWindow
        |
   [engine.py]  calls every configured detector, in config order
        |
        +--> acoustic detector    --> DetectorResult (score or abstain)
        +--> prosodic detector    --> DetectorResult
        +--> third-signal router  --> DetectorResult (contextual | consistency | auto)
        |
   [fusion: weighted_sum.py]
        |  renormalises weights of non-abstained detectors,
        |  EMA-smooths against the session's previous score,
        |  bands the result (low/elevated/high)
        v
   FusedScore  (full breakdown)  --> logged (structlog, JSON) --> returned to caller
```

Two entrypoints call the exact same `Engine.score_window()`:

- `POST /v1/score/file` — windows a whole uploaded file and returns every
  window's score plus the final one. **This is the one to use for
  debugging** — no WebSocket client needed.
- `WS /v1/stream/{session_id}` — one message in, one score out, for a real
  live client (a browser capturing mic audio, or a telephony bridge).

## Adding a new risk factor

1. Implement `DetectorPort` (`app/ports/detector.py`):

   ```python
   class MyDetector:
       name = "my_detector"
       version = "0.1.0"

       def score(self, window, context):
           # return DetectorResult(detector_name=self.name, detector_version=self.version,
           #                        score=0.0-1.0 or None, detail={...})
           ...
   ```

2. Add it to `config/risk_formula.yaml`:

   ```yaml
   detectors:
     - name: my_detector
       class: app.adapters.detectors.my_module:MyDetector
       weight: 0.1
       params: {}
   ```

   (Reduce the other weights so they still make sense together — the
   fusion step doesn't require weights to sum to 1, but a formula that's
   easy to reason about usually should.)

3. Restart the service. `GET /v1/config` will show it; `POST
   /v1/score/file` will include it in `components`.

No change to `pipeline/engine.py`, `app/api/*.py`, or any other detector —
that's the whole point of the port.

## Changing the fusion formula itself

Weights are a config change (above). Changing the *math* — different
smoothing, a non-linear combination, a learned meta-model instead of a
weighted sum — means a new class against `FusionPort`
(`app/ports/fusion.py`), pointed at from `fusion.class` in
`config/risk_formula.yaml`. `WeightedSumFusion` is not special; it's just
the default.

## The pluggable third signal

`app/adapters/detectors/third_signal_router.py` is itself a `DetectorPort`
— the engine and registry never special-case it. It composes two other
detectors (contextual rules, voiceprint consistency) and picks between
them per `third_signal.params.mode` in config:

| Mode | Behaviour |
|---|---|
| `contextual` | Always use the rule-based detector (no enrollment needed). |
| `consistency` | Always use the voiceprint detector (currently a stub — see below). |
| `auto` (default) | Prefer consistency; fall back to contextual if it abstains. |

Full pros/cons of each mode are in the project blueprint's §03.

## Central logging

Every log line is one JSON object on stdout (`app/logging_setup.py`,
built on `structlog`). `docker compose logs -f live-call-api` gives you
raw JSON lines you can pipe through `jq`; **Dozzle**
(`http://localhost:8888`) gives a zero-config web UI over the same
container logs with no log-shipping infrastructure to run. Every
`risk_score_computed` log line carries the *entire* FusedScore
breakdown — you can debug a score from the logs alone, without
re-running anything.

To move to a real log-shipping stack later (Loki+Grafana, ELK): point a
log driver or a sidecar at the containers' stdout. Application code
doesn't change — it only ever calls `logging_setup.get_logger()`.

## The acoustic detector: a real pretrained model, not a heuristic

`app/adapters/detectors/acoustic_aasist.py` wraps **AASIST**
(clovaai/aasist, MIT licensed, NAVER Corp.), a real published spoof-
countermeasure model, trained on ASVspoof2019 LA. The model architecture
is vendored unmodified in `vendor/aasist_model.py` (only the outer class
was renamed, to keep `load_state_dict(strict=True)` matching the
checkpoint); the checkpoint itself is fetched and checksum-verified by
`vendor/fetch_checkpoint.py` — not committed to git, pulled at Docker
build time (or once locally before running tests without Docker).

**What it does and doesn't prove:** the model was trained to tell English
bonafide speech apart from the specific TTS/voice-conversion attacks in
one 2019 dataset. It has no exposure to Hindi, Marathi, or a held-out
generator — measuring that gap is exactly the work the project
blueprint's §04 describes, and hasn't happened. It also has no exposure
to non-speech audio: our own synthetic test fixtures (a sine tone, white
noise — see `scripts/gen_test_audio.py`) both score as ~99.99% "not
bonafide", because neither one is real speech to begin with. That's the
model working correctly, not a bug — but it does mean those fixtures no
longer demonstrate a meaningful genuine-vs-suspicious divergence now that
a real speech-trained classifier is in the loop. A real demo needs real
recorded speech (and a real cloned counterpart), which is exactly the
`docs/risk-model.md`, "What would change this" section describes.

The lightweight DSP baseline (`acoustic_spectral_flatness.py`) is still
there and still wired to the same `DetectorPort` — swap `config/
risk_formula.yaml`'s `acoustic` entry back to it for a torch-free, no-
download quick run (see the comment right above that entry in the file).

`prosody_pitch_variance.py` is still a heuristic v0 baseline — same
caveat as before, not yet swapped for a real model.

## What's not built yet

- Voiceprint enrollment + a real consistency detector (SASV-style
  AASIST+ECAPA-TDNN fusion is the reference implementation to start from —
  notably, the same AASIST checkpoint now wired in here is literally half
  of that fusion).
- Persistent session state (currently in-memory only — a restart resets
  every session's smoothing; fine for one process, not for multiple
  replicas — swap `SessionStore` for Redis when that matters).
- The Indian-language dataset/held-out-generator evaluation pipeline —
  including fine-tuning this same AASIST model on that data, per §04.
- The operator dashboard UI and the mock banking approval flow.
- A real model behind the prosodic detector.

None of this changes the shape of `services/live-call-api/app/` — each
item above is a new adapter (or a new service) behind an existing or new
port.
