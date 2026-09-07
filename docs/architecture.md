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

## Session history

Every window's `FusedScore` — the full explainability breakdown, not just
the number — is persisted right after fusion, via `HistoryStorePort`
(`app/ports/history_store.py`). `app/pipeline/engine.py` calls
`history.save(fused)` after every `score_window()`, wrapped so a storage
failure is logged (`history_save_failed`) but never breaks the live
scoring path — a session you can't look up later is a much smaller
problem than a call that fails to score at all.

The default implementation is `SqliteHistoryStore`
(`app/adapters/history/sqlite_store.py`) — one file
(`data/sessions.db`, path configurable via `HISTORY_DB_PATH`), no server
process, no extra dependency (`sqlite3` is in the standard library).
Mounted as a docker-compose volume so it survives `docker compose up
--build` and container restarts — see `docker-compose.yml`. This is
enough for a single-instance demo; the moment more than one API replica
needs to share history, swap this for a Postgres-backed implementation of
the same port — nothing in `Engine` or the API routes changes.

Three endpoints read it (`app/api/history_router.py`):

- `GET /v1/sessions` — most recent sessions first, summarised (window
  count, time range, final score/band) — cheap enough to call on every
  page load.
- `GET /v1/sessions/{id}` — the full trace, same shape as
  `POST /v1/score/file`'s response, so the UI replays a past session
  through the identical rendering code as a live one.
- `DELETE /v1/sessions/{id}` — no automatic retention/TTL yet (a known
  gap, consistent with the honesty pattern elsewhere in this repo) —
  deletion today is manual, from the History tab or this endpoint.

## The browser dashboard (`app/ui/`)

A plain HTML/CSS/JS page — no build step, no framework, no npm — mounted
at `/` via Starlette's `StaticFiles(html=True)` in `app/main.py`. It's a
real client of the entrypoints above, nothing more:

- **Upload a file** tab → `POST /v1/score/file`, then replays the
  returned `trace` array through the same rendering code a live session
  uses.
- **Microphone (live)** tab → captures mic audio via the Web Audio API
  (`AudioContext` + `ScriptProcessorNode`, chosen over the more modern
  `AudioWorkletNode` because it needs no separate module file — a
  reasonable trade-off for an internal tool), windows it client-side to
  match `config/risk_formula.yaml`'s `window_ms`/`hop_ms`, and streams
  each window over `WS /v1/stream/{session_id}` exactly like a real
  telephony bridge would.
- **History** tab → `GET /v1/sessions` for the list, `GET
  /v1/sessions/{id}` to replay one (through the same rendering code as
  the other two tabs — one render path, three ways to feed it), `DELETE
  /v1/sessions/{id}` per row.

Mount order matters: `app.mount("/", StaticFiles(...))` is registered
**after** the API routers in `app/main.py`, specifically so `/healthz`,
`/v1/*`, and FastAPI's own `/docs`/`/redoc` all resolve first — the
static mount only ever catches what's left (`/`, `/app.js`). See
`tests/integration/test_ui_served.py`, which guards this ordering.

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
- **Persistent *history* is built (above); persistent *live smoothing
  state* is not** — these are two different things. `Engine.sessions`
  (the EMA state used mid-call, in `SessionStore`) is still in-memory
  only, so a restart mid-call resets that call's smoothing to a fresh
  start. Fine for one process; swap `SessionStore` for Redis when more
  than one API replica needs to share an *in-progress* call's state.
- The Indian-language dataset/held-out-generator evaluation pipeline —
  including fine-tuning this same AASIST model on that data, per §04.
- The mock banking approval flow (the dashboard UI itself is built —
  see below).
- A real model behind the prosodic detector.

None of this changes the shape of `services/live-call-api/app/` — each
item above is a new adapter (or a new service) behind an existing or new
port.
