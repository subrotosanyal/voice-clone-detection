# Architecture

## Layering

```
domain/     framework-free types (AudioWindow, DetectorResult, FusedScore, Band,
            SpeakerSegment, EnrollmentSummary)
ports/      interfaces — DetectorPort, FusionPort, HistoryStorePort,
            EnrollmentStorePort, DiarizerPort, TranscriberPort,
            IntentClassifierPort. Nothing outside adapters/ ever imports a
            concrete detector/fusion/diarizer/transcriber/intent-classifier
            class directly.
adapters/   concrete implementations of the ports, plus the registry that
            builds them from config/risk_formula.yaml by dotted path.
            embeddings/     shared ECAPA-TDNN speaker-embedding extractor,
                            used by both the voiceprint consistency detector
                            and the diarizer (not itself a port — see below).
            transcription/  Whisper-based TranscriberPort implementation +
                            the transparent urgency/financial-language/
                            authority-claim keyword detector it feeds (see
                            "Transcription" below).
            intent/         zero-shot IntentClassifierPort implementation —
                            active by default, weighted low, with a known
                            calibration caveat (see "Intent detection"
                            below).
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
  debugging** — no WebSocket client needed. Pass `diarize=true` to also
  split the call by speaker first — see "Diarization" below.
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
- **Voiceprints** tab → `POST /v1/enroll` (identity + a voice sample),
  `GET /v1/enrollments` to list, `DELETE /v1/enrollments/{identity}` per
  row. Enroll once, then set "Claimed caller identity" (in either tab's
  Call details) to that same name — the third signal's consistency mode
  compares the live voice against it instead of abstaining.
- **History** tab → `GET /v1/sessions` for the list, `GET
  /v1/sessions/{id}` to replay one (through the same rendering code as
  the other tabs — one render path, several ways to feed it), `DELETE
  /v1/sessions/{id}` per row.

Every component name and risk band in the results panel has a small "?"
icon (hover/focus) with a plain-language explanation — see `COMPONENT_HELP`
/ `BAND_HELP` in `app/ui/app.js` if you need to update the wording.

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
| `consistency` | Always use the voiceprint detector — real ECAPA-TDNN comparison, abstains until an identity is enrolled (see below). |
| `auto` (default) | Prefer consistency; fall back to contextual if it abstains. |

Full pros/cons of each mode are in the project blueprint's §03 and
`docs/risk-model.md`.

## Voiceprint consistency — real, enrollment-based

`app/adapters/detectors/voiceprint_consistency.py` (`VoiceprintConsistencyDetector`)
replaced `consistency_stub.py` as the active `consistency_class` in
`config/risk_formula.yaml` (the stub is still there, swappable back in via
the comment right above that config entry, for a dependency-light run with
no speechbrain/torch model download).

How it works: a live call window's speaker embedding — SpeechBrain's
ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, see
`app/adapters/embeddings/ecapa_embedding.py` for why this model, not
pyannote or Resemblyzer, was chosen) — is cosine-compared against the
embedding enrolled for `context.claimed_identity`. Low similarity -> higher
risk. It abstains whenever there's nothing to compare: no
`claimed_identity` in context, no enrollment on file for that identity, or
a near-silent window.

Enrollment (`app/adapters/enrollment/sqlite_enrollment_store.py`, backed by
`data/voiceprints.db`) stores only the derived embedding vector, never raw
audio — `POST /v1/enroll`, `GET /v1/enrollments`, `DELETE
/v1/enrollments/{identity}` (`app/api/enrollment_router.py`). These
endpoints reuse the SAME `VoiceprintConsistencyDetector` instance the
scoring pipeline already built (found by walking `pipeline.detectors` for
`ThirdSignalRouter.consistency_detector` in `app/main.py`'s lifespan) —
one loaded model, shared between scoring and enrollment, not two.

**Honesty note**: the similarity->risk mapping is a placeholder heuristic
(anchor points, not a calibrated classifier) — see the module's docstring.
The underlying embedding comparison is real; the exact risk thresholds are
not yet validated against a labeled dataset.

## Diarization — who spoke when

`app/ports/diarizer.py` (`DiarizerPort`) + `app/adapters/diarization/
embedding_cluster_diarizer.py` (`EmbeddingClusterDiarizer`) answer "how many
people spoke, and which stretches were whose" for an uploaded recording.
Deliberately NOT pyannote.audio (its pretrained pipelines are all gated on
HuggingFace — would need every user to manage an auth token just to build
the image). Instead: pause-aligned segments (see below), skip near-silent
ones, extract an ECAPA-TDNN embedding per segment (the same shared
extractor voiceprint consistency uses), agglomerative clustering
(`scipy.cluster.hierarchy`, cosine distance) groups same-voice segments,
consecutive same-cluster segments merge into a `SpeakerSegment`.

Wired into `POST /v1/score/file` only, via an optional `diarize=true` form
field (`app/api/http_router.py::_diarize_and_score`) — **not** the live
WebSocket path, where diarization would mean discovering speaker clusters
incrementally from partial audio, a substantially harder problem left for
later. When requested, each detected speaker's audio is concatenated and
run through the exact same `Engine.score_call()` as a normal whole-call
score, under a derived session id (`{session_id}::{speaker_label}`) — so
every per-speaker window still flows through the normal fusion + history-
save machinery, and each speaker's trace is independently browsable later
from the History tab like any other session.

**Honesty note**: the clustering distance threshold and segment length were
recalibrated 2026-09-08, twice, after a real over-counting bug (one
speaker's natural voice variation was splitting into several — a real
report saw over 100 phantom speakers for one call). The first same-day fix
moved the threshold the wrong direction and made it worse, confirmed
immediately by that same report; see the module's CALIBRATION HISTORY
docstring note for the full account. A `max_speakers` cap (default 8) was
added as a real safety net independent of whatever the distance threshold
turns out to be wrong about — this diarizer will never report more
speakers than that, regardless. The threshold itself remains an
unvalidated guess (no real labeled multi-speaker corpus exists in this
repo). Real signal processing, reproducible, not turn-by-turn
transcription-grade diarization.

**Segmentation redesigned, same day**: fixed-length segmentation (a plain
clock, no voice-activity/change-point front end) turned out to have a
second, distinct over-counting mechanism — a segment straddling a real
speaker change produces an embedding resembling NEITHER speaker (measured
by hand: cosine similarity 0.04-0.34 against either pure voice, far below
the ~0.67-0.70 normal cross-speaker baseline), a phantom cluster of its
own. `app/adapters/diarization/pause_segmentation.py` now cuts boundaries
at real acoustic pauses detected by Praat's own silence detector (no new
dependency — Parselmouth is already required for the prosodic detector),
so a segment no longer straddles a detected turn; `max_segment_ms` is now
only a cap for sub-slicing long uninterrupted stretches, not the primary
segmentation unit. Verified by hand: a ~150ms gap between speakers is
reliably detected; a genuine zero-gap turn-take (immediate back-to-back
speech, or overlap) is not — no acoustic silence exists for any pause-based
method to find there, so `max_segment_ms` bounds (not eliminates) the
damage from that case, same as the old fixed-grid fallback. See
`pause_segmentation.py`'s own HONESTY NOTE for the measurements.

**Also fixed the same day**: diarized calls were re-transcribing the
whole call once per detected speaker (N+1 Whisper calls for N speakers,
compounding badly whenever the diarizer over-counted) — `Engine.
score_call()`'s transcription guard only skips re-transcribing when the
context it's given already carries a `transcript`, and each per-speaker
call in `_diarize_and_score()` previously didn't. `app/api/http_router.py`
now transcribes once and passes the same context to every score_call
below it, whole-call and per-speaker alike.

## Transcription — feeding real call language into the contextual signal

`app/ports/transcriber.py` (`TranscriberPort`) + `app/adapters/transcription/
whisper_transcriber.py` (`WhisperTranscriber`) answer "what did the caller
actually say" for an uploaded recording, using OpenAI's Whisper (MIT,
confirmed ungated). Wired into `POST /v1/score/file` only (see the port's
scope note — same file-upload-only reasoning as diarization), the whole
buffer is transcribed once by `Engine.score_call()` and merged into
`context["transcript"]` before windowing, so every window's third signal
sees the same transcript.

**Model size — verified by hand, not assumed**: the smaller "base" model
mis-transcribed real Hindi speech into Urdu script (a real, reproducible
finding, not a guess — the two languages are spoken near-identically but
written differently, and "base" isn't reliable enough to keep them
straight). "small" was tested on the same audio and got it right, so
that's the default — still fast on CPU (~1-2s once loaded).

`app/adapters/transcription/urgency_language.py` then scans the transcript
for fraud-relevant language — urgency phrases, financial-request words,
and authority-claim phrases ("this is your bank", "cyber crime cell"), in
English/Hindi/Marathi — and `ContextualRulesDetector`
(`app/adapters/detectors/contextual_rules.py`) merges whatever it finds
with any manually-supplied `urgency_keywords`/`is_financial_request`/
`authority_claim`, never replacing them. An authority claim paired with a
financial request also fires its own `combined_authority_financial_pressure`
rule — the classic fraud script, scored explicitly rather than left as an
implicit sum. Every match is traceable to an actual word in the actual
transcript, shown in the component's `detail` — see `docs/risk-model.md`
for why a transparent keyword list was chosen here over a black-box
sentiment model.

**Not built**: transcription on the live WebSocket path (same "substantially
harder, deferred" reasoning as live diarization), and the keyword lists
themselves aren't validated against real fraud-call transcripts — see the
module's own honesty note.

## Intent detection — built, verified, active with a known caveat

`app/ports/intent_classifier.py` (`IntentClassifierPort`) +
`app/adapters/intent/zero_shot_intent_classifier.py`
(`ZeroShotIntentClassifier`) classify a transcript against a fixed list of
fraud-relevant candidate labels using zero-shot NLI classification
(`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, MIT, confirmed ungated, real
Hindi coverage via XNLI) — meant to generalise beyond
`urgency_language.py`'s exact keyword matches, while staying explainable
(every score traces to "the model judged this X% consistent with
candidate Y", for every candidate — and unlike most components, the
dashboard shows the FULL breakdown, every candidate's own score, not just
the winner; see "Intent" in the browser dashboard).

Wired the same way as transcription: `Engine.score_call()` (and
`http_router.py`, for the diarization fan-out) computes the classification
ONCE per call and merges `intent_label`/`intent_top_score`/
`intent_label_scores` into `context`; `app/adapters/detectors/
intent_risk.py` (`IntentRiskDetector`) only ever reads those precomputed
fields, never calling the model itself — an NLI forward pass is
expensive, and calling it once per 2-second window would repeat the exact
class of redundant-computation bug fixed the same day for
transcription-during-diarization (see "Diarization" above).

**Why this is enabled despite a real calibration problem**: a hand-run
check (a fraud script, an impersonation script, two ordinary sentences)
found the model can misclassify completely benign text as fraud-relevant,
under three different configurations tried — see
`zero_shot_intent_classifier.py`'s HONESTY NOTE for the full account. It
was enabled anyway, at explicit user instruction, with two real
mitigations rather than none: a deliberately low fusion weight (0.15,
auto-renormalised against the others — see "Fusion" above) so it can't
single-handedly push a genuine call into High, and full UI transparency
(the dashboard shows every candidate label's own score, so a false
positive here is visible and inspectable, not hidden inside one number).
See `docs/risk-model.md`, "Intent detection" for the full writeup.

## Concurrency: keeping the event loop free during a long score

`Engine.score_call()`/`score_window()` are plain synchronous, CPU-bound
calls (AASIST + Parselmouth + maybe ECAPA-TDNN inference, run per window).
Calling them directly inside an `async def` FastAPI route would block
uvicorn's single event loop for however long that takes — for a long
uploaded recording, potentially minutes — during which even a trivial
concurrent `GET /healthz` or another user's request would hang.

`app/api/http_router.py` (`POST /v1/score/file`, including the diarize
fan-out) and `app/api/ws_router.py` (`WS /v1/stream/{id}`) both offload
these calls via FastAPI's `run_in_threadpool`, so the event loop stays free
to serve other requests/WS sessions while one is still crunching. Verified
by hand: a 20s clip took ~11s to score, and `/healthz` stayed sub-100ms
throughout (see `docs/testing.md`).

This introduces real concurrent calls into libraries not all originally
written with that in mind:
- PyTorch (AASIST) — safe: CPU inference under `torch.no_grad()` from
  multiple threads on one model instance is a standard, documented
  serving pattern.
- SpeechBrain's `EncoderClassifier` (ECAPA-TDNN) — not documented as
  thread-safe, so `EcapaEmbeddingExtractor` (`app/adapters/embeddings/
  ecapa_embedding.py`) serialises calls with its own lock.
- Parselmouth/Praat — Praat's C++ core predates any notion of being called
  from multiple threads; `prosody_parselmouth.py` serialises its calls
  with a module-level lock for the same reason.

`SessionStore`'s per-session EMA dict and both SQLite stores' single-lock
connections were already safe for this (see their own docstrings).

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
one 2019 dataset. It had no exposure to Hindi or a held-out generator at
training time — measuring that gap is exactly the work the project
blueprint's §04 describes, and it has partially happened: see
`eval/indian_language/README.md` for a real first EER number on Hindi and
what's still missing (scope there is English + Hindi only, for now). It
also has no exposure
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

`prosody_pitch_variance.py` (the autocorrelation heuristic) is still there,
swappable back in via config, but is no longer the default — see "The
prosodic detector: Parselmouth" below.

## The prosodic detector: Parselmouth

`app/adapters/detectors/prosody_parselmouth.py` (`ParselmouthProsodyDetector`)
replaced `prosody_pitch_variance.py` as the default `prosodic` class in
`config/risk_formula.yaml` (the old one is still there, swappable back in
via the comment above that config entry, for a Praat-free run). It computes
jitter, shimmer, and harmonics-to-noise ratio via Parselmouth (the official
Praat Python binding, GPLv3) — the same validated acoustic-phonetic
algorithms used in clinical voice-quality research, a meaningfully more
validated *front end* than the previous autocorrelation coefficient-of-
variation estimate.

**Honesty note**: the jitter/shimmer/HNR features themselves are real and
Praat-validated; the risk-score MAPPING built on top of them
(`_JITTER_FLOOR`/`_SHIMMER_FLOOR`/`_HNR_CEILING` in the module) is still an
unvalidated heuristic hypothesis ("unnaturally smooth voice = suspicious"),
not a trained classifier — same caveat class as everywhere else in this
repo that says so.

## What's not built yet

- **Real-time diarization on the live WebSocket path.** File-upload
  diarization is built (see "Diarization" above); doing the same
  incrementally, from partial streamed audio, is a substantially harder
  problem and hasn't been attempted.
- **Persistent *history* is built (above); persistent *live smoothing
  state* is not** — these are two different things. `Engine.sessions`
  (the EMA state used mid-call, in `SessionStore`) is still in-memory
  only, so a restart mid-call resets that call's smoothing to a fresh
  start. Fine for one process; swap `SessionStore` for Redis when more
  than one API replica needs to share an *in-progress* call's state.
- **A genuinely held-out second synthesis system for the Indian-language
  eval (§04).** Scope there is deliberately **English and Hindi only, for
  now** — Marathi and Malvi were dropped as required deliverables (see the
  scope note in `eval/indian_language/README.md`). The pipeline itself is
  built and has run end to end (`eval/indian_language/`): real, license-
  verified CC BY 4.0 Hindi genuine speech, real XTTS-v2-cloned Hindi spoof
  audio, and an actual fine-tune of AASIST's final layer, giving a first
  real number — 10% EER for Hindi, unchanged before/after fine-tuning (an
  honest finding, not an improvement). What's still missing: a second
  synthesis system never used in training (without it, "after fine-tuning"
  is only measured on data the model already saw), and an independently-
  measured English baseline EER against a public benchmark. See
  `eval/indian_language/README.md`.
- The mock banking approval flow (the dashboard UI itself is built —
  see above).
- Calibrated risk thresholds for voiceprint consistency, diarization
  clustering, and the Parselmouth prosodic mapping — all three are real
  signal-processing/model pipelines with honestly-documented placeholder
  heuristic score mappings, not yet tuned against a labeled dataset.

None of this changes the shape of `services/live-call-api/app/` — each
item above is a new adapter (or a new service) behind an existing or new
port.
