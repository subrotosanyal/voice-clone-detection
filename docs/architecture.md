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
field (`app/api/http_router.py::_diarize_and_score`) — this diarizer
itself is NOT used on the live WebSocket path (it needs a COMPLETE
recording before it can cluster). The live path has its own, genuinely
different incremental tracker instead — see "Live diarization" below.
When requested, each detected speaker's audio is concatenated and
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
actually say", using OpenAI's Whisper (MIT, confirmed ungated). On
`POST /v1/score/file`, the whole buffer is transcribed once by
`Engine.score_call()` and merged into `context["transcript"]` before
windowing, so every window's third signal sees the same transcript.

**Live WebSocket path (added 2026-09-08)**: `app/pipeline/
live_transcription.py` (`LiveTranscriptionBuffer`) brings the same
signal to `/v1/stream/{session_id}` — see "Live transcription" below for
how.

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

**Still not built/validated**: the keyword lists themselves aren't
validated against real fraud-call transcripts — see the module's own
honesty note. (Live-path transcription itself IS now built — see below.)

## Live transcription — periodic, off-hot-path, for the WebSocket path

`app/pipeline/live_transcription.py` (`LiveTranscriptionBuffer`) answers
the same "what did the caller actually say" question on
`/v1/stream/{session_id}`, where there's no natural "whole buffer" to
transcribe until the call ends. `app/api/ws_router.py` holds one instance
per connection; `ingest()` is called once per incoming window, appending
only the TRAILING hop-worth of new audio to a rolling buffer (a window
overlaps the previous one by construction, so only the newest hop is
actually new — this assumes the client's hop matches
`config/risk_formula.yaml`'s `windowing.hop_ms`, the same assumption
`app/ui/app.js`'s own comment already makes). Every 4s of new audio, a
transcription of the trailing 8s fires in a background `asyncio.Task` via
`run_in_threadpool`, so it never blocks the per-window score response;
the latest COMPLETED result (and, if configured, an intent classification
of it) is merged into every subsequent window's context via `setdefault`
— never overwriting a caller-supplied value, same rule
`Engine.score_call()` already applies on the file-upload path. This is
also what lets the already-active `intent` detector (weight 0.15) start
contributing on the live path too, not just file uploads.

**Real bug found and fixed 2026-09-08** (user report: "I see it for the
first sentence, then nothing"): intent classification used to run INSIDE
the same task as transcription, sequentially after it, before the next
transcription cycle was allowed to start. Measured inside the actual
deployed container (not the faster local dev machine that motivated the
original "1-2s" estimate below): Whisper takes ~3-5s and the intent
classifier a further ~5s on top — combined, ~8-10s per cycle. A short
live-mic session would complete exactly one cycle and never start a
second before the user stopped, reading as broken. Fixed by giving intent
classification its OWN independently-scheduled task
(`_intent_task`, separate from `_transcribe_task`): a slow intent
classification no longer blocks the next transcription cycle. Verified
against the real running container: transcript updates now arrive every
~3.5-8s instead of ~10-11s.

**Honesty note**: deliberately never real-time-exact. A transcript lags
real speech by up to 4s plus however long Whisper actually takes — measured
inside the real deployed container: roughly 3-5s for an 8s window (notably
slower than a faster local dev machine) — the very first several seconds
of any live call have no transcript at all. Intent classification is a
further, independently-timed ~5s delay on top (no longer blocking
transcription, but still real latency for that specific signal). If a WS
client's hop doesn't match `windowing.hop_ms`, the reconstructed buffer
stretches or compresses relative to real time — a documented limitation,
not a silent one. See the module's own docstring for the full account.

## Live diarization — incremental "who's talking now" for the WebSocket path

`app/pipeline/live_diarization.py` (`LiveSpeakerTracker`) answers the
same "who spoke when" question the file-upload diarizer answers, above
— but incrementally, one window at a time, for a call that has no
"after the fact" the way a complete recording does. `app/api/ws_router.py`
holds one instance per connection; `ingest()` runs once per window,
extracting an ECAPA-TDNN embedding (the same shared model the file-
upload diarizer and voiceprint consistency both use, via a SEPARATE
instance — see the module's own docstring for why sharing an instance
across features wasn't done) and comparing it against every speaker
centroid seen so far this call by cosine similarity. Above
`similarity_threshold`: same speaker, the matched centroid is nudged
toward the new embedding via EMA so it can drift with natural voice
variation. Below threshold: a new speaker, up to `max_speakers` (past
that cap, same safety-net reasoning as the file-upload diarizer's own
cap — attribute to the closest existing speaker rather than minting an
unbounded number of new ones).

Unlike live transcription, this does NOT merge into `context` and does
NOT feed any detector's score — it's attached directly onto the
outgoing `FusedScore` as `live_speaker` (`speaker_label`,
`is_new_speaker`, `speaker_count`), purely informational. Optional, same
"omit the config section to disable" shape as transcription/intent
classification — see `config/risk_formula.yaml`'s `live_diarization:`
section.

**Honesty note, same caveat class as the file-upload diarizer's own**:
`similarity_threshold` (0.75) and `ema_alpha` (0.1) are reasonable-
looking defaults, not validated against any real labeled multi-speaker
LIVE-call dataset — no such corpus exists in this repo. 0.75 is chosen
deliberately above the file-upload diarizer's own effective threshold
(0.6), reasoning from that diarizer's own measured ~0.67-0.70 cross-
speaker cosine-similarity baseline for this embedding space — seeing
0.6 sit below that baseline is exactly why 0.75 was picked here instead,
but it remains a starting point, not a tuned result. Not persisted:
session history does not record `live_speaker` — replaying a past live
session from the History tab shows the score breakdown, not who was
talking at each point. Per-window, not per-utterance: unlike the file-
upload diarizer (which segments on detected pauses first), every window
is attributed independently, with no pause-aware segment boundary.

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

## Watermark check — a narrow, high-precision complement to AASIST

`app/adapters/detectors/perth_watermark.py` (`PerthWatermarkDetector`)
checks for Resemble AI's Perth neural watermark (MIT, resemble-ai/Perth)
— the fingerprint Chatterbox and other Perth-integrated voice-cloning
tools embed in every clip they generate. Real, standalone `get_watermark()`
API (`resemble-perth` on PyPI); verified by hand that genuine speech and
a different, non-Perth TTS system (XTTS-v2) both read near-zero, while a
clip actually watermarked by Perth reads 1.0 — confirms this is specific
to Perth's own signature, not a generic "sounds synthetic" trigger that
would just duplicate AASIST. Weighted low (0.15, auto-renormalised) for
the same reason `intent` is: a real but narrow signal that should never
single-handedly dominate the score. See `docs/risk-model.md`, "Watermark
check" for the full writeup, including a real false-positive edge case
on non-speech audio (a pure tone reads 0.92) found and documented while
verifying this.

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
training time — measuring that gap (and, as of 2026-09-08, partially
closing it) is exactly the work the project blueprint's §04 describes:
see `eval/indian_language/README.md` for the full reproducible process
and what's still missing (scope there is English + Hindi only, for now).
The result is now wired into this detector itself, not just measured
offline — `config/risk_formula.yaml`'s `finetuned_out_layer_path` param
swaps in a small, in-house-produced recalibration of AASIST's final
layer, cutting the false-positive rate on real, held-out Hindi speech
from 75% to 20% (see `docs/risk-model.md`, "Hindi recalibration", and
every score's `detail.hindi_finetuned_out_layer` flag for whether it was
active). It also has no exposure
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

- **Persistent *history* is built (above); persistent *live smoothing
  state* is not** — these are two different things. `Engine.sessions`
  (the EMA state used mid-call, in `SessionStore`) is still in-memory
  only, so a restart mid-call resets that call's smoothing to a fresh
  start. Fine for one process; swap `SessionStore` for Redis when more
  than one API replica needs to share an *in-progress* call's state.
- **An independently-measured English baseline EER for the Indian-language
  eval (§04)**, against a public benchmark split — AASIST currently runs
  on its published checkpoint, unvalidated by this team on English. Scope
  there is deliberately **English and Hindi only, for now** — Marathi and
  Malvi were dropped as required deliverables (see the scope note in
  `eval/indian_language/README.md`). The pipeline itself is built and has
  run end to end (`eval/indian_language/`): real, license-verified CC BY
  4.0 Hindi genuine speech, real XTTS-v2-cloned Hindi spoof audio, and an
  actual fine-tune of AASIST's final layer — **17.75% → 3.0% EER** on the
  trained-on synthesis system (XTTS-v2), plus a SECOND, genuinely held-out
  synthesis system never used in training (Resemble AI's Chatterbox)
  confirming this generalises rather than just memorising XTTS-v2:
  **17.75% → 10.25% EER** held-out. What's still missing: the English
  baseline above, and more held-out spoof volume (currently 40 examples,
  a calibration-scale sample, not yet a statistically solid one). See
  `eval/indian_language/README.md`.
- The mock banking approval flow (the dashboard UI itself is built —
  see above).
- Calibrated risk thresholds for voiceprint consistency, diarization
  clustering (both file-upload and live), and the Parselmouth prosodic
  mapping — all real signal-processing/model pipelines with honestly-
  documented placeholder heuristic score mappings, not yet tuned against
  a labeled dataset.

None of this changes the shape of `services/live-call-api/app/` — each
item above is a new adapter (or a new service) behind an existing or new
port.
