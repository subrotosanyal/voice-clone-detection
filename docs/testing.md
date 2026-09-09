# Testing & debugging

## Running the tests

```bash
cd services/live-call-api
python3 -m venv .venv && source .venv/bin/activate   # first time only
pip install -r requirements-dev.txt                   # first time only
pytest tests/ -v
```

157 tests, no Docker needed — unit tests for each detector (including
`test_acoustic_aasist.py`'s coverage of the optional Hindi-recalibrated
`finetuned_out_layer_path`, verifying it actually changes the score, not
just that the parameter is accepted; and `test_perth_watermark.py`,
which watermarks a real spoken clip via Perth's own `apply_watermark()`
and confirms the detector actually distinguishes it from the
unwatermarked original, not just that it runs — plus a regression test
for a real bug found 2026-09-09 via dogfooding external audio: a very
short trailing window used to crash the whole request instead of
abstaining), degraded-audio
robustness checks against AASIST and the prosodic detector
(`test_degraded_audio_robustness.py` — noise at two SNR levels, a
telephone-bandwidth filter, and a real low-bitrate Opus re-encode,
verifying no detector returns NaN or swings unreasonably on real,
if degraded, genuine speech — motivated by a real report of convincing
real-time vishing over "poor" quality audio), the windowing
math, the fusion formula (including the abstain/renormalisation
behaviour), the SQLite history and enrollment stores, the ECAPA-TDNN-based
diarizer and its pause-aware segmentation front end
(`test_pause_segmentation.py`, `test_embedding_cluster_diarizer.py`), the
live WebSocket path's incremental speaker tracker
(`test_live_diarization.py` — a fake embedder for the clustering-logic
tests, plus one test through the real ECAPA-TDNN model proving actual
same/different-speaker separation, not just that the arithmetic is
right; see app/pipeline/live_diarization.py), the
Whisper transcriber and its urgency/financial/authority-claim keyword
detection, the zero-shot intent classifier (`test_zero_shot_
intent_classifier.py` — includes a KNOWN LIMITATION test that documents,
rather than hides, the real calibration problem keeping it out of the
active formula) and its detector (`test_intent_risk.py`), the live
WebSocket path's periodic transcription buffer (`test_live_transcription.py`
— fake transcriber/intent-classifier objects, no real model load, async
internals driven via plain `asyncio.run()`), plus integration tests that
drive the real FastAPI app with `TestClient`
(`tests/integration/test_api_score_file.py`, `test_ui_served.py`,
`test_history_api.py`, `test_enrollment_api.py`, `test_diarization_api.py`,
`test_transcription_api.py`, `test_live_transcription_ws.py`,
`test_live_diarization_ws.py`).

**Real-speech tests use macOS's `say` command.** `test_whisper_transcriber.py`
and `test_transcription_api.py` need genuinely intelligible speech, not a
synthetic tone — Whisper transcribes real words, not spectral shape. Rather
than adding a TTS dependency, they generate a spoken fixture at test time
via `say` (already on macOS, zero new dependencies) and skip gracefully
wherever it isn't available (any non-macOS CI/Docker environment). Verified
by hand: "Please transfer the money immediately, it's urgent." synthesized
this way and fed through the real pipeline correctly detected the language,
transcribed it, and matched the urgency/financial keywords.

**Pure sine tones don't exercise voiceprint/diarization tests.** A speaker-
embedding model needs real vocal-tract-like spectral variation to tell
voices apart — two pure tones at different frequencies still cosine-
similarity almost as high as two takes of the *same* tone. Tests for
`voiceprint_consistency.py` and `embedding_cluster_diarizer.py` instead use
`tests/audio_fixtures.py::formant_voice()` — a pulsed source filtered
through a few band-pass "formants", giving genuinely different spectral
envelopes between presets. Still synthetic, not real speech — same
standing caveat as `scripts/gen_test_audio.py`'s fixtures, just with enough
timbre variation to prove the clustering/comparison logic is real rather
than a constant.

Test runs share the same local `data/sessions.db` file the app itself
uses (see `docs/running-locally.md`) — each test uses a fresh
`uuid`-based session id, so rows never collide across runs, but the file
does accumulate history from every test run. That's harmless (it's
gitignored, and nothing asserts on the *total* row count), but if it
bothers you, delete `services/live-call-api/data/sessions.db` any time.

**One caveat, not fully offline any more:** the app's default config loads
the real AASIST checkpoint, the ECAPA-TDNN speaker-embedding model, AND
the Whisper transcription model at startup. `tests/conftest.py`'s
`aasist_checkpoint` fixture fetches AASIST automatically on first run if
it's missing (needs internet once; cached afterward, checksum-verified
every time); the `ecapa_extractor` fixture does the analogous thing for
the speechbrain model; `test_whisper_transcriber.py`'s own `transcriber`
fixture does the same for Whisper (all cached after the first download —
not checksum-verified for the latter two, only the confirmed-ungated
download itself). Any test that doesn't request one of these fixtures
(windowing, fusion math, the contextual/spectral-flatness detectors, the
transcript-merging tests in `test_contextual_rules.py`, which pass a
literal `transcript` string and never touch Whisper itself) stays fully
offline and unaffected. If a fetch fails, only the tests that need that
model skip; nothing else breaks.

Run just one file while iterating: `pytest tests/unit/test_fusion_weighted_sum.py -v`.

## Continuous integration

`.github/workflows/ci.yml` runs on every push/PR to `master` (and
manually, via workflow_dispatch), two jobs:

- **`test`** — the exact `pytest tests/ -v` command above, on a plain
  Python 3.12 venv (not Docker). Installs CPU-only torch/torchaudio first
  (same reasoning as the Dockerfile's own install step — a plain `pip
  install torch` on Linux resolves the ~2GB CUDA build this CPU-only
  runner never uses), then fetches the same AASIST/ECAPA-TDNN/Whisper
  models described above via the same `conftest.py` fixtures — cached
  across runs (`actions/cache`) so only the first run after a model
  version bump pays the download cost.
- **`docker-compose-build`** — runs the real `docker compose build` /
  `up`, waits for the container's own healthcheck to pass, then scores
  `samples/genuine_tone.wav` through the running container exactly like
  README.md's Quickstart curl command — proving the committed Dockerfile/
  docker-compose.yml combination still works end to end, not just that
  the test suite passes outside Docker.

**Honest accounting of why `test` needs a generous timeout (45min, not a
round number picked in advance)**: a real CI run hit an earlier 20min
limit and got cancelled at 16% of tests, having spent 1-4 real minutes on
several individual integration tests. Two compounding reasons, not a
fluke: every one of the 25 `with TestClient(app) as client:` blocks across
`tests/integration/` (8 files) reloads AASIST + Whisper + ECAPA-TDNN from
disk into memory from scratch (the app's lifespan runs fresh each time —
and, as of `live_diarization:` being enabled by default alongside
`diarization:`, this now loads TWO separate ECAPA-TDNN copies per
startup, one per feature — see app/pipeline/live_diarization.py's own
docstring for why that's a deliberate, not accidental, trade-off), and
every `/v1/score/file` call also runs a real Whisper decode regardless of
whether that particular test has anything to do with transcription — a
GitHub-hosted CPU runner is slower than a dev machine for this. Sharing
one long-lived app/TestClient across a test module (loading each model
once for the whole run, not once per test) would meaningfully cut this,
but touches every integration test file and needs care around tests
that share SQLite-backed state — not done yet; the honest workaround for
now is enough timeout headroom to let the real work finish.

## The fastest way to debug a score

```bash
curl -F "file=@services/live-call-api/samples/genuine_tone.wav" \
     http://localhost:8020/v1/score/file | python3 -m json.tool
```

No WebSocket client, no audio hardware — one file in, the full breakdown
back, including every detector's `detail` and why any of them abstained.
Same code path (`Engine.score_window`) as the live WebSocket stream, just
fed from a file instead of a socket, so a bug reproduced here is the same
bug that would show up live.

## Reading the logs

```bash
docker compose logs -f live-call-api | jq .
```

or open **Dozzle** at http://localhost:8888 for a searchable web UI over
the same stdout. Every `risk_score_computed` line has the full
`components` breakdown — usually enough to answer "why did this score
come out the way it did" without touching a debugger.

## Common errors

**`OSError: cannot load library 'libsndfile.so'`** — only happens if
you're running outside the provided Docker image (e.g. a bare Linux
venv without `libsndfile1` installed). The Dockerfile installs it via
apt; on macOS, `brew install libsndfile` (usually not needed — the
soundfile wheel bundles it on macOS/Windows, just not reliably on all
Linux architectures).

**A detector's weight looks like it's not affecting the score** —
`WeightedSumFusion.fuse()` matches detectors to their configured weight
*by position*, not by name (see the comment in
`app/adapters/fusion/weighted_sum.py`) — this relies on
`Engine.score_window()` calling detectors in exactly the order they
appear in `config/risk_formula.yaml`. If you're calling `fuse()` directly
in a script (not through `Engine`), make sure your `results` list is in
that same order.

**Port already in use** — `docker-compose.yml` maps host port 8020, not
8000, specifically to avoid clashing with other projects on a shared dev
machine. Check `docker ps` for what's actually using a port before
assuming it's this project.

**`FileNotFoundError: AASIST checkpoint not found ...`** — run `python
app/adapters/detectors/vendor/fetch_checkpoint.py` once (needs internet;
idempotent). Under pytest this happens automatically via the
`aasist_checkpoint` fixture in `conftest.py` — you'd only see this
running the app directly without having fetched it first.

**`CMake Error: CMake was unable to find a build program ...` while
building `praat-parselmouth`** — only happens building this Docker image
on linux/aarch64 (Apple Silicon Docker Desktop, or an arm64 host):
praat-parselmouth publishes prebuilt wheels for x86_64/i686 Linux and
macOS/Windows, but not linux/aarch64, so pip falls back to compiling
Parselmouth's bundled Praat C++ source from an sdist — which needs a real
build toolchain. The Dockerfile's `parselmouth-builder` stage installs
`build-essential cmake ninja-build` and builds just that one wheel there,
so the final image never needs a C/C++ toolchain at all. If you're
installing outside Docker on a bare linux/aarch64 venv, install the same
three apt packages first, or install a specific working version if one
publishes an aarch64 wheel by the time you're reading this
(`pip index versions praat-parselmouth`).

**Sample fixtures both score as high-risk** — expected now that the
acoustic detector is a real speech classifier; see the note at the top of
`scripts/gen_test_audio.py`. Neither fixture is real speech, so neither
reads as "bonafide" — that's correct model behaviour, not a broken test.

**`test_repeated_calls_with_same_input_are_reproducible` fails
intermittently (fixed 2026-09-08)** — this was a real, genuinely
non-deterministic bug, not flaky infrastructure: Whisper's `transcribe()`
defaults to a temperature FALLBACK tuple, and on non-speech audio (this
test's synthetic fixture) the initial greedy pass could fail Whisper's own
quality gates and fall back to sampling from PyTorch's unseeded global
RNG — producing a different transcript call to call (observed directly:
empty on most calls, a hallucinated `"MMMMMMMMMMMMMM"` on others), which
then changed the intent classifier's output and the final fused score.
Fixed by pinning `temperature=0.0` in `whisper_transcriber.py` (forces
pure greedy decoding, deterministic by construction) — see that file's
REPRODUCIBILITY FIX docstring note and
`test_repeated_calls_on_non_speech_audio_are_deterministic` in
`test_whisper_transcriber.py` for the regression coverage. If a similar
"same input, different score" report ever resurfaces, suspect any model
call that isn't pinned to greedy/deterministic decoding first.

## Debugging the browser dashboard

It's plain JS with no build step, so browser devtools are the whole
toolchain: open http://localhost:8020/, then devtools' Console tab for
JS errors and Network tab to inspect the exact request/response of
`POST /v1/score/file` or the `WS /v1/stream/{id}` messages — the same
JSON `curl` would get back. The "▸ view raw JSON" toggle under any result
shows the full `FusedScore` the UI rendered, so a UI bug ("the bar looks
wrong") and a scoring bug ("the number is wrong") are easy to tell apart:
if the raw JSON already has the wrong number, it's not the UI's fault.

## Writing a new test for a new detector

Copy `tests/unit/test_contextual_rules.py` — it's the simplest example: no
audio math, just a table of inputs to expected `score`/`detail` values.
For a detector that touches audio, `tests/unit/test_acoustic_detector.py`
shows the pattern (construct a known signal — a pure tone, white noise —
and assert a property you can reason about by hand, not a magic number).
