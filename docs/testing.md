# Testing & debugging

## Running the tests

```bash
cd services/live-call-api
python3 -m venv .venv && source .venv/bin/activate   # first time only
pip install -r requirements-dev.txt                   # first time only
pytest tests/ -v
```

79 tests, no Docker needed — unit tests for each detector, the windowing
math, the fusion formula (including the abstain/renormalisation
behaviour), the SQLite history and enrollment stores, the ECAPA-TDNN-based
diarizer, plus integration tests that drive the real FastAPI app with
`TestClient` (`tests/integration/test_api_score_file.py`,
`test_ui_served.py`, `test_history_api.py`, `test_enrollment_api.py`,
`test_diarization_api.py`).

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
the real AASIST checkpoint AND the ECAPA-TDNN speaker-embedding model at
startup. `tests/conftest.py`'s `aasist_checkpoint` fixture fetches AASIST
automatically on first run if it's missing (needs internet once; cached
afterward, checksum-verified every time); the `ecapa_extractor` fixture
does the analogous thing for the speechbrain model (also cached after the
first download — HuggingFace's own cache, not checksum-verified since
there's no published hash to check against, only the confirmed-ungated
download itself). Any test that doesn't request either fixture (windowing,
fusion math, the contextual/spectral-flatness detectors) stays fully
offline and unaffected. If a fetch fails, only the tests that need that
model skip; nothing else breaks.

Run just one file while iterating: `pytest tests/unit/test_fusion_weighted_sum.py -v`.

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
