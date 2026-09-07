# Testing & debugging

## Running the tests

```bash
cd services/live-call-api
python3 -m venv .venv && source .venv/bin/activate   # first time only
pip install -r requirements-dev.txt                   # first time only
pytest tests/ -v
```

33 tests, all offline, no Docker needed — unit tests for each detector,
the windowing math, and the fusion formula (including the abstain/
renormalisation behaviour), plus integration tests that drive the real
FastAPI app with `TestClient` (`tests/integration/test_api_score_file.py`).

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

## Writing a new test for a new detector

Copy `tests/unit/test_contextual_rules.py` — it's the simplest example: no
audio math, just a table of inputs to expected `score`/`detail` values.
For a detector that touches audio, `tests/unit/test_acoustic_detector.py`
shows the pattern (construct a known signal — a pure tone, white noise —
and assert a property you can reason about by hand, not a magic number).
