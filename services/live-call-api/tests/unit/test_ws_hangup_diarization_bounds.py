"""Tests for app/api/ws_router.py's hangup-diarization resource bounds —
see that module's own 2026-09-11 REAL BUG note (a real CI SIGABRT traced
to unbounded, undrained "diarize on hangup" background tasks). Exercises
the private _diarize_on_hangup/_schedule_diarize_on_hangup/
drain_pending_hangup_diarizations functions directly with a fake engine —
no real AASIST/ECAPA/Whisper/Perth inference, no real WebSocket — same
"fake the expensive port, test the real concurrency logic" style as
test_whisper_transcriber.py's own concurrency regression test
(test_concurrent_transcribe_calls_are_serialized_not_overlapping).
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import numpy as np
import pytest

from app.api import ws_router
from app.pipeline.live_session_recorder import LiveSessionRecorder

SR = 16_000


class _FakePipeline:
    diarizer = object()  # anything not None — _diarize_on_hangup only checks identity


class _SlowFakeEngine:
    """Stands in for the real Engine. Sleeps briefly inside
    diarize_and_score() — called via run_in_threadpool, so this really
    runs on a worker thread — so a concurrency test can reliably observe
    overlap if the semaphore isn't actually bounding it, same pattern as
    test_whisper_transcriber.py's _SlowFakeModel."""

    def __init__(self, sleep_s: float = 0.05) -> None:
        self.pipeline = _FakePipeline()
        self.sleep_s = sleep_s
        self.max_concurrent = 0
        self.call_count = 0
        self._current = 0
        self._lock = threading.Lock()

    def diarize_and_score(self, **kwargs):
        with self._lock:
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
            self.call_count += 1
        time.sleep(self.sleep_s)
        with self._lock:
            self._current -= 1
        return []


def _recorder_with_audio() -> LiveSessionRecorder:
    recorder = LiveSessionRecorder(hop_ms=500)
    recorder.ingest(np.zeros(SR, dtype=np.float32), SR)  # 1s of "audio", first-call full-window path
    return recorder


class _FakeLiveTranscription:
    """Stands in for LiveTranscriptionBuffer — just enough surface for
    _finish_live_transcription_then_diarize: an async flush() that takes
    real (if brief) time, same shape as a real WhisperLive cycle that can
    now legitimately take up to result_timeout_s (75s, see that module's
    own 2026-09-11 REAL BUG note) to produce a transcript."""

    def __init__(self, flush_delay_s: float = 0.05) -> None:
        self.flush_delay_s = flush_delay_s
        self.flushed = False
        self.closed = False
        self.latest_context: dict = {}

    async def flush(self) -> None:
        await asyncio.sleep(self.flush_delay_s)
        self.flushed = True
        self.latest_context = {"transcript": "hello from flush", "transcript_language": "en"}

    async def aclose(self) -> None:
        self.closed = True


class _FakeSessionTranscriber:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _isolated_pending_set(monkeypatch):
    """Every test gets its own _pending_hangup_diarizations set — the
    real one is process-global (see the module's own REAL BUG note on
    why: it must survive across many `with TestClient(app)` blocks in
    the real integration suite), which would otherwise leak scheduled
    tasks between tests in this file."""
    monkeypatch.setattr(ws_router, "_pending_hangup_diarizations", set())


@pytest.mark.asyncio
async def test_concurrent_hangup_diarizations_are_bounded_by_the_semaphore(monkeypatch):
    """Regression test for the 2026-09-11 REAL BUG: schedules MORE hangup
    diarizations than _HANGUP_DIARIZE_MAX_CONCURRENT permits and proves
    the fake engine's diarize_and_score() never runs more of them at once
    than that cap allows."""
    monkeypatch.setattr(
        ws_router, "_hangup_diarize_semaphore", asyncio.Semaphore(ws_router._HANGUP_DIARIZE_MAX_CONCURRENT)
    )
    engine = _SlowFakeEngine(sleep_s=0.1)
    n_sessions = ws_router._HANGUP_DIARIZE_MAX_CONCURRENT + 3

    tasks = [
        asyncio.create_task(ws_router._diarize_on_hangup(engine, f"session-{i}", _recorder_with_audio(), {}))
        for i in range(n_sessions)
    ]
    await asyncio.gather(*tasks)

    assert engine.call_count == n_sessions
    assert engine.max_concurrent <= ws_router._HANGUP_DIARIZE_MAX_CONCURRENT, (
        f"{engine.max_concurrent} hangup diarizations ran concurrently — "
        f"the semaphore (cap {ws_router._HANGUP_DIARIZE_MAX_CONCURRENT}) did not bound them"
    )
    assert engine.max_concurrent > 1, "test is meaningless if concurrency never actually happened"


@pytest.mark.asyncio
async def test_drain_waits_for_in_flight_tasks_to_actually_finish():
    """Regression test for the 2026-09-11 REAL BUG's other half: without
    draining at shutdown, a scheduled hangup task silently outlives the
    caller. Proves drain_pending_hangup_diarizations() really waits for
    it (not just returns immediately) by checking the engine's own
    call_count only AFTER draining."""
    engine = _SlowFakeEngine(sleep_s=0.1)
    ws_router._schedule_diarize_on_hangup(engine, "sess-1", _recorder_with_audio(), {})

    assert engine.call_count == 0  # scheduled, not yet run
    await ws_router.drain_pending_hangup_diarizations(timeout_s=5.0)

    assert engine.call_count == 1
    assert len(ws_router._pending_hangup_diarizations) == 0


@pytest.mark.asyncio
async def test_drain_gives_up_after_its_timeout_instead_of_hanging_forever():
    """A genuinely stuck task must not hang app shutdown forever — see
    drain_pending_hangup_diarizations()'s own docstring. Uses a real
    (short) sleep longer than the drain timeout to prove drain returns
    promptly rather than blocking until the task finishes."""
    engine = _SlowFakeEngine(sleep_s=0.5)
    ws_router._schedule_diarize_on_hangup(engine, "sess-slow", _recorder_with_audio(), {})

    started = time.monotonic()
    await ws_router.drain_pending_hangup_diarizations(timeout_s=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, "drain waited for the slow task instead of giving up at its own timeout"
    # Let the still-running background task actually finish so it doesn't
    # leak into a later test — same cleanup concern the module's own
    # docstring names (a thread can't be forcibly cancelled).
    await asyncio.sleep(0.6)


@pytest.mark.asyncio
async def test_drain_is_a_no_op_when_nothing_is_pending():
    await ws_router.drain_pending_hangup_diarizations(timeout_s=1.0)  # must not raise


@pytest.mark.asyncio
async def test_post_hangup_work_survives_cancellation_of_the_caller():
    """Regression test for a real bug found 2026-09-11, via a fresh user
    report (a live call's own logs showed a real, correct transcript
    computed — but the post-hangup summary UI still showed intent/
    semantic_risk "abstained: no transcript-derived ... available"):
    flush() used to be awaited INLINE in the WS handler's own `finally`
    block — the exact same shape already found and fixed for
    _diarize_on_hangup() (see this module's docstring), just not also
    applied to flush() at the time. The instant a WebSocket's underlying
    connection is fully closed, Starlette/anyio tears down the request's
    cancel scope, killing anything still running in it, `finally` block
    included. flush() finishing in well under a second (the OLD, buggy
    WhisperLive timeouts) never gave this enough rope to matter; now that
    a real live cycle can legitimately take up to result_timeout_s (75s),
    flush() routinely got killed mid-flight before final_context was ever
    built. Proves the fix: cancelling the CALLER right after it schedules
    the work (simulating the WS connection closing) must not stop
    flush()/aclose()/session-close/diarize from all still running to
    completion in the background."""
    engine = _SlowFakeEngine(sleep_s=0.01)
    live_transcription = _FakeLiveTranscription(flush_delay_s=0.1)
    session_transcriber = _FakeSessionTranscriber()
    scheduled = asyncio.Event()

    async def _caller():
        ws_router._schedule_post_hangup_work(
            engine, "sess-cancel", _recorder_with_audio(), live_transcription, session_transcriber
        )
        scheduled.set()
        await asyncio.sleep(10)  # stands in for the rest of the WS handler's own scope

    caller_task = asyncio.create_task(_caller())
    await scheduled.wait()
    caller_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await caller_task

    await ws_router.drain_pending_hangup_diarizations(timeout_s=2.0)

    assert live_transcription.flushed, "flush() must survive even though the calling task was cancelled"
    assert live_transcription.closed
    assert session_transcriber.closed
    assert engine.call_count == 1, "diarize_and_score must still run, using flush()'s own transcript"
