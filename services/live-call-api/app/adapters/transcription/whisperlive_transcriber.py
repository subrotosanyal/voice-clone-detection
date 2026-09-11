"""WhisperLiveTranscriber — bridges the LIVE WebSocket path's periodic
transcription (app/pipeline/live_transcription.py) to a collabora/
WhisperLive server (https://github.com/collabora/WhisperLive, MIT
licensed) instead of running Whisper in-process.

WHY: a real, measured bug (see live_transcription.py's own docstring,
"REAL BUG found 2026-09-09, live-mic parity investigation"): every
~500ms window on the live path already runs several CPU-bound model
calls in-process (AASIST, Parselmouth/Praat, ECAPA-TDNN) with no thread
limit, oversubscribing this container's CPU. The periodic Whisper
transcription task shared that same CPU/thread pool, so it got starved
— real live sessions measured 0 transcripts in an ~11s session, 1 in a
~15s session (12.7s in), transcription cycles 13s+ apart instead of the
configured 4s. Moving the actual Whisper forward pass OUT of this
process entirely, into WhisperLive's own server/container, removes it
from that contention regardless of how the in-process detectors are
tuned — see services/live-call-api's own Dockerfile OMP_NUM_THREADS
note for the complementary in-process fix for the detectors that stay
here.

SCOPE: this is the LIVE path only (config/risk_formula.yaml's
`live_transcription:` section, wired through app/api/ws_router.py's
LiveTranscriptionBuffer). POST /v1/score/file (file upload,
app/pipeline/engine.py's score_call()) keeps using the local
WhisperTranscriber (see whisper_transcriber.py) — that path transcribes
once per call, not every ~4s of a live session, so it was never the one
starved for CPU; switching it too would just add a new external-service
dependency to a path that already works.

PROTOCOL — verified against WhisperLive's own client.py (not guessed):
connect to `ws://{host}:{port}` (no path suffix), send one JSON
handshake message first (uid/language/task/model/use_vad/... — see
_build_config_message below for the exact fields this client sends),
wait for `{"message": "SERVER_READY", ...}`, then stream raw float32 PCM
audio as BINARY frames (their own client chunks at 4096 samples; matched
here), followed by the literal UTF-8 bytes `"END_OF_AUDIO"` as a final
binary frame to signal the end of this buffer. The server replies with
JSON messages carrying `segments` (each `{"text": ..., "completed":
bool, ...}`), `language`, `status`, or `{"message": "DISCONNECT"}` to
end the exchange.

CALLING PATTERN — one WS connection per transcribe() call, not a
persistent per-session connection: LiveTranscriptionBuffer re-transcribes
a fresh ~8s TRAILING window every ~4s (see its own docstring) — each
call is an independent buffer, not a continuation of the previous one,
so there is no "session" to keep a connection open for. A fresh
handshake per call also means a fresh `uid`, sidestepping any
per-connection state the server keeps between one buffer and the next.

FAILURE MODE — same "abstain, never crash" discipline as
WhisperTranscriber.transcribe() and every other best-effort adapter in
this project (see e.g. acoustic_aasist.py's floor_rms abstain, or
local_llm_semantic_classifier.py's graceful-abstain-on-load-failure
fix): a connection refused (server not up yet), a timeout, or a
malformed server message all return an empty TranscriptResult rather
than raising — LiveTranscriptionBuffer's own _transcribe() already
treats an empty transcript as "nothing to report this cycle", the same
shape as real silence.

REAL BUG found and fixed 2026-09-10, via a real user report ("the
live-mic transcript never appears") plus vexyl-stt-side connection-churn
errors during the SAME session, which led here on closer inspection —
traced directly in the actual collabora/whisperlive-cpu image's own
source (not guessed), then reproduced and the fix verified by hand
against the real running sidecar: `server.py`'s main receive loop, on
seeing our `END_OF_AUDIO` frame, breaks immediately, calls the (no-op,
for us) end-of-stream finalizer, then `cleanup()` + `websocket.close()`
in a `finally` block — and `cleanup()` (`backend/base.py`) only sets an
exit flag and an event; it never `.join()`s the background
`trans_thread` that's still running real inference and about to call
`self.websocket.send(...)` with the segment. That send then races
against (or lands after) the socket already closing, which server-side
logs as `[ERROR]: Failed to transcribe audio chunk: sent 1000 (OK);
then received 1000 (OK)` (or `Client gone before the last segments were
sent`) — and client-side, we get back exactly zero segments, every
time, regardless of how patiently we wait AFTER sending END_OF_AUDIO
(verified: even a 20s post-EOF wait finds nothing, because the socket is
already gone within milliseconds of the server receiving END_OF_AUDIO).
Confirmed by hand with a raw diagnostic script talking directly to the
sidecar: waiting up to 15s BEFORE ever sending END_OF_AUDIO reliably
produced a real segment (`" Hello, this is"`); the old code, waiting the
same 15s but AFTER sending END_OF_AUDIO, got nothing. Fixed by
reordering: collect whatever segments/language arrive while the
connection is still alive on the SERVER's own terms (i.e. before we
ever signal end-of-stream), bounded to `result_timeout_s` total, THEN
send END_OF_AUDIO best-effort (its own send wrapped so a
by-then-already-closing connection can't raise past this method) purely
to end the exchange cleanly — no longer relying on anything arriving
after it. HONESTY NOTE, not fully solved: this trades against catching
the very LAST moment of speech in a window (whatever's said after the
last chunk we send but before the collection deadline elapses won't be
captured, since we don't wait for MORE audio here — that's inherent to
this project's "whole trailing window per call" design, not new here),
and a genuinely slow/cold model load can still eat into the same
`result_timeout_s` budget this method now spends BEFORE learning
anything came back at all, rather than after.
"""
from __future__ import annotations

import json
import time
import uuid
from math import gcd
from typing import Any, Optional

import numpy as np
from websockets.sync.client import connect

from app.domain.models import TranscriptResult
from app.logging_setup import get_logger

logger = get_logger(component="whisperlive_transcriber")

_MODEL_SAMPLE_RATE = 16_000
_SEND_CHUNK_SAMPLES = 4096  # matches WhisperLive's own client.py chunk size
_END_OF_AUDIO = b"END_OF_AUDIO"


class WhisperLiveTranscriber:
    name = "whisperlive_transcriber"

    def __init__(
        self,
        host: str,
        port: int,
        model: str = "small",
        language: Optional[str] = None,
        use_vad: bool = True,
        # REAL BUG found and fixed 2026-09-11, alongside _collect_segments'
        # own idle-timeout fix (see that method's docstring): both of these
        # defaults were measured, by hand, against real speech on both the
        # local and home-lab whisper-live sidecars, to be too tight for
        # this integration's actual real-world cost — a FRESH model load
        # on every single connection (see this class's own CALLING PATTERN
        # note), not a cheap warm call. Measured end-to-end round trips on
        # an already-warm container: ~13-16s for a real ~7s clip in the
        # best case, up to ~70s in a slower one (concurrent load on a
        # shared host, same "model reloads from scratch every cycle" cost
        # each time) — nowhere close to the old 5s/20s ceilings, which is
        # why every single live call was failing, not just some. Raised to
        # comfortably cover the slower end of what was actually measured,
        # not a guess: a live call now (best-effort) waits longer per
        # cycle rather than silently producing nothing, matching this
        # project's "abstain never, empty only when genuinely nothing
        # came back" discipline elsewhere. HONEST COST: a slow cycle can
        # now legitimately take up to result_timeout_s, so the live path's
        # transcript can lag further behind real time than before on a
        # loaded host — a latency/correctness trade explicitly accepted
        # here, not something to silently tune back down without
        # re-measuring first.
        connect_timeout_s: float = 20.0,
        result_timeout_s: float = 75.0,
        idle_timeout_s: float = 3.0,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        vad_parameters: Optional[dict[str, Any]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.language = language
        self.use_vad = use_vad
        self.connect_timeout_s = connect_timeout_s
        self.result_timeout_s = result_timeout_s
        # Added 2026-09-11 alongside the idle-based collection change below
        # — a fixed budget MEASURED FROM THE FIRST MESSAGE onward, distinct
        # from result_timeout_s's TOTAL budget. See _collect_segments's own
        # comment for why both are needed together, not one or the other.
        self.idle_timeout_s = idle_timeout_s
        # Added 2026-09-11 — these three fields were already part of the
        # handshake message (_build_config_message below), always sent as
        # None/hardcoded defaults; now real constructor params so a
        # deployment can actually use them. initial_prompt/hotwords: no
        # decoding-quality controls at all were configurable here before
        # (unlike whisper_transcriber.py's pinned temperature/degenerate-
        # segment filtering) — domain-vocabulary priming is a real,
        # low-risk accuracy lever WhisperLive/faster-whisper supports.
        # vad_parameters: left None (server/library default) unless a
        # caller sets it — this project has no vendored copy of
        # WhisperLive's server source to verify field-level VAD tuning
        # against (unlike vexyl-stt, which IS vendored here), so no
        # specific values are guessed; only the wiring is added.
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
        self.vad_parameters = vad_parameters
        self.version = f"whisperlive@{model}"

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptResult:
        if samples.size == 0:
            return self._empty()

        if sample_rate != _MODEL_SAMPLE_RATE:
            samples = _resample(samples, sample_rate, _MODEL_SAMPLE_RATE)

        try:
            return self._transcribe_via_server(samples.astype(np.float32))
        except Exception:  # noqa: BLE001 — best-effort, must never break live scoring
            logger.exception("whisperlive_transcription_failed")
            return self._empty()

    def _transcribe_via_server(self, samples: np.ndarray) -> TranscriptResult:
        uri = f"ws://{self.host}:{self.port}"
        with connect(uri, open_timeout=self.connect_timeout_s) as ws:
            ws.send(json.dumps(self._build_config_message()))

            # Wait for SERVER_READY before sending audio — sending audio
            # first would race the server's own per-connection setup
            # (confirmed against WhisperLive's own client.py, which does
            # the same wait before recording).
            if not self._wait_for_server_ready(ws):
                return self._empty()

            for offset in range(0, samples.size, _SEND_CHUNK_SAMPLES):
                chunk = samples[offset : offset + _SEND_CHUNK_SAMPLES]
                ws.send(chunk.tobytes())

            # Collect BEFORE sending END_OF_AUDIO — see this module's own
            # REAL BUG note (2026-09-10) for why: the server closes the
            # connection almost immediately after receiving END_OF_AUDIO,
            # without waiting for in-flight transcription to finish, so
            # anything we'd wait for AFTER sending it is unreliable.
            segment_texts, detected_language = self._collect_segments(ws)

            try:
                ws.send(_END_OF_AUDIO)
            except Exception:  # noqa: BLE001 — connection may already be
                # finishing up server-side; we already have whatever
                # we're going to get, this send is just to end cleanly.
                pass

            text = " ".join(t.strip() for t in segment_texts if t.strip())
            return TranscriptResult(
                text=text,
                language=detected_language,
                detector_name=self.name,
                detector_version=self.version,
            )

    def _build_config_message(self) -> dict[str, Any]:
        """Every field WhisperLive's own client always sends on connect —
        confirmed against its client.py, not guessed. Fields this
        integration doesn't use a non-default value for (translation,
        diarization, hotwords, word timestamps) are passed at their
        library defaults so the server's own handshake validation
        doesn't reject a message missing an expected key."""
        return {
            "uid": str(uuid.uuid4()),
            "language": self.language,
            "task": "transcribe",
            "model": self.model,
            "use_vad": self.use_vad,
            "send_last_n_segments": 10,
            "no_speech_thresh": 0.45,
            "clip_audio": False,
            "same_output_threshold": 10,
            "enable_translation": False,
            "target_language": None,
            "hotwords": self.hotwords,
            "enable_diarization": False,
            "max_speakers": 1,
            "known_speakers": None,
            "word_timestamps": False,
            "initial_prompt": self.initial_prompt,
            "vad_parameters": self.vad_parameters,
        }

    def _wait_for_server_ready(self, ws) -> bool:
        message = self._recv_json(ws, timeout=self.connect_timeout_s)
        return bool(message) and message.get("message") == "SERVER_READY"

    def _collect_segments(self, ws) -> tuple[list[str], Optional[str]]:
        """Collects whatever segment/language messages arrive, bounded by
        TWO independent budgets, not one:

        - `result_timeout_s` TOTAL from the start of collection (unchanged
          from before this method had an idle budget too) — a hard ceiling
          so one call can never run arbitrarily long.
        - `idle_timeout_s` since the LAST message actually received, reset
          on every new message. Added 2026-09-11 to narrow (not fully
          close — see this module's own HONESTY NOTE, still not fully
          solved) the documented gap where the very last moment of speech
          in a window could be missed: without an idle budget, a message
          arriving late in the total window (e.g. because a cold/slow
          model finally started producing output) got the same shrinking
          `remaining` time as everything before it; with one, a server
          that's ACTIVELY still sending things gets a fresh
          `idle_timeout_s` grace each time, right up until `result_timeout_s`
          total is reached — so a burst of trailing segments near the
          overall deadline isn't cut off just because the wall-clock total
          happened to run out at an inconvenient moment.

        REAL BUG found and fixed 2026-09-11 (same day as the idle-timeout
        feature above — a fresh user report, "live transcribing is not
        working at all anymore", confirmed with real audio against both
        the local and home-lab whisper-live sidecars): the idle deadline
        used to be seeded ONE `idle_timeout_s` after collection STARTS, not
        after the first message arrives — so it governed the wait for the
        very FIRST message too, not just gaps BETWEEN messages as the
        paragraph above describes. Measured by hand against a real,
        already-warm whisper-live server: ~13s from connecting to its
        FIRST message (language detection) on a real ~7s speech window —
        routine cold-model-per-connection latency (see this class's own
        CALLING PATTERN note: a fresh connection, and therefore a fresh
        model load, every single cycle), nowhere close to a stall. Against
        `idle_timeout_s`'s default of 3.0s, EVERY call's very first wait
        expired before the server could ever say anything at all — not an
        occasional miss, a 100%-of-calls failure, exactly matching the
        report. Fixed by not creating an idle deadline until AFTER the
        first message is received; before that, only the full
        `result_timeout_s` budget applies, same as before idle_timeout_s
        existed — restoring the original intent (guard against a stall
        AFTER the model starts talking, don't also gate how long a cold
        model takes to start).

        Whichever budget is tighter at any given instant wins (once both
        are active) — this can only ever SHORTEN or match the old
        total-only wait, never lengthen it past `result_timeout_s`."""
        deadline = time.monotonic() + self.result_timeout_s
        idle_deadline: Optional[float] = None  # set only once the first message arrives
        segment_texts: list[str] = []
        detected_language: Optional[str] = None

        while True:
            now = time.monotonic()
            remaining = (min(deadline, idle_deadline) if idle_deadline is not None else deadline) - now
            if remaining <= 0:
                break
            message = self._recv_json(ws, timeout=remaining)
            if message is None:
                break  # nothing new before whichever budget ran out first
            idle_deadline = time.monotonic() + self.idle_timeout_s
            if message.get("status") == "ERROR":
                logger.warning("whisperlive_server_error", detail=message.get("message"))
                break
            if "language" in message:
                detected_language = message.get("language")
            if "segments" in message:
                segment_texts = [s.get("text", "") for s in message["segments"] if s.get("text")]

        return segment_texts, detected_language

    @staticmethod
    def _recv_json(ws, timeout: float) -> Optional[dict[str, Any]]:
        try:
            raw = ws.recv(timeout=timeout)
        except Exception:  # noqa: BLE001 — TimeoutError, ConnectionClosed, etc.
            return None
        if isinstance(raw, (bytes, bytearray)):
            return None  # WhisperLive's own replies are always JSON text frames
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def _empty(self) -> TranscriptResult:
        return TranscriptResult(text="", language=None, detector_name=self.name, detector_version=self.version)


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(samples, up, down).astype(np.float32)
