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
"""
from __future__ import annotations

import json
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
        connect_timeout_s: float = 5.0,
        result_timeout_s: float = 20.0,
    ) -> None:
        self.host = host
        self.port = port
        self.model = model
        self.language = language
        self.use_vad = use_vad
        self.connect_timeout_s = connect_timeout_s
        self.result_timeout_s = result_timeout_s
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
            ws.send(_END_OF_AUDIO)

            return self._collect_result(ws)

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
            "hotwords": None,
            "enable_diarization": False,
            "max_speakers": 1,
            "known_speakers": None,
            "word_timestamps": False,
            "initial_prompt": None,
            "vad_parameters": None,
        }

    def _wait_for_server_ready(self, ws) -> bool:
        message = self._recv_json(ws, timeout=self.connect_timeout_s)
        return bool(message) and message.get("message") == "SERVER_READY"

    def _collect_result(self, ws) -> TranscriptResult:
        segment_texts: list[str] = []
        detected_language: Optional[str] = None

        while True:
            message = self._recv_json(ws, timeout=self.result_timeout_s)
            if message is None:
                break  # timed out or connection closed — return what we have

            if message.get("message") == "DISCONNECT":
                break
            if "language" in message:
                detected_language = message.get("language")
            if "segments" in message:
                segment_texts = [s.get("text", "") for s in message["segments"] if s.get("text")]
            if message.get("status") == "ERROR":
                logger.warning("whisperlive_server_error", detail=message.get("message"))
                break

        text = " ".join(t.strip() for t in segment_texts if t.strip())
        return TranscriptResult(
            text=text,
            language=detected_language,
            detector_name=self.name,
            detector_version=self.version,
        )

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
