"""VexylSttTranscriber — bridges either transcription path to a
vexyl-ai/vexyl-stt server (https://github.com/vexyl-ai/vexyl-stt, Apache
2.0 licensed), wrapping ai4bharat/indic-conformer-600m-multilingual (MIT
licensed, but a GATED HuggingFace model — see services/vexyl-stt/
fetch_model.py's own docstring for what that means at build time).

WHY: evaluated by hand against this project's own deployed Whisper
adapter, on 4 real Hindi/Marathi files plus 2 synthesized English/Hinglish
clips (see the conversation that led to this file, not reproduced here —
docs/risk-model.md's own transcription section has the durable summary).
Findings: on real Hindi/Marathi audio, this model was both consistently
MORE ACCURATE (Whisper's Hindi/Marathi output had real word-level errors;
this one didn't, on the same clips) and 6-20x FASTER on CPU (CTC decoding
is a single forward pass, no autoregressive generation loop — structurally
immune to the repetition-loop failure mode whisper_transcriber.py's own
REAL BUG notes document). But it ONLY supports 14 Indian languages, and
en/unsupported input isn't rejected — it's silently mis-routed to
Malayalam and confidently transliterated into nonsense (verified by hand:
English audio came back as Malayalam-script phonetic gibberish, no error,
no abstain). That failure mode is exactly why this class is never used
directly as `transcription:`/`live_transcription:` on its own — see
routing_transcriber.py, which gates on a detected language BEFORE ever
calling this class, using only the language codes in SUPPORTED_LANGUAGES
below (the real vexyl-stt server's own LANG_MAP, verified against its
source in services/vexyl-stt/vendor/vexyl_stt_server.py).

PROTOCOL — verified against vexyl-stt's own vexyl_stt_server.py (vendored
copy in services/vexyl-stt/vendor/, not guessed): connect to
`ws://{host}:{port}` (no path suffix), server sends `{"type":"ready",...}`
first, client sends one JSON `{"type":"start","lang":...,"session_id":...}`
handshake, server replies `{"type":"started",...}`, client then streams
raw 16-bit PCM audio as BINARY frames, client sends
`{"type":"stop"}` to end, server replies with zero or more
`{"type":"final","text":...}` messages (the server's own energy-based VAD
decides when to emit one — a single long buffer can come back as several
final messages, not one) followed by `{"type":"stopped"}`.

CALLING PATTERN — one WS connection per transcribe()/transcribe_lang()
call, same reasoning as whisperlive_transcriber.py's own docstring: this
project always hands over one whole buffer per call (a full file, or one
live window), never a continuation of a previous stream, so there's no
"session" worth keeping a connection open for.

FAILURE MODE — same "abstain, never crash" discipline as every other
best-effort adapter here (whisperlive_transcriber.py, acoustic_aasist.py's
floor_rms abstain, local_llm_semantic_classifier.py's graceful-abstain-
on-load-failure): a connection refused (server not up yet), a timeout, or
a malformed server message all return an empty TranscriptResult rather
than raising.
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

logger = get_logger(component="vexyl_stt_transcriber")

_MODEL_SAMPLE_RATE = 16_000

# The real vexyl-stt server's own LANG_MAP keys (services/vexyl-stt/
# vendor/vexyl_stt_server.py) — both the "-IN" and bare short-code forms
# it accepts. routing_transcriber.py's default `indic_languages` comes
# from this, so the two never drift apart silently.
SUPPORTED_LANGUAGES = frozenset(
    {
        "ml", "hi", "ta", "te", "kn", "bn", "gu", "mr", "pa", "or", "as", "ur", "sa", "ne",
    }
)


class VexylSttTranscriber:
    name = "vexyl_stt_transcriber"
    version = "vexyl-stt@indic-conformer-600m"

    def __init__(
        self,
        host: str,
        port: int,
        default_lang: str = "hi",
        connect_timeout_s: float = 5.0,
        result_timeout_s: float = 20.0,
    ) -> None:
        self.host = host
        self.port = port
        self.default_lang = default_lang
        self.connect_timeout_s = connect_timeout_s
        self.result_timeout_s = result_timeout_s

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptResult:
        """Generic TranscriberPort entry point — uses `default_lang`.
        Only correct to wire directly as `transcription:`/
        `live_transcription:` when every call is known to be in one
        language (e.g. a single-language deployment); routing_transcriber.py
        is the one that picks a language per call and should call
        transcribe_lang() instead."""
        return self.transcribe_lang(samples, sample_rate, self.default_lang)

    def transcribe_lang(self, samples: np.ndarray, sample_rate: int, lang: str) -> TranscriptResult:
        """Same as transcribe(), but with an explicit language — the
        entry point routing_transcriber.py actually uses, once it has
        already decided (via a cheap Whisper language-ID pass) that this
        buffer belongs to one of SUPPORTED_LANGUAGES. `lang` isn't
        validated against SUPPORTED_LANGUAGES here — the server itself
        silently falls back to Malayalam on an unrecognised code (see
        this module's own WHY note); callers are expected to have
        already checked, same as this project's other ports never
        re-validate what an upstream caller already guaranteed."""
        if samples.size == 0:
            return self._empty(lang)

        if sample_rate != _MODEL_SAMPLE_RATE:
            samples = _resample(samples, sample_rate, _MODEL_SAMPLE_RATE)

        try:
            return self._transcribe_via_server(samples, lang)
        except Exception:  # noqa: BLE001 — best-effort, must never break scoring
            logger.exception("vexyl_stt_transcription_failed")
            return self._empty(lang)

    def _transcribe_via_server(self, samples: np.ndarray, lang: str) -> TranscriptResult:
        uri = f"ws://{self.host}:{self.port}"
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

        with connect(uri, open_timeout=self.connect_timeout_s) as ws:
            # Wait for the server's own "ready" before sending the start
            # handshake — verified against vexyl_stt_server.py's
            # handle_connection(): it sends this unprompted, immediately
            # on connect, before reading anything from the client.
            if not self._wait_for_type(ws, "ready", self.connect_timeout_s):
                return self._empty(lang)

            ws.send(json.dumps({"type": "start", "lang": lang, "session_id": str(uuid.uuid4())}))
            if not self._wait_for_type(ws, "started", self.connect_timeout_s):
                return self._empty(lang)

            chunk_bytes = 4096 * 2  # int16 = 2 bytes/sample, matches the WhisperLive adapter's own chunk size
            pcm_bytes = pcm16.tobytes()
            for offset in range(0, len(pcm_bytes), chunk_bytes):
                ws.send(pcm_bytes[offset : offset + chunk_bytes])

            ws.send(json.dumps({"type": "stop"}))
            return self._collect_result(ws, lang)

    def _wait_for_type(self, ws, expected_type: str, timeout: float) -> bool:
        message = self._recv_json(ws, timeout=timeout)
        return bool(message) and message.get("type") == expected_type

    def _collect_result(self, ws, lang: str) -> TranscriptResult:
        segment_texts: list[str] = []

        while True:
            message = self._recv_json(ws, timeout=self.result_timeout_s)
            if message is None:
                break  # timed out or connection closed — return what we have

            msg_type = message.get("type")
            if msg_type == "final" and message.get("text"):
                segment_texts.append(message["text"])
            elif msg_type == "stopped":
                break
            elif msg_type == "error":
                logger.warning("vexyl_stt_server_error", detail=message.get("message"))
                break

        text = " ".join(t.strip() for t in segment_texts if t.strip())
        return TranscriptResult(text=text, language=lang, detector_name=self.name, detector_version=self.version)

    @staticmethod
    def _recv_json(ws, timeout: float) -> Optional[dict[str, Any]]:
        try:
            raw = ws.recv(timeout=timeout)
        except Exception:  # noqa: BLE001 — TimeoutError, ConnectionClosed, etc.
            return None
        if isinstance(raw, (bytes, bytearray)):
            return None  # vexyl-stt's own replies are always JSON text frames
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def _empty(self, lang: str) -> TranscriptResult:
        return TranscriptResult(text="", language=lang, detector_name=self.name, detector_version=self.version)


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(samples, up, down).astype(np.float32)
