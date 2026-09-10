"""Tests for VexylSttTranscriber — the bridge to a vexyl-ai/vexyl-stt
server (see that module's own docstring for why).

Fakes `websockets.sync.client.connect` throughout — no real vexyl-stt
server, same "fake the expensive/external port, test the real logic"
style already used for WhisperLiveTranscriber
(test_whisperlive_transcriber.py), which this file's structure mirrors
closely since both adapters share the same "one WS connection per
transcribe() call" design.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.adapters.transcription import vexyl_stt_transcriber as vs_module
from app.adapters.transcription.vexyl_stt_transcriber import VexylSttTranscriber

SR = 16_000


class _FakeWS:
    def __init__(self, incoming: list) -> None:
        self._incoming = list(incoming)
        self.sent: list = []

    def send(self, data):
        self.sent.append(data)

    def recv(self, timeout=None):
        if not self._incoming:
            raise TimeoutError("fake server: no more scripted messages")
        return self._incoming.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _fake_connect(incoming: list):
    ws = _FakeWS(incoming)

    def _connect(uri, open_timeout=None):
        return ws

    return _connect, ws


def _ready() -> str:
    return json.dumps({"type": "ready", "model": "indic-conformer-600m-multilingual"})


def _started(session_id: str = "sess", lang: str = "hi") -> str:
    return json.dumps({"type": "started", "session_id": session_id, "lang": lang})


def _final(text: str, lang: str = "hi") -> str:
    return json.dumps({"type": "final", "text": text, "lang": lang, "duration": 1.0, "latency_ms": 10})


def _stopped() -> str:
    return json.dumps({"type": "stopped"})


@pytest.fixture
def transcriber() -> VexylSttTranscriber:
    return VexylSttTranscriber(host="vexyl-stt", port=8091, default_lang="hi")


def test_happy_path_returns_concatenated_final_segments(transcriber, monkeypatch):
    connect_fn, ws = _fake_connect(
        [_ready(), _started(), _final("namaste"), _final("kaise ho"), _stopped()]
    )
    monkeypatch.setattr(vs_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == "namaste kaise ho"
    assert result.language == "hi"
    assert result.detector_name == "vexyl_stt_transcriber"
    assert result.detector_version == "vexyl-stt@indic-conformer-600m"


def test_sends_start_handshake_before_any_audio(transcriber, monkeypatch):
    connect_fn, ws = _fake_connect([_ready(), _started(), _final("hi"), _stopped()])
    monkeypatch.setattr(vs_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    transcriber.transcribe(samples, SR)

    first_sent = json.loads(ws.sent[0])
    assert first_sent["type"] == "start"
    assert first_sent["lang"] == "hi"
    assert "session_id" in first_sent and first_sent["session_id"]
    # Everything between the start handshake and the stop message is raw
    # binary int16 PCM.
    audio_frames = ws.sent[1:-1]
    assert all(isinstance(f, (bytes, bytearray)) for f in audio_frames)
    assert json.loads(ws.sent[-1])["type"] == "stop"


def test_transcribe_lang_uses_the_given_language_not_the_default(transcriber, monkeypatch):
    connect_fn, ws = _fake_connect([_ready(), _started(lang="mr"), _final("ho", lang="mr"), _stopped()])
    monkeypatch.setattr(vs_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe_lang(samples, SR, "mr")

    assert result.language == "mr"
    handshake = json.loads(ws.sent[0])
    assert handshake["lang"] == "mr"


def test_two_calls_use_different_session_ids(transcriber, monkeypatch):
    seen_ids = []

    def _connect(uri, open_timeout=None):
        ws = _FakeWS([_ready(), _started(), _final("ok"), _stopped()])
        real_send = ws.send

        def _send(data):
            if isinstance(data, str):
                msg = json.loads(data)
                if msg.get("type") == "start":
                    seen_ids.append(msg["session_id"])
            real_send(data)

        ws.send = _send
        return ws

    monkeypatch.setattr(vs_module, "connect", _connect)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    transcriber.transcribe(samples, SR)
    transcriber.transcribe(samples, SR)

    assert len(seen_ids) == 2
    assert seen_ids[0] != seen_ids[1]


def test_abstains_when_server_never_sends_ready(transcriber, monkeypatch):
    connect_fn, ws = _fake_connect([])  # recv() raises TimeoutError immediately
    monkeypatch.setattr(vs_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == ""
    assert ws.sent == [], "must not send the start handshake if the server never said ready"


def test_abstains_when_server_never_confirms_started(transcriber, monkeypatch):
    connect_fn, ws = _fake_connect([_ready()])  # no "started" reply
    monkeypatch.setattr(vs_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == ""
    # Handshake was sent, but no audio — must not proceed without confirmation.
    assert len(ws.sent) == 1
    assert json.loads(ws.sent[0])["type"] == "start"


def test_never_raises_when_connection_fails(transcriber, monkeypatch):
    def _connect(uri, open_timeout=None):
        raise ConnectionRefusedError("vexyl-stt not up yet")

    monkeypatch.setattr(vs_module, "connect", _connect)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == ""


def test_empty_samples_abstain_without_connecting(transcriber, monkeypatch):
    def _connect(uri, open_timeout=None):
        raise AssertionError("must not connect for an empty buffer")

    monkeypatch.setattr(vs_module, "connect", _connect)

    result = transcriber.transcribe(np.array([], dtype=np.float32), SR)

    assert result.text == ""


def test_resamples_non_16k_audio_before_sending(transcriber, monkeypatch):
    connect_fn, ws = _fake_connect([_ready(), _started(), _final("ok"), _stopped()])
    monkeypatch.setattr(vs_module, "connect", connect_fn)
    samples_48k = np.random.default_rng(0).uniform(-0.5, 0.5, size=48_000).astype(np.float32)

    result = transcriber.transcribe(samples_48k, 48_000)

    assert result.text == "ok"
    total_audio_bytes = sum(len(f) for f in ws.sent[1:-1])
    # int16 = 2 bytes/sample; resampled 48k->16k roughly a third the sample count.
    assert total_audio_bytes == pytest.approx(16_000 * 2, rel=0.05)


def test_empty_final_text_is_not_included(transcriber, monkeypatch):
    """A 'final' message with no text (server-side VAD triggered but
    produced nothing) must not become a stray blank entry in the joined
    transcript."""
    connect_fn, ws = _fake_connect(
        [_ready(), _started(), _final(""), _final("real words"), _stopped()]
    )
    monkeypatch.setattr(vs_module, "connect", connect_fn)
    samples = np.random.default_rng(0).uniform(-0.5, 0.5, size=SR).astype(np.float32)

    result = transcriber.transcribe(samples, SR)

    assert result.text == "real words"
