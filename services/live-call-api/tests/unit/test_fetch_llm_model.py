"""Tests for app/adapters/semantic_risk/fetch_llm_model.py's
ensure_model() logic — the first fetch_*.py script in this project with
enough real conditional logic (legacy-path reuse, cache_dir resolution,
checksum handling) to be worth a dedicated unit test; the other four
fetch_*.py scripts just delegate entirely to a library's own lazy-
download call (whisper.load_model(), EncoderClassifier.from_hparams(),
etc.) with no logic of their own to test.

Fakes huggingface_hub.hf_hub_download() throughout — no real ~2.4GB
network download, same "fake the expensive port, test the real logic"
style used everywhere else in this suite (e.g.
test_whisper_transcriber.py's fake model). EXPECTED_SHA256 is
monkeypatched per test to match small, fast, in-test fixture bytes
instead of the real pinned GGUF checksum.
"""
from __future__ import annotations

import hashlib

import pytest

from app.adapters.semantic_risk import fetch_llm_model


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_returns_legacy_flat_path_without_calling_hf_hub_download(tmp_path, monkeypatch):
    """REAL BUG found while writing this change: an old-layout flat file
    from before this fix already existed in this exact repo's own
    working copy — without this legacy check, everyone who already had
    it would silently pay for a redundant ~2.4GB re-download the next
    time they ran this script."""
    legacy_path = tmp_path / "legacy.gguf"
    legacy_path.write_bytes(b"fake gguf bytes")
    monkeypatch.setattr(fetch_llm_model, "_LEGACY_MODEL_PATH", legacy_path)
    monkeypatch.setattr(fetch_llm_model, "EXPECTED_SHA256", _sha256_bytes(b"fake gguf bytes"))
    monkeypatch.setattr(
        fetch_llm_model,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not be called — legacy path was valid")),
    )

    result = fetch_llm_model.ensure_model()

    assert result == legacy_path


def test_falls_through_to_hf_hub_download_when_legacy_path_absent(tmp_path, monkeypatch):
    legacy_path = tmp_path / "does-not-exist.gguf"
    downloaded_path = tmp_path / "downloaded.gguf"
    downloaded_path.write_bytes(b"real download bytes")
    monkeypatch.setattr(fetch_llm_model, "_LEGACY_MODEL_PATH", legacy_path)
    monkeypatch.setattr(fetch_llm_model, "EXPECTED_SHA256", _sha256_bytes(b"real download bytes"))

    calls = []

    def _fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded_path)

    monkeypatch.setattr(fetch_llm_model, "hf_hub_download", _fake_hf_hub_download)

    result = fetch_llm_model.ensure_model()

    assert result == downloaded_path
    assert len(calls) == 1
    assert calls[0]["repo_id"] == fetch_llm_model.REPO_ID
    assert calls[0]["filename"] == fetch_llm_model.FILENAME


@pytest.mark.parametrize(
    "env_vars,expect_explicit_cache_dir",
    [
        ({}, True),  # nothing set — falls back to the repo-tree default
        ({"HF_HUB_CACHE": "/tmp/somewhere"}, False),
        ({"HF_HOME": "/tmp/somewhere"}, False),
        ({"HUGGINGFACE_HUB_CACHE": "/tmp/somewhere"}, False),
    ],
)
def test_cache_dir_resolution_honours_huggingface_hubs_own_env_vars(
    tmp_path, monkeypatch, env_vars, expect_explicit_cache_dir
):
    """See this module's own CACHE LOCATION docstring note: the
    repo-tree default is used UNLESS the developer has opted into
    huggingface_hub's own standard env vars, in which case cache_dir is
    left as None so huggingface_hub resolves its own (now shared)
    default from those same variables."""
    for var in fetch_llm_model._HF_CACHE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    legacy_path = tmp_path / "does-not-exist.gguf"
    downloaded_path = tmp_path / "downloaded.gguf"
    downloaded_path.write_bytes(b"content")
    monkeypatch.setattr(fetch_llm_model, "_LEGACY_MODEL_PATH", legacy_path)
    monkeypatch.setattr(fetch_llm_model, "EXPECTED_SHA256", _sha256_bytes(b"content"))

    captured = {}

    def _fake_hf_hub_download(**kwargs):
        captured.update(kwargs)
        return str(downloaded_path)

    monkeypatch.setattr(fetch_llm_model, "hf_hub_download", _fake_hf_hub_download)

    fetch_llm_model.ensure_model()

    if expect_explicit_cache_dir:
        assert captured["cache_dir"] == str(fetch_llm_model.MODEL_DIR)
    else:
        assert captured["cache_dir"] is None


def test_checksum_mismatch_raises_and_does_not_delete_the_file(tmp_path, monkeypatch):
    """REAL BUG found while writing this change: the OLD version deleted
    a checksum-mismatched file automatically — safe when the cache was
    always repo-tree-local and owned outright by this one script, but
    now the cache directory can be huggingface_hub's own SHARED layout
    (see CACHE LOCATION note), which this script must not reach into and
    mutate. Proves the new behaviour: raise, leave the file alone."""
    legacy_path = tmp_path / "does-not-exist.gguf"
    downloaded_path = tmp_path / "downloaded.gguf"
    downloaded_path.write_bytes(b"corrupted or unexpected content")
    monkeypatch.setattr(fetch_llm_model, "_LEGACY_MODEL_PATH", legacy_path)
    monkeypatch.setattr(fetch_llm_model, "EXPECTED_SHA256", _sha256_bytes(b"something else entirely"))
    monkeypatch.setattr(fetch_llm_model, "hf_hub_download", lambda **kwargs: str(downloaded_path))

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        fetch_llm_model.ensure_model()

    assert downloaded_path.exists(), "the mismatched file must be left in place, not deleted"
