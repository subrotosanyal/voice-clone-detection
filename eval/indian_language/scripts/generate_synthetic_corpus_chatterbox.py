#!/usr/bin/env python3
"""Generates synthetic (spoof) Hindi speech via Resemble AI's Chatterbox
(Multilingual) — the SECOND, independent synthesis system the blueprint's
held-out-generator design needs, alongside generate_synthetic_corpus.py's
XTTS-v2 output. Same reference-speaker/sentence-set pattern (shared via
_corpus_utils.py), a genuinely different model family and training data,
so "does the fine-tuned recalibration generalise, or did it just memorise
XTTS-v2's specific artifacts" finally has real evidence to answer with.

MUST be run with its own dedicated venv, NOT eval/indian_language's main
one:

    python3 -m venv .venv-chatterbox
    source .venv-chatterbox/bin/activate
    pip install chatterbox-tts "setuptools<81"  # see REAL BUG note below
    python scripts/generate_synthetic_corpus_chatterbox.py

WHY A SEPARATE VENV, verified by hand not assumed: chatterbox-tts hard-pins
torch==2.6.0 and transformers==5.2.0. The main eval venv already has
torch==2.11.0/transformers==4.57.6 for coqui-tts (XTTS-v2) — and this
project's own requirements.txt already documents "transformers>=5 breaks
coqui-tts's bundled XTTS code". `pip install chatterbox-tts` in the main
venv was confirmed (dry-run) to downgrade torch to 2.6.0 and upgrade
transformers to 5.2.0, which would break the existing XTTS-v2 setup —
not a hypothetical risk, a measured one.

REAL BUG found and fixed while setting this up (2026-09-08): Chatterbox
embeds its own output with Resemble's Perth watermark internally (see
services/live-call-api/app/adapters/detectors/perth_watermark.py for the
DETECTION side of this same library) — and a fresh `python3 -m venv`
lacks `pkg_resources` by default in recent Python/setuptools, which
`perth`'s code still imports. That ImportError gets silently swallowed
(perth/__init__.py sets PerthImplicitWatermarker = None instead of
raising), which then surfaces deep inside Chatterbox as a cryptic
"'NoneType' object is not callable". Fixed by pinning `setuptools<81`
(the exact fix resemble-perth's own deprecation warning recommends) in
the dedicated venv — not needed in the main eval venv or
services/live-call-api's venv, which both already had a compatible
setuptools/pkg_resources from earlier, unrelated installs.

Licence note: chatterbox-tts is MIT (github.com/resemble-ai/chatterbox,
confirmed via the GitHub API's own license detection).

VERIFIED BY HAND before adopting: generated one real Hindi clip from a
genuine reference speaker and fed it back through this project's own
already-verified Whisper transcriber — detected language 'hi', transcript
closely matched the intended text (minor phonetic-spelling variance
only), confirming this produces genuinely intelligible Hindi, not just
audio that doesn't crash.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _corpus_utils import parse_hindi_sentences, pick_reference_speakers  # noqa: E402

DATA_DIR = _EVAL_DIR / "data" / "synthetic_chatterbox"
GENUINE_MANIFEST = _EVAL_DIR / "data" / "genuine" / "manifest.json"
SENTENCES_MD = _EVAL_DIR / "prompts" / "sentences.md"

_LICENSE_NOTE = (
    "Resemble AI Chatterbox Multilingual (MIT, github.com/resemble-ai/chatterbox); "
    "voice-cloned from a genuine speaker in the CC BY 4.0 IndicTTS-Hindi corpus"
)


def main() -> None:
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    genuine = json.loads(GENUINE_MANIFEST.read_text())
    sentences = parse_hindi_sentences(SENTENCES_MD)
    print(f"Loaded {len(sentences)} Hindi sentences from {SENTENCES_MD.name}")

    reference_speakers = pick_reference_speakers(genuine, _EVAL_DIR)
    print(f"Reference speakers ({len(reference_speakers)} total): {list(reference_speakers.keys())}")
    for key, ref in reference_speakers.items():
        print(f"  {key} (gender={ref['gender']}): {ref['utterance_id']} ({ref['_duration']:.1f}s)")

    print("\nLoading Chatterbox Multilingual model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    print(f"Loaded — native sample rate {model.sr}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    i = 0
    for key, ref in reference_speakers.items():
        reference_wav = _EVAL_DIR / ref["wav_path"]
        for text in sentences:
            utterance_id = f"hi_chatterbox_{i:05d}"
            wav_path = DATA_DIR / f"{utterance_id}.wav"
            print(f"  synthesizing {utterance_id} (ref={key}): {text[:40]}...")
            wav = model.generate(text, language_id="hi", audio_prompt_path=str(reference_wav))
            ta.save(str(wav_path), wav, model.sr)
            manifest_entries.append(
                {
                    "utterance_id": utterance_id,
                    "language": "hi",
                    "text": text,
                    "source_dataset": "resemble-ai/chatterbox",
                    "license": _LICENSE_NOTE,
                    "gender": ref["gender"],
                    "reference_utterance_id": ref["utterance_id"],
                    "wav_path": str(wav_path.relative_to(_EVAL_DIR)),
                    "sample_rate": model.sr,
                }
            )
            i += 1

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(manifest_entries)} synthetic entries to {manifest_path}")


if __name__ == "__main__":
    main()
