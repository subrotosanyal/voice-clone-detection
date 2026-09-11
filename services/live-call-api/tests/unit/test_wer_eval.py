"""Tests for eval/wer_eval.py's word_error_rate()/normalize() — the
standalone Levenshtein-based WER implementation this project uses since
no jiwer/similar dependency is already installed (see that module's own
docstring for why this script exists at all: nothing else here measures
live-mic transcription ACCURACY, only latency).

Does not test simulate_live_path()/run() themselves — those drive real
transcriber objects and the real LiveTranscriptionBuffer end-to-end,
already covered indirectly by test_routing_transcriber.py,
test_whisperlive_transcriber.py, test_vexyl_stt_transcriber.py, and
test_live_transcription.py; this file only locks in the metric math
itself, which those don't touch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))

from wer_eval import normalize, word_error_rate  # noqa: E402


def test_identical_transcripts_have_zero_wer():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_one_substitution_out_of_two_words():
    assert word_error_rate("hello world", "hello there") == 0.5


def test_one_deletion_counts_against_the_reference_length():
    assert abs(word_error_rate("please transfer money", "please money") - (1 / 3)) < 1e-9


def test_one_insertion_still_counted_against_reference_length_not_hypothesis():
    assert word_error_rate("hello world", "hello big world") == 0.5


def test_punctuation_and_case_are_normalized_away():
    assert word_error_rate("Hello, World!", "hello world") == 0.0


def test_empty_reference_and_empty_hypothesis_is_zero_wer():
    assert word_error_rate("", "") == 0.0


def test_empty_reference_but_nonempty_hypothesis_is_full_wer():
    """An empty reference with SOME hypothesis text is scored as the
    worst case (1.0), not a division-by-zero crash or a 0.0 false
    "perfect match" — there's nothing to divide by, so this is a
    deliberate convention, not derived from the edit-distance formula."""
    assert word_error_rate("", "oops") == 1.0


def test_completely_wrong_hypothesis_can_exceed_100_percent():
    """A hypothesis much longer than the reference (e.g. hallucinated
    extra words) is a real, worse-than-total-mismatch outcome — WER is
    allowed to exceed 1.0, same as any standard ASR WER metric."""
    wer = word_error_rate("hi", "this is a very different much longer sentence")
    assert wer > 1.0


def test_normalize_collapses_whitespace_and_strips_punctuation():
    assert normalize("  Hello,   World!!  ") == "hello world"
