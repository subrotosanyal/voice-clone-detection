"""LocalLLMSemanticClassifier — a generative LLM reading the call
transcript for social-engineering patterns, in place of (or alongside)
the zero-shot NLI classifier's fixed candidate-label scoring.

WHY THIS EXISTS: app/adapters/intent/zero_shot_intent_classifier.py's
own HONESTY NOTE documents a real, measured false positive — "are we
still on for dinner tonight?" scored 38.9% "creating urgency" against
4.5% "ordinary conversation". A zero-shot NLI model is forced to
distribute probability across a fixed candidate-label set without any
actual reasoning about the specific conversation; a small instruction-
tuned generative model, asked to directly judge the transcript against
explicit criteria, may do better — see
tests/unit/test_local_llm_semantic_classifier.py's real-model
comparison against that exact same "dinner tonight" sentence before
trusting this over the zero-shot classifier.

Also captures `isolation_request` ("don't hang up the phone", "don't
tell anyone") — a real social-engineering tactic neither the zero-shot
classifier's candidate list nor the keyword-based contextual rules
(app/adapters/detectors/contextual_rules.py) currently look for at all.

MODEL — verified before adoption (see docs/risk-model.md's standing
discipline):
- Source: microsoft/Phi-3-mini-4k-instruct-gguf (the
  Phi-3-mini-4k-instruct-q4.gguf file — Microsoft's own official
  4-bit-quantized GGUF release, not a third-party requantization)
- Licence: MIT (confirmed via the HF API — `license: "mit"`)
- Confirmed ungated on HuggingFace (`gated: false`, plain unauthenticated
  download, no token needed — same "no auth, it just works" bar every
  other model in this project has to clear)
- Runs entirely locally, CPU-only, via llama-cpp-python (a llama.cpp
  binding) — no external API call at runtime, matching every other
  model in this pipeline.

HONESTY NOTE — read before trusting `reasoning`: it is a plain-language
explanation for a human to read, NOT itself a computation this system
can verify or replay the way a raw feature (jitter, a cosine similarity,
a keyword match) can be. This is precisely why SemanticRiskAssessment's
own docstring says reasoning must never be used to override another
detector's score — doing that (the "Explainable Fusion Arbitrator" idea
discussed and explicitly rejected) would let an attacker influence their
own risk assessment just by how they phrase things, and would break
this project's reproducibility guarantee (a free-form LLM narrative is
not "reproducibly derived from raw feature values" the way
FusedScore's own docstring requires).

FAILURE MODE, verified not assumed: a small quantized model does not
always produce valid JSON on the first try, even when explicitly
instructed to. On any parse failure, this returns a NEUTRAL assessment
(urgency_level=0.0, every bool False) with the parse failure recorded in
`reasoning` — the same "never let an infrastructure failure manufacture
a false positive" discipline every detector in this project follows
(e.g. abstaining, not scoring high, on a near-silent window).

REAL BUG found and fixed 2026-09-09, via a real user-uploaded ~4-minute
conversation recording (the exact same upload that surfaced the Whisper
long-file crash fixed in whisper_transcriber.py the same day): once
that Whisper fix let a real, long transcript (~3400 characters, ~840
tokens) through, THIS classifier crashed instead — `ValueError:
Requested tokens (1062) exceed context window of 1024`. The original
`n_ctx=1024` default only had headroom for a short example sentence,
not a real multi-minute conversation. Fixed two ways: `n_ctx` raised to
4096 (Phi-3-mini-4k-instruct's own native context size — using the full
capacity the model was actually built for, not an arbitrary guess), and
the transcript itself is truncated to `_MAX_TRANSCRIPT_CHARS` before
building the prompt, so an even longer real call still can't overflow
the window. Truncation keeps the LAST N characters, not the first — an
honest, undecided trade-off documented here, not a validated choice:
a scam's actual financial ask often lands late in the call, but an
opening authority claim could be lost if the call is long enough to
truncate. Also now wraps the model call itself in a broad try/except,
same neutral-fallback discipline as the JSON-parse failure above — this
project's own track record this session (three separate transcript-fed
models crashing on real, unanticipated inputs) is reason enough not to
assume this is the last failure mode either.

LATENCY, measured by hand on this same real transcript after the fix:
~18.7s (vs ~1-1.5s for the short example sentences this was originally
verified against) — a real cost of the larger n_ctx and the longer
prompt itself, not a regression to chase down. Acceptable for a one-
shot file-upload analysis (Whisper's own transcription of a multi-
minute file already takes a comparable amount of time), but this is the
real number, not the ~1-1.5s figure quoted elsewhere in this module for
short inputs — don't assume that number holds for a long transcript.

STATUS (2026-09-09): wired into Engine.score_call() and the active risk
formula (config/risk_formula.yaml's `semantic_risk` entry, weight 0.15,
running alongside — not replacing — `intent`). See that config entry
and app/ports/semantic_risk_classifier.py's own STATUS note for the
real evaluation evidence behind this.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

from app.domain.models import SemanticRiskAssessment

_SYSTEM_PROMPT = (
    "You are a fraud-detection assistant analysing a live phone call transcript "
    "(which may be in Hindi, English, or Hinglish code-switched text). Determine "
    "whether the speaker is exhibiting social-engineering / scam-call tactics.\n\n"
    'Respond with ONLY a single JSON object, no other text, using EXACTLY these keys:\n'
    '{"urgency_level": <float 0.0-1.0>, "financial_solicitation": <true/false>, '
    '"authority_claim": <true/false>, "isolation_request": <true/false>, '
    '"reasoning": "<one short sentence>"}\n\n'
    "financial_solicitation: is the speaker asking for money, a transfer, an OTP, or a PIN?\n"
    "authority_claim: is the speaker claiming to be a bank, police, government, or other authority?\n"
    "isolation_request: is the speaker telling the listener not to hang up, not to tell anyone, "
    "or not to consult someone else?\n"
    "urgency_level: how much manufactured time-pressure or urgency is present?\n"
    "An ordinary, benign conversation should score LOW on all of these — do not treat normal "
    "small talk, plans, or requests as suspicious just because it involves a specific topic."
)

# See the FAILURE MODE note above the class docstring — returned whenever
# the model's output can't be parsed as the required JSON shape.
_NEUTRAL_REASONING_PREFIX = "could not parse a structured judgment from the model"

# See the REAL BUG note above the class docstring: ~8000 characters is a
# generous margin under n_ctx=4096 even after the ~260-token system
# prompt and the completion budget — the real transcript that overflowed
# the OLD 1024-token window was only ~3400 characters.
_MAX_TRANSCRIPT_CHARS = 8000


class LocalLLMSemanticClassifier:
    """Wraps llama-cpp-python's Llama class. See this module's own
    HONESTY NOTE before wiring this into the active risk formula."""

    name = "local_llm_semantic_classifier"
    version = "0.1.0-phi3-mini-q4"

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        max_tokens: int = 150,
    ) -> None:
        # `model_path` is an explicit override (tests pin a specific file via
        # this). Left unset by default rather than hardcoding the old flat
        # "app/adapters/semantic_risk/.cache/Phi-3-mini-4k-instruct-q4.gguf"
        # path here — REAL BUG, found via a real user upload crashing every
        # /v1/score/file call with `ValueError: Model path does not exist`:
        # fetch_llm_model.py's migration to huggingface_hub's own cache
        # (see that module's MIGRATION NOTE) nests the downloaded file under
        # models--.../snapshots/<rev>/<filename> instead of that flat path,
        # but config/risk_formula.yaml's `model_path:` param — and this
        # class's old default — were never updated to match, so a fresh
        # Docker build (no pre-existing flat-layout file) always pointed at
        # a path that no longer existed. tests/conftest.py's
        # phi3_llm_model_path fixture already resolved the real path
        # correctly via fetch_llm_model.ensure_model(), which is exactly why
        # this regression passed CI while breaking every real deployment.
        # _ensure_loaded() below now does the same resolution the fixture
        # does, so the two can't drift apart again.
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self._llm = None  # lazy-loaded — see _ensure_loaded()

    def _ensure_loaded(self) -> None:
        # REAL BUG found 2026-09-11 (CI): this used to be a plain,
        # unsynchronized `if self._llm is None:` check — safe as long as
        # only one caller at a time ever reached analyze() on a given
        # instance, which was true until Engine.diarize_and_score()
        # started scoring different speakers CONCURRENTLY (see that
        # method's own REAL BUG note) via a thread pool, all sharing the
        # SAME Engine (and therefore the SAME semantic_risk_classifier
        # instance). Two threads could both see self._llm is None and
        # both start constructing a NEW Llama(...) — each a full ~2.4GB
        # GGUF load — concurrently: not a crash (only one survives as
        # self._llm, and self._lock already served every actual inference
        # call), but real wasted CPU/disk/memory from a redundant
        # simultaneous model load, observed in a real CI run as the
        # underlying diarize_and_score() work taking so long it blew past
        # its own 120s internal timeout (app/api/ws_router.py's
        # _HANGUP_DIARIZE_TIMEOUT_S) and kept running for several more
        # minutes in the background afterward. Double-checked locking
        # with the SAME self._lock analyze()'s inference call already
        # uses — sequential loading and sequential inference on one
        # instance, same "one shared expensive model, one lock" pattern
        # this project already applies to Parselmouth (PRAAT_LOCK) and
        # Whisper (WhisperTranscriber._lock).
        if self._llm is not None:
            return
        with self._lock:
            if self._llm is not None:  # another thread may have just finished loading
                return
            from llama_cpp import Llama  # local import: heavy, only needed if actually used

            model_path = self.model_path
            if not model_path or not Path(model_path).exists():
                # No explicit override, or an override that's gone stale
                # (e.g. the pre-migration flat path — see __init__'s note
                # above) — resolve the real, current path the same way
                # tests/conftest.py's phi3_llm_model_path fixture does.
                # Idempotent and cheap when already cached: see
                # fetch_llm_model.ensure_model()'s own docstring.
                from app.adapters.semantic_risk.fetch_llm_model import ensure_model

                model_path = str(ensure_model())

            self._llm = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )

    def analyze(self, text: str) -> SemanticRiskAssessment:
        if not text or not text.strip():
            return self._neutral_assessment(text, reason="empty transcript — nothing to analyse")

        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001 — same "never let an infra
            # failure crash the whole request instead of abstaining"
            # discipline as the inference-call except below (and every other
            # detector in this pipeline): a missing/corrupted model file
            # must not take down every OTHER detector's result along with
            # this one. See __init__'s REAL BUG note for the actual failure
            # this was written for.
            return self._neutral_assessment(text, reason=f"model load failed: {exc}")

        # See the REAL BUG note above the class docstring: a real, long
        # transcript can overflow the context window on its own — this
        # truncates the PROMPT input, not the `text` field recorded on
        # the returned assessment, so detail/logging still show the full
        # original transcript even though the model only saw part of it.
        prompt_text = text[-_MAX_TRANSCRIPT_CHARS:] if len(text) > _MAX_TRANSCRIPT_CHARS else text
        prompt = f"<|user|>\n{_SYSTEM_PROMPT}\n\nTranscript: \"{prompt_text}\"<|end|>\n<|assistant|>"

        try:
            with self._lock:
                # temperature=0.0: this project's whole design point is
                # explainable, REPLAY-IDENTICAL scoring (see docs/risk-
                # model.md's "Reproducibility" section) — a nonzero
                # temperature would make this the one non-deterministic
                # signal in an otherwise fully reproducible pipeline.
                completion = self._llm(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    stop=["<|end|>", "<|user|>"],
                )
        except Exception as exc:  # noqa: BLE001 — see the REAL BUG note above the
            # class docstring: this project's track record this session is
            # three separate transcript-fed models crashing on real,
            # unanticipated inputs — never assume this is the last one.
            return self._neutral_assessment(text, reason=f"model call failed: {exc}")

        raw_text = completion["choices"][0]["text"]
        parsed = _extract_json(raw_text)
        if parsed is None:
            return self._neutral_assessment(
                text, reason=f"{_NEUTRAL_REASONING_PREFIX}: {raw_text[:200]!r}"
            )

        try:
            return SemanticRiskAssessment(
                text=text,
                urgency_level=float(parsed["urgency_level"]),
                financial_solicitation=bool(parsed["financial_solicitation"]),
                authority_claim=bool(parsed["authority_claim"]),
                isolation_request=bool(parsed["isolation_request"]),
                reasoning=str(parsed.get("reasoning", "")),
                detector_name=self.name,
                detector_version=self.version,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._neutral_assessment(
                text, reason=f"{_NEUTRAL_REASONING_PREFIX} (missing/invalid field: {exc})"
            )

    def _neutral_assessment(self, text: str, reason: str) -> SemanticRiskAssessment:
        return SemanticRiskAssessment(
            text=text,
            urgency_level=0.0,
            financial_solicitation=False,
            authority_claim=False,
            isolation_request=False,
            reasoning=reason,
            detector_name=self.name,
            detector_version=self.version,
        )


def _extract_json(raw_text: str) -> Optional[dict[str, Any]]:
    """Small quantized models don't always emit ONLY the JSON object even
    when instructed to — this pulls out the first balanced-looking
    {...} block rather than requiring the whole completion to be clean
    JSON. Returns None (not an exception) on any failure — see the
    FAILURE MODE note above the class docstring for why that matters."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None
