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


class LocalLLMSemanticClassifier:
    """Wraps llama-cpp-python's Llama class. See this module's own
    HONESTY NOTE before wiring this into the active risk formula."""

    name = "local_llm_semantic_classifier"
    version = "0.1.0-phi3-mini-q4"

    def __init__(
        self,
        model_path: str = "app/adapters/semantic_risk/.cache/Phi-3-mini-4k-instruct-q4.gguf",
        n_ctx: int = 1024,
        n_threads: Optional[int] = None,
        max_tokens: int = 150,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self._lock = threading.Lock()
        self._llm = None  # lazy-loaded — see _ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._llm is None:
            from llama_cpp import Llama  # local import: heavy, only needed if actually used

            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )

    def analyze(self, text: str) -> SemanticRiskAssessment:
        if not text or not text.strip():
            return self._neutral_assessment(text, reason="empty transcript — nothing to analyse")

        self._ensure_loaded()
        prompt = f"<|user|>\n{_SYSTEM_PROMPT}\n\nTranscript: \"{text}\"<|end|>\n<|assistant|>"

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
