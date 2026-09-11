"""Zero-shot fraud-intent classification over a real transcript.

WHY ZERO-SHOT, NOT A FIXED-CLASS FINE-TUNED CLASSIFIER: a zero-shot NLI
model scores the transcript against an explicit, visible list of
candidate hypotheses (see _CANDIDATE_LABELS below) rather than a fixed,
opaque set of classes baked into training — every score this produces can
be traced back to "the model judged this transcript X% consistent with
hypothesis Y", for every candidate, not just the winner. That keeps the
same transparency discipline the rest of this project's detectors follow
(see contextual_rules.py, urgency_language.py) while, in principle,
generalising beyond the exact keyword matches those two modules rely on.

MODEL — verified before adoption (see docs/risk-model.md's standing
discipline):
- Source: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
- Licence: MIT (explicit tag on the model card, confirmed via the HF API
  — unlike the sentiment-analysis candidate discussed but not adopted
  elsewhere in this project, this one has no ambiguity)
- Confirmed ungated on HuggingFace (`gated: false`)
- Real language coverage, not assumed: fine-tuned on the XNLI dataset,
  which includes Hindi among its 15 languages — real supervised coverage,
  not a guess. Marathi is NOT in XNLI; any Marathi result here is
  unvalidated cross-lingual transfer from the base model's 100-language
  CC100 pretraining, same caveat class as the cardiffnlp sentiment model
  discussed for the separate, still-unbuilt sentiment feature.

HONESTY NOTE — READ THIS BEFORE TOUCHING THE CALIBRATION (2026-09-08,
enabled 2026-09-08): this module is wired into config/risk_formula.yaml's
active `detectors:` list (the `intent` entry) at explicit user
instruction, despite a real, documented calibration problem — see below.
It was NOT enabled lightly: the finding was surfaced in full first.

Before this was first built, three real transcripts were run through it
by hand: a clear fraud script ("this is your bank... your account has
been compromised... transfer your funds immediately"), a clear
impersonation script ("this is inspector from cyber crime cell... share
the OTP"), and two completely ordinary sentences ("are we still on for
dinner tonight?", "the weather has been really nice this week"). The
fraud/impersonation examples classified sensibly. Both ordinary sentences
did NOT: "dinner tonight" scored just 4.5% for "ordinary conversation"
against 38.9% for "creating urgency or time pressure" — a severe false
positive on completely benign text. This was tested three ways (default
settings, `multi_label=True`, and a phone-call-specific
`hypothesis_template`) and the miscalibration persisted in all three. The
likely cause: a 4-candidate "risky" set against only 1 "ordinary"
candidate structurally disadvantages the negative class in
`multi_label=False` mode, and `multi_label=True` scores each hypothesis
independently against the whole premise, which this model does not do
reliably for short, informal spoken-style text.

**What mitigates the risk of shipping this anyway**: a deliberately low
fusion weight (0.15, see config/risk_formula.yaml — auto-renormalised
against acoustic/prosodic/third_signal, so it can meaningfully raise a
score but is unlikely to single-handedly push a genuine bonafide call
into High on its own), and full UI transparency — every candidate label's
own score is shown in the dashboard (app/ui/app.js), not just the winner,
so a false positive here is visible and inspectable, not hidden inside
one opaque number. This does NOT fix the miscalibration; it means an
operator sees "creating urgency or time pressure: 39%, ordinary
conversation: 4%" and can judge for themselves, the same trust model this
project's other honestly-flagged heuristic thresholds (prosodic,
voiceprint consistency) already rely on.

Real follow-up work, not done yet: several distinct "ordinary" hypotheses
instead of one, a larger held-out evaluation set before trusting any
threshold, or a different base model.
"""
from __future__ import annotations

import threading

from app.domain.models import IntentClassificationResult

# A starting point, not validated against real fraud-call transcripts —
# same class of caveat as urgency_language.py's keyword lists. See the
# HONESTY NOTE above: the "ordinary conversation" negative class in
# particular is known not to work reliably yet.
_CANDIDATE_LABELS = [
    "requesting a money transfer or payment",
    "requesting an OTP or PIN",
    "impersonating a bank or government official",
    "creating urgency or time pressure",
    "ordinary conversation",
]

_ORDINARY_LABEL = "ordinary conversation"


class ZeroShotIntentClassifier:
    """Wraps transformers' zero-shot-classification pipeline. See this
    module's own HONESTY NOTE before wiring this into the active
    risk formula."""

    name = "zero_shot_intent_classifier"
    version = "0.1.0-mdeberta-v3-mnli-xnli"

    def __init__(
        self,
        model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        cache_dir: str = "app/adapters/intent/.cache",
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._pipeline = None  # lazy-loaded — see _ensure_loaded()

    def _ensure_loaded(self) -> None:
        # REAL BUG found 2026-09-11, via a home-lab production log
        # (`ImportError: cannot import name 'pipeline' from 'transformers'`,
        # raised inside Engine.diarize_and_score()'s per-speaker scoring):
        # this used to be a plain, unsynchronized `if self._pipeline is
        # None:` check — safe only as long as a single Engine (and
        # therefore a single shared ZeroShotIntentClassifier instance,
        # see app/main.py's one-time app.state.engine construction) was
        # never asked to classify() from more than one thread at once.
        # That stopped being true once Engine.diarize_and_score() started
        # scoring different speakers CONCURRENTLY via a thread pool (see
        # that method's own REAL BUG note) — the exact same shape of race
        # already found and fixed in
        # app/adapters/semantic_risk/local_llm_semantic_classifier.py's
        # own _ensure_loaded(), just missed here at the time. Two threads
        # both seeing self._pipeline is None and both racing into
        # `from transformers import pipeline` for the first time in this
        # process is consistent with the observed failure: transformers'
        # top-level `__init__.py` is a lazy module (attributes are
        # resolved into `sys.modules`/internal caches on first access, not
        # eagerly at import time), and that lazy resolution is not
        # documented as safe against two threads triggering it
        # concurrently for the same name. Fixed the same way as the LLM
        # classifier: double-checked locking around the load, using the
        # SAME lock classify() below now also takes for the actual
        # inference call — "one shared expensive model, one lock", the
        # same pattern this project already applies to Whisper, Parselmouth,
        # and the local LLM classifier.
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:  # another thread may have just finished loading
                return
            from transformers import pipeline  # local import: heavy, only needed if actually used

            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # CPU-only, same reasoning as every other model in this project
                # Only the model weights land in cache_dir — transformers'
                # pipeline() has no equivalent kwarg for the tokenizer, so
                # it falls back to the default ~/.cache/huggingface. Not
                # worth over-engineering for a feature that isn't wired
                # into the live formula yet — see this module's HONESTY
                # NOTE.
                model_kwargs={"cache_dir": self.cache_dir},
            )

    def classify(self, text: str) -> IntentClassificationResult:
        self._ensure_loaded()
        # Same lock as the load above — untested against concurrent
        # inference on one shared pipeline instance, so serialize actual
        # calls too, same "one shared expensive model, one lock"
        # discipline as local_llm_semantic_classifier.py's analyze().
        with self._lock:
            result = self._pipeline(text, candidate_labels=_CANDIDATE_LABELS)
        label_scores = {label: float(score) for label, score in zip(result["labels"], result["scores"])}
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        return IntentClassificationResult(
            text=text,
            top_label=top_label,
            top_score=top_score,
            label_scores=label_scores,
            detector_name=self.name,
            detector_version=self.version,
        )
