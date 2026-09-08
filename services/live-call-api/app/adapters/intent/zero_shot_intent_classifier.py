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

HONESTY NOTE — READ THIS BEFORE ENABLING THIS DETECTOR (2026-09-08):
this module is built and the model loads and runs correctly, but it is
NOT wired into config/risk_formula.yaml's active `detectors:` list, and
should not be enabled without addressing the finding below first.

Before wiring this up, three real transcripts were run through it by
hand: a clear fraud script ("this is your bank... your account has been
compromised... transfer your funds immediately"), a clear impersonation
script ("this is inspector from cyber crime cell... share the OTP"), and
two completely ordinary sentences ("are we still on for dinner tonight?",
"the weather has been really nice this week"). The fraud/impersonation
examples classified sensibly. Both ordinary sentences did NOT: "dinner
tonight" scored just 4.5% for "ordinary conversation" against 38.9% for
"creating urgency or time pressure" — a severe false positive on
completely benign text. This was tested three ways (default settings,
`multi_label=True`, and a phone-call-specific `hypothesis_template`) and
the miscalibration persisted in all three. The likely cause: a 4-candidate
"risky" set against only 1 "ordinary" candidate structurally disadvantages
the negative class in `multi_label=False` mode, and `multi_label=True`
scores each hypothesis independently against the whole premise, which this
model does not do reliably for short, informal spoken-style text.

This is NOT a reason to delete the work — the port, adapter, detector,
and wiring are all real and correct, and this stands as a genuine,
verified option once the calibration issue is addressed (candidates:
several distinct "ordinary" hypotheses instead of one, a larger held-out
evaluation set before trusting any threshold, or a different base model).
It IS a reason not to let an unvalidated signal silently score real calls
in a fraud-detection product. Re-run the three-sentence check above by
hand before ever adding an `intent:` entry to config/risk_formula.yaml's
`detectors:` list.
"""
from __future__ import annotations

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
        self._pipeline = None  # lazy-loaded — see _ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
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
