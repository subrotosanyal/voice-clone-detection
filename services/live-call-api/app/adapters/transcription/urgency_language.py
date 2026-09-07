"""Urgency/financial-request language detection over a real transcript.

WHY KEYWORD MATCHING, NOT A SENTIMENT MODEL: the "sentiment" this project
actually needs isn't generic positive/negative polarity — it's specific,
fraud-relevant language ("transfer the money now", "don't tell anyone").
A transparent keyword list keeps every match traceable to an actual word
in the actual transcript (shown in ContextualRulesDetector's `detail`),
matching the same "legible to a non-technical reader" design goal
contextual_rules.py's own docstring already states for its other rules —
a black-box sentiment score wouldn't explain itself the same way. See
docs/risk-model.md for the full reasoning and for what a real sentiment
model would add if this is ever revisited.

HONESTY NOTE: these keyword lists are a starting point, not validated
against real fraud-call transcripts — same class of caveat as the
prosodic/voiceprint detectors' placeholder thresholds elsewhere in this
project. The Hindi/Marathi lists were drafted for coverage (and share
vocabulary with eval/indian_language/prompts/sentences.md's fixed
sentences, deliberately, for consistency) and have not been reviewed by
a native speaker — flag the same way sentences.md already does.
"""
from __future__ import annotations

_URGENCY_KEYWORDS: dict[str, list[str]] = {
    "en": [
        "immediately", "urgent", "urgently", "right now", "hurry", "as soon as possible",
        "asap", "emergency", "don't tell anyone", "before it's too late", "quickly",
    ],
    "hi": [
        "तुरंत", "जल्दी", "अभी", "जरूरी", "आपातकाल", "किसी को न बताएं", "तत्काल",
    ],
    "mr": [
        "लगेच", "तातडीने", "तातडीचे", "आत्ता", "आणीबाणी", "कोणालाही सांगू नका",
    ],
}

_FINANCIAL_KEYWORDS: dict[str, list[str]] = {
    "en": [
        "transfer", "account", "money", "payment", "funds", "bank", "transaction",
        "wire", "otp", "pin", "balance",
    ],
    "hi": [
        "पैसे", "ट्रांसफर", "खाता", "खाते", "भुगतान", "बैंक", "लेन-देन", "राशि",
    ],
    "mr": [
        "पैसे", "ट्रान्सफर", "खाते", "बँक", "व्यवहार", "पेमेंट",
    ],
}


def detect_urgency_signals(text: str, language: str | None = None) -> tuple[list[str], bool]:
    """Returns (matched_urgency_phrases, is_financial_request_detected) for
    a transcript. Checks every language's list rather than only the
    detected `language`, since Whisper's language ID isn't perfectly
    reliable and a phrase match is cheap and harmless either way."""
    if not text:
        return [], False

    text_lower = text.lower()
    matched_urgency: list[str] = []
    for keywords in _URGENCY_KEYWORDS.values():
        for phrase in keywords:
            if phrase.lower() in text_lower and phrase not in matched_urgency:
                matched_urgency.append(phrase)

    is_financial = any(
        phrase.lower() in text_lower for keywords in _FINANCIAL_KEYWORDS.values() for phrase in keywords
    )

    return matched_urgency, is_financial
