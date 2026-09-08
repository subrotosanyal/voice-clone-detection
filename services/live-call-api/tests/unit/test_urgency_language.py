from app.adapters.transcription.urgency_language import detect_urgency_signals


def test_empty_text_returns_nothing():
    keywords, is_financial, authority = detect_urgency_signals("")
    assert keywords == []
    assert is_financial is False
    assert authority == []


def test_detects_english_urgency_and_financial_language():
    keywords, is_financial, authority = detect_urgency_signals("Please transfer the money immediately, it's urgent.")
    assert "immediately" in keywords
    assert "urgent" in keywords
    assert is_financial is True
    assert authority == []


def test_detects_hindi_urgency_and_financial_language():
    keywords, is_financial, _ = detect_urgency_signals("मुझे चाहिए कि आप तुरंत पैसे ट्रांसफर करें")
    assert "तुरंत" in keywords
    assert is_financial is True


def test_detects_marathi_urgency_language():
    keywords, is_financial, _ = detect_urgency_signals("तुम्ही लगेच पैसे ट्रान्सफर करा")
    assert "लगेच" in keywords
    assert is_financial is True


def test_neutral_sentence_detects_nothing():
    keywords, is_financial, authority = detect_urgency_signals("Good morning, how are you today?")
    assert keywords == []
    assert is_financial is False
    assert authority == []


def test_matching_is_case_insensitive():
    keywords, _, _ = detect_urgency_signals("URGENT: please respond IMMEDIATELY")
    assert "urgent" in keywords
    assert "immediately" in keywords


def test_no_duplicate_keywords_across_languages():
    # "पैसे" appears in both the Hindi and Marathi financial lists —
    # matching should still only report each distinct phrase once.
    keywords, _, _ = detect_urgency_signals("तुरंत लगेच पैसे")
    assert keywords.count("तुरंत") <= 1
    assert keywords.count("लगेच") <= 1


def test_detects_english_authority_claim_language():
    _, _, authority = detect_urgency_signals("This is your bank calling, please cooperate.")
    assert "this is your bank" in authority


def test_detects_hindi_authority_claim_language():
    _, _, authority = detect_urgency_signals("मैं पुलिस से बोल रहा हूं")
    assert "पुलिस" in authority


def test_authority_claim_alone_does_not_imply_financial_or_urgency():
    # Deliberately avoids the word "bank" — it's also a financial keyword,
    # which would make this a bad example of "authority claim alone".
    keywords, is_financial, authority = detect_urgency_signals(
        "This is the police calling about a noise complaint at your address."
    )
    assert authority == ["police"]
    assert is_financial is False
    assert keywords == []
