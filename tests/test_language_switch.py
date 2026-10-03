"""agent/language_switch.py (KCD-438): an explicit request to speak
another language, detected deterministically in the language the request
itself was made in. Pure and offline.

    python -m pytest tests/test_language_switch.py -v
"""

from agent.language_switch import detect_language_switch_request


def test_bengali_caller_asks_for_hindi_or_english():
    assert detect_language_switch_request("আপনি হিন্দিতে বলতে পারবেন?", "bn") == "hi"
    assert detect_language_switch_request("ইংরেজিতে বলুন প্লিজ", "bn") == "en"


def test_hindi_caller_asks_for_english_or_bengali():
    assert detect_language_switch_request("क्या आप अंग्रेज़ी में बोल सकते हैं", "hi") == "en"
    assert detect_language_switch_request("बंगाली में बोलिए", "hi") == "bn"


def test_english_caller_asks_for_hindi_or_bengali():
    assert detect_language_switch_request("Could you speak in Hindi please", "en") == "hi"
    assert detect_language_switch_request("Can you talk in Bengali", "en") == "bn"


def test_ordinary_utterances_are_not_mistaken_for_a_switch_request():
    assert detect_language_switch_request("আমার একটা টেস্ট বুক করতে হবে", "bn") is None
    assert detect_language_switch_request("What is the price of uric acid", "en") is None
    assert detect_language_switch_request("मुझे अपॉइंटमेंट बुक करनी है", "hi") is None


def test_request_is_understood_in_the_language_it_was_made_in_only():
    # An English phrase spoken while lang="hi" (LID's own job, not this
    # module's) is not matched against the English phrase table -- this
    # function trusts the caller's ACTUAL detected language for lookup.
    assert detect_language_switch_request("speak in hindi", "hi") is None
