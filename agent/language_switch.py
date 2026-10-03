"""KCD-438: a caller explicitly asks the agent to speak a different
language ("please speak in Hindi", "bengali te bolun", "आप अंग्रेज़ी में
बोल सकते हैं?").

Deterministic keyword detection, no LLM call -- the same zero-extra-
latency, no-hallucination-risk reasoning as agent/booking_flow.py's
classify_yes_no. Scope, stated plainly: this detects the REQUEST and
main.py uses it to (a) speak the very next reply in the requested
language and (b) bias the language-ID router's ambiguous-turn prior
toward it via ASRLanguageRouter.note_response_language(), so a later
turn where LID itself is unsure leans toward what the caller asked for.
It does NOT force every later reply into the requested language
regardless of what the caller actually says next -- this project's
per-utterance language architecture (KCD-437) means a caller who goes
back to speaking their original language is still heard correctly, and
overriding that safely would mean threading a second language variable
through every reply call site in main.py's dispatch loop, a change too
large to make blind against a file with no local test coverage (it
needs the pod's ASR/TTS venv to import at all). Revisit with real call
data once that refactor can be verified live.
"""

from __future__ import annotations

_SWITCH_PHRASES: dict[str, dict[str, str]] = {
    # utterance language -> {phrase substring: target language code}
    "bn": {
        "হিন্দিতে বল": "hi",
        "হিন্দিতে কথা": "hi",
        "হিন্দি ভাষায়": "hi",
        "ইংরেজিতে বল": "en",
        "ইংরেজিতে কথা": "en",
        "ইংলিশে বল": "en",
        "বাংলায় বল": "bn",
        "বাংলায় কথা": "bn",
    },
    "hi": {
        "हिंदी में बोल": "hi",
        "हिंदी में बात": "hi",
        "अंग्रेज़ी में बोल": "en",
        "इंग्लिश में बोल": "en",
        "अंग्रेजी में बात": "en",
        "बंगाली में बोल": "bn",
        "बांग्ला में बोल": "bn",
    },
    "en": {
        "speak in hindi": "hi",
        "talk in hindi": "hi",
        "hindi please": "hi",
        "speak in bengali": "bn",
        "talk in bengali": "bn",
        "bengali please": "bn",
        "speak in english": "en",
        "talk in english": "en",
        "english please": "en",
    },
}


def detect_language_switch_request(text: str, lang: str) -> str | None:
    """`lang` is the ASR-detected language of THIS utterance (the request
    itself is understood correctly regardless of which language it was
    made in). Returns the requested target language code, or None if this
    utterance was not a language-switch request."""
    phrases = _SWITCH_PHRASES.get(lang, {})
    t = text.strip().lower()
    for phrase, target in phrases.items():
        if phrase.lower() in t:
            return target
    return None
