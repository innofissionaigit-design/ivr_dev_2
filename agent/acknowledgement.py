"""KCD-155: deterministic acknowledgement templates per language.

Blueprint 4.5: "Deterministic acknowledgement templates per language ('I
can hear this is difficult - I am here, take your time')" -- selected by
the policy engine, never composed by a model. They acknowledge the FEELING
first and only then proceed or escalate, so the caller is not met with
indifference or a procedural question.

Selection is driven by agent/speech_policy.py: only a state whose Appendix
C row says acknowledge_first gets one, and a row that says offer_human gets
the variant that offers a person as its closing question.

WORDING is provisional. The story requires it to be reviewed by the
clinical lead -- that review is not something this codebase can do, so
every entry is marked pending in REVIEW_STATUS and the count is exported
for the release gate rather than being quietly assumed fine.

No colons or brackets (agent/spoken_text_lint.py): these are spoken.
"""

from __future__ import annotations

from agent.speech_policy import SpeechPolicy

# state -> lang -> text
_ACK: dict[str, dict[str, str]] = {
    "distressed": {
        "bn": "আমি বুঝতে পারছি আপনার পক্ষে এটা কঠিন। আমি আছি, আপনি নিজের মতো করে সময় নিন।",
        "hi": "मैं समझ सकती हूँ कि यह आपके लिए मुश्किल है। मैं यहीं हूँ, आप अपना समय लीजिए।",
        "en": "I can hear this is difficult. I am here, so please take your time.",
    },
    "angry": {
        "bn": "আপনার অসুবিধার জন্য আমি সত্যিই দুঃখিত। আমি এটা ঠিক করতে সাহায্য করতে চাই।",
        "hi": "आपको जो परेशानी हुई उसके लिए मुझे सचमुच खेद है। मैं इसे ठीक करने में मदद करना चाहती हूँ।",
        "en": "I am sorry for the trouble this has caused. I want to help put it right.",
    },
    "confused": {
        "bn": "ঠিক আছে, কোনো তাড়া নেই। আমরা একটা একটা করে এগোব।",
        "hi": "ठीक है, कोई जल्दी नहीं है। हम एक-एक करके आगे बढ़ते हैं।",
        "en": "That is quite all right, there is no hurry. We will take it one step at a time.",
    },
}

# Spoken after the acknowledgement when the policy says offer_human. One
# question, so it still satisfies "one question at a time".
_OFFER_HUMAN: dict[str, str] = {
    "bn": "আপনি চাইলে আমি আপনাকে আমাদের স্টাফের সাথে যুক্ত করে দিতে পারি। করে দেব?",
    "hi": "आप चाहें तो मैं आपको हमारे स्टाफ़ से जोड़ सकती हूँ। जोड़ दूँ?",
    "en": "If you like, I can connect you with our staff. Shall I do that?",
}

REVIEW_STATUS = {(state, lang): "pending_clinical_review" for state, t in _ACK.items() for lang in t}


def acknowledgement_for(caller_state: str, lang: str, offer_human: bool = False) -> str | None:
    table = _ACK.get(caller_state)
    if not table:
        return None
    text = table.get(lang) or table["bn"]
    if offer_human:
        text = f"{text} {_OFFER_HUMAN.get(lang, _OFFER_HUMAN['bn'])}"
    return text


def select_acknowledgement(policy: SpeechPolicy, caller_state: str, lang: str) -> str | None:
    """The acknowledgement the POLICY calls for, or None. The policy engine
    decides whether one is due; this only supplies the words."""
    if policy.emergency or not policy.acknowledge_first:
        return None
    return acknowledgement_for(caller_state, lang, offer_human=policy.offer_human)


def pending_review_count() -> int:
    return sum(1 for status in REVIEW_STATUS.values() if status != "approved")


_ECHO = {"bn": "ঠিক আছে, {value}।", "hi": "ठीक है, {value}।", "en": "Alright, {value}."}


def slot_echo(value: str, lang: str) -> str:
    """Senior mode's "one question, listen, CONFIRM, next question": the
    value the caller just gave, said back before the next question, so a
    misheard value is caught the moment it happens rather than in a
    summary minutes later. A statement, not a question -- the caller can
    correct it on their next turn (the correction path already exists)."""
    return _ECHO.get(lang, _ECHO["bn"]).format(value=value)
