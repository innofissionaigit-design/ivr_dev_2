"""The caller wants the call to end -- decided by words, not by the model.

A live call (2026-09-25): the agent asked "anything else? if not I will end the call", the caller said "শেষ করে দিন"
("finish it / end it"), and the agent went to the model, which heard a booking request and asked for a confirmation number.
Ending a call is a binary, low-risk, high-annoyance decision: it is decided here, deterministically, with no model.

Two strengths of cue, because the same words mean different things in different places:

  * EXPLICIT -- unmistakable at any moment: "end the call", "কল কেটে দিন", "फोन रखता हूँ", "goodbye". Mid-booking too;
    the unfinished booking is saved for the next call (main._save_unfinished_draft).
  * WHEN ASKED -- only as the answer to "is there anything else?": a bare "শেষ করে দিন", "nothing", "that's all",
    "no thanks". The same words mid-booking ("finish the booking") are not a request to hang up.

Whole-token matching for short cues (a substring hit would fire "bye" inside "maybe"), substring matching for longer
Bengali phrases (Bengali is written without spaces between a word and its endings). REASONED cue lists, not measured;
a native speaker should review them (docs/pod-verification-checklist.md).
"""

from __future__ import annotations

import re
import unicodedata

_EXPLICIT = {
    "bn": (
        "কল শেষ",
        "কল কেটে",
        "কল রেখে",
        "কল বন্ধ",
        "ফোন রাখ",
        "ফোন রেখে",
        "ফোন কেটে",
        "ফোন ছেড়ে",
        "লাইন কেটে",
        "রাখছি",
        "রেখে দিচ্ছি",
        "বাই",
        "গুডবাই",
        "টা টা",
        # "cut" on its own is unambiguous for ending a CALL (unlike "শেষ"/finish, which also means finishing a
        # booking -- that stays in _WHEN_ASKED below, gated by _TASK_WORDS). A live call (2026-09-28): the caller
        # said "কেটে দিন" mid-conversation, not right after the "anything else?" prompt, and it was not recognised.
        "কেটে দিন",
        "কেটে দাও",
        "কাটো",
        "কেটে দে",
    ),
    "hi": (
        "कॉल खत्म",
        "कॉल काट",
        "कॉल बंद",
        "फोन रख",
        "फोन काट",
        "लाइन काट",
        "रखता हूं",
        "रखती हूं",
        "रखता हूँ",
        "रखती हूँ",
        "बाय",
        "अलविदा",
        "गुडबाय",
        "टाटा",
        # Same reasoning as the Bengali "কাটো" entries above.
        "काट दीजिए",
        "काट दो",
        "काटो",
    ),
    "en": (
        "end the call",
        "end this call",
        "end call",
        "hang up",
        "hanging up",
        "goodbye",
        "good bye",
        "bye",
        "that's all",
        "thats all",
        "that is all",
        "i'm done",
        "im done",
        "i am done",
    ),
}
_WHEN_ASKED = {
    "bn": (
        "শেষ করে দিন",
        "শেষ করে দাও",
        "শেষ করুন",
        "শেষ করে দেন",
        "শেষ",
        "বন্ধ করে দিন",
        "বন্ধ করুন",
        "আর কিছু না",
        "আর কিছু নেই",
        "আর কিছু লাগবে না",
        "কিছু জানার নেই",
        "কিছু জানতে চাই না",
        "কিছু না",
        "থাক",
        "দরকার নেই",
        "লাগবে না",
        "না ধন্যবাদ",
        "না, ধন্যবাদ",
    ),
    "hi": (
        "खत्म कर दीजिए",
        "खत्म कर दो",
        "खत्म करें",
        "खत्म करिए",
        "खत्म",
        "बंद कर दीजिए",
        "बंद करें",
        "और कुछ नहीं",
        "कुछ नहीं",
        "कुछ और नहीं",
        "बस",
        "रहने दीजिए",
        "रहने दो",
        "नहीं चाहिए",
        "नहीं, धन्यवाद",
        "नहीं धन्यवाद",
    ),
    "en": (
        "end it",
        "go ahead and end",
        "finish it",
        "nothing",
        "nothing else",
        "no thanks",
        "no thank you",
        "that's it",
        "thats it",
        "nope",
        "i'm good",
        "im good",
        "all good",
        "no more",
    ),
}
# A bare "শেষ" can also mean "finish the booking": with a task word in the sentence it is not a request to hang up.
_TASK_WORDS = (
    "বুকিং",
    "অ্যাপয়েন্টমেন্ট",
    "টেস্ট",
    "সিরিয়াল",
    "बुकिंग",
    "अपॉइंटमेंट",
    "टेस्ट",
    "booking",
    "appointment",
    "test",
    "slot",
)
_PUNCT = re.compile(r"[?!.,;:।\"'()\[\]]")


def _normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").replace("़", "").lower().replace("’", "'")
    t = _PUNCT.sub(lambda m: "'" if m.group(0) == "'" else " ", t)
    return " ".join(t.split())


def _has(text: str, cue: str) -> bool:
    cue = _normalise(cue)
    if " " in cue or len(cue) >= 5:
        return cue in text  # a phrase, or a long enough word to be unambiguous
    return f" {cue} " in f" {text} "  # a short word: whole tokens only ("bye" is not in "maybe")


def wants_to_end(text: str, lang: str, after_prompt: bool = False) -> bool:
    """True when the caller is asking to end the call. `after_prompt` is True right after the agent asked whether
    there was anything else (agent/phrases silence_prompt), which is when the bare cues count."""
    t = _normalise(text)
    if not t:
        return False
    languages = (lang,) + tuple(x for x in _EXPLICIT if x != lang)  # a caller may end it in any language
    if any(_has(t, c) for lg in languages for c in _EXPLICIT.get(lg, ())):
        return True
    if after_prompt and not any(w in t for w in _TASK_WORDS):
        return any(_has(t, c) for lg in languages for c in _WHEN_ASKED.get(lg, ()))
    return False
