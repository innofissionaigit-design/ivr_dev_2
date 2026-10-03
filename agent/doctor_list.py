"""A caller asking WHICH doctors there are, naming none of them.

Ported from `merged_code/agent/doctor_list.py` (story: "Caller asks which doctors
are available"). Two shapes of the same question:

    "কে কে ডাক্তার আছেন"            -> every doctor (no department named)
    "which doctors are in cardiology" -> one department's doctors

WHY THIS IS NOT COVERED BY ANY EXISTING GUARD OR INTENT:

`doctor_availability` and `doctor_schedule`-style questions all require a NAMED
doctor, and a caller asking for the list cannot name one. `department_query`
takes a SYMPTOM and routes it to a department -- it cannot answer "what
departments do you have". So before this module the question reached the model
with nothing to look up, and this repository had no endpoint to look it up with
either; `/api/v1/doctors` and `/api/v1/doctors/by-department` were added for it.

WHY THIS MODULE IS ~150 LINES AND NOT THE ORIGINAL'S 423:

The module this was ported from carried its own department-name extractor --
plural detection, reduplicated question words, Bengali case endings, fuzzy
department scoring with its own floor and margin, a `_NOT_A_DEPARTMENT` stop
list. All of it existed to pull a department name out of the sentence BEFORE
asking the API, which meant a text matcher in `agent/` was guessing at a name it
had no list to check itself against.

None of that is needed here. `/api/v1/doctors/by-department` matches the query
against the real department table in BOTH directions -- the department name
inside the caller's sentence, or the caller's short form inside the department
name -- so the whole utterance can simply be handed over, and the clinic's own
data decides. Several matches come back `ambiguous` with the names offered, a
near match comes back as `did_you_mean`, and neither is ever resolved silently.
That is the same no-guessing discipline `_find_doctor` and `_find_test` already
enforce, and it is enforced against the catalogue rather than re-implemented
against a hand-written list.

So this module answers exactly one question -- "is this a request for the list?"
-- and `main.py` asks the API the rest.

REASONED, not measured: the Hindi list is new in this repository (the original
covered Bengali, English and Latin transliteration only). A native speaker
should review the Bengali and Hindi.
"""

from __future__ import annotations

import unicodedata

# A plural/interrogative ask for doctors, with NO specific doctor named. Each
# entry pairs a "which/who/what" sense with a doctor word, because either alone
# is far too broad: "which day" and "the doctor" are both ordinary turns.
_EN = (
    "which doctors",
    "what doctors",
    "who are the doctors",
    "who all are the doctors",
    "list of doctors",
    "list the doctors",
    "doctors list",
    "all the doctors",
    "all doctors",
    "what are the doctors",
    "which doctor do you have",
    "which doctors do you have",
    "what doctors do you have",
    "who are your doctors",
    "your doctors",
    "doctors available",
    "doctors are available",
    "which specialists",
    "what specialists",
    "list of departments",
    "which departments",
    "what departments",
)

_LATIN = (
    "ke ke doctor",
    "kon kon doctor",
    "kara kara doctor",
    "doctor der list",
    "doctorder list",
    "ki ki doctor",
    "kon doctor ache",
    "kaun kaun doctor",
    "kaun kaun se doctor",
    "konse doctor hain",
    "kaun se doctor hain",
    "doctor ki list",
    "doctoron ki list",
    "sare doctor",
    "sab doctor",
    "kon kon bibhag",
    "kaun kaun vibhag",
)

_BN = (
    "কে কে ডাক্তার",
    "কারা কারা ডাক্তার",
    "কোন কোন ডাক্তার",
    "কি কি ডাক্তার",
    "কী কী ডাক্তার",
    "ডাক্তারদের তালিকা",
    "ডাক্তারের তালিকা",
    "সব ডাক্তারের নাম",
    "কোন ডাক্তাররা আছেন",
    "কোন ডাক্তাররা বসেন",
    "কে কে বসেন",
    "কোন কোন বিভাগ",
    "কি কি বিভাগ",
    "বিভাগের তালিকা",
)

# NEW in this repository: the original had no Devanagari list.
_HI = (
    "कौन कौन डॉक्टर",
    "कौन कौन से डॉक्टर",
    "कौन से डॉक्टर हैं",
    "कौन डॉक्टर हैं",
    "क्या क्या डॉक्टर",
    "डॉक्टरों की सूची",
    "डॉक्टर की लिस्ट",
    "डॉक्टरों की लिस्ट",
    "सभी डॉक्टरों के नाम",
    "सब डॉक्टर",
    "आपके डॉक्टर कौन",
    "कौन कौन विभाग",
    "कौन से विभाग",
    "विभागों की सूची",
)

_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("en", _EN),
    ("latin", _LATIN),
    ("bn", _BN),
    ("hi", _HI),
)

# WHY THERE IS NO "a doctor was named, so this is not a list request" CHECK:
#
# One was written first, as a regex for a doctor title followed by a name across
# the three scripts, and it was WRONG in a way worth recording. In Bengali and
# Hindi the words after the title are ordinary grammar, not a surname --
# "ডাক্তার আছেন" ("are there doctors"), "डॉक्टर हैं" ("are doctors") -- so the
# check read almost every genuine list request in those languages as naming a
# doctor and silently suppressed it. It suppressed 8 of 15 real requests, all of
# them non-English: precisely the trilingual failure this merge exists to avoid.
#
# It is also unnecessary. The phrase lists above all pair a plural or
# interrogative sense with a doctor word, so an availability or booking question
# about a NAMED doctor matches none of them -- verified in
# tests/test_doctor_list.py across all three scripts, including "is Dr Sen
# available tomorrow", "ডক্টর সেনের কাছে অ্যাপয়েন্টমেন্ট" and
# "डॉक्टर सेन के साथ अपॉइंटमेंट". Specific phrases do the job a loose regex
# could not.


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


def detect_doctor_list_request(text: str) -> tuple[bool, str | None]:
    """True, plus which phrase family matched, when the caller is asking which
    doctors (or departments) the clinic has, without naming a doctor.

    The tag is for logging only, never the reply language.
    """
    if not text or not text.strip():
        return False, None
    raw = unicodedata.normalize("NFC", text)
    lowered = raw.lower()
    for tag, phrases in _FAMILIES:
        if any((p in lowered) if p.isascii() else (p in raw) for p in phrases):
            return True, tag
    return False, None


def asks_for_the_doctor_list(text: str) -> bool:
    """The one question this module answers for the orchestrator."""
    matched, _ = detect_doctor_list_request(text)
    return matched
