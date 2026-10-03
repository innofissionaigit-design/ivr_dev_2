"""A caller filing a complaint: hear it, record it as said, pass it to a person.

Ported from `merged_code/agent/complaint_flow.py` (story: "Caller wants to make a
complaint"). Its acceptance criteria were:

  1. A complaint is recognised as a complaint, not as an enquiry.
  2. It is acknowledged without argument.
  3. It is captured verbatim.
  4. It is routed to a person.
  5. The agent never defends the clinic or explains the complaint away.

WHY THIS IS A DETERMINISTIC GUARD AHEAD OF THE MODEL:

Acceptance criterion 5 is a zero-tolerance policy, and a policy the model is
asked to honour is a policy the model can be argued out of. Worse, this
repository's intent set has no "complaint" member at all (agent/intent_schema.py),
so an unguarded complaint is classified as `smalltalk`, `clinic_faq` or `unclear`
and answered as an enquiry -- which fails criterion 1 outright and usually
criterion 2 with it. So the check runs before intent extraction, exactly like
agent/abuse.py, agent/emergency.py and agent/human_request.py, and the reply is
fixed text from agent/phrases.py, which the model never gets a turn to write.

WHY IT IS CHECKED BEFORE agent/anger.py:

A caller filing a complaint is frequently angry as well, and both guards would
match. Without a fixed order they would produce two acknowledgements and two
routes for one utterance. The complaint path is the more specific and the more
complete of the two -- it records the words verbatim and routes to a person --
so it is checked FIRST and returns, and the anger guard never runs for that turn.
That ordering is asserted in tests/test_orchestrator_complaint.py rather than
left as a comment. The two phrase lists are deliberately kept separate (and
proven non-overlapping in tests/test_complaint.py) so each story's own vocabulary
stays auditable against its own criteria.

WHERE THE VERBATIM TEXT GOES, AND WHERE IT MUST NOT GO:

Criterion 3 wants the caller's own words kept; this repository's logging
discipline forbids caller free text in `logs/` and in the call record
(agent/call_record.py stores counts, flags and ids, "never any text"). Both
constraints are satisfied the same way the module this was ported from satisfied
them: the verbatim complaint goes to ONE narrow-access durable store -- the
clinic API's `complaints` table, via agent/tools_client.submit_complaint() -- and
the escalation ledger and call record continue to carry only the reason code.

REASONED, not measured: a native speaker should review all four lists. The Hindi
Devanagari list is new in this repository -- the module this was ported from
covered Bengali, English and Latin-script transliteration only.
"""

from __future__ import annotations

import unicodedata

# ---- English -------------------------------------------------------------
# Substrings of the lower-cased utterance. Two families are present on purpose:
# naming the ACT of complaining ("I want to file a complaint"), and an
# unambiguous statement of dissatisfaction about the clinic or its staff ("your
# staff were rude to me"). A bare "bad" or "problem" is on neither list.
_EN = (
    "i want to file a complaint",
    "i want to make a complaint",
    "i want to lodge a complaint",
    "i want to register a complaint",
    "i want to raise a complaint",
    "i'd like to file a complaint",
    "i'd like to make a complaint",
    "i'd like to lodge a complaint",
    "i'd like to register a complaint",
    "i'd like to raise a complaint",
    "i have a complaint",
    "this is a complaint",
    "let me file a complaint",
    "let me make a complaint",
    "i need to complain",
    "i want to complain",
    "i'd like to complain",
    "i am here to complain",
    "filing a complaint",
    "making a complaint",
    "lodging a complaint",
    "registering a complaint",
    "raising a complaint",
    "file a formal complaint",
    "make a formal complaint",
    "i want to report bad service",
    "i want to report poor service",
    "i want to report how i was treated",
    "i was treated very badly",
    "i was treated so badly",
    "your staff was very rude to me",
    "your staff were rude to me",
    "the doctor was rude to me",
    "nobody helped me at the clinic",
    "i am extremely dissatisfied",
    "i am very dissatisfied",
    "i am extremely unhappy with the service",
    "i am very unhappy with the service",
    "i am extremely unhappy with this service",
    "i am very unhappy with this service",
    "i am not happy with the service",
    "i am not happy with this service",
    "this service is unacceptable",
    "this is completely unacceptable",
    "this is really unacceptable",
    "i had a terrible experience",
    "i had a horrible experience",
    "i had a very bad experience at your clinic",
    "i had a very bad experience with this service",
    "worst experience i have ever had",
    "this is the worst service",
    "i want to speak to someone about a complaint",
    "i want this complaint noted down",
    "please note my complaint",
    "please register my complaint",
    "i am complaining about",
    # A complaint added mid-utterance after something else ("...and I also want
    # to file a complaint") is an ordinary caller phrasing; a substring match
    # that assumed "I" came first missed every one of them.
    "i also want to file a complaint",
    "i also want to make a complaint",
    "i also have a complaint",
    "and i have a complaint",
    "and i want to file a complaint",
    "and i want to make a complaint",
    "and i want to complain",
    "i also want to complain",
)

# ---- Hindi and Bengali written in Latin script ---------------------------
_LATIN = (
    "complaint karna chahta hoon",
    "complaint karna chahti hoon",
    "complaint file karna chahta hoon",
    "complaint file karna hai",
    "ek complaint hai",
    "mujhe complaint karni hai",
    "mujhe ek complaint darj karni hai",
    "complaint darj karo",
    "complaint darj karna chahta hoon",
    "shikayat karna chahta hoon",
    "shikayat karni hai",
    "meri ek shikayat hai",
    "mujhe shikayat darj karni hai",
    "shikayat darj karo",
    "shikayat darj karna hai",
    "aapki service se bahut naraz hoon",
    "bahut kharab service thi",
    "bahut buri service thi",
    "staff ne bahut badtameezi ki",
    "mujhe bahut bura anubhav hua",
    "amar ekta complaint ache",
    "amar ekta obhijog ache",
    "complaint korte chai",
    "obhijog korte chai",
    "ami complaint korte chai",
    "ami obhijog korte chai",
    "complaint janate chai",
    "obhijog janate chai",
    "ekta complaint korbo",
    "ekta obhijog korbo",
    "apnader service niye amar complaint ache",
    "khub kharap byabohar korechilo",
    "staff khub kharap byabohar korlo",
    "khub baje experience hoyeche",
)

# ---- Bengali -------------------------------------------------------------
# Substring match on NFC-normalised text, safe because every entry is
# multi-word: the embedded space is the word boundary Python's \b cannot
# provide for a script whose vowel signs are combining marks.
_BN = (
    "আমার একটা অভিযোগ আছে",
    "আমি অভিযোগ করতে চাই",
    "অভিযোগ জানাতে চাই",
    "একটা অভিযোগ করতে চাই",
    "কমপ্লেইন করতে চাই",
    "কমপ্লেইন জানাতে চাই",
    "আমি কমপ্লেইন করতে চাই",
    "আমার একটা কমপ্লেইন আছে",
    "খুব খারাপ ব্যবহার করেছে",
    "খুব খারাপ পরিষেবা পেয়েছি",
    "আমি খুব অসন্তুষ্ট",
    "এটা একদম ঠিক হয়নি",
    "আমার সাথে খুব খারাপ আচরণ করা হয়েছে",
    "স্টাফ খুব খারাপ ব্যবহার করেছে",
    "ডাক্তার খুব খারাপ ব্যবহার করেছেন",
    "এই পরিষেবা মোটেও ভালো না",
)

# ---- Hindi ---------------------------------------------------------------
# NEW in this repository: the module this was ported from had no Devanagari
# list. Both families from the English list are covered -- naming the act of
# complaining, and an unambiguous statement about the clinic's conduct.
_HI = (
    "मुझे शिकायत करनी है",
    "मुझे एक शिकायत करनी है",
    "मैं शिकायत करना चाहता हूँ",
    "मैं शिकायत करना चाहती हूँ",
    "मेरी एक शिकायत है",
    "मुझे शिकायत दर्ज करनी है",
    "शिकायत दर्ज कीजिए",
    "शिकायत दर्ज करनी है",
    "मैं शिकायत दर्ज कराना चाहता हूँ",
    "मेरी शिकायत लिख लीजिए",
    "मेरी शिकायत नोट कर लीजिए",
    "मुझे कंप्लेंट करनी है",
    "एक कंप्लेंट है",
    "आपकी सेवा से मैं बहुत असंतुष्ट हूँ",
    "मैं सेवा से खुश नहीं हूँ",
    "बहुत खराब सेवा मिली",
    "बहुत ख़राब सेवा मिली",
    "स्टाफ ने बहुत बदतमीजी की",
    "स्टाफ़ ने बहुत बदतमीज़ी की",
    "डॉक्टर ने बहुत बुरा व्यवहार किया",
    "मेरे साथ बहुत बुरा व्यवहार हुआ",
    "मुझे बहुत बुरा अनुभव हुआ",
    "यह सेवा बिलकुल ठीक नहीं है",
)

_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("en", _EN),
    ("latin", _LATIN),
    ("bn", _BN),
    ("hi", _HI),
)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


def detect_complaint(text: str) -> tuple[bool, str | None]:
    """True, plus which phrase family matched, when `text` is a caller filing or
    stating a complaint.

    The returned tag ("en" | "latin" | "bn" | "hi") says which list matched and
    is for logging only -- never the language to reply in. The reply language is
    the sticky one the orchestrator already resolved for the turn.
    """
    t = _normalise(text)
    if not t.strip():
        return False, None
    for tag, phrases in _FAMILIES:
        if any(p in t for p in phrases):
            return True, tag
    return False, None


def is_complaint(text: str) -> bool:
    """Bare boolean form, matching agent/abuse.py::is_abusive()'s shape at the
    call site that uses it."""
    matched, _ = detect_complaint(text)
    return matched
