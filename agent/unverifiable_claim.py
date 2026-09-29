"""ADDED BY SOURAV -- "Caller states something the agent cannot verify"
story.

story title: Caller states something the agent cannot verify
user story: As a caller asserting a fact about my record, I want the
    agent to check rather than accept it, so that a mistake is not
    compounded.
acceptance criteria: A caller assertion never becomes a system fact. The
    agent checks the system of record where verification is possible.
    Where the system cannot verify the claim, the agent clearly says it
    cannot confirm it and offers a human. A caller claim is never echoed
    back as though it were verified.

WHY THIS IS A SEPARATE, PRE-CLASSIFIER GUARD, STRUCTURED THE SAME WAY AS
agent/clinical_safety.py / agent/human_fast_path.py / agent/complaint_flow.py
/ agent/doctor_personal_request.py / agent/symptom_routing.py:

The investigation for this story (see the delivered report,
"Caller_unverifiable_assertion_investigation_report.txt") found that
every intent with a real clinic-api tool behind it (report status/
delivery/OTP, billing, doctor availability/schedule, booking) ALREADY
verifies before speaking -- that architecture is correct and this story
does not touch it (see agent/report_flow.py, agent/tools_client.py,
agent/reply_templates.py's tool-backed reply functions). The one real gap
the investigation found is narrower: "smalltalk" is the ONE path in this
whole system where the model's own free Bengali text is spoken with no
downstream check (agent/llm.py's _validate() strips direct_reply_bn for
every other intent). Nothing stops a caller's factual-sounding assertion
about their OWN record ("my appointment is tomorrow", "I already paid")
from being classified smalltalk and getting a free-text reply that
sounds like agreement, because nothing in that lane's prompt or code
distinguishes "pure social content" from "an assertion the caller wants
treated as already true."

This module is the deterministic fix for exactly that gap -- not a
second verification system, and not a redesign of anything that already
verifies correctly. It recognises the SHAPE of a caller asserting one of
a small, closed set of facts this system has NO way to check today
(an existing appointment's date/doctor, a report's clinical result, a
doctor's "approval", a past payment event -- see this module's own
CLAIM_PATTERNS docstring for exactly why these four and not others), and
hands every one of them to a fixed, human-offering reply -- never to
smalltalk, never to anything generative. So this module runs BEFORE the
classifier -- checked in main.py/main_pcm.py's _resolve_intent(), after
every existing guard (immediate-human, clinical-interpretation,
complaint, doctor-personal-request, symptom-routing), and BEFORE
fast_path, the semantic cache, and Ollama. A caller's assertion is never
shown to the model at all once this guard fires.

WHY THIS RUNS AFTER EVERY EXISTING GUARD, NOT BEFORE OR INSTEAD OF THEM:

A caller who states a claim AND asks a danger/diagnosis question ("my
report is normal, right, so I don't need to worry?") must still be
caught by is_clinical_interpretation() first -- that guard is checked
earlier in _resolve_intent() and is left completely untouched here. A
caller filing a complaint, asking for a doctor personally, or describing
a symptom takes priority over this guard for the identical reason those
guards already take priority over each other -- see each of their own
module docstrings. This module only ever runs for utterances every guard
above it has already looked at and let through.

WHY THIS DOES NOT INTERCEPT EVERY QUESTION ABOUT A RECORD -- ONLY A
CLOSED SET OF ASSERTION SHAPES WITH NO BACKING LOOKUP:

"the guard should be narrow and deterministic" and "do not treat every
normal question as an unverifiable claim" are this story's own explicit
constraints. A caller ASKING "is my report ready?" or "do I have any
dues?" already routes correctly to report_status/billing_balance -- a
real, tested, verified lookup -- and must keep doing so; intercepting
those here would be exactly the "second verification system where one
already exists" this story explicitly forbids. So two things keep this
guard narrow:

  1. CLAIM_PATTERNS below only lists DECLARATIVE assertion shapes about
     the four topics this system genuinely cannot check (see that
     constant's own docstring) -- not the many phrasings of the
     QUESTIONS that already route correctly elsewhere.
  2. _REAL_REQUEST_MARKERS below is a small, separate exclusion list: an
     utterance that also asks the system to check, confirm, or book
     something ("can you confirm", "please check my balance", "I want to
     book") is deliberately left UNINTERCEPTED here, even if it also
     contains a CLAIM_PATTERNS phrase, so it falls through to the
     ordinary classifier chain and reaches the real, already-verified
     tool-backed intent instead. This is the mirror image of
     agent/symptom_routing.py's own _DIAGNOSIS_REQUEST_MARKERS exclusion
     (there: decline when a diagnosis is being asked for; here: decline
     when a real check/booking is being asked for) -- same reasoning,
     applied to the opposite failure mode.

WHY THIS DOES NOT BUILD A NEW APPOINTMENT-LOOKUP API:

The investigation found no GET endpoint anywhere in clinic-api for
reading back an existing appointment by phone -- only POST
/api/v1/appointments (create) exists. Building that lookup is a separate,
larger product decision (a new endpoint, model, and tool-client method),
explicitly out of this story's scope per its own instruction. Because no
such lookup exists, an "appointment_existing" claim can only ever be
answered honestly with "I cannot confirm that" -- which is exactly this
story's own AC3, not a gap this module papers over.
"""
from __future__ import annotations

from agent.slot_parse import _bn_bounded

NONE = "none"
UNVERIFIABLE = "unverifiable"

# Audit reason codes -- see agent/outcomes.py's record_unverifiable_claim()
# for where these are logged. Fixed, non-sensitive strings only, never a
# sentence built from caller speech (same discipline
# agent/outcomes.py::record_report_access_denied()'s own docstring
# already establishes for its own `reason` values).
CATEGORY_APPOINTMENT_EXISTING = "APPOINTMENT_EXISTING_CLAIM"
CATEGORY_REPORT_RESULT = "REPORT_RESULT_CLAIM"
CATEGORY_DOCTOR_APPROVAL = "DOCTOR_APPROVAL_CLAIM"
CATEGORY_PAYMENT_ALREADY = "PAYMENT_ALREADY_CLAIM"


def _is_bengali_script(phrase: str) -> bool:
    """True if `phrase` contains at least one Bengali-script character (the
    Bengali Unicode block, U+0980-U+09FF). A small, self-contained helper
    -- deliberately NOT imported from agent/symptom_routing.py's own
    identically-named one, matching this codebase's existing convention
    of keeping each deterministic guard module (clinical_safety.py,
    human_fast_path.py, complaint_flow.py, doctor_personal_request.py,
    symptom_routing.py) self-contained and only sharing genuinely common,
    public utilities (agent/slot_parse.py's _bn_bounded, imported above)."""
    return any("ঀ" <= ch <= "৿" for ch in phrase)


# =============================================================================
# CLAIM_PATTERNS -- the closed set of assertion shapes this guard acts on.
#
# Every category below is a topic the investigation confirmed has NO
# system-of-record check available today (see this module's own docstring,
# "WHY THIS DOES NOT BUILD A NEW APPOINTMENT-LOOKUP API"). This is
# deliberately NOT an attempt to catch every way a caller might mention an
# appointment, report, or payment -- a caller ASKING about any of these
# (a question, not an assertion) already reaches a real, verified,
# tool-backed intent (report_status, billing_balance, book_appointment,
# doctors_by_department) and must keep doing so unchanged.
#
# Languages covered: English, Hindi/Hinglish, Bengali/Banglish (romanised),
# and Bengali script -- the same four-way coverage this codebase's other
# guard modules carry (see agent/symptom_routing.py's own "Story 6
# validation cleanup" docstring on why Bengali script was added there
# after being missed the first time; this module starts with all four).
#
# KNOWN LIMITATION, FLAGGED NOT SOLVED: this is a fixed phrase list, not a
# semantic classifier -- a caller who asserts one of these facts in
# wording not listed here will not be intercepted (it falls through to
# the ordinary classifier chain, most likely landing on "unclear" or
# "out_of_scope", both of which are ALREADY SAFE -- see this module's own
# docstring's "smalltalk is the one unsafe lane" reasoning; only smalltalk
# needed a deterministic guard, not every possible phrasing of a claim).
# Widening this vocabulary is a judgement call for whoever reviews real
# call transcripts, the same discipline already established for
# agent/symptom_routing.py's own prototype vocabulary.
# =============================================================================
CLAIM_PATTERNS: dict[str, list[str]] = {
    CATEGORY_APPOINTMENT_EXISTING: [
        # English
        "my appointment is tomorrow", "my appointment is today",
        "my appointment is on", "i already have an appointment",
        "i have an appointment tomorrow", "i have an appointment today",
        "my appointment is with dr", "my appointment is with doctor",
        "i already have an appointment with dr",
        "i have an appointment with doctor",
        # Hindi / Hinglish
        "mera appointment kal hai", "mera appointment aaj hai",
        "mera appointment already hai", "mera appointment fix hai",
        "mera appointment doctor ke saath hai",
        "mera appointment dr ke saath pehle se hai",
        # Bengali / Banglish (romanised)
        "amar appointment kal ache", "amar appointment ajke ache",
        "amar appointment already ache", "amar appointment fix kora ache",
        "amar appointment doctor er sathe ache",
        "amar appointment dr er sathe age theke fix ache",
        # Bengali script
        "আমার অ্যাপয়েন্টমেন্ট কাল আছে", "আমার অ্যাপয়েন্টমেন্ট আজ আছে",
        "আমার অ্যাপয়েন্টমেন্ট আগে থেকেই ঠিক করা আছে",
        "আমার অ্যাপয়েন্টমেন্ট ডাক্তারের সাথে আছে",
    ],

    CATEGORY_REPORT_RESULT: [
        # English
        "my report is normal", "my report was normal",
        "my report came back normal", "my report is fine",
        "my test result is normal", "my results are normal",
        "my report was fine",
        # Hindi / Hinglish
        "mera report normal hai", "mera report normal aaya hai",
        "mera result normal hai", "mera test normal aaya",
        # Bengali / Banglish (romanised)
        "amar report normal ache", "amar report normal eshechhe",
        "amar result normal", "amar test normal eshechhe",
        # Bengali script
        "আমার রিপোর্ট নরমাল", "আমার রিপোর্ট নরমাল এসেছে",
        "আমার রেজাল্ট নরমাল",
    ],

    CATEGORY_DOCTOR_APPROVAL: [
        # English
        "the doctor already approved", "doctor already approved it",
        "doctor has already approved", "my doctor approved this",
        "doctor already agreed", "doctor already said yes",
        # Hindi / Hinglish
        "doctor ne already approve kar diya",
        "doctor already approve kar chuke hain",
        "doctor ne pehle hi haan bol diya",
        # Bengali / Banglish (romanised)
        "doctor already approve kore diyechen",
        "doctor age thekei approve korechen",
        "doctor age thekei raji hoyechen",
        # Bengali script
        "ডাক্তার আগে থেকেই অনুমোদন দিয়েছেন",
        "ডাক্তার ইতিমধ্যে অনুমতি দিয়েছেন",
    ],

    CATEGORY_PAYMENT_ALREADY: [
        # English
        "i already paid", "i have already paid",
        "i already made the payment", "payment is already done",
        "i paid already",
        # Hindi / Hinglish
        "maine already payment kar diya hai",
        "maine paise de diye hain already",
        "payment already ho chuka hai",
        # Bengali / Banglish (romanised)
        "ami already payment kore diyechi",
        "ami age thekei taka diye diyechi",
        "payment already hoye geche",
        # Bengali script
        "আমি আগেই পেমেন্ট করে দিয়েছি",
        "টাকা আগে থেকেই দেওয়া হয়ে গেছে",
        "পেমেন্ট আগেই হয়ে গেছে",
    ],
}


# ADDED BY SOURAV -- see this module's own docstring above ("WHY THIS DOES
# NOT INTERCEPT EVERY QUESTION ABOUT A RECORD", point 2) for exactly why
# this exists: a caller whose utterance ALSO asks the system to check,
# confirm, or book something must fall through to the real, verified,
# tool-backed intent, never be quietly answered with a generic decline
# instead. Mirrors agent/symptom_routing.py's own
# _DIAGNOSIS_REQUEST_MARKERS shape exactly, just guarding the opposite
# failure mode.
_REAL_REQUEST_MARKERS_LATIN = (
    "can you check", "can you confirm", "please check", "please confirm",
    "check my", "confirm my", "verify my", "could you check",
    "book", "schedule", "i want an appointment", "i need an appointment",
    "can i get an appointment", "want to book",
    "check kar sakte", "confirm kar sakte", "check karo", "confirm karo",
    "book karna hai", "appointment chahiye", "appointment book karna",
    "check korte paren", "confirm korte paren", "check korun",
    "confirm korun", "book korte chai", "appointment chai",
    "appointment book korte chai",
)
_REAL_REQUEST_MARKERS_BENGALI = (
    "চেক করতে পারবেন", "কনফার্ম করতে পারবেন", "চেক করুন", "কনফার্ম করুন",
    "বুক করতে চাই", "অ্যাপয়েন্টমেন্ট চাই",
)


def detect_unverifiable_claim(text: str | None) -> tuple[str, str | None]:
    """-> (verdict, a fixed category reason code, or None).

    UNVERIFIABLE is returned only when the utterance contains a phrase
    from CLAIM_PATTERNS above AND none of _REAL_REQUEST_MARKERS. Anything
    else -- no claim phrase at all, or a claim phrase alongside a real
    check/booking request -- returns NONE, None, so the ordinary
    classifier chain (and, downstream of it, a real tool-backed intent
    where one applies) handles the turn exactly as it already does today.
    This module never itself decides an utterance is FALSE or TRUE; a
    NONE verdict is not "the claim is fine", and an UNVERIFIABLE verdict
    is not "the claim is wrong" -- both are silent on that question by
    design, which is this story's own AC3/AC4.

    Latin-script phrases (English/Hinglish/Banglish) are checked as a
    lowercased substring match, the same treatment every other guard
    module in this codebase uses for that script. Bengali-script phrases
    are checked with `_bn_bounded` (shared from agent/slot_parse.py)
    instead, for the identical word-boundary-safety reason
    agent/clinical_safety.py's and agent/symptom_routing.py's own Bengali
    sets already document: Bengali vowel signs and the nukta are
    combining marks Python's `\\b` does not treat as word characters, so
    a bare substring check could otherwise fire inside an unrelated
    longer word.
    """
    if not text or not text.strip():
        return NONE, None

    lowered = text.lower()

    for marker in _REAL_REQUEST_MARKERS_LATIN:
        if marker in lowered:
            return NONE, None
    for marker in _REAL_REQUEST_MARKERS_BENGALI:
        if _bn_bounded(marker, text):
            return NONE, None

    for category, phrases in CLAIM_PATTERNS.items():
        for phrase in phrases:
            if _is_bengali_script(phrase):
                hit = _bn_bounded(phrase, text)
            else:
                hit = phrase in lowered
            if hit:
                return UNVERIFIABLE, category

    return NONE, None
