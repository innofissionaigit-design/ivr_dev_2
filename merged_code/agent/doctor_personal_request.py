"""ADDED BY SOURAV -- "Caller wants to speak to a doctor personally" story.

story title: Caller wants to speak to a doctor personally
user story: As a caller who wants clinical reassurance, I want a realistic
    answer about what is possible, so that I am not left waiting for
    something that will not happen.
acceptance criteria:
    1. The agent states the actual process for reaching a clinician.
    2. The agent offers the appropriate route.
    3. The agent never promises a call from a named doctor it cannot
       schedule.
    4. Where a clinical callback exists in policy it is offered.

INVESTIGATION FINDING THIS MODULE EXISTS TO CLOSE ("no such handling"):
there is no distinct clinical/doctor callback system anywhere in this
repository. The real, existing capabilities are book_appointment
(schedules an actual VISIT with a named doctor), request_callback (a
generic pending callback with no doctor field at all -- see
clinic-api/models.py's CallbackRequest docstring), human_direct_request
(a generic escalation-ledger entry, never a live connection), and
clinical_interpretation's own "connect you with our clinician right now"
wording (which, per that story's own dispatch, resolves to the SAME
generic escalation-ledger entry as human_direct_request -- no live
clinician connection exists to perform). None of these four is a
"personally speak to a doctor" mechanism, and none of them was ever
checked deterministically for THIS specific ask -- a caller saying "I
want to speak to a doctor personally" fell through to the LLM classifier
with no intent cleanly describing it, most likely landing on
book_appointment's slot-filling flow (the wrong route for someone who
wants reassurance now, not a future visit) or "unclear" (human_fallback_
reply()'s vague "connecting you right away", which is not the real
process either).

STRUCTURED DIRECTLY ON agent/human_fast_path.py AND agent/complaint_flow.py,
the same shape for the same reason: "the agent never promises a call from
a named doctor it cannot schedule" is a zero-tolerance policy, not
something safe to leave to a 7B model's prompt-following under phrasing
pressure (see agent/human_fast_path.py's own "WHY THIS IS A SEPARATE,
PRE-CLASSIFIER CHECK" section -- the argument applies here without change).
A caller who asks this must never be able to land on "smalltalk" (the one
intent whose reply the model composes itself with zero downstream check --
see agent/clinical_safety.py's own docstring on this exact loophole) or on
any LLM misclassification that might produce a "Dr X will call you"-shaped
sentence. So this guard runs BEFORE the classifier -- before the fast path,
the semantic cache, and Ollama -- and again at the top of
main.py/main_pcm.py's _continue_pending(), so a caller mid-booking who
suddenly asks for a doctor personally is heard, not silently parsed as an
answer to whatever field was pending.

WHAT THIS MODULE DELIBERATELY DOES NOT DO, AND WHY (do not "fix" these
without re-reading this section):

  - It does NOT match a bare mention of the word "doctor". "What are the
    doctor's visiting hours?", "I want to book an appointment with Dr
    Sen", and "Is Dr Sen available tomorrow?" must all continue through
    the existing doctor_availability/doctor_schedule/book_appointment
    flows unchanged -- this module only recognises a request for PERSONAL
    COMMUNICATION with a doctor/clinician (a "speak to"/"talk to"/"call
    me"-shaped verb next to a doctor/clinician reference), never a
    schedule question or a booking request. See _EN_CONTACT_VERBS/
    _EN_DOCTOR_NOUNS below for exactly which verb+noun combinations count,
    and tests/test_doctor_personal_request.py's own "remains unaffected"
    classes for the negative proof.

  - It does NOT attempt to extract WHICH doctor was named. A caller who
    says "I want Dr Sen to call me" is recognised as this story's target
    (a name-shaped token next to a personal-contact verb, matched by
    regex since a doctor's name is open vocabulary, not a fixed phrase
    list) but the name itself is never captured or spoken back -- the
    fixed reply (agent/reply_templates.doctor_personal_request_reply())
    never names any doctor, named or not, which is what makes AC 3 true
    by construction rather than by hoping the model does not compose a
    promise. There is no code path in this module, or in the reply it
    triggers, that could ever say "Dr Sen will call you" -- that sentence
    simply does not exist anywhere in the fixed template set.

  - It does NOT decide callback availability itself. main.py's dispatch
    calls the SAME agent/callback_flow.check_callback_availability() the
    "request_callback" intent's own branch already uses, with the SAME
    agent/callback_config.CALLBACKS_ENABLED and the SAME already-cached
    get_clinic_info() hours lookup -- see main.py's _finish_doctor_
    personal_request() for the call site. This module has no callback
    logic of its own to duplicate or drift out of sync with the real one.

  - It does NOT introduce a new conversation state. The fixed reply
    states both real routes in one turn (book an appointment; a generic
    callback, only when actually available right now) and leaves
    session.pending unset afterward, same as human_direct_request's own
    "no offer, no choice" branch -- but unlike that branch, this ONE does
    offer routes, in words, because a caller's very next ordinary
    utterance ("book an appointment with Dr Sen" / "please call me back")
    is already correctly handled by the book_appointment/request_callback
    intents this codebase already has. Reusing those whole, existing,
    already-tested intents AS the "pending choice mechanism" is the
    "reuse it where appropriate" the story note asks for, without
    building a second, parallel state machine that would have to parse a
    caller's free-text choice between two named routes.
"""
from __future__ import annotations

import re

from agent.slot_parse import _bn_bounded

# --------------------------------------------------------------------- #
# English -- generic (unnamed) doctor/clinician personal-contact phrases.
# Built as verb x noun combinations (same "CORES x CARRIERS" spirit the
# test suite already uses, applied here directly in production code)
# rather than hand-written one at a time, so the combination is auditable
# as two short lists instead of dozens of near-duplicate sentences. Every
# resulting phrase already names BOTH a personal-contact verb AND a
# doctor/clinician noun -- never a bare "doctor" alone -- which is what
# keeps this from firing on "the doctor's visiting hours" or "available
# tomorrow".
# --------------------------------------------------------------------- #
_EN_CONTACT_VERBS = (
    "speak to", "speak with", "talk to", "talk with",
    "speak directly to", "speak directly with",
    "talk directly to", "talk directly with",
    "connect me to", "connect me with",
)
_EN_DOCTOR_NOUNS = (
    "a doctor", "the doctor", "my doctor",
    "a clinician", "the clinician", "my clinician",
)
_EN_VERB_FIRST_PHRASES = tuple(
    f"{verb} {noun}" for verb in _EN_CONTACT_VERBS for noun in _EN_DOCTOR_NOUNS
)

# Reversed order: "I want A DOCTOR TO call/talk/speak/reassure me" -- the
# noun comes first, the personal-contact verb trails it. Covers the
# story's own "I want a doctor to talk to me" / "please ask a doctor to
# call me" / "I want my doctor to call me" / "I need my doctor to
# reassure me" shape, which _EN_VERB_FIRST_PHRASES above does not (that
# set only covers the verb appearing BEFORE the noun).
_EN_TRAILING_CONTACT = (
    "to call me", "to ring me",
    "to talk to me", "to speak to me", "to speak with me",
    "to reassure me",
)
_EN_NOUN_FIRST_PHRASES = tuple(
    f"{noun} {trail}" for noun in _EN_DOCTOR_NOUNS for trail in _EN_TRAILING_CONTACT
)

_EN_GENERIC_PHRASES = _EN_VERB_FIRST_PHRASES + _EN_NOUN_FIRST_PHRASES

# Two more fixed phrases the story names explicitly that do not fit either
# combination shape above (no doctor/clinician noun immediately adjacent
# to the contact verb).
_EN_EXTRA_PHRASES = (
    "i want clinical reassurance",
    "i need clinical reassurance",
)

# --------------------------------------------------------------------- #
# English -- NAMED doctor personal-contact requests ("I want Dr Sen to
# call me", "Can Dr Sen speak to me now?", "Please ask Dr Sen to call
# me"). A doctor's name is open vocabulary (this system has no fixed list
# it could match against without depending on the live catalogue), so
# this is a regex over a "Dr <name>"-shaped token plus a nearby personal-
# contact trigger, in either order, rather than a literal phrase list.
# The name itself is never captured -- see this module's own docstring
# above for why: the fixed reply never repeats it, so there is nothing
# for AC 3 to protect by extracting it correctly in the first place.
# --------------------------------------------------------------------- #
_DR_NAME_RE = r"dr\.?\s+[a-z]+"
_CONTACT_TRIGGER_RE = r"(?:call me|ring me|speak to me|speak with me|talk to me|talk with me)"

_EN_NAMED_DOCTOR_CONTACT_RE = re.compile(
    r"(?:" + _DR_NAME_RE + r".{0,40}" + _CONTACT_TRIGGER_RE + r")"
    r"|(?:" + _CONTACT_TRIGGER_RE + r".{0,40}" + _DR_NAME_RE + r")"
    r"|(?:(?:speak to|speak with|talk to|talk with|speak directly to|talk directly to"
    r"|connect me to|connect me with).{0,10}" + _DR_NAME_RE + r")"
    r"|(?:(?:ask|tell).{0,15}" + _DR_NAME_RE + r".{0,20}(?:to call me|to ring me|to speak to me))"
    r"|(?:can.{0,5}" + _DR_NAME_RE + r".{0,20}(?:speak to me|speak with me|talk to me|call me))",
    re.IGNORECASE,
)

# --------------------------------------------------------------------- #
# Hinglish / Banglish (Hindi and Bengali written in Latin script). Same
# substring-of-lowercased-text treatment as the English generic set.
# --------------------------------------------------------------------- #
_LATIN_TRANSLIT_GENERIC = (
    "doctor se baat karna chahta hoon", "doctor se baat karni hai",
    "doctor se baat karna chahti hoon", "mujhe doctor se baat karni hai",
    "doctor se seedhe baat karna chahta hoon", "doctor se seedhe baat karni hai",
    "doctor se connect karo", "doctor se connect kar dijiye",
    "clinician se baat karna chahta hoon", "clinician se baat karni hai",
    "doctor mujhe call kare", "doctor mujhe phone kare",
    "doctor se call karwa dijiye", "doctor ko bolo mujhe call kare",
    "doctor ko kahiye mujhe call kare",
    "mera doctor mujhe call kare", "mujhe apne doctor se baat karni hai",
    "daktarer sathe kotha bolte chai", "daktarer sathe sojasuji kotha bolte chai",
    "daktar amake call korun", "daktar amake phone korun",
    "amar daktar amake call korun", "daktar ke bolben amake call korte",
    "daktarer sathe directly kotha bolte chai",
)
_LATIN_TRANSLIT_NAMED_DOCTOR_RE = re.compile(
    r"(?:dr\.?\s+[a-z]+.{0,30}(?:call kare|phone kare|call karo|baat kare|amake call korun|amake phone korun))"
    r"|(?:(?:bolo|bolen|bolben|kahiye).{0,20}dr\.?\s+[a-z]+.{0,20}(?:call kare|phone kare))",
    re.IGNORECASE,
)

# --------------------------------------------------------------------- #
# Bengali script. Word-bounded via _bn_bounded() for the generic phrases
# (see agent/human_fast_path.py's own comment on why a bare substring
# check is unsafe for Bengali). The named-doctor case is matched with a
# direct regex against the raw text instead, the same "open vocabulary"
# reasoning as the English/Latin named-doctor patterns above -- "ডা."/
# "ডক্টর" plus a name is not something a fixed, word-bounded phrase list
# could ever enumerate.
# --------------------------------------------------------------------- #
_BN_GENERIC = (
    "ডাক্তারের সাথে কথা বলতে চাই", "ডাক্তারের সাথে সরাসরি কথা বলতে চাই",
    "একজন ডাক্তারের সাথে কথা বলতে চাই", "ডাক্তারের সাথে যোগাযোগ করতে চাই",
    "ডাক্তার আমাকে কল করুন", "ডাক্তার আমাকে ফোন করুন",
    "আমার ডাক্তার আমাকে কল করুন", "আমার ডাক্তার আমাকে ফোন করুন",
    "একজন ডাক্তার আমাকে কল করুন", "ডাক্তারকে বলুন আমাকে কল করতে",
    "ডাক্তারকে বলুন আমাকে ফোন করতে",
)
_BN_NAMED_DOCTOR_RE = re.compile(
    r"(?:ডা\.?\s*\S+.{0,20}(?:কল করুন|ফোন করুন|কল করবেন|ফোন করবেন))"
    r"|(?:ডক্টর\s*\S+.{0,20}(?:কল করুন|ফোন করুন|কল করবেন|ফোন করবেন))"
)


def detect_doctor_personal_request(text: str) -> tuple[bool, str | None]:
    """True (plus which script/language family matched) when `text` is a
    caller asking for PERSONAL communication with a doctor/clinician --
    "speak to a doctor", "Dr Sen call me", "I need my doctor to reassure
    me" -- in any supported language. Deliberately never fires on a bare
    "doctor" mention, a schedule/availability question, or a booking
    request -- see this module's own docstring for the reasoning and
    tests/test_doctor_personal_request.py for the negative proof against
    exactly those three cases.

    The returned language tag ("english" | "latin_translit" | "bengali")
    identifies which phrase set matched, for logging only -- same
    convention agent/complaint_flow.detect_complaint() and
    agent/human_fast_path.detect_immediate_human_request() already
    established; it is NOT the language the reply should be spoken in.
    """
    if not text or not text.strip():
        return False, None
    lowered = text.lower()

    for phrase in _EN_GENERIC_PHRASES:
        if phrase in lowered:
            return True, "english"
    for phrase in _EN_EXTRA_PHRASES:
        if phrase in lowered:
            return True, "english"
    if _EN_NAMED_DOCTOR_CONTACT_RE.search(lowered):
        return True, "english"

    for phrase in _LATIN_TRANSLIT_GENERIC:
        if phrase in lowered:
            return True, "latin_translit"
    if _LATIN_TRANSLIT_NAMED_DOCTOR_RE.search(lowered):
        return True, "latin_translit"

    for phrase in _BN_GENERIC:
        if _bn_bounded(phrase, text):
            return True, "bengali"
    if _BN_NAMED_DOCTOR_RE.search(text):
        return True, "bengali"

    return False, None


def is_doctor_personal_request(text: str) -> bool:
    """Bare boolean form of detect_doctor_personal_request(), matching the
    naming convention agent/complaint_flow.is_complaint() and
    agent/human_fast_path.is_immediate_human_request() already established
    for this exact call site (main.py/main_pcm.py's _resolve_intent() and
    _continue_pending()) -- callers there only ever need the yes/no
    answer."""
    matched, _ = detect_doctor_personal_request(text)
    return matched
