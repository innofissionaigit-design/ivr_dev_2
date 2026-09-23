"""ADDED BY SOURAV -- "Caller wants to make a complaint" story.

story title: Caller wants to make a complaint
user story: As a dissatisfied patient, I want my complaint recorded and
    routed to a person, so that it is not absorbed by a machine.
acceptance criteria:
    1. Complaint is recognised.
    2. Complaint is acknowledged exactly once.
    3. Complaint is captured verbatim.
    4. Complaint is routed to the complaints path/person-handling path.
    5. Agent does NOT attempt to resolve, explain, defend, justify, or
       argue about the complaint.

STRUCTURED DIRECTLY ON agent/human_fast_path.py, THE SAME SHAPE FOR THE
SAME REASON: detect_immediate_human_request() there is a deterministic,
literal, multi-word phrase match, checked BEFORE the fast path, the
semantic cache, and Ollama -- because "escalate ... with no negotiation"
is a policy, and a prompt instruction handed to a 7B model is advice that
model can and does ignore under phrasing pressure (see that module's own
"WHY THIS IS A SEPARATE, PRE-CLASSIFIER CHECK" section for the argument in
full; it applies here without any change). AC 5's "does NOT attempt to
resolve, explain, defend, justify, or argue" is exactly the same kind of
zero-tolerance policy -- the model must never even get a turn to try, so
this has to be a guard the classifier cannot outvote, not a bullet in its
prompt it can talk itself out of.

WHY THIS IS ALSO ITS OWN MODULE, NOT JUST MORE PHRASES ADDED TO
agent/human_fast_path.py: a complaint is not a request for a human in the
"connect me to a person" sense -- a caller filing a complaint is not
asking to bypass this assistant, they are reporting that something went
wrong. The two stories route to different final actions (this one stores
the verbatim text in a durable record before handing off; human_fast_path
never stores any caller text at all) and are tested and reasoned about
separately, so keeping them in separate modules keeps each one's own
phrase list auditable against its own story instead of a merged list
nobody can tell apart by story anymore.

WHY THIS ALSO HAS TO BE CHECKED INSIDE main.py/main_pcm.py's
_continue_pending(), NOT ONLY IN _resolve_intent():

Exactly the reasoning agent/human_fast_path.py's own docstring gives for
its own second call site: _resolve_intent() is only ever reached for a
turn NOT already owned by an in-progress flow -- _continue_pending() is
tried FIRST whenever session.pending is set. A caller mid-booking who
suddenly says "actually I want to file a complaint about how I was
treated last time" would otherwise have that sentence parsed as an
attempted answer to whatever field was pending (a date, a phone digit),
which is the opposite of AC 4's "routed to the complaints path" -- the
complaint would be silently swallowed by the booking flow instead. So this
guard is checked again, first, at the very top of _continue_pending, ahead
of every field-specific parser -- see main.py's own comment at that call
site for why it interrupts (rather than defers) whatever flow was already
in progress: AC 5 rules out doing anything BUT recording the complaint on
this turn, so there is nothing to finish first.

WHAT THIS MODULE DOES NOT DO: it makes no attempt to judge whether the
complaint is valid, to summarise it, to guess its category, or to soften
it -- the caller's words reach agent/tools_client.py's submit_complaint()
exactly as detected, untouched. It also never matches on bare, generic
words like "problem", "bad", or "issue" alone -- every phrase below
explicitly names the ACT of complaining (filing/making/lodging/raising a
complaint) or an unambiguous statement of dissatisfaction about the
clinic/service/treatment, for the same reason human_fast_path.py's own
docstring gives for avoiding a bare "person"/"human"/"agent" match: a
generic word alone is too easily part of an unrelated sentence on a phone
line that also discusses test results, doctors, and billing.
"""
from __future__ import annotations

from agent.slot_parse import _bn_bounded

# --------------------------------------------------------------------- #
# English. Checked as substrings of the lowercased utterance -- every
# entry is already a multi-word phrase naming either the ACT of
# complaining or an unambiguous statement of dissatisfaction, deliberately
# never a bare "problem", "bad", or "issue" on its own (see this module's
# own docstring above for why).
# --------------------------------------------------------------------- #
_EN_COMPLAINT = (
    "i want to file a complaint", "i want to make a complaint",
    "i want to lodge a complaint", "i want to register a complaint",
    "i want to raise a complaint", "i'd like to file a complaint",
    "i'd like to make a complaint", "i'd like to lodge a complaint",
    "i'd like to register a complaint", "i'd like to raise a complaint",
    "i have a complaint", "i have a complaint to make",
    "this is a complaint", "let me file a complaint",
    "let me make a complaint", "i need to complain",
    "i want to complain", "i want to complain about",
    "i'd like to complain", "i'd like to complain about",
    "i am here to complain", "filing a complaint",
    "making a complaint", "lodging a complaint", "registering a complaint",
    "raising a complaint", "file a formal complaint",
    "make a formal complaint", "i want to report bad service",
    "i want to report poor service", "i want to report how i was treated",
    "i was treated very badly", "i was treated so badly",
    "your staff was very rude to me", "your staff were rude to me",
    "the doctor was rude to me", "nobody helped me at the clinic",
    "i am extremely dissatisfied", "i am very dissatisfied",
    "i am extremely unhappy with the service",
    "i am very unhappy with the service",
    "i am extremely unhappy with this service",
    # FIXED BY SOURAV -- final validation pass. "this service"/"this
    # experience" is at least as natural a caller phrasing as "the
    # service"/"the experience" (a caller is usually describing THEIR
    # own just-had experience, not a generic one), and the original
    # list only had the latter -- a real AC-1 miss found by validating
    # the story's own literal first test case ("I am very unhappy with
    # THIS service.").
    "i am very unhappy with this service",
    "i am not happy with the service", "i am not happy with this service",
    "this service is unacceptable",
    "this is completely unacceptable", "this is really unacceptable",
    "i had a terrible experience", "i had a horrible experience",
    "i had a very bad experience at your clinic",
    "i had a very bad experience with this service",
    "worst experience i have ever had", "this is the worst service",
    "i want to speak to someone about a complaint",
    "i want this complaint noted down", "please note my complaint",
    "please register my complaint", "i am complaining about",
    # FIXED BY SOURAV -- final validation pass. A complaint tacked onto
    # a previous sentence with "also"/"and" ("I also want to file a
    # complaint...", "and I have a complaint...") is a very ordinary
    # way a caller adds a second thing mid-utterance; the original
    # phrases all assumed "I" was the very first word, which a
    # substring match then missed the moment anything (even one word)
    # came before it.
    "i also want to file a complaint", "i also want to make a complaint",
    "i also have a complaint", "and i have a complaint",
    "and i want to file a complaint", "and i want to make a complaint",
    "and i want to complain", "i also want to complain",
)

# --------------------------------------------------------------------- #
# Hinglish / Banglish (Hindi and Bengali written in Latin script). Same
# substring-of-lowercased-text treatment as English.
# --------------------------------------------------------------------- #
_LATIN_TRANSLIT_COMPLAINT = (
    "complaint karna chahta hoon", "complaint karna chahti hoon",
    "complaint file karna chahta hoon", "complaint file karna hai",
    "ek complaint hai", "mujhe complaint karni hai",
    "mujhe ek complaint darj karni hai", "complaint darj karo",
    "complaint darj karna chahta hoon", "shikayat karna chahta hoon",
    "shikayat karni hai", "meri ek shikayat hai",
    "mujhe shikayat darj karni hai", "shikayat darj karo",
    "shikayat darj karna hai", "shikayat darj karni chahta hoon",
    "aapki service se bahut naraz hoon", "bahut kharab service thi",
    "bahut buri service thi", "staff ne bahut badtameezi ki",
    "mujhe bahut bura anubhav hua", "yeh bilkul galat hai",
    "amar ekta complaint ache", "amar ekta obhijog ache",
    "complaint korte chai", "obhijog korte chai",
    "ami complaint korte chai", "ami obhijog korte chai",
    "complaint janate chai", "obhijog janate chai",
    "ekta complaint korbo", "ekta obhijog korbo",
    "apnader service niye amar complaint ache",
    "khub kharap byabohar korechilo", "staff khub kharap byabohar korlo",
    "khub baje experience hoyeche", "eta ekdom thik hoyni",
)

# --------------------------------------------------------------------- #
# Bengali script. Word-bounded via _bn_bounded() (shared from
# agent/slot_parse.py -- see agent/human_fast_path.py's own comment on why
# a bare substring check is unsafe for Bengali: vowel signs and the nukta
# are combining marks Python's \b does not treat as word characters, so a
# short syllable could otherwise match inside an unrelated longer word).
# --------------------------------------------------------------------- #
_BN_COMPLAINT = (
    "আমার একটা অভিযোগ আছে", "আমি অভিযোগ করতে চাই",
    "অভিযোগ জানাতে চাই", "একটা অভিযোগ করতে চাই",
    "কমপ্লেইন করতে চাই", "কমপ্লেইন জানাতে চাই",
    "আমি কমপ্লেইন করতে চাই", "আমার একটা কমপ্লেইন আছে",
    "খুব খারাপ ব্যবহার করেছে", "খুব খারাপ পরিষেবা পেয়েছি",
    "আমি খুব অসন্তুষ্ট", "এটা একদম ঠিক হয়নি",
    "আমার সাথে খুব খারাপ আচরণ করা হয়েছে", "স্টাফ খুব খারাপ ব্যবহার করেছে",
    "ডাক্তার খুব খারাপ ব্যবহার করেছেন", "এই পরিষেবা মোটেও ভালো না",
)


def detect_complaint(text: str) -> tuple[bool, str | None]:
    """True (plus which script/language family matched) when `text` is a
    caller filing or stating a complaint, in any supported language.

    The returned language tag ("english" | "latin_translit" | "bengali")
    identifies which phrase set matched, for logging only -- it is NOT the
    language the reply should be spoken in, same convention
    agent/human_fast_path.detect_immediate_human_request() already
    established: main.py/main_pcm.py's dispatch already runs
    agent/bn_normalize.detect_language() on every turn for that purpose.
    """
    if not text or not text.strip():
        return False, None
    lowered = text.lower()

    for phrase in _EN_COMPLAINT:
        if phrase in lowered:
            return True, "english"

    for phrase in _LATIN_TRANSLIT_COMPLAINT:
        if phrase in lowered:
            return True, "latin_translit"

    for phrase in _BN_COMPLAINT:
        if _bn_bounded(phrase, text):
            return True, "bengali"

    return False, None


def is_complaint(text: str) -> bool:
    """Bare boolean form of detect_complaint(), matching the naming
    convention agent/human_fast_path.is_immediate_human_request() and
    agent/clinical_safety.is_clinical_interpretation() already established
    for this exact call site (main.py/main_pcm.py's _resolve_intent() and
    _continue_pending()) -- callers there only ever need the yes/no
    answer."""
    matched, _ = detect_complaint(text)
    return matched
