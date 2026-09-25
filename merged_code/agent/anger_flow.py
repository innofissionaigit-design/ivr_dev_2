"""ADDED BY SOURAV -- "Caller is angry about a previous experience" story.

story title: Caller is angry about a previous experience
user story: As a frustrated caller, I want to be heard and offered a
    person, so that my frustration is not compounded by a machine.
acceptance criteria:
    1. Anger lowers the escalation threshold.
    2. Delivery slows.
    3. A human is offered explicitly rather than continuing to transact.
    4. The agent apologises once.
    5. The agent does not argue.
    6. The agent does not defend the hospital.
    7. The complaint/frustration is captured in the context packet.

WHY THIS IS A SEPARATE, PRE-CLASSIFIER GUARD, STRUCTURED DIRECTLY ON
agent/complaint_flow.py's OWN SHAPE, FOR THE SAME REASON:

The investigation for this story ("Caller_anger_investigation_report.txt")
found that NO existing guard catches a plain expression of anger or
frustration -- agent/complaint_flow.py's own phrase list requires the
caller to name the ACT of complaining or an unambiguous statement of
dissatisfaction about the clinic/service/treatment ("I am very
dissatisfied", "I had a terrible experience"); it does not match a bare
"I am so angry" or "this is ridiculous". agent/human_fast_path.py
likewise requires an explicit request to be connected to a human being.
Neither guard fires for pure anger, so today that utterance falls
straight through to fast_path/the semantic cache/the LLM -- the exact
same "the model is the one place a policy can be talked out of" gap
agent/unverifiable_claim.py's own docstring names for a different
failure mode. AC 5/AC 6's "does not argue"/"does not defend the
hospital" are zero-tolerance policies for the identical reason AC 5 is
one in agent/complaint_flow.py's own docstring: the model must never
even get a turn to try, so this has to be a guard the classifier cannot
outvote, not a bullet in its prompt it can talk itself out of. Per this
story's own explicit instruction, the LLM is never the primary anger
detector -- this deterministic module is.

WHY THIS IS ALSO ITS OWN MODULE, NOT JUST MORE PHRASES ADDED TO
agent/complaint_flow.py: a caller expressing anger is not necessarily
filing a complaint in the sense Story 2 already handles (naming the ACT
of complaining, or naming a specific bad experience) -- they may simply
be frustrated in the moment ("I am fed up with this", "this is
extremely frustrating") without ever saying anything complaint_flow.py
would recognise. The two stories deliberately stay distinguishable
(their own reason codes in the shared escalation ledger, their own
reply templates) for the same reason agent/complaint_flow.py's own
docstring gives for staying separate from agent/human_fast_path.py: a
merged phrase list would make each story's own vocabulary impossible to
audit against its own AC set.

WHY THIS IS CHECKED AFTER agent/complaint_flow.py, NOT BEFORE OR
INSTEAD OF IT -- "ANGER ENHANCES THE COMPLAINT FLOW, IT DOES NOT COMPETE
WITH IT":

This story's own explicit instruction is that Story 2 (complaint) must
not be redesigned, and that an utterance which is BOTH angry and
clearly a complaint must not produce two competing flows or a double
acknowledgement/double human-routing. main.py's/main_pcm.py's
_resolve_intent() and _continue_pending() both check is_complaint(text)
strictly BEFORE is_anger(text) (this module), in the same fixed-order,
first-match-wins chain every guard pair in this codebase already uses
to resolve overlap (e.g. complaint is itself checked before
doctor_personal_request, symptom_routing before unverifiable_claim).
Because of that ordering alone -- with no code added to
agent/complaint_flow.py or _finish_complaint() at all -- an utterance
that matches BOTH modules' phrase lists is routed through the complaint
flow exactly as Story 2 already built it (one acknowledgement, one
verbatim record, one handoff), and this module's own guard never even
runs for it. Story 2's own acknowledgement reply already contains an
apology ("I'm sorry to hear that"), which already satisfies this
story's own AC 4 for that turn without this module doing anything
further. This is "enhance, don't compete" implemented as guard order,
the same technique that already resolves every other overlapping pair
of guards in this codebase, not a new merging mechanism.

WHY THIS DOES NOT INTERCEPT ORDINARY DISSATISFACTION OR ORDINARY
REQUESTS:

"the guard should be narrow enough to avoid treating normal
dissatisfaction or ordinary requests as anger" is this story's own
explicit constraint. Every phrase in _EN_ANGER/_LATIN_TRANSLIT_ANGER/
_BN_ANGER below is already a strong, explicit, multi-word statement of
anger or frustration ("I am very angry", "I am fed up with this", "you
people are useless", "this is extremely frustrating") -- never a bare
word like "bad", "annoyed", or "issue" on its own, the same discipline
agent/complaint_flow.py's own docstring holds itself to for the
identical reason: a generic word alone is too easily part of an
unrelated sentence on a phone line that also discusses test results,
doctors, and billing. A caller merely asking an ordinary question in an
impatient tone with nothing else stated ("why is this taking so long"
alone) matches nothing here, mirroring agent/llm.py's own existing
"complaint" intent note making the identical distinction for Story 2.

WHY THIS DOES NOT BUILD A SECOND COMPLAINT-STORAGE SYSTEM OR A SECOND
SPEECH-RATE SYSTEM: per this story's own explicit design decisions,
main.py's _finish_anger() (the only consumer of this module) reuses
clinic-api's existing ComplaintRecord table via agent/tools_client.py's
existing submit_complaint() -- the same call site agent/complaint_flow.py
already uses -- rather than a second table, and rebuilds
session.call_state via the existing, already-built
agent/call_state.py::build()/responses/speech_policy.py machinery
(whose "angry" row already maps to speech_rate="slow_normal",
escalation_threshold="low") rather than inventing a second speech-rate
mechanism. This module itself is pure detection only -- it makes no
tool call, sets no state, and speaks nothing; see main.py's
_finish_anger() for all of that.
"""
from __future__ import annotations

from agent.slot_parse import _bn_bounded

# --------------------------------------------------------------------- #
# English. Checked as substrings of the lowercased utterance -- every
# entry is already a strong, explicit, multi-word statement of anger or
# frustration, deliberately never a bare "bad", "annoyed", or "issue" on
# its own (see this module's own docstring above for why).
# --------------------------------------------------------------------- #
_EN_ANGER = (
    "i am very angry", "i'm very angry", "i am so angry", "i'm so angry",
    "i am extremely angry", "i'm extremely angry", "i am really angry",
    "i'm really angry", "i am angry", "i'm angry",
    "i am furious", "i'm furious", "i am absolutely furious",
    "i am fed up with this", "i'm fed up with this", "i am fed up",
    "i'm fed up", "i am so fed up", "i'm so fed up",
    "you people are useless", "you guys are useless",
    "your staff are useless", "this hospital is useless",
    "this is extremely frustrating", "this is so frustrating",
    "this is very frustrating", "this is really frustrating",
    "i am extremely frustrated", "i'm extremely frustrated",
    "i am so frustrated", "i'm so frustrated", "i am very frustrated",
    "i'm very frustrated",
    "i have had enough of this", "i've had enough of this",
    "i have had enough", "i've had enough",
    "this is ridiculous", "this is absolutely ridiculous",
    "this is completely ridiculous",
    "i am sick of this", "i'm sick of this",
    "i am sick and tired of this", "i'm sick and tired of this",
    "i am losing my patience", "i'm losing my patience",
    "this is infuriating", "i am outraged", "i'm outraged",
)

# --------------------------------------------------------------------- #
# Hinglish / Banglish (Hindi and Bengali written in Latin script). Same
# substring-of-lowercased-text treatment as English.
# --------------------------------------------------------------------- #
_LATIN_TRANSLIT_ANGER = (
    "mujhe bahut gussa aa raha hai", "mujhe gussa aa raha hai",
    "mujhe bahut gussa hai", "mujhe bohot gussa aa raha hai",
    "main bahut gussa hoon", "main bahut naraz hoon",
    "main bahut pareshan ho gaya hoon", "main bahut pareshan ho gayi hoon",
    "main bahut frustrated hoon", "yeh bahut frustrating hai",
    "mujhe bahut irritation ho raha hai", "mujhe bahut irritation horaha hai",
    "bas bahut ho gaya", "bahut ho gaya ab", "bahut ho chuka hai ab",
    "yeh bilkul bakwas hai", "aap log bekar hain", "aap log nikamme hain",
    "mera sabr khatam ho gaya hai",
    "ami khub raag korchi", "amar khub raag hocche", "amar khub raag uthche",
    "ami onek frustrated", "ami khub frustrated", "eta khub frustrating",
    "eta ekdom frustrating", "ami ar shoy korte parchi na",
    "ami r shoy korte parchi na", "onek hoyeche ekhon",
    "onek shoyjo korechi ar na", "tomra kono kajer na",
    "apnara kono kajer na",
)

# --------------------------------------------------------------------- #
# Bengali script. Word-bounded via _bn_bounded() (shared from
# agent/slot_parse.py -- see agent/complaint_flow.py's own comment on why
# a bare substring check is unsafe for Bengali: vowel signs and the nukta
# are combining marks Python's \b does not treat as word characters, so a
# short syllable could otherwise match inside an unrelated longer word).
# --------------------------------------------------------------------- #
_BN_ANGER = (
    "আমি খুব রাগান্বিত", "আমার খুব রাগ হচ্ছে", "আমি প্রচণ্ড রেগে আছি",
    "আমি খুব রেগে গেছি", "আমার প্রচণ্ড রাগ হচ্ছে",
    "এটা খুবই বিরক্তিকর", "এটা একদম বিরক্তিকর",
    "আমি আর সহ্য করতে পারছি না", "আমি আর সহ্য করতে পারছি না এটা",
    "অনেক হয়েছে এখন", "অনেক সহ্য করেছি আর না",
    "আমি খুব বিরক্ত", "আমি অত্যন্ত বিরক্ত",
    "আপনারা কোনো কাজের না", "তোমরা কোনো কাজের না",
)


def detect_anger(text: str) -> tuple[bool, str | None]:
    """True (plus which script/language family matched) when `text` is a
    caller expressing anger or frustration, in any supported language.

    The returned language tag ("english" | "latin_translit" | "bengali")
    identifies which phrase set matched, for logging only -- it is NOT
    the language the reply should be spoken in, same convention
    agent/complaint_flow.py's own detect_complaint() already established:
    main.py/main_pcm.py's dispatch already runs
    agent/bn_normalize.detect_language() on every turn for that purpose.
    """
    if not text or not text.strip():
        return False, None
    lowered = text.lower()

    for phrase in _EN_ANGER:
        if phrase in lowered:
            return True, "english"

    for phrase in _LATIN_TRANSLIT_ANGER:
        if phrase in lowered:
            return True, "latin_translit"

    for phrase in _BN_ANGER:
        if _bn_bounded(phrase, text):
            return True, "bengali"

    return False, None


def is_anger(text: str) -> bool:
    """Bare boolean form of detect_anger(), matching the naming
    convention agent/complaint_flow.py::is_complaint() and every other
    guard module in this codebase already established for this exact
    call site (main.py/main_pcm.py's _resolve_intent() and
    _continue_pending()) -- callers there only ever need the yes/no
    answer."""
    matched, _ = detect_anger(text)
    return matched
