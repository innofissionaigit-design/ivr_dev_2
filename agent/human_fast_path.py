"""ADDED BY SOURAV -- "Caller asks for a person immediately" story (Epic:
Conversation -- Difficult, Sensitive and Edge Cases).

story title: Caller asks for a person immediately
user story: As a caller who wants a person, I want one without negotiating,
    so that I do not feel trapped.
acceptance criteria: A request for a human in any supported language
    escalates on the same turn with no retention attempt and no question
    about why. The phrase set is tested per language and the rate is
    reported as a quality signal rather than something to minimise.

NAME NOTE: this module is deliberately NOT part of agent/fast_path.py,
despite the similar name. agent/fast_path.py's own module docstring is
explicit that its whole design purpose is efficiency -- skip the LLM for a
routine catalogue lookup, and "abstaining is a first-class result" the
moment it is not confident. That is the wrong design for a caller
explicitly demanding a human: this module must fire with total recall on
every phrase it lists, in every supported language, with zero tolerance
for "abstain and let the LLM decide" -- the exact opposite trade-off. See
agent/clinical_safety.py's own module docstring for the same reasoning
applied to a different story; this module is that story's sibling, kept
separate from agent/fast_path.py for the identical reason.

WHY THIS IS A SEPARATE, PRE-CLASSIFIER CHECK, NOT JUST A NEW INTENT
DESCRIBED IN agent/llm.py's PROMPT TEXT:

"escalates ... with no retention attempt" is a policy, and a prompt
instruction is advice a 7B model can and does ignore under phrasing
pressure -- the same argument agent/clinical_safety.py's docstring makes at
length. So this module runs BEFORE the classifier -- before the fast path,
before the semantic cache, before Ollama -- see main.py/main_pcm.py's
_resolve_intent(), which checks this first (ahead of even the clinical-
interpretation guard, since "give me a human right now" is the more
absolute, zero-tolerance-for-negotiation request of the two, and an
utterance that could plausibly read as either is safest resolved as
immediate escalation rather than an offer). A misclassification is
structurally impossible here, not merely unlikely: the model is never
given the chance to call this turn anything else.

WHY THIS ALSO HAS TO BE CHECKED INSIDE main.py/main_pcm.py's
_continue_pending(), NOT ONLY IN _resolve_intent():

_resolve_intent() is only ever reached for a turn that is NOT already
owned by an in-progress flow -- _dispatch_turn_inner calls
_continue_pending(session, text) FIRST, and only falls through to
_resolve_intent() when that returns False (see main_pcm.py's own docstring
on _continue_pending for the full flow-ownership model). A caller three
questions into a booking, or mid-OTP-verification, who suddenly says "just
connect me to a person" would otherwise have that sentence parsed as an
attempted answer to whatever field was pending (a date, a phone digit, an
OTP code) -- misreading the one sentence in the whole call that most
needs to be heard as itself. That is the literal shape of "feeling
trapped" the user story names, so this guard is checked again, first, at
the very top of _continue_pending, ahead of every field-specific parser
and ahead of the existing "না/thak" universal escape hatch (which only
covers ABANDONING a flow, not asking for a human specifically).

WHAT THIS MODULE DOES NOT DO: it makes no attempt to guess or resolve
WHY the caller wants a person -- the acceptance criterion is explicit that
no question about why is ever asked. It also never matches on the bare
word "doctor": a caller asking to talk to a doctor is already served,
deliberately differently, by the existing "book_appointment" /
"doctor_availability" flows (booking a consultation) and by the
"clinical_interpretation" story (a worried patient offered a doctor
specifically, with a fixed reassuring line) -- collapsing all of those
into "connect me to any human, right now, no questions asked" would be a
regression for callers who are not actually demanding to bypass this
system, merely asking about a doctor's schedule.
"""
from __future__ import annotations

from agent.slot_parse import _bn_bounded

# --------------------------------------------------------------------- #
# English. Checked as substrings of the lowercased utterance -- every
# entry is already a multi-word phrase naming BOTH the desire to talk/
# connect AND a human counterpart, deliberately never a bare "person",
# "human" or "agent" on its own (those words alone are too generic for a
# phone line that also says "our clinic staff" and "one of our experts" in
# its own replies -- a bare-word match would risk firing on an unrelated
# sentence that merely contains one of them).
# --------------------------------------------------------------------- #
_EN_HUMAN_REQUEST = (
    "talk to a person", "talk to a human", "talk to someone",
    "speak to a person", "speak to a human", "speak with a person",
    "speak with a human", "connect me to a person", "connect me to a human",
    "connect me to an agent", "connect me to a representative",
    "connect me to an operator", "connect me with a person",
    "get me a person", "get me a human", "get me an agent",
    "get me a representative", "get me a real person",
    "transfer me to a person", "transfer me to a human",
    "transfer me to an agent", "transfer me to a representative",
    "transfer me to an operator", "put me through to a person",
    "put me through to a human", "put me through to someone",
    "put me through to an agent", "put me through to a representative",
    "let me talk to a person", "let me talk to someone",
    "let me speak to someone", "let me speak to a person",
    "let me speak to a human", "i want a real person",
    "i want to talk to a person", "i want to talk to a human",
    "i want to speak to a person", "i want to speak to a human",
    "i want to speak with someone", "i want to talk to someone",
    "i want an agent", "i want a human", "i want a representative",
    "i want an operator", "can you connect me to a person",
    "can you connect me to a human", "can you connect me to an agent",
    "can you get me a person", "can you get me a human",
    "can you get me an agent", "can you get me a representative",
    "can you transfer me to a person", "can you transfer me to a human",
    "can you transfer me to an agent",
    "can you transfer me to a representative",
    "can you put me through to a person",
    "can you put me through to a human",
    "can you put me through to someone",
    "could you connect me to a person", "could you get me a human",
    "could you transfer me to an agent",
    "is there a real person", "no bot", "not a bot", "no robot",
    "not a robot", "stop the bot", "human agent please",
    "real person please", "give me a human", "give me a person",
    "give me an agent", "give me an operator", "give me a representative",
)

# --------------------------------------------------------------------- #
# Hinglish / Banglish (Hindi and Bengali written in Latin script). Same
# substring-of-lowercased-text treatment as English.
# --------------------------------------------------------------------- #
_LATIN_TRANSLIT_HUMAN_REQUEST = (
    "insan se baat", "insaan se baat", "kisi insan se baat",
    "kisi insaan se baat", "kisi bande se baat", "banda se baat karo",
    "bande se baat karao", "kisi se baat karni hai", "kisi se baat karvao",
    "kisi se connect karo", "kisi ko line pe do", "kisi ko phone do",
    "kisi ko de do", "human se baat", "human se connect",
    "real insan se baat", "real banda chahiye", "agent se baat",
    "agent se connect", "executive se baat", "executive se connect",
    "representative se baat", "operator se baat", "operator se connect",
    "mujhe insan chahiye", "mujhe koi insan chahiye", "mujhe agent chahiye",
    "mujhe kisi se baat karni hai", "bot nahi", "robot nahi",
    "manush er sathe kotha", "kono manush er sathe", "keu ekjon manush",
    "manush chai", "manush ke dao", "agent er sathe kotha",
    "executive er sathe kotha", "kauke line e dao", "kauke phone dao",
    "ekta manusher sathe kotha", "bot na", "robot na",
)

# --------------------------------------------------------------------- #
# Bengali script. Word-bounded via _bn_bounded() (shared from
# agent/slot_parse.py -- see agent/clinical_safety.py's own comment on why
# a bare substring check is unsafe for Bengali: vowel signs and the nukta
# are combining marks Python's \b does not treat as word characters, so a
# short syllable could otherwise match inside an unrelated longer word).
# --------------------------------------------------------------------- #
_BN_HUMAN_REQUEST = (
    "মানুষের সাথে কথা বলতে চাই", "মানুষের সাথে কথা বলাও",
    "কারো সাথে কথা বলতে চাই", "কারো সাথে কথা বলাও",
    "একজন মানুষের সাথে কথা", "সত্যিকারের মানুষ চাই", "মানুষ চাই",
    "এজেন্টের সাথে কথা বলতে চাই", "এজেন্টের সাথে সংযুক্ত করুন",
    "কাউকে লাইনে দিন", "কাউকে ফোন দিন", "কাউকে দিন",
    "বট না", "রোবট না", "কোনো মানুষ দিন",
)


def detect_immediate_human_request(text: str) -> tuple[bool, str | None]:
    """True (plus which script/language family matched) when `text` is a
    caller directly asking to be connected to a human being, in any
    supported language -- however plainly phrased (this story's AC names
    no roleplay/hypothetical trickery to guard against, unlike the
    clinical-interpretation story's adversarial framings, so the phrase
    set here is deliberately literal rather than pattern-generalised).

    The returned language tag ("english" | "latin_translit" | "bengali")
    identifies which phrase set matched, for logging only -- it is NOT the
    language the reply should be spoken in. main.py/main_pcm.py's dispatch
    already runs agent/bn_normalize.detect_language() on every turn for
    that purpose (the same function every other intent's reply-language
    choice already goes through); reusing that, rather than inventing a
    second language detector here, is what keeps this story's spoken
    reply consistent with how every other intent picks its language.
    """
    if not text or not text.strip():
        return False, None
    lowered = text.lower()

    for phrase in _EN_HUMAN_REQUEST:
        if phrase in lowered:
            return True, "english"

    for phrase in _LATIN_TRANSLIT_HUMAN_REQUEST:
        if phrase in lowered:
            return True, "latin_translit"

    for phrase in _BN_HUMAN_REQUEST:
        if _bn_bounded(phrase, text):
            return True, "bengali"

    return False, None


def is_immediate_human_request(text: str) -> bool:
    """Bare boolean form of detect_immediate_human_request(), matching the
    naming convention agent/clinical_safety.is_clinical_interpretation()
    already established for this exact call site
    (main.py/main_pcm.py's _resolve_intent() and _continue_pending()) --
    callers there only ever need the yes/no answer."""
    matched, _ = detect_immediate_human_request(text)
    return matched
