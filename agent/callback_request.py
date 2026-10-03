"""A caller asking to be rung back: write it down, and never imply we will ring.

Ported from `merged_code/agent/callback_flow.py` (story: "Caller asks to be
called back"). The evidence behind that story holds here unchanged: **there is
no outbound calling capability in this stack** -- no dialer, no telephony
egress. So the only honest thing the agent can do is record who to call and why,
for a human to act on, and say exactly that.

WHAT IS PORTED, AND THE ONE PART THAT DELIBERATELY IS NOT:

`build_callback_reason()` is ported as it stood -- it is pure, and its priority
order (the caller's own stated reason, else the entity actually discussed and
confirmed earlier in the call, else nothing) is right for this codebase too,
including its refusal to invent a generic "general enquiry" when there is
genuinely nothing to record.

`check_callback_availability()` is **not** ported. It decides whether a callback
can be promised by comparing the current time against the clinic's opening
hours, and it needs those hours as a structured
`{weekday: {"closed": bool, "open": "HH:MM", "close": "HH:MM"}}` mapping. **No
such data exists anywhere in this repository.** Hours here are localised FREE
TEXT: `clinic-api/models.py`'s `DepartmentHours` stores `hours_bn`, `hours_hi`
and `hours_en` as strings, and the clinic-wide hours are a FAQ answer string
(`clinic-api/enquiry_service.py`'s `department_hours`). Parsing a sentence like
"সোম-শনি সকাল ৮টা - রাত ৮টা" in order to decide whether a callback may be
promised would be **inferring a fact from prose**, which is the one thing this
project forbids outright (CLAUDE.md rule 1). Promising a callback the clinic
cannot make, or refusing one it could, are both worse than not making the
time-of-day judgement at all -- so this port records the request and says
plainly that a colleague will call, without claiming when.

Adding structured hours is a real and worthwhile piece of schema work; it is
named in merge_report.md rather than faked here.

WHY THE DETECTOR IS NEW CODE RATHER THAN A PORT:

In the module this came from, "caller asks to be called back" was an **LLM
intent** (`request_callback` in that repo's `agent/llm.py`), so there were no
phrase lists to bring over. This repository's intent set and prompt are
deliberately untouched by this merge, so detection is deterministic here, in the
same shape as every other guard: fixed phrases, all languages checked at once,
before the model runs.

Checked AFTER agent/doctor_personal_request.py, which is the right precedence: a
caller asking for a DOCTOR to ring them back is answered by that guard, because
no specific doctor can be promised. The phrases here are the generic ones --
"call me back", "have someone call me" -- and the two lists do not overlap.

REASONED, not measured: all four lists are new, and a native speaker should
review the Bengali and Hindi.
"""

from __future__ import annotations

import unicodedata

# ---- English -------------------------------------------------------------
_EN = (
    "call me back", "call me later", "ring me back", "ring me later",
    "please call me", "can you call me", "could you call me",
    "can someone call me", "have someone call me", "get someone to call me",
    "i want a callback", "i need a callback", "i would like a callback",
    "give me a call back", "give me a callback", "call me when",
    "someone should call me", "ask someone to call me",
    "can i get a call back", "please ring me",
)

# ---- Hindi and Bengali in Latin script -----------------------------------
_LATIN = (
    "mujhe call back karo", "mujhe call back kijiye", "mujhe wapas call karo",
    "mujhe wapas call kijiye", "wapas call kijiye", "baad me call kijiye",
    "baad mein call kijiye", "mujhe baad me call karo",
    "koi mujhe call kare", "koi mujhe phone kare", "mujhe phone kar dijiye",
    "callback chahiye", "call back chahiye",
    "amake call back korun", "amake abar phone korun", "amake pore phone korun",
    "pore call korun", "keu amake phone korun", "amake phone kore janaben",
    "amake ekta call korun",
)

# ---- Bengali -------------------------------------------------------------
_BN = (
    "আমাকে কল ব্যাক করুন", "আমাকে কলব্যাক করুন", "আমাকে আবার ফোন করুন",
    "আমাকে পরে ফোন করুন", "আমাকে পরে কল করুন", "পরে কল করুন",
    "কেউ আমাকে ফোন করুন", "কেউ আমাকে কল করুন",
    "আমাকে ফোন করে জানাবেন", "আমাকে একটা কল করুন",
    "আমাকে ফোন করতে বলুন", "একটা কলব্যাক চাই",
)

# ---- Hindi (Devanagari) --------------------------------------------------
_HI = (
    "मुझे कॉल बैक कीजिए", "मुझे कॉलबैक कीजिए", "मुझे वापस कॉल कीजिए",
    "मुझे वापस फ़ोन कीजिए", "मुझे बाद में फ़ोन कीजिए", "मुझे बाद में कॉल कीजिए",
    "बाद में कॉल कीजिए", "बाद में फ़ोन कीजिए",
    "कोई मुझे फ़ोन करे", "कोई मुझे कॉल करे", "मुझे फ़ोन कर दीजिए",
    "मुझे एक कॉल कीजिए", "कॉलबैक चाहिए",
)

_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("en", _EN),
    ("latin", _LATIN),
    ("bn", _BN),
    ("hi", _HI),
)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


def detect_callback_request(text: str) -> tuple[bool, str | None]:
    """True, plus which phrase family matched, when the caller is asking to be
    rung back. The tag is for logging only, never the reply language.

    An ASCII phrase is matched against the lower-cased text; an Indic one against
    the text as spoken, since lower-casing means nothing for those scripts and
    every Indic entry here is multi-word (the embedded space is the word
    boundary Python's \\b cannot provide for a combining script).
    """
    if not text or not text.strip():
        return False, None
    raw = unicodedata.normalize("NFC", text)
    lowered = raw.lower()
    for tag, phrases in _FAMILIES:
        if any((p in lowered) if p.isascii() else (p in raw) for p in phrases):
            return True, tag
    return False, None


def asks_for_a_callback(text: str) -> bool:
    matched, _ = detect_callback_request(text)
    return matched


def build_callback_reason(
    stated_reason: str | None,
    active_test: str | None = None,
    active_doctor: str | None = None,
    active_package: str | None = None,
) -> str | None:
    """-> the reason to store beside a callback request, or None.

    Ported unchanged in behaviour. Priority, each step a real fact and never a
    guess: the caller's own words if they said why; else whichever entity was
    actually discussed and confirmed to exist earlier in this call; else None.

    Deliberately never falls back to an invented "general enquiry": an honest
    empty reason is preserved as empty, not papered over.
    """
    if stated_reason and stated_reason.strip():
        return stated_reason.strip()
    for entity in (active_test, active_doctor, active_package):
        if entity:
            return f"Regarding {entity}"
    return None
