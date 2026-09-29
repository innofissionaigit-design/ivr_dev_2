"""Caller asks WHICH doctors there are, without naming one.

WHY THIS EXISTS
---------------
Live transcript:

    Caller: ডাক্তার অবৈলাবল আছে
    Agent:  কোন ডাক্তারের কথা জিজ্ঞেস করছেন?
    Caller: কে কে ডাক্তার অবৈলাবল আছে সেটা বলুন
    Agent:  কোন ডাক্তারের কথা জিজ্ঞেস করছেন?

Every doctor intent this agent had was keyed on something the caller must
already know: doctor_availability / doctor_schedule on a doctor's NAME,
doctors_by_department on a DEPARTMENT. "Who are your doctors?" fitted none of
them, so the classifier chose the nearest (doctor_availability), its
doctor_name slot came back empty, and the missing-slot prompt asked for the
very thing the caller was asking us for. Re-asking changed nothing, so the
same question came back word for word.

It was worse one turn later. With a doctor discussed earlier in the call,
agent/state.py backfilled that doctor into the empty slot -- "আপনাদের কি কি
ডাক্তার এখানে বসেন" was answered with Dr Sen's next free day. A confident
answer to a question nobody asked.

WHAT THIS DOES
--------------
is_doctor_list_request() recognises the question -- a doctor noun together
with a plural/list cue ("কে কে", "কি কি", "কোন কোন", "সব ডাক্তার", "ডাক্তারদের
লিস্ট", "which doctors", ...). It is deliberately a CUE check and not a
classifier: it says nothing about what else the sentence contains.

apply_list_doctors_override() is applied to whatever the three intent tiers
(fast path, semantic cache, LLM) returned. It rewrites a part to
"list_doctors" ONLY when that part's classification could not have been
answered anyway -- a doctor intent with no doctor, a department intent with no
department, or one of the no-lookup intents (unclear / smalltalk /
out_of_scope). A part that names a doctor or a department is never touched,
so "কার্ডিওলজিতে কে কে ডাক্তার আছেন" with department=Cardiology still goes to
doctors_by_department exactly as before, and "ডাক্তার সেন কবে বসেন" is not
affected by any cue here. Running AFTER the tiers, rather than as one more
pre-classification guard, is what makes that distinction possible: the tiers
decide what was named; this only corrects the one shape they get wrong.

The answer itself comes from clinic-api (/api/v1/departments), never from
here and never from the model: main.py lists the departments that have
doctors and asks which one, then hands over to the existing
doctors_by_department flow. Reading out every doctor in the clinic would be
thirty-odd names on a phone line.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

LIST_DOCTORS = "list_doctors"


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


_BN = r"ঀ-৿"

# The doctor noun, as said and as the ASR spells it. Plural/possessive
# endings (ডাক্তাররা, ডাক্তারদের, ডাক্তারবাবুরা) are covered because the stem
# match below is not anchored on the right.
_DOCTOR_NOUNS_BN = tuple(_nfc(w) for w in ("ডাক্তার", "ডক্টর", "ডাক্তারবাবু", "চিকিৎসক"))
_DOCTOR_NOUNS_LATIN = ("doctor", "doctors", "daktar", "dactor", "dr")

# Cues that make it a LIST question. Reduplicated wh-words are the Bengali
# plural question ("কে কে" = who all); a quantifier or "which" must sit right
# next to the noun, since "সব ঠিক আছে, ডাক্তার সেন কবে বসেন" also contains
# both words.
_REDUPLICATED_WH_BN = tuple(_nfc(w) for w in ("কে কে", "কি কি", "কী কী", "কোন কোন", "কোনো কোনো",
                                             "কারা কারা", "কাকে কাকে"))
_PLURAL_WH_BN = tuple(_nfc(w) for w in ("কারা", "কাদের", "কতজন", "কয়জন", "ক'জন"))
_LIST_WORDS_BN = tuple(_nfc(w) for w in ("লিস্ট", "তালিকা", "নামগুলো", "নামগুলি"))
_ADJACENT_QUANTIFIERS_BN = tuple(_nfc(w) for w in ("সব", "সকল", "সমস্ত", "কোন", "কোন্‌", "কোনো"))
_PLURAL_DOCTOR_BN = tuple(_nfc(w) for w in ("ডাক্তাররা", "ডাক্তারেরা", "ডাক্তারদের",
                                           "ডাক্তারবাবুরা", "ডাক্তারবাবুদের", "ডক্টররা", "ডক্টরদের"))

_LATIN_LIST_PATTERNS = (
    r"\b(?:which|what|all|list of|all the|how many)\s+(?:the\s+)?(?:doctors?|drs?)\b",
    r"\bdoctors?\s+(?:list|available|are there|do you have)\b",
    r"\bwho\s+are\s+(?:the|your)\s+doctors\b",
    r"\b(?:ke ke|ki ki|kon kon|kaun kaun|kaun se|konse|kon se|sab|sob|shob)\s+(?:doctor|daktar|dactor)s?\b",
    r"\b(?:doctor|daktar)s?\s+(?:ke ke|kaun kaun|kara)\b",
)


# "Which DEPARTMENTS do you have?" is answered by the same reply as "which
# doctors" -- the department list, then "which one?". Live, it was not
# recognised at all: "আপনাদের এখানে কোন কোন ডিপার্টমেন্ট রয়েছে" / "বিভাগগুলোর
# নাম বলুন" have no doctor word, so the model's nearest guess --
# doctors_by_department with no department -- asked "কোন বিভাগের ডাক্তার
# খুঁজছেন?", i.e. asked the caller the very list they had asked for.
_DEPARTMENT_NOUNS_BN = tuple(_nfc(w) for w in ("বিভাগ", "ডিপার্টমেন্ট", "ডিপার্টমেন্ট", "স্পেশালিটি"))
_DEPARTMENT_NOUNS_LATIN = ("department", "departments", "dept", "speciality", "specialty", "specialities")
# A plural or "the names" said with the noun is a list request by itself:
# "বিভাগগুলোর নাম বলুন", "বিভাগগুলো কী কী", "ডিপার্টমেন্টের লিস্ট".
_PLURAL_ENDINGS_BN = tuple(_nfc(w) for w in ("গুলো", "গুলি", "গুলোর", "গুলির", "সমূহ"))
_NAME_ASK_BN = tuple(_nfc(w) for w in ("নাম বলুন", "নাম বলবেন", "নামগুলো", "নাম বল", "নাম জানান"))


def _has_department_noun(key: str) -> bool:
    if any(n in key for n in _DEPARTMENT_NOUNS_BN):
        return True
    return re.search(r"\b(?:" + "|".join(_DEPARTMENT_NOUNS_LATIN) + r")\b", key) is not None


def is_department_list_request(text: str) -> bool:
    """Is the caller asking WHICH departments there are?"""
    key = _nfc(text).lower()
    if not key.strip() or not _has_department_noun(key):
        return False
    if re.search(r"\b(?:which|what|all|list of|all the|how many)\s+(?:the\s+)?(?:departments?|specialit(?:y|ies))\b", key):
        return True
    if re.search(r"\bdepartments\b", key):
        return True
    if any(noun + pl in key for noun in _DEPARTMENT_NOUNS_BN for pl in _PLURAL_ENDINGS_BN):
        return True
    if any(w in key for w in _REDUPLICATED_WH_BN + _LIST_WORDS_BN + _NAME_ASK_BN):
        return True
    if any(re.search(rf"(?<![{_BN}]){re.escape(w)}(?![{_BN}])", key) for w in _PLURAL_WH_BN):
        return True
    nouns = "|".join(re.escape(n) for n in _DEPARTMENT_NOUNS_BN)
    return any(re.search(rf"(?<![{_BN}]){re.escape(w)}\s+(?:{nouns})", key)
               for w in _ADJACENT_QUANTIFIERS_BN)


def _has_doctor_noun(key: str) -> bool:
    if any(noun in key for noun in _DOCTOR_NOUNS_BN):
        return True
    return re.search(r"\b(?:" + "|".join(_DOCTOR_NOUNS_LATIN) + r")s?\b", key) is not None


def _adjacent_to_doctor(word: str, key: str) -> bool:
    """`word` immediately before a doctor noun, as a whole word."""
    nouns = "|".join(re.escape(n) for n in _DOCTOR_NOUNS_BN)
    return re.search(rf"(?<![{_BN}]){re.escape(word)}\s+(?:{nouns})", key) is not None


def is_doctor_list_request(text: str) -> bool:
    """Is the caller asking which/who the doctors are, rather than about one
    they have named? A cue check only -- see the module docstring for why the
    decision to ACT on it lives in apply_list_doctors_override()."""
    key = _nfc(text).lower()
    if not key.strip():
        return False
    if any(re.search(p, key) for p in _LATIN_LIST_PATTERNS):
        return True
    if not _has_doctor_noun(key):
        return False
    if any(p in key for p in _PLURAL_DOCTOR_BN):
        return True
    if any(w in key for w in _REDUPLICATED_WH_BN + _LIST_WORDS_BN):
        return True
    if any(re.search(rf"(?<![{_BN}]){re.escape(w)}(?![{_BN}])", key) for w in _PLURAL_WH_BN):
        return True
    return any(_adjacent_to_doctor(w, key) for w in _ADJACENT_QUANTIFIERS_BN)


# The classifications that can be corrected, and the slot whose ABSENCE
# makes them correctable. None = no slot to check (the intent answers nothing
# a list request could have wanted, and smalltalk is the one lane where the
# model composes its own reply -- never a place for a factual question).
_CORRECTABLE = {
    "doctor_availability": "doctor_name",
    "doctor_schedule": "doctor_name",
    "doctors_by_department": "department",
    "unclear": None,
    "smalltalk": None,
    "out_of_scope": None,
}


# What the model sometimes files as a department when the caller asked for
# the list: "আপনাদের কি কি বিভাগ আছে" came back department="কি কি". A value
# made only of question/list words is no department at all.
_NOT_A_DEPARTMENT = frozenset(_nfc(w).lower() for w in (
    "কি", "কী", "কোন", "কোনো", "কে", "কারা", "কি কি", "কী কী", "কোন কোন", "সব", "সকল", "সমস্ত",
    "বিভাগ", "বিভাগগুলো", "ডিপার্টমেন্ট", "লিস্ট", "তালিকা", "নাম",
    "which", "what", "all", "list", "department", "departments", "any",
))


def _real_department(value) -> bool:
    words = [w for w in _nfc(str(value or "")).lower().split() if w]
    return bool(words) and not all(w in _NOT_A_DEPARTMENT for w in words)


def _should_override(part: dict) -> bool:
    intent = part.get("intent")
    if intent not in _CORRECTABLE:
        return False
    slot = _CORRECTABLE[intent]
    if slot is None:
        return True
    value = (part.get("slots") or {}).get(slot)
    return not (_real_department(value) if slot == "department" else value)


def _as_list_part(part: dict) -> dict:
    slots = dict(part.get("slots") or {})
    # A doctor name can only have been backfilled or hallucinated here --
    # _should_override() already established none was extracted -- so it is
    # removed rather than carried into a question about ALL doctors.
    slots["doctor_name"] = None
    # A DEPARTMENT, though, is the caller narrowing the question. Live:
    # "অর্থোপেডিক্সে কোন কোন ডক্টর বসছেন কালকে" was rewritten to list_doctors
    # and answered with all eight departments -- the one fact the caller gave
    # thrown away. With a department it is doctors_by_department, which lists
    # THAT department's doctors for the day asked. (When the model did not
    # fill the slot, main.py's _answer_doctor_list finds it in the words.)
    if not _real_department(slots.get("department")):
        slots["department"] = None
    intent = "doctors_by_department" if slots.get("department") else LIST_DOCTORS
    return {**part, "intent": intent, "slots": slots}


# Intents that cannot be answered without a doctor, arriving with NO doctor
# but WITH a department: "অর্থোপেডিক্সের ডাক্তার কাল বসেন?" classified as
# doctor_availability{doctor_name: None, department: "orthopaedics"} asked
# "which doctor?" of a caller who had just said which department -- and a
# list_doctors the model emits with a department is the same question.
# book_appointment is NOT here: _open_booking lists the department's doctors
# itself and then carries on into the booking, which this would lose.
_NEEDS_A_DOCTOR = frozenset({"doctor_availability", "doctor_schedule", LIST_DOCTORS,
                             "unclear", "smalltalk"})


def _department_instead_of_doctor(part: dict) -> bool:
    slots = part.get("slots") or {}
    return (part.get("intent") in _NEEDS_A_DOCTOR
            and not slots.get("doctor_name") and _real_department(slots.get("department")))


def _as_department_part(part: dict) -> dict:
    slots = dict(part.get("slots") or {})
    slots["doctor_name"] = None
    return {**part, "intent": "doctors_by_department", "slots": slots}


def apply_department_correction(data: dict) -> dict:
    """-> `data` with every part that names a department but no doctor, under
    an intent that needs a doctor, rewritten to doctors_by_department; `data`
    itself (same object) when nothing needs it. Never mutates its input: the
    semantic cache may hold it."""
    if not isinstance(data, dict):
        return data
    top = {"intent": data.get("intent"), "slots": data.get("slots")}
    parts = data.get("parts")
    change_top = _department_instead_of_doctor(top)
    new_parts = None
    if isinstance(parts, list):
        rewritten = [_as_department_part(p) if isinstance(p, dict) and _department_instead_of_doctor(p)
                     else p for p in parts]
        if any(a is not b for a, b in zip(rewritten, parts)):
            new_parts = rewritten
    if not change_top and new_parts is None:
        return data
    out = dict(data)
    if change_top:
        fixed = _as_department_part(top)
        out["intent"], out["slots"], out["direct_reply_bn"] = fixed["intent"], fixed["slots"], None
    if new_parts is not None:
        out["parts"] = new_parts
    return out


# "<something>'s doctor" -- the way a caller names a KIND of doctor rather than
# a person: চোখের ডাক্তার (eye), দাঁতের ডাক্তার (teeth), বাচ্চাদের ডাক্তার
# (children's), skin er doctor. When the doctor lookup finds nobody and the
# department lookup finds nothing either, a sentence of this shape is still a
# request for a department we do not have -- and the useful answer is the list
# of departments we do have, not "no doctor by that name", which is true and
# helps nobody. A person's name is not said this way ("ডাক্তার নোবডি" is not).
_SPECIALTY_BN = re.compile(
    rf"[{_BN}](?:ের|এর|র)\s+(?:ডাক্তার|ডক্টর|ডাক্তারবাবু|বিশেষজ্ঞ|স্পেশালিস্ট)")
_SPECIALTY_LATIN = re.compile(
    r"\b\w+\s+(?:er|ka|ke|ki|r)\s+(?:doctor|dr|specialist|daktar)s?\b"
    r"|\b(?:specialist|speciality|specialty)\b")


def looks_like_specialty_request(text: str) -> bool:
    """Does `text` name a kind of doctor ("চোখের ডাক্তার") rather than one?"""
    key = _nfc(text).lower()
    return bool(_SPECIALTY_BN.search(key) or _SPECIALTY_LATIN.search(key))


# Words in a department request that say nothing about WHICH department --
# excluded from the fuzzy step below, where "ডাক্তার" would otherwise make
# every "<X>ের ডাক্তার" alias look alike.
_DEPT_FILLER = frozenset(_nfc(w).lower() for w in (
    "ডাক্তার", "ডাক্তারের", "ডাক্তারবাবু", "ডাক্তাররা", "ডাক্তারদের", "ডক্টর", "ডঃ", "ডাঃ",
    "বিভাগ", "বিভাগের", "বিভাগে", "বিভাগটা", "ডিপার্টমেন্ট", "স্পেশালিস্ট", "বিশেষজ্ঞ",
    "কে", "কে কে", "কি", "কী", "কোন", "কোন কোন", "কারা", "আছেন", "আছে", "বসেন", "বসছেন",
    "বসবেন", "চাই", "দেখাতে", "দরকার", "লাগবে", "আমার", "একজন", "এর", "টা", "কাল",
    "কালকে", "আজ", "আজকে", "doctor", "doctors", "dr", "department", "specialist",
    "chai", "er", "ke", "ki", "kon", "the", "a", "in", "please",
))
# Fuzzy thresholds. 0.75 sits above the one near-collision between two REAL
# departments in the seed ("ডার্মাটোলজি" vs "ডায়াবেটোলজি" is 0.70) and below
# the ASR garbles actually heard ("অর্থ পেডিটস" vs "অর্থোপেডিক্স" is 0.82 with
# the ASR's split closed up). The margin keeps two similar departments from
# being settled by a coin flip.
_DEPT_FUZZY_FLOOR = 0.75
_DEPT_FUZZY_MARGIN = 0.10


def _dept_forms(dept: dict) -> list[str]:
    forms = [dept.get("name") or "", dept.get("department_bn") or ""] + list(dept.get("aliases_bn") or [])
    return [f for f in (_nfc(f).strip().lower() for f in forms) if f]


def match_department(text: str, departments: list[dict]) -> dict | None:
    """-> the ONE department (as clinic-api listed it) the caller names, or None.

    Works on a whole sentence ("অর্থোপেডিক্সে কোন কোন ডক্টর বসছেন কালকে") as
    well as a bare answer ("অর্থো", "হাড়ের", "অর্থ পেডিটস"). Matched only
    against clinic-api's own aliases (/api/v1/departments), so this module
    keeps no vocabulary of its own. Three steps, strictest first; the first
    that finds exactly one department decides:

      1. an alias said in the sentence. A Bengali alias may carry a case
         ending ("কার্ডিওলজিতে", "অর্থোর"), so only its left edge is bounded;
         a Latin one is a whole word ("ent" must not fire in "appointment").
         The longest alias wins, so "জেনারেল মেডিসিন" beats "মেডিসিন".
      2. the answer is the START of an alias ("হাড়ের" -> "হাড়ের ডাক্তার",
         "অর্থোপেডিক" -> "অর্থোপেডিক্স") -- a caller cutting a name short.
      3. fuzzy, for what the ASR mangles ("অর্থ পেডিটস", "কার্ডিয়োলজি"):
         each word, each pair of words, and the pair with its split closed
         up, against each alias, ignoring filler. Only above
         _DEPT_FUZZY_FLOOR and clear of the runner-up by _DEPT_FUZZY_MARGIN.

    Two different departments tied at any step -> None: asked again, never
    guessed."""
    key = _nfc(text).lower().strip()
    if not key or not departments:
        return None

    # 1. an alias inside the sentence
    best_len, best = 0, []
    for dept in departments:
        for form in _dept_forms(dept):
            if re.search(r"[a-z]", form):
                hit = re.search(rf"\b{re.escape(form)}s?\b", key)
            else:
                hit = re.search(rf"(?<![{_BN}]){re.escape(form)}", key)
            if not hit:
                continue
            if len(form) > best_len:
                best_len, best = len(form), [dept]
            elif len(form) == best_len and all(d is not dept for d in best):
                best.append(dept)
    if best:
        return best[0] if len(best) == 1 else None

    words = [w for w in re.split(r"[\s,।?!.]+", key) if w and w not in _DEPT_FILLER]
    if not words:
        return None

    # 2. the answer cut short: a word of 3+ letters that an alias starts with
    starts = []
    for dept in departments:
        if any(len(w) >= 3 and any(f.startswith(w) for f in _dept_forms(dept)) for w in words):
            starts.append(dept)
    if len(starts) == 1:
        return starts[0]

    # 3. fuzzy against what the ASR heard
    probes = set(words)
    for a, b in zip(words, words[1:]):
        probes.update((f"{a} {b}", a + b))
    scores = []
    for dept in departments:
        s = max((difflib.SequenceMatcher(None, p, f).ratio()
                 for p in probes if len(p) >= 3 for f in _dept_forms(dept)), default=0.0)
        scores.append((s, dept))
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[0][0]
    second = scores[1][0] if len(scores) > 1 else 0.0
    if top >= _DEPT_FUZZY_FLOOR and top - second >= _DEPT_FUZZY_MARGIN:
        return scores[0][1]
    return None


def apply_list_doctors_override(text: str, data: dict) -> dict:
    """-> `data` with every correctable part rewritten to list_doctors when
    the utterance is a doctor-list request -- or to doctors_by_department
    where that part names a department; `data` itself (same object) when
    nothing needs correcting. Never mutates its input: the semantic cache may
    hold it."""
    if not isinstance(data, dict):
        return data
    if not (is_doctor_list_request(text) or is_department_list_request(text)):
        return apply_department_correction(data)
    top = {"intent": data.get("intent"), "slots": data.get("slots")}
    parts = data.get("parts")
    new_top = _as_list_part(top) if _should_override(top) else None
    new_parts = None
    if isinstance(parts, list):
        rewritten = [_as_list_part(p) if isinstance(p, dict) and _should_override(p) else p
                     for p in parts]
        if any(a is not b for a, b in zip(rewritten, parts)):
            new_parts = rewritten
    if new_top is None and new_parts is None:
        return apply_department_correction(data)
    out = dict(data)
    if new_top is not None:
        out["intent"], out["slots"] = new_top["intent"], new_top["slots"]
        # direct_reply_bn only exists for smalltalk; a factual answer must
        # never be spoken from the model's own text.
        out["direct_reply_bn"] = None
    if new_parts is not None:
        out["parts"] = new_parts
    elif new_top is not None and isinstance(parts, list) and parts:
        # parts[0] mirrors the top level (turn_parts.normalise's guarantee).
        out["parts"] = [_as_list_part(parts[0])] + list(parts[1:]) \
            if _should_override(parts[0]) else parts
    # A part the list cue did not touch may still need the department fix.
    return apply_department_correction(out)
