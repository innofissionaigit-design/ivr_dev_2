"""Epic E33 on the agent side: what the agent may say about the patient on the line, and
the rules that keep it from saying more (KCD-493..500).

The read model is clinic-api/patient_context.py. This module is the POLICY in front of it:

  * NOTHING beyond the existence of a record is spoken before verification
    (KCD-495) -- `history_gate()` is the one function that decides, and it reads
    agent/identity.py, never a phone string;
  * a shared number is settled by an OPEN question, never by listing who is on it
    (KCD-494) -- `resolve_question()`;
  * continuity is OFFERED, never assumed, and its details wait for verification
    (KCD-496) -- `continuity_offer()`;
  * preferences are offered for confirmation and never applied silently, and a stored
    language bias never overrides what the caller actually says (KCD-499);
  * every statement about history is rendered from a retrieved field by
    agent/history_templates.py, carries its provenance id, and a missing, stale or
    ambiguous field becomes an explicit "I cannot see that" (KCD-500);
  * a test's last date and whether it is DUE may be stated; a result, an
    interpretation, or advice for or against a test may not (KCD-498) -- there is no
    code path here that could produce one, because no field holds one.

Pure Python over already-fetched dicts: no I/O, clock passed in, testable off-pod.
"""

from __future__ import annotations

import dataclasses
import datetime

from agent import history_templates as ht
from agent.identity import IdentityState

# A timeline older than this is not confirmed to the caller. Mirrors clinic-api's
# STALE_AFTER_HOURS (asserted equal in tests/test_patient_context.py). REASONED.
STALE_AFTER_HOURS = 24
MAX_APPOINTMENTS_SPOKEN = 3


# ================================================================ KCD-495: the gate


@dataclasses.dataclass
class GateDecision:
    allowed: bool
    reason: str
    statement: tuple[str, str] | None = None  # what to say if not allowed


def history_gate(
    identity: IdentityState, patient_ref: str | int | None, lang: str, record_exists: bool | None = None
) -> GateDecision:
    """May anything personal about `patient_ref` be spoken? Only after a verification for
    THAT patient. Before it, the one thing that may be said is that a record exists (or
    does not), and only when the caller has already given the number it is on."""
    if patient_ref is not None and identity.may_disclose_for(str(patient_ref)):
        return GateDecision(True, "verified")
    if record_exists is False:
        return GateDecision(False, "no_record", ht.no_record(lang))
    return GateDecision(False, "not_verified", ht.needs_verification(lang))


# ============================================================ KCD-494: shared numbers

RESOLVE_QUESTION = {
    "bn": "এই নম্বরে একাধিক রোগীর রেকর্ড আছে। কার জন্য ফোন করছেন, নামটা বলবেন?",
    "hi": "इस नंबर पर एक से ज़्यादा मरीज़ों के रिकॉर्ड हैं। आप किसके लिए फ़ोन कर रहे हैं, नाम बताइए।",
    "en": "There is more than one patient record on this number. Who is the call for? Please say the name.",
}
RESOLVE_AGAIN = {
    "bn": "এখনও ঠিক বুঝতে পারলাম না। রোগীর পুরো নাম আর বয়সটা বলবেন?",
    "hi": "मैं अभी भी ठीक से नहीं समझ पाई। मरीज़ का पूरा नाम और उम्र बताइए।",
    "en": "I still cannot tell which record you mean. Please give the patient's full name and age.",
}


def resolve_question(lang: str, attempt: int = 0) -> str:
    """The question that settles a shared number. It never lists the names on it: naming
    them would disclose who is on the number to whoever is holding the phone."""
    return (RESOLVE_AGAIN if attempt else RESOLVE_QUESTION).get(lang) or RESOLVE_QUESTION["bn"]


def identify_outcome(result: dict) -> str:
    """'new' | 'single' | 'ambiguous' -> what the agent does next. A failed lookup is a NEW
    caller, never a guess."""
    status = (result or {}).get("status")
    return status if status in ("single", "ambiguous") else "new"


# ============================================================ KCD-496: continuity

CONTINUITY_GENERIC = {
    "bn": "আগের বার আপনার একটা কাজ শেষ হয়নি বলে মনে রাখছি। সেটা কি চালিয়ে যেতে চান, নাকি নতুন করে শুরু করব?",
    "hi": "पिछली बार आपका एक काम अधूरा रह गया था। क्या उसे जारी रखूँ, या नए सिरे से शुरू करूँ?",
    "en": "Something was left unfinished last time. Would you like to continue it, or start fresh?",
}


def continuity_offer(
    context: dict, identity: IdentityState, patient_ref, lang: str, draft_text: str | None = None
) -> tuple[str, str] | None:
    """(id, text) for what the agent believes was unfinished, or None. It STATES the belief
    and lets the caller correct it -- never assumes it. Before verification the offer is
    generic (that something was left, not what); the specifics (`draft_text`, rendered by the
    caller from the verified draft) are spoken only after."""
    if not (context or {}).get("unfinished"):
        return None
    if identity.may_disclose_for(str(patient_ref) if patient_ref is not None else None) and draft_text:
        return "continuity:specific", draft_text
    return "continuity:generic", CONTINUITY_GENERIC.get(lang) or CONTINUITY_GENERIC["bn"]


# ============================================================ KCD-499: preferences


def effective_language(spoken: str | None, bias: str | None, default: str = "bn") -> str:
    """The language for this turn. What the caller ACTUALLY SAID always wins; a stored
    bias is used only when nothing was identified this turn -- it never overrides speech."""
    if spoken in ("bn", "hi", "en"):
        return spoken
    return bias if bias in ("bn", "hi", "en") else default


_PREF_LABEL = {
    "branch": {"bn": "শাখা", "hi": "शाखा", "en": "branch"},
    "delivery_channel": {"bn": "বার্তা পাঠানোর মাধ্যম", "hi": "संदेश भेजने का माध्यम", "en": "message channel"},
    "accessibility_mode": {"bn": "কথা বলার ধরন", "hi": "बात करने का तरीक़ा", "en": "way of speaking"},
    "collection_address": {"bn": "নমুনা নেওয়ার ঠিকানা", "hi": "नमूना लेने का पता", "en": "collection address"},
}


@dataclasses.dataclass
class PreferenceOffer:
    text: str
    fields: dict  # what WOULD be applied if the caller says yes

    def apply(self, slots: dict, confirmed: bool) -> dict:
        """Slots with the preferences merged in -- only when the caller confirmed."""
        return {**slots, **self.fields} if confirmed else dict(slots)


def preference_offer(prefs: dict, identity: IdentityState, patient_ref, lang: str) -> PreferenceOffer | None:
    """An offer to use the patient's usual branch/channel, spoken only after verification, with
    the values read out so they can be corrected. The address is never read aloud: it is offered
    by name ("your usual collection address") and confirmed, not recited to the room."""
    if not identity.may_disclose_for(str(patient_ref) if patient_ref is not None else None):
        return None
    usable = {k: v for k, v in (prefs or {}).items() if k in _PREF_LABEL and v}
    if not usable:
        return None
    parts = []
    for k, v in usable.items():
        label = _PREF_LABEL[k].get(lang) or _PREF_LABEL[k]["en"]
        parts.append(label if k == "collection_address" else f"{label} {v}")
    joined = ", ".join(parts)
    text = {
        "bn": f"আপনার আগের পছন্দ অনুযায়ী {joined} ব্যবহার করব? ঠিক থাকলে হ্যাঁ বলুন।",
        "hi": f"आपकी पिछली पसंद के अनुसार {joined} इस्तेमाल करूँ? ठीक हो तो हाँ कहिए।",
        "en": f"Shall I use your usual {joined}? Say yes if that is right.",
    }.get(lang) or ""
    return PreferenceOffer(text, usable)


# ====================================================== KCD-498/500: answering from fields


@dataclasses.dataclass
class HistoryAnswer:
    statements: list[tuple[str, str]]  # (provenance id, text)

    @property
    def text(self) -> str:
        return " ".join(t for _, t in self.statements)

    @property
    def ids(self) -> list[str]:
        return [i for i, _ in self.statements]


def _is_stale(as_of: str | None, now: datetime.datetime) -> bool:
    if not as_of:
        return True
    try:
        return (now - datetime.datetime.fromisoformat(as_of)) > datetime.timedelta(hours=STALE_AFTER_HOURS)
    except ValueError:
        return True


def _matching_tests(events: list[dict], spoken: str) -> list[dict]:
    spoken_l = (spoken or "").strip().lower()
    return [
        e
        for e in events
        if e["kind"] == "test_performed"
        and spoken_l
        and (
            spoken_l == (e["fields"].get("test_name") or "").lower()
            or spoken_l in (e["fields"].get("test_name") or "").lower()
        )
    ]


def answer_appointments(timeline: dict | None, lang: str, now: datetime.datetime) -> HistoryAnswer:
    if not timeline or not timeline.get("success"):
        return HistoryAnswer([ht.cannot_see(lang)])
    if _is_stale(timeline.get("as_of"), now):
        return HistoryAnswer([ht.cannot_confirm(lang)])
    today = now.date().isoformat()
    upcoming = sorted(
        (
            e
            for e in timeline["events"]
            if e["kind"] == "appointment" and e["fields"].get("status") == "confirmed" and e["fields"]["date"] >= today
        ),
        key=lambda e: (e["fields"]["date"], e["fields"].get("time_slot") or ""),
    )
    if not upcoming:
        return HistoryAnswer([ht.cannot_see(lang)])
    return HistoryAnswer([ht.appointment_statement(e, lang, today) for e in upcoming[:MAX_APPOINTMENTS_SPOKEN]])


def answer_last_test(
    timeline: dict | None, test_name: str, status: dict | None, lang: str, now: datetime.datetime
) -> HistoryAnswer:
    """When was this test last done, and is it due. Never what it showed."""
    if not timeline or not timeline.get("success"):
        return HistoryAnswer([ht.cannot_see(lang)])
    if _is_stale(timeline.get("as_of"), now):
        return HistoryAnswer([ht.cannot_confirm(lang)])
    hits = _matching_tests(timeline["events"], test_name)
    if not hits:
        return HistoryAnswer([ht.cannot_see(lang)])
    if len({(e["fields"].get("test_name") or "").lower() for e in hits}) > 1:
        return HistoryAnswer([ht.ambiguous(lang)])  # "lipid" fits two tests: ask, never pick
    latest = max(hits, key=lambda e: e["fields"]["performed_on"])
    out = [ht.test_performed_statement(latest, lang)]
    due = ht.due_statement(status or {}, lang) if status and status.get("known") else None
    if due:
        out.append(due)
    return HistoryAnswer(out)


def answer_medicines(timeline: dict | None, lang: str, now: datetime.datetime) -> HistoryAnswer:
    """The most recent prescription of up to three different medicines. Names and dates only."""
    if not timeline or not timeline.get("success"):
        return HistoryAnswer([ht.cannot_see(lang)])
    if _is_stale(timeline.get("as_of"), now):
        return HistoryAnswer([ht.cannot_confirm(lang)])
    latest: dict[str, dict] = {}
    for e in timeline["events"]:
        if e["kind"] != "medicine_prescribed":
            continue
        name = (e["fields"].get("medicine_name") or "").lower()
        if name and (name not in latest or e["fields"]["prescribed_on"] > latest[name]["fields"]["prescribed_on"]):
            latest[name] = e
    if not latest:
        return HistoryAnswer([ht.cannot_see(lang)])
    ordered = sorted(latest.values(), key=lambda e: e["fields"]["prescribed_on"], reverse=True)
    return HistoryAnswer([ht.medicine_statement(e, lang) for e in ordered[:MAX_APPOINTMENTS_SPOKEN]])


def answer_recent_tests(timeline: dict | None, lang: str, now: datetime.datetime) -> HistoryAnswer:
    """The last date of up to three DIFFERENT tests, most recent first. Dates only -- never
    what any of them showed."""
    if not timeline or not timeline.get("success"):
        return HistoryAnswer([ht.cannot_see(lang)])
    if _is_stale(timeline.get("as_of"), now):
        return HistoryAnswer([ht.cannot_confirm(lang)])
    latest: dict[str, dict] = {}
    for e in timeline["events"]:
        if e["kind"] != "test_performed":
            continue
        name = (e["fields"].get("test_name") or "").lower()
        if name and (name not in latest or e["fields"]["performed_on"] > latest[name]["fields"]["performed_on"]):
            latest[name] = e
    if not latest:
        return HistoryAnswer([ht.cannot_see(lang)])
    ordered = sorted(latest.values(), key=lambda e: e["fields"]["performed_on"], reverse=True)
    return HistoryAnswer([ht.test_performed_statement(e, lang) for e in ordered[:MAX_APPOINTMENTS_SPOKEN]])
