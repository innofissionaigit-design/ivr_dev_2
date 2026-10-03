"""KCD-498 / KCD-500: every historical statement the agent makes is a TEMPLATE filled from
a retrieved field -- never generated -- and a field that is missing, stale or ambiguous
produces an explicit statement that the agent cannot see it.

Each renderer returns (statement_id, text). The id is `kind:record_id` (or a fixed
label for the "cannot see" statements) and is written into the call record
(KCD-501), so an audit of a call can match every sentence about a patient's history
to the retrieved field it came from -- and find any that has no such origin
(tools/history_audit.py, and the fabricated-history cases in
tests/test_history_hallucination.py).

What these never contain: a result value, an interpretation, a recommendation for or
against a test. "Due" is stated only where a clinician-approved interval exists
(clinic-api RetestInterval); otherwise the date is stated and nothing about need.
Dates are ISO strings the speech normaliser turns into words (agent/bn_normalize).

Wording is provisional and pending native review (agent/persona.py). No colons or
brackets, one apology at most, no hedging -- the persona checks scan this file too.
"""

from __future__ import annotations

import re

# ---- "I cannot see it" statements: the only honest answer to a missing/stale/ambiguous field
# KCD-500, as the operator worded it: nothing found in the cache or the tables is said plainly, with one
# simple apology, and the caller is asked to guide the agent. The operator can change it from the
# database (agent/messages.py, key "cannot_find"); clinic-api/agent_messages.py holds the same default.
CANNOT_FIND = {
    "bn": "দুঃখিত। আমি তথ্যটা খুঁজে পাচ্ছি না। আপনি কি আমাকে একটু বলবেন, যাতে আমি সাহায্য করতে পারি?",
    "hi": "माफ़ कीजिए। मुझे जानकारी नहीं मिल रही। क्या आप मुझे थोड़ा बताएँगे, ताकि मैं मदद कर सकूँ?",
    "en": "Sorry. I am unable to find the details. Could you please guide me, so that I can help you?",
}
CANNOT_SEE = CANNOT_FIND
CANNOT_CONFIRM = {
    "bn": "আমার কাছে থাকা তথ্যটা এখনকার নয়, তাই আমি এটা নিশ্চিত করতে পারছি না।",
    "hi": "मेरे पास मौजूद जानकारी अभी की नहीं है, इसलिए मैं इसकी पुष्टि नहीं कर सकती।",
    "en": "The information I have is not recent enough for me to confirm it.",
}
AMBIGUOUS = {
    "bn": "এর জন্য আমি একাধিক এন্ট্রি দেখছি, তাই কোনটা বলছেন বুঝতে পারছি না। তারিখটা বলবেন?",
    "hi": "इसके लिए मुझे एक से ज़्यादा प्रविष्टियाँ दिख रही हैं। मैं समझ नहीं पा रही कि आप कौन सी पूछ रहे हैं। क्या आप तारीख़ बताएँगे?",
    "en": "I see more than one entry for that, so I cannot tell which you mean. Could you tell me the date?",
}
NEEDS_VERIFICATION = {
    "bn": "ব্যক্তিগত তথ্য বলার আগে আমাকে পরিচয় যাচাই করতে হবে। এই কলে সেটা আমি এখন পারছি না। স্টাফের সাথে যুক্ত করে দেব?",
    "hi": "निजी जानकारी बताने से पहले मुझे पहचान की पुष्टि करनी होगी। इस कॉल पर अभी मैं यह नहीं कर सकती। क्या स्टाफ़ से जोड़ दूँ?",
    "en": "I need to verify who I am speaking with before I share anything personal. I cannot do that on this call yet. Shall I connect you with our staff?",
}
NO_RECORD = CANNOT_FIND


ASK_PHONE = {
    "bn": "আপনার রেকর্ড খুঁজে দেখতে আমাদের কাছে নথিভুক্ত ফোন নম্বরটা বলবেন?",
    "hi": "आपका रिकॉर्ड खोजने के लिए हमारे पास दर्ज फ़ोन नंबर बताइए।",
    "en": "To look for your record, please tell me the phone number registered with us.",
}
ASK_WHICH_TEST = {
    "bn": "কোন টেস্টের কথা জানতে চাইছেন?",
    "hi": "आप किस टेस्ट के बारे में पूछ रहे हैं?",
    "en": "Which test do you mean?",
}
OK_ANYTHING_ELSE = {
    "bn": "ঠিক আছে। আর কী সাহায্য করতে পারি?",
    "hi": "ठीक है। और मैं क्या मदद कर सकती हूँ?",
    "en": "All right. What else can I help you with?",
}


def _pick(table: dict, lang: str) -> str:
    return table.get(lang) or table["bn"]


def ask_phone(lang: str) -> tuple[str, str]:
    return "ask_phone", _pick(ASK_PHONE, lang)


def ask_which_test(lang: str) -> tuple[str, str]:
    return "ask_which_test", _pick(ASK_WHICH_TEST, lang)


def ok_anything_else(lang: str) -> tuple[str, str]:
    return "ok_anything_else", _pick(OK_ANYTHING_ELSE, lang)


def _msg(key: str, table: dict, lang: str) -> str:
    from agent import messages

    return messages.text(key, lang if lang in table else "bn", _pick(table, lang))


def cannot_see(lang: str) -> tuple[str, str]:
    return "cannot_see", _msg("cannot_find", CANNOT_FIND, lang)


def cannot_confirm(lang: str) -> tuple[str, str]:
    return "cannot_confirm", _pick(CANNOT_CONFIRM, lang)


def ambiguous(lang: str) -> tuple[str, str]:
    return "ambiguous", _pick(AMBIGUOUS, lang)


def needs_verification(lang: str) -> tuple[str, str]:
    return "needs_verification", _pick(NEEDS_VERIFICATION, lang)


def no_record(lang: str) -> tuple[str, str]:
    return "no_record", _msg("cannot_find", CANNOT_FIND, lang)


# ---- statements rendered from retrieved fields ---------------------------------------


def _name(f: dict, field: str, lang: str) -> str:
    """A name in the language's own script when the record carries one (a Latin name inside Bengali
    text is dropped by the Bengali voice), else the canonical name."""
    return (f.get(f"{field}_{lang}") if lang in ("bn", "hi") else None) or f.get(field) or ""


def appointment_statement(event: dict, lang: str, today_iso: str) -> tuple[str, str]:
    f = event["fields"]
    upcoming = f["date"] >= today_iso and f.get("status") == "confirmed"
    sid = f"appointment:{event['id'].split(':')[1]}"
    doc, date, time = _name(f, "doctor_name", lang), f["date"], f.get("time_slot") or ""
    if lang == "hi":
        return sid, (
            f"आपकी {doc} डॉक्टर के साथ {date} को {time} बजे अपॉइंटमेंट है।"
            if upcoming
            else f"आपकी {doc} डॉक्टर के साथ {date} को अपॉइंटमेंट थी।"
        )
    if lang == "en":
        return sid, (
            f"You have an appointment with {doc} on {date} at {time}."
            if upcoming
            else f"You had an appointment with {doc} on {date}."
        )
    return sid, (
        f"আপনার {doc} ডাক্তারের সঙ্গে {date} তারিখে {time}-এ একটা অ্যাপয়েন্টমেন্ট আছে।"
        if upcoming
        else f"আপনার {doc} ডাক্তারের সঙ্গে {date} তারিখে একটা অ্যাপয়েন্টমেন্ট ছিল।"
    )


def test_performed_statement(event: dict, lang: str) -> tuple[str, str]:
    f = event["fields"]
    sid = f"test_performed:{event['id'].split(':')[1]}"
    test, date = _name(f, "test_name", lang), f["performed_on"]
    if lang == "hi":
        return sid, f"आपका {test} आख़िरी बार {date} को किया गया था।"
    if lang == "en":
        return sid, f"Your {test} was last done on {date}."
    return sid, f"আপনার {test} শেষবার {date} তারিখে করা হয়েছিল।"


def medicine_statement(event: dict, lang: str) -> tuple[str, str]:
    """THAT a medicine was prescribed and when -- never a dose, a reason or what to do."""
    f = event["fields"]
    sid = f"medicine_prescribed:{event['id'].split(':')[1]}"
    med, date = _name(f, "medicine_name", lang), f["prescribed_on"]
    if lang == "hi":
        return sid, f"{date} को आपको {med} लिखी गई थी।"
    if lang == "en":
        return sid, f"{med} was prescribed to you on {date}."
    return sid, f"{date} তারিখে আপনাকে {med} লেখা হয়েছিল।"


def due_statement(status: dict, lang: str) -> tuple[str, str] | None:
    """Whether the test is due -- ONLY where a clinician approved an interval. None otherwise:
    with no approved interval the agent says nothing about need, not even "not due"."""
    if status.get("due") is None:
        return None
    if status["due"]:
        table = {
            "bn": "ক্লিনিকের নির্ধারিত সময় পেরিয়ে গেছে, তাই এটা এখন করানোর সময় হয়েছে।",
            "hi": "क्लिनिक का तय किया हुआ समय बीत चुका है, इसलिए यह अब दोबारा कराने का समय है।",
            "en": "The interval our clinicians have approved for it has passed, so it is due.",
        }
        return "due:yes", _pick(table, lang)
    table = {
        "bn": "ক্লিনিকের নির্ধারিত সময় এখনও পেরোয়নি, তাই এটা এখন করানোর সময় হয়নি।",
        "hi": "क्लिनिक का तय किया हुआ समय अभी नहीं बीता है, इसलिए यह अभी कराने का समय नहीं है।",
        "en": "The interval our clinicians have approved for it has not passed yet, so it is not due.",
    }
    return "due:no", _pick(table, lang)


def report_ready_statement(event: dict, lang: str) -> tuple[str, str]:
    f = event["fields"]
    sid = f"report_ready:{event['id'].split(':')[1]}"
    conf = f["confirmation_id"]
    if lang == "hi":
        return sid, f"बुकिंग {conf} की रिपोर्ट तैयार है।"
    if lang == "en":
        return sid, f"The report for booking {conf} is ready."
    return sid, f"{conf} বুকিংয়ের রিপোর্ট তৈরি আছে।"


# ---- detecting a model-composed history claim (the guard for the one text the model writes) ----
_HISTORY_CLAIM = {
    "en": re.compile(
        r"\b(you (had|have had|visited|were (here|tested|seen)|last (came|visited))|your (last|previous|earlier|most recent)( [\w-]+){0,3} "
        r"(test|visit|appointment|report|booking|result|check)|last time you|previous(ly)? (visit|appointment|test)|"
        r"you'?ve (been|had|done))\b",
        re.I,
    ),
    "bn": re.compile(r"(আপনার (শেষ|আগের|পূর্বের)|আপনি (আগে|গতবার|শেষবার)|গতবার আপনি|আগেরবার আপনি|শেষবার আপনি)"),
    "hi": re.compile(r"(आपका (पिछला|आख़िरी|आखिरी|पहले का)|आप (पिछली बार|पहले|आख़िरी बार|आखिरी बार)|पिछली बार आप)"),
}


def mentions_personal_history(text: str, lang: str | None = None) -> bool:
    """True if `text` makes a claim about what THIS caller did or had before. The model
    writes only small talk; small talk may not contain one (KCD-500) -- history is
    rendered from retrieved fields by the functions above, never composed."""
    if lang:
        return bool(_HISTORY_CLAIM.get(lang, _HISTORY_CLAIM["en"]).search(text or ""))
    return any(p.search(text or "") for p in _HISTORY_CLAIM.values())
