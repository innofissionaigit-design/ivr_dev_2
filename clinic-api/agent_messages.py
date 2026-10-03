"""Wording the operator can change from the database (KCD-353, KCD-500, KCD-513).

The agent reads these at the start of a call. To change what callers hear -- the disclosure, the
"I cannot find that" message, the thank-you -- update the row (SQL, or PUT /api/v1/agent/messages);
no deploy. The agent falls back to its built-in text if this API is unreachable, so an outage
never leaves it without words, and `version` rises on every change so a call record can say which
wording was spoken.

What the database can override is deliberately open: any key in agent/phrases.py can be given a
row, plus the keys below that live outside that table. What it cannot do is add a FACT -- prices,
times and confirmation numbers are template substitutions from live API answers, not messages.

The operator owns the wording. `tools/check_messages.py` runs the persona checks (register, one
apology, sentence length, no hedging or reassurance) over what is in the database so a change can
be checked before callers hear it.
"""

from __future__ import annotations

from booking_service import _now
from models import AgentMessage
from registry import audit
from sqlalchemy.orm import Session

LANGS = ("bn", "hi", "en")
MAX_TEXT_CHARS = 600

# Built-in defaults for the keys that do not live in agent/phrases.py. tests/test_e33_registry.py
# asserts these equal the agent's built-in text, so the two cannot drift apart.
DEFAULTS: dict[str, dict[str, str]] = {
    "disclosure": {
        "bn": "আপনি সোনোস্ক্যান বাণীর সঙ্গে কথা বলছেন। আমি একটি স্বয়ংক্রিয় সহকারী, মানুষ নই। চাইলে যেকোনো সময় স্টাফের সঙ্গে কথা বলতে পারবেন।",
        "hi": "आप सोनोस्कैन वाणी से बात कर रहे हैं। मैं एक स्वचालित सहायक हूँ, इंसान नहीं। आप चाहें तो कभी भी स्टाफ़ से बात कर सकते हैं।",
        "en": "You are speaking with Sonoscan Vaani. I am an automated assistant, not a person. You can ask for our staff at any time.",
    },
    "cannot_find": {
        "bn": "দুঃখিত। আমি তথ্যটা খুঁজে পাচ্ছি না। আপনি কি আমাকে একটু বলবেন, যাতে আমি সাহায্য করতে পারি?",
        "hi": "माफ़ कीजिए। मुझे जानकारी नहीं मिल रही। क्या आप मुझे थोड़ा बताएँगे, ताकि मैं मदद कर सकूँ?",
        "en": "Sorry. I am unable to find the details. Could you please guide me, so that I can help you?",
    },
    "thanks_ack": {
        "bn": "ধন্যবাদ।",
        "hi": "धन्यवाद।",
        "en": "Thank you.",
    },
}

# Earlier built-in wording, per (key, lang). seed_defaults() moves a row still holding one of these AND still owned by
# the seed to the current default; a row an operator has edited is never touched.
SUPERSEDED: dict[tuple[str, str], tuple[str, ...]] = {
    ("thanks_ack", "bn"): ("বলার জন্য ধন্যবাদ।",),
    ("thanks_ack", "hi"): ("बताने के लिए धन्यवाद।",),
    ("thanks_ack", "en"): ("Thank you for telling me.",),
}


def seed_defaults(db: Session) -> int:
    """Insert the default rows that are missing, and move a seeded row that still holds a superseded default to the
    current one. Never overwrites a row an operator has changed."""
    n = 0
    for key, by_lang in DEFAULTS.items():
        for lang, text in by_lang.items():
            row = db.query(AgentMessage).filter_by(key=key, lang=lang).first()
            if row is None:
                db.add(
                    AgentMessage(
                        key=key, lang=lang, text=text, version=1, active=True, updated_at=_now(), updated_by="seed"
                    )
                )
                n += 1
            elif row.updated_by == "seed" and row.text in SUPERSEDED.get((key, lang), ()):
                row.text, row.version, row.updated_at = text, row.version + 1, _now()
                n += 1
    db.commit()
    return n


def get_messages(db: Session, lang: str | None = None) -> dict:
    """Every ACTIVE message: {"messages": {key: {lang: {"text": ..., "version": n}}}, "version": highest}."""
    q = db.query(AgentMessage).filter(AgentMessage.active.is_(True))
    if lang:
        q = q.filter(AgentMessage.lang == lang)
    out: dict[str, dict] = {}
    top = 0
    for m in q.all():
        out.setdefault(m.key, {})[m.lang] = {"text": m.text, "version": m.version}
        top = max(top, m.version)
    return {"messages": out, "version": top}


def set_message(db: Session, key: str, lang: str, text: str, updated_by: str = "operator", active: bool = True) -> dict:
    key = (key or "").strip()
    if not key or lang not in LANGS:
        return {"success": False, "reason": "invalid_key_or_language"}
    text = (text or "").strip()
    if not text or len(text) > MAX_TEXT_CHARS:
        return {"success": False, "reason": "invalid_text"}
    row = db.query(AgentMessage).filter_by(key=key, lang=lang).first()
    if row is None:
        row = AgentMessage(
            key=key, lang=lang, text=text, version=1, active=active, updated_at=_now(), updated_by=updated_by
        )
        db.add(row)
    elif row.text != text or row.active != active:
        row.text, row.active, row.version = text, active, row.version + 1
        row.updated_at, row.updated_by = _now(), updated_by
    db.commit()
    audit(db, "message_changed", "ok", actor="operator", key=key, lang=lang, version=row.version, by=updated_by)
    return {"success": True, "key": key, "lang": lang, "version": row.version}
