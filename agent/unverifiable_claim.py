"""A caller asserting something about their own record that this system cannot check.

Ported from `merged_code/agent/unverifiable_claim.py` (story: "Caller states
something the agent cannot verify"). Its acceptance criteria were:

  * a caller assertion never becomes a system fact;
  * **the agent checks the system of record where verification is possible**;
  * where it cannot verify, it says so clearly and offers a human;
  * a caller claim is never echoed back as though it were verified.

WHY ONLY TWO OF THE ORIGINAL'S FOUR CATEGORIES ARE PORTED:

The original intercepted four assertion categories -- an existing appointment, a
report result, a doctor's approval, and a past payment -- and answered all four
with "I cannot confirm that". Its docstring justifies the appointment case
explicitly: "The investigation found no GET endpoint anywhere in clinic-api for
reading back an existing appointment by phone -- only POST /api/v1/appointments
(create) exists."

**That is not true of this repository.** It has:

  * `GET /api/v1/bookings/lookup` (by phone, confirmation id or name) and a
    `lookup_booking` intent that already reaches it -- so "my appointment is
    tomorrow" CAN be checked;
  * `GET /api/v1/billing/outstanding` -- so "I already paid" CAN be checked.

Porting those two categories faithfully would make the agent answer "I cannot
confirm that" to a caller whose appointment or balance it is perfectly able to
look up. That does not just lose a feature -- it **violates this story's own
second acceptance criterion**, which requires checking the system of record
wherever verification is possible. So those two categories are deliberately NOT
intercepted here: they fall through to the ordinary classifier chain and reach
the real, verified, tool-backed lookup, which is the correct answer to them in
this codebase.

What remains genuinely unverifiable here, and is what this module intercepts:

  * `DOCTOR_APPROVAL` -- "the doctor already approved this". There is no notion
    of a doctor approving anything anywhere in this schema, and no endpoint that
    could confirm it. Nothing can ever check this claim.
  * `REPORT_RESULT` -- "my report is normal". Report STATUS is checkable
    (`/api/v1/reports/...`), but what a result MEANS is not something this
    system may state at all -- which is agent/clinical_safety.py's rule, not a
    gap. So a claim about a result is answered with that module's own reply: a
    doctor will explain it, and here is one.

WHY IT IS A GUARD AND WHERE IT SITS:

It runs before the classifier for the same reason every guard here does, and
runs AFTER all of them -- emergency, abuse, clinical interpretation, complaint,
doctor-personal-request -- because every one of those is a more specific reading
of an utterance that may also contain an assertion. It only ever sees text they
have all looked at and let through.

`_REAL_REQUEST_MARKERS` is the narrowing rule that keeps this from becoming a
second verification system: an utterance that ALSO asks the system to check,
confirm or book something falls through uninterrupted even when it contains a
claim phrase, so it reaches the real lookup rather than a generic decline.

REASONED, not measured: the Devanagari phrases are new in this repository.
"""

from __future__ import annotations

import unicodedata

_BN_CHAR_LO, _BN_CHAR_HI = "ঀ", "৿"

NONE = "none"

CATEGORY_DOCTOR_APPROVAL = "DOCTOR_APPROVAL_CLAIM"
CATEGORY_REPORT_RESULT = "REPORT_RESULT_CLAIM"

# Deliberately not intercepted -- this repository can verify both. Named here so
# the decision is discoverable from the code, not only from the docstring.
VERIFIABLE_HERE = {
    "APPOINTMENT_EXISTING_CLAIM": "GET /api/v1/bookings/lookup (intent: lookup_booking)",
    "PAYMENT_ALREADY_CLAIM": "GET /api/v1/billing/outstanding",
}

CLAIM_PATTERNS: dict[str, tuple[str, ...]] = {
    CATEGORY_DOCTOR_APPROVAL: (
        # English
        "the doctor already approved", "doctor already approved it",
        "doctor has already approved", "my doctor approved this",
        "doctor already agreed", "doctor already said yes",
        "the doctor has agreed to this", "my doctor has approved it",
        # Hindi / Bengali in Latin script
        "doctor ne already approve kar diya",
        "doctor already approve kar chuke hain",
        "doctor ne pehle hi haan bol diya",
        "doctor already approve kore diyechen",
        "doctor age thekei approve korechen",
        "doctor age thekei raji hoyechen",
        # Bengali
        "ডাক্তার আগে থেকেই অনুমোদন দিয়েছেন",
        "ডাক্তার ইতিমধ্যে অনুমতি দিয়েছেন",
        "ডাক্তার আগেই রাজি হয়েছেন",
        # Hindi (new here)
        "डॉक्टर ने पहले ही मंज़ूरी दे दी है",
        "डॉक्टर ने पहले ही मंजूरी दे दी है",
        "डॉक्टर ने पहले ही हाँ कह दिया है",
        "डॉक्टर पहले ही राज़ी हो गए हैं",
        "डॉक्टर ने अनुमति दे दी है",
    ),
    CATEGORY_REPORT_RESULT: (
        # English
        "my report is normal", "my report was normal",
        "my report came back normal", "my report is fine",
        "my test result is normal", "my results are normal",
        "my report was fine", "my result is normal",
        # Hindi / Bengali in Latin script
        "mera report normal hai", "mera report normal aaya hai",
        "mera result normal hai", "mera test normal aaya",
        "amar report normal ache", "amar report normal eshechhe",
        "amar result normal", "amar test normal eshechhe",
        # Bengali
        "আমার রিপোর্ট নরমাল", "আমার রিপোর্ট নরমাল এসেছে",
        "আমার রেজাল্ট নরমাল", "আমার রিপোর্ট ঠিক আছে",
        # Hindi (new here)
        "मेरी रिपोर्ट नॉर्मल है", "मेरी रिपोर्ट नॉर्मल आई है",
        "मेरा रिजल्ट नॉर्मल है", "मेरी रिपोर्ट ठीक है",
        "मेरा टेस्ट नॉर्मल आया है",
    ),
}

# An utterance that ALSO asks for a real check or a booking must reach the real,
# verified lookup -- never a generic decline. This is what keeps the guard narrow.
_REAL_REQUEST_MARKERS = (
    "can you check", "can you confirm", "please check", "please confirm",
    "check my", "confirm my", "verify my", "could you check", "is it still",
    "book", "schedule", "i want an appointment", "i need an appointment",
    "can i get an appointment", "want to book",
    "check kar sakte", "confirm kar sakte", "check karo", "confirm karo",
    "book karna hai", "appointment chahiye", "appointment book karna",
    "zara check", "dekh lijiye", "pata kar",
    "check korte paren", "confirm korte paren", "check korun",
    "confirm korun", "book korte chai", "appointment chai",
    "appointment book korte chai", "dekhe nin",
)
_REAL_REQUEST_MARKERS_BN = (
    "দেখে নিন", "একটু দেখুন", "চেক করুন", "নিশ্চিত করুন", "বুক করতে চাই",
    "অ্যাপয়েন্টমেন্ট চাই",
)
_REAL_REQUEST_MARKERS_HI = (
    "ज़रा देखिए", "जरा देखिए", "चेक कीजिए", "पुष्टि कीजिए", "बुक करना है",
    "अपॉइंटमेंट चाहिए", "पता कीजिए",
)


def _has_bengali(s: str) -> bool:
    return any(_BN_CHAR_LO <= ch <= _BN_CHAR_HI for ch in s)


def asks_for_a_real_check(text: str) -> bool:
    """True when the caller is also asking the system to check or book something.

    Such a turn is left alone so it reaches the real lookup, which is this
    story's own second acceptance criterion.
    """
    t = unicodedata.normalize("NFC", text or "")
    lowered = t.lower()
    return (
        any(m in lowered for m in _REAL_REQUEST_MARKERS)
        or any(m in t for m in _REAL_REQUEST_MARKERS_BN)
        or any(m in t for m in _REAL_REQUEST_MARKERS_HI)
    )


def detect_unverifiable_claim(text: str | None) -> tuple[str, str | None]:
    """-> (category, matched phrase family) or (NONE, None).

    Only the two categories nothing in this repository can check are reported;
    see the module docstring for the two that are deliberately left to their
    real lookups.
    """
    if not text or not text.strip():
        return NONE, None
    t = unicodedata.normalize("NFC", text)
    lowered = t.lower()

    if asks_for_a_real_check(t):
        return NONE, None

    for category, phrases in CLAIM_PATTERNS.items():
        for phrase in phrases:
            hit = phrase in t if _has_bengali(phrase) or not phrase.isascii() else phrase in lowered
            if hit:
                return category, phrase
    return NONE, None


def is_unverifiable_claim(text: str | None) -> bool:
    category, _ = detect_unverifiable_claim(text)
    return category != NONE
