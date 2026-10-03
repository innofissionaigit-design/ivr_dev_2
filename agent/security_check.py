"""The security questions, as a small conversation (KCD-495, KCD-497).

History and medicines are spoken only to a caller who has PASSED a check, and the check is: the
caller answers questions about the patient -- date of birth, patient id, full name, address -- and the
SERVER compares them with its registry (clinic-api/registry.py). Two matching facts, one of them
strong (patient id or date of birth). This module is the agent's half: which question to ask next,
turning the caller's words into the values the server compares (agent/security_input.py), and
knowing when to stop. It cannot decide anyone is verified, and it never learns which answer was
wrong: the server says only yes or no and how many tries are left.

ORDER. First the strongest thing a caller can usually give: "your date of birth, or your patient ID"
(whichever they say is taken). Then the full name. Then, if the pair did not verify, the address,
and after that whichever of date of birth or id has not been given, or a repeat. At most three
evaluations (the server's limit); after that, or when the caller cannot answer, the call goes to a
person -- the check is never softened.

The wording is short and warm: one question at a time, one simple sentence each, and it says why it
asks. Provisional and pending native review (agent/persona.py).
"""

from __future__ import annotations

import dataclasses

from agent import security_input as si

INTRO = {
    "bn": "আপনার নিরাপত্তার জন্য আমি কয়েকটা প্রশ্ন করব।",
    "hi": "आपकी सुरक्षा के लिए मैं कुछ सवाल पूछूँगी।",
    "en": "For your safety, I will ask a few questions.",
}
QUESTION = {
    "strong": {
        "bn": "আপনার জন্মতারিখ বলবেন? অথবা পেশেন্ট আইডি।",
        "hi": "क्या आप अपनी जन्मतिथि बताएँगे? या पेशेंट आईडी।",
        "en": "Please tell me your date of birth. Or your patient ID.",
    },
    "dob": {
        "bn": "আপনার জন্মতারিখ বলবেন? দিন, মাস আর সাল।",
        "hi": "क्या आप अपनी जन्मतिथि बताएँगे? दिन, महीना और साल।",
        "en": "Please tell me your date of birth. The day, month and year.",
    },
    "patient_id": {
        "bn": "আপনার পেশেন্ট আইডিটা বলবেন?",
        "hi": "क्या आप अपनी पेशेंट आईडी बताएँगे?",
        "en": "Please tell me your patient ID.",
    },
    "name": {
        "bn": "আপনার পুরো নামটা বলবেন?",
        "hi": "क्या आप अपना पूरा नाम बताएँगे?",
        "en": "Please tell me your full name.",
    },
    "address": {
        "bn": "আপনার ঠিকানাটা বলবেন? এলাকা আর পিন কোড।",
        "hi": "क्या आप अपना पता बताएँगे? इलाक़ा और पिन कोड।",
        "en": "Please tell me your address. The area and the pin code.",
    },
}
DID_NOT_UNDERSTAND = {
    "bn": "দুঃখিত। আমি বুঝতে পারিনি। আবার বলবেন?",
    "hi": "माफ़ कीजिए। मैं समझ नहीं पाई। क्या आप फिर बोलेंगे?",
    "en": "Sorry. I did not understand. Could you say it again?",
}
NOT_MATCHED = {
    "bn": "দুঃখিত। তথ্যগুলো মিলছে না। আরেকটা প্রশ্ন করছি।",
    "hi": "माफ़ कीजिए। जानकारी मेल नहीं खा रही। एक और सवाल पूछती हूँ।",
    "en": "Sorry. The details do not match. Let me ask one more.",
}
VERIFIED = {
    "bn": "ধন্যবাদ। আপনার পরিচয় যাচাই হয়েছে।",
    "hi": "धन्यवाद। आपकी पहचान की पुष्टि हो गई।",
    "en": "Thank you. Your identity is confirmed.",
}
FAILED = {
    "bn": "দুঃখিত। আমি যাচাই করতে পারছি না। আমি আপনাকে স্টাফের সঙ্গে যুক্ত করছি।",
    "hi": "माफ़ कीजिए। मैं पुष्टि नहीं कर पा रही। मैं आपको स्टाफ़ से जोड़ रही हूँ।",
    "en": "Sorry. I cannot confirm your identity. I am connecting you with our staff.",
}
FIND_BY_DETAILS = {
    "bn": "আমি এই নম্বরে রেকর্ড পাচ্ছি না। আপনার জন্মতারিখ আর পুরো নামটা বলবেন?",
    "hi": "मुझे इस नंबर पर रिकॉर्ड नहीं मिल रहा। क्या आप अपनी जन्मतिथि और पूरा नाम बताएँगे?",
    "en": "I cannot find a record on this number. Please tell me your date of birth and your full name.",
}


def _t(table: dict, lang: str) -> str:
    return table.get(lang) or table["bn"]


def intro(lang: str) -> str:
    return _t(INTRO, lang)


def did_not_understand(lang: str) -> str:
    return _t(DID_NOT_UNDERSTAND, lang)


def not_matched(lang: str) -> str:
    return _t(NOT_MATCHED, lang)


def verified_text(lang: str) -> str:
    return _t(VERIFIED, lang)


def failed_text(lang: str) -> str:
    return _t(FAILED, lang)


def find_by_details_text(lang: str) -> str:
    return _t(FIND_BY_DETAILS, lang)


MAX_PARSE_FAILURES = 2
MAX_EVALUATIONS = 3  # the server's own limit (registry.MAX_ATTEMPTS_PER_CALL)


@dataclasses.dataclass
class SecurityCheck:
    """One per attempt to verify one patient on one call. `patient_ref` is None while the record is
    still being FOUND (the phone number found nobody): the caller's details then locate it
    (`find_patient`), and the same answers are submitted for verification."""

    patient_ref: str | None
    answers: dict = dataclasses.field(default_factory=dict)  # factor -> the value to submit
    expecting: list[str] = dataclasses.field(default_factory=list)  # the factors the last question asked for
    parse_failures: int = 0
    evaluations: int = 0
    introduced: bool = False
    finds: int = 0  # find attempts that located nobody
    finished: bool = False
    verified: bool = False
    age_years: int | None = None
    is_senior: bool = False

    # ------------------------------------------------------------------ questions
    def _missing(self) -> list[str]:
        return [f for f in ("dob", "patient_id", "name", "address") if f not in self.answers]

    def next_question(self, lang: str) -> str:
        """The next thing to ask, and remember what it asks for."""
        have = self.answers
        if not ({"dob", "patient_id"} & set(have)):
            self.expecting = ["dob", "patient_id"]
            key = "strong"
        elif "name" not in have:
            self.expecting, key = ["name"], "name"
        elif "address" not in have:
            self.expecting, key = ["address"], "address"
        elif "dob" not in have:
            self.expecting, key = ["dob"], "dob"
        elif "patient_id" not in have:
            self.expecting, key = ["patient_id"], "patient_id"
        else:  # everything was given and did not match: one repeat
            self.expecting, key = ["dob"], "dob"
            self.answers.pop("dob", None)
        text = _t(QUESTION[key], lang)
        if not self.introduced:
            self.introduced = True
            return f"{intro(lang)} {text}"
        return text

    # ------------------------------------------------------------------ answers
    def absorb(self, text: str, today=None) -> bool:
        """Read the caller's answer for what was asked. True if at least one expected factor was
        understood; a failed read is counted, and the agent asks again (or gives up)."""
        got = False
        for factor in self.expecting:
            if factor == "dob":
                d = si.parse_dob(text, today)
                if d is not None:
                    self.answers["dob"], got = d.isoformat(), True
            elif factor == "patient_id":
                # a date read out ("12 May 1980") is a date, never also an id
                pid = (
                    None
                    if "dob" in self.answers and si.parse_dob(text, today) is not None
                    else si.parse_patient_id(text)
                )
                if pid is not None:
                    self.answers["patient_id"], got = pid, True
            elif factor == "name":
                name = si.clean_name(text)
                if len(name.split()) >= 2:  # a full name, not one word
                    self.answers["name"], got = name, True
            elif factor == "address":
                addr = si.address_text(text)
                if len(addr.split()) >= 2:
                    self.answers["address"], got = addr, True
        self.parse_failures = 0 if got else self.parse_failures + 1
        return got

    @property
    def gave_up_reading(self) -> bool:
        return self.parse_failures > MAX_PARSE_FAILURES

    def ready_to_submit(self) -> bool:
        """Two facts, one strong: the smallest set the server will evaluate."""
        return len(self.answers) >= 2 and bool({"dob", "patient_id"} & set(self.answers))

    def record_result(self, result: dict) -> None:
        self.evaluations += 1
        if result.get("verified"):
            self.verified, self.finished = True, True
            self.age_years = result.get("age_years")
            self.is_senior = bool(result.get("is_senior"))
        elif result.get("locked") or self.evaluations >= MAX_EVALUATIONS or result.get("attempts_left", 1) <= 0:
            self.finished = True

    @property
    def finding(self) -> bool:
        return self.patient_ref is None

    def find_payload(self) -> dict:
        """What locates a record: a patient id, or a date of birth with a name."""
        a = self.answers
        if "patient_id" in a:
            return {"patient_id": a["patient_id"]}
        if "dob" in a and "name" in a:
            return {"dob": a["dob"], "name": a["name"]}
        return {}

    def record_find_miss(self) -> bool:
        """Nobody matched. Forget the strong answer so it is asked again; True when it is time to stop."""
        self.finds += 1
        self.answers.pop("dob", None)
        self.answers.pop("patient_id", None)
        return self.finds >= 2

    def payload(self) -> dict:
        return {k: v for k, v in self.answers.items() if k in ("dob", "patient_id", "name", "address")}
