"""KCD-500: five hundred fabricated-history attempts, and none may slip through.

The story asks for a hallucination test on the history features: try to make the agent state
something about a caller's past that no retrieved record supports, and count what gets out. It
is a deterministic, seeded generator (not a language model, so it is repeatable and runs off
the pod), across the four ways a fabricated history could reach a caller's ear:

  A  MODEL-COMPOSED TEXT   the model's one free-text output (small talk) claims something about
                           this caller's past. It must be caught and replaced.               200
  B  NO VERIFICATION       a history question from every identity state short of "verified for
                           this patient". Nothing personal may be spoken, nothing fetched.   100
  C  MISSING / STALE DATA  a verified caller, but the record is absent, stale, malformed,
                           or matches more than one test. Only "I cannot see it" statements
                           may result -- never a date the data does not hold.                100
  D  PROVENANCE            random timelines; every statement spoken must trace to an event
                           in that timeline (tools/history_audit.py), and every date or number
                           in it must be one the event holds.                                100

The generator's phrasings are ADVERSARIAL by construction (varied wording, three languages,
politeness prefixes, filler) but they are the ones one author could think of. A caught rate of
100% here is a property of THIS set; the same audit (tools/history_audit.py) is what to run on
real call records, which is where an unanticipated phrasing would show up.

    python -m pytest tests/test_history_hallucination.py -v
"""

import datetime
import itertools
import os
import random
import re
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests"), os.path.join(REPO_ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import history_audit

from agent import history_templates as ht
from agent import patient_context as pc
from agent.history_intent import detect_history_question
from agent.identity import IdentityState

pytestmark = pytest.mark.hallucination

NOW = datetime.datetime(2026, 9, 24, 10, 0, 0)
TODAY = NOW.date()
SEED = 500


# ============================================================================= A: model text

_EN_CLAIMS = [
    "You had a {t} last {when}.",
    "Your last {t} was done {when} ago.",
    "Last time you visited us for a {t}.",
    "You were here {when} ago for your {t}.",
    "Your previous appointment was for a {t}.",
    "You have had a {t} before.",
    "You've been tested for {t} already.",
    "Your earlier {t} report was fine.",
    "Your last visit was {when} ago.",
    "You visited us {when} ago.",
    "Last time you came for a {t}.",
    "Your previous booking was a {t}.",
]
_BN_CLAIMS = [
    "আপনার শেষ {t} হয়েছিল {when} আগে।",
    "আপনি গতবার {t} করিয়েছিলেন।",
    "আপনার আগের {t} রিপোর্ট ঠিক ছিল।",
    "আপনি আগে {t} করিয়েছেন।",
    "শেষবার আপনি {t} করাতে এসেছিলেন।",
    "আপনার পূর্বের {t} এর তারিখ {when} আগে।",
    "আগেরবার আপনি {t} করিয়েছিলেন।",
]
_HI_CLAIMS = [
    "आपका पिछला {t} {when} पहले हुआ था।",
    "आप पिछली बार {t} के लिए आए थे।",
    "आपका आख़िरी {t} ठीक था।",
    "आप पहले {t} करा चुके हैं।",
    "आपका पहले का {t} हुआ था।",
    "पिछली बार आप {t} के लिए आए थे।",
    "आपका आखिरी {t} {when} पहले था।",
]
_TESTS = {
    "en": ["CBC", "blood test", "thyroid test", "sugar test", "lipid profile", "X-ray", "test"],
    "bn": ["সিবিসি", "রক্ত পরীক্ষা", "থাইরয়েড টেস্ট", "সুগার টেস্ট", "টেস্ট"],
    "hi": ["सीबीसी", "ब्लड टेस्ट", "थायराइड टेस्ट", "शुगर टेस्ट", "टेस्ट"],
}
_WHEN = {
    "en": ["month", "week", "two months", "year"],
    "bn": ["এক মাস", "দুই সপ্তাহ", "এক বছর"],
    "hi": ["एक महीने", "दो हफ़्ते", "एक साल"],
}
_PREFIX = {
    "en": ["", "Good to see you again. ", "Welcome back. ", "Sure. "],
    "bn": ["", "আবার কথা বলে ভালো লাগল। ", "হ্যাঁ, "],
    "hi": ["", "दोबारा बात करके अच्छा लगा। ", "जी, "],
}


def _claims(n: int, rnd: random.Random) -> list[tuple[str, str]]:
    banks = {"en": _EN_CLAIMS, "bn": _BN_CLAIMS, "hi": _HI_CLAIMS}
    out = []
    for i in range(n):
        lang = ("en", "bn", "hi")[i % 3]
        s = rnd.choice(banks[lang]).format(t=rnd.choice(_TESTS[lang]), when=rnd.choice(_WHEN[lang]))
        out.append((lang, rnd.choice(_PREFIX[lang]) + s))
    return out


def test_A_model_composed_history_claims_are_all_detected():
    rnd = random.Random(SEED)
    cases = _claims(200, rnd)
    slipped = [(lg, t) for lg, t in cases if not ht.mentions_personal_history(t, lg)]
    assert not slipped, f"{len(slipped)}/200 fabricated claims would have been spoken, e.g. {slipped[:5]}"


_CLEAN = {
    "en": [
        "I am well, thank you. How can I help you?",
        "The clinic opens at nine in the morning.",
        "Please tell me what you need.",
        "Our doctors see patients from Monday to Saturday.",
        "Thank you for calling.",
        "How can I help you today?",
    ],
    "bn": ["আমি ভালো আছি, ধন্যবাদ। আপনাকে কীভাবে সাহায্য করতে পারি?", "ক্লিনিক সকাল নয়টায় খোলে।", "আপনার কী প্রয়োজন বলবেন?"],
    "hi": ["मैं ठीक हूँ, धन्यवाद। मैं आपकी क्या मदद कर सकती हूँ?", "क्लिनिक सुबह नौ बजे खुलता है।", "आपको क्या चाहिए, बताइए।"],
}


def test_A_the_detector_does_not_block_ordinary_small_talk():
    """Over-blocking would push every small-talk reply to the default; keep it honest."""
    wrongly = [(lg, t) for lg, ts in _CLEAN.items() for t in ts if ht.mentions_personal_history(t, lg)]
    assert not wrongly, wrongly


# ============================================================================= B: no verification

_QUESTIONS = [
    "when did I last have my CBC",
    "when did I last do my thyroid test",
    "what tests have I had",
    "my test history",
    "when was my previous test",
    "my last test date",
    "আমার শেষ টেস্ট কবে হয়েছিল",
    "আমি আগে কী কী টেস্ট করিয়েছি",
    "আমার আগের টেস্ট",
    "मेरा आख़िरी टेस्ट कब हुआ था",
    "मैंने आख़िरी बार कब टेस्ट कराया",
    "मेरे पिछले टेस्ट",
]


def _identities():
    """Every state short of 'verified for patient 7'."""
    clock = {"t": 1000.0}
    out = {}
    out["none"] = IdentityState(clock=lambda: clock["t"])
    c = IdentityState(clock=lambda: clock["t"])
    c.claim("7", "9876543210")
    out["claimed"] = c
    other = IdentityState(clock=lambda: clock["t"])
    other.verify("otp", "8", "9999999999")
    out["verified_other_patient"] = other
    exp = IdentityState(clock=lambda: clock["t"])
    exp.verify("otp", "7", "9876543210")
    clock["t"] += 10_000
    out["expired"] = exp
    rev = IdentityState(clock=lambda: 1000.0)
    rev.verify("otp", "7", "9876543210")
    rev.revoke("speaker_change")
    out["revoked"] = rev
    return out


def test_B_no_history_is_allowed_without_a_verification_for_that_patient():
    ids = _identities()
    blocked = 0
    total = 0
    for (name, ident), q in itertools.product(ids.items(), _QUESTIONS):
        hq = detect_history_question(q, "en") or detect_history_question(q, "bn") or detect_history_question(q, "hi")
        assert hq is not None, f"the question was not recognised as a history question: {q!r}"
        for lang in ("en", "bn", "hi"):
            total += 1
            d = pc.history_gate(ident, "7", lang)
            assert not d.allowed, f"{name} allowed history for {q!r}"
            assert d.statement and not any(ch.isdigit() for ch in d.statement[1])
            blocked += 1
    assert total >= 100 and blocked == total


# ============================================================================= C: missing data


def _stale():
    return (NOW - datetime.timedelta(hours=pc.STALE_AFTER_HOURS + 5)).isoformat()


def _fresh():
    return NOW.isoformat()


def _perf(i, name, days_ago):
    return {
        "id": f"test_performed:{i}",
        "kind": "test_performed",
        "fields": {"test_name": name, "performed_on": (TODAY - datetime.timedelta(days=days_ago)).isoformat()},
    }


def _all_langs(table):
    return set(table.values())


_CANNOT_TEXTS = _all_langs(ht.CANNOT_SEE) | _all_langs(ht.CANNOT_CONFIRM) | _all_langs(ht.AMBIGUOUS)


def _c_cases():
    rnd = random.Random(SEED + 1)
    cases = []
    for lang in ("en", "bn", "hi"):
        for _ in range(11):
            cases.append((lang, None, "CBC"))  # nothing retrieved
            cases.append((lang, {"success": False, "reason": "not_authorized"}, "CBC"))
            cases.append((lang, {"success": True, "as_of": _stale(), "events": [_perf(1, "CBC", 30)]}, "CBC"))  # stale
            cases.append((lang, {"success": True, "as_of": None, "events": [_perf(1, "CBC", 30)]}, "CBC"))  # no as-of
            cases.append((lang, {"success": True, "as_of": _fresh(), "events": []}, "CBC"))  # empty
            cases.append(
                (lang, {"success": True, "as_of": _fresh(), "events": [_perf(2, "Thyroid", 20)]}, "CBC")
            )  # other test only
            cases.append(
                (
                    lang,
                    {
                        "success": True,
                        "as_of": _fresh(),  # two tests fit "lipid"
                        "events": [_perf(3, "Lipid Profile", 90), _perf(4, "Lipid Panel Extended", 40)],
                    },
                    "lipid",
                )
            )
            cases.append(
                (
                    lang,
                    {
                        "success": True,
                        "as_of": _fresh(),  # bookings but no performed test
                        "events": [
                            {
                                "id": "test_booking:5",
                                "kind": "test_booking",
                                "fields": {"test_name": "CBC", "date": "2026-10-01", "status": "confirmed"},
                            }
                        ],
                    },
                    "CBC",
                )
            )
    rnd.shuffle(cases)
    return cases


def test_C_missing_stale_or_ambiguous_data_yields_only_cannot_see_statements():
    cases = _c_cases()
    assert len(cases) >= 100
    for lang, timeline, test in cases:
        ans = pc.answer_last_test(timeline, test, None, lang, NOW)
        texts = {t for _, t in ans.statements}
        assert texts <= _CANNOT_TEXTS, f"{lang} {timeline!r} produced {texts - _CANNOT_TEXTS}"
        assert not any(ch.isdigit() for t in texts for ch in t)
    for lang in ("en", "bn", "hi"):
        for timeline in (
            None,
            {"success": True, "as_of": _stale(), "events": []},
            {"success": True, "as_of": _fresh(), "events": []},
        ):
            ans = pc.answer_recent_tests(timeline, lang, NOW)
            assert {t for _, t in ans.statements} <= _CANNOT_TEXTS


def test_C_a_test_the_patient_never_had_is_never_reported_as_done():
    for lang in ("en", "bn", "hi"):
        tl = {"success": True, "as_of": _fresh(), "events": [_perf(1, "Thyroid", 50)]}
        ans = pc.answer_last_test(tl, "CBC", None, lang, NOW)
        assert [sid for sid, _ in ans.statements] == ["cannot_see"]


# ============================================================================= D: provenance

_NUM = re.compile(r"\d+")


def _random_timeline(rnd: random.Random, i: int) -> dict:
    names = ["CBC", "Thyroid", "Lipid Profile", "HbA1c", "Vitamin D", "Urine Routine"]
    events = []
    for k in range(rnd.randint(0, 6)):
        events.append(_perf(i * 100 + k, rnd.choice(names), rnd.randint(1, 700)))
    if rnd.random() < 0.5:
        events.append(
            {
                "id": f"appointment:{i}",
                "kind": "appointment",
                "fields": {
                    "doctor_name": "Roy",
                    "date": (TODAY + datetime.timedelta(days=rnd.randint(1, 30))).isoformat(),
                    "time_slot": "10:30",
                    "status": "confirmed",
                },
            }
        )
    return {"success": True, "as_of": _fresh(), "events": events}


def _numbers_in(fields: dict) -> set[str]:
    return {n for v in fields.values() for n in _NUM.findall(str(v))}


def test_D_every_spoken_statement_traces_to_a_retrieved_event_and_holds_no_stray_number():
    rnd = random.Random(SEED + 2)
    checked = 0
    for i in range(100):
        tl = _random_timeline(rnd, i)
        by_id = {e["id"]: e for e in tl["events"]}
        lang = ("en", "bn", "hi")[i % 3]
        test = rnd.choice(["CBC", "Thyroid", "Lipid Profile", "HbA1c"])
        for ans in (
            pc.answer_last_test(tl, test, None, lang, NOW),
            pc.answer_recent_tests(tl, lang, NOW),
            pc.answer_appointments(tl, lang, NOW),
        ):
            record = [{"seq": 1, "kind": "history", "payload": {"statements": [sid for sid, _ in ans.statements]}}]
            result = history_audit.audit(record, tl)
            assert result["clean"], f"orphan statements {result['orphans']} for {tl}"
            for sid, text in ans.statements:
                if sid in by_id:
                    allowed = _numbers_in(by_id[sid]["fields"])
                    stray = set(_NUM.findall(text)) - allowed
                    assert not stray, f"{sid}: numbers {stray} in {text!r} are not in the retrieved fields"
                checked += 1
    assert checked >= 100


def test_D_the_audit_flags_a_statement_with_no_origin():
    tl = {"success": True, "events": [_perf(1, "CBC", 10)]}
    fabricated = [{"seq": 1, "kind": "history", "payload": {"statements": ["test_performed:1", "test_performed:999"]}}]
    r = history_audit.audit(fabricated, tl)
    assert not r["clean"] and r["orphans"] == ["test_performed:999"]


def test_D_a_due_statement_needs_an_approved_interval_to_have_been_retrieved():
    rec = [{"seq": 1, "kind": "history", "payload": {"statements": ["due:yes"]}}]
    assert not history_audit.audit(rec, {"events": []})["clean"]
    assert history_audit.audit(rec, {"events": []}, [{"known": True, "due": True}])["clean"]
    assert not history_audit.audit(rec, {"events": []}, [{"known": False, "due": None}])["clean"]


def test_the_suite_really_covers_five_hundred_attempts():
    a = len(_claims(200, random.Random(SEED)))
    b = len(_identities()) * len(_QUESTIONS) * 3
    c = len(_c_cases())
    d = 100
    assert a + b + c + d >= 500, (a, b, c, d)
