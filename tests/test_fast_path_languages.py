"""KCD-095 / KCD-094 / KCD-096: the deterministic fast path serves Bengali, Hindi and English through one interface,
abstains on a language it has no table for, reports serve rate per language, and gets a language by data alone.

    python -m pytest tests/test_fast_path_languages.py -v

Bengali is compared, utterance by utterance, with the frozen pre-change module (tests/_fast_path_legacy.py) so that
"Bengali did not change" is a check and not a claim. Hindi and English cue tables are REASONED, not measured.
"""

import datetime
import os
import random
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "tools"), os.path.join(REPO_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.append(_p)

import _fast_path_legacy as legacy  # noqa: E402
import gazetteer_eval as ev  # noqa: E402

from agent import fast_path_cues as cues  # noqa: E402
from agent.fast_path import (  # noqa: E402
    AMBIGUITY_MARGIN,
    COMMIT_FLOOR,
    LANGUAGE_GAP_MARGIN,
    MIN_TURNS_FOR_GAP,
    Catalogue,
    FastPath,
    serve_rate_gap,
)
from agent.fast_path import _normalize as _normalize_text  # noqa: E402
from agent.outcome_metrics import abstentions  # noqa: E402

TODAY = datetime.date(2026, 9, 25)


@pytest.fixture(scope="module")
def catalogue_payload():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        return client.get("/api/v1/catalogue").json()


@pytest.fixture
def fp(catalogue_payload):
    return FastPath(Catalogue(catalogue_payload), today=TODAY)


def _served(fp, text, lang):
    r = fp.resolve(text, lang)
    return None if r is None else (r.intent, {k: v for k, v in r.slots.items() if v})


# ======================================================================= Bengali is unchanged (against the oracle)


def _bengali_corpus(cat):
    rng = random.Random(5)
    frames = [
        "{} টেস্টের রেট কত",
        "{} এর দাম কত",
        "{} টেস্টের আগে কী প্রস্তুতি নিতে হবে",
        "{} এর জন্য খালি পেটে থাকতে হবে",
        "ডাক্তার {} কবে চেম্বারে বসবেন",
        "ডাক্তার {} আজ আছেন",
        "ডাক্তার {} কাল কখন বসবেন",
        "{} বুক করতে চাই",
        "{} আর সুগার টেস্টের রেট",
        "ডাক্তার {} সোমবার আছেন",
        "{}",
    ]
    fixed = [
        "হ্যালো",
        "ধন্যবাদ",
        "নমস্কার",
        "ক্লিনিক কখন খোলে",
        "আপনাদের ঠিকানা কোথায়",
        "পার্কিং আছে",
        "রিপোর্ট কীভাবে পাব",
        "পেমেন্ট কীভাবে করব",
        "ইনসিওরেন্স চলে",
        "আজ কেমন আছেন",
        "কিছু বুঝলাম না",
        "সবগুলো টেস্টের তালিকা দিন",
        "hello",
        "",
        "   ",
    ]
    forms = (
        [a for t in cat["tests"] for a in t["aliases_bn"]]
        + [d["surname"] for d in cat["doctors"]]
        + [a for d in cat["doctors"] for a in d["aliases_bn"]]
    )
    out = list(fixed)
    for f in forms:
        vs = [f] + [v for vv in ev.variants(f, rng).values() for v in vv]
        for v in vs[:4]:
            out += [fr.format(v) for fr in rng.sample(frames, 3)]
    return out


def test_bengali_decisions_match_the_frozen_pre_change_module_except_where_a_generic_word_gave_a_wrong_answer(
    catalogue_payload,
):
    """DELIBERATE SPEC CHANGE, marked: the frozen module answered a garbled turn with a WRONG test whenever the
    generic word "test" plus a shared syllable scored above the commit floor (found on the live pod: "hon ak ei test
    dam koto" matched the HIV test at 0.89; "sin test rate" matched the kidney test). A match on a test form must
    now also hold on the words that name the test. So on this corpus the two modules must agree on everything EXCEPT:
      * the same answer at a lower confidence (the generic word no longer counts toward it), and
      * an old answer that is now an abstain (the turn goes to the model), where the old matched form contained the
        generic word.
    A DIFFERENT answer is never allowed: the change may only take answers away, never swap one for another."""
    old = legacy.FastPath(legacy.Catalogue(catalogue_payload), today=TODAY)
    new = FastPath(Catalogue(catalogue_payload), today=TODAY)
    corpus = _bengali_corpus(catalogue_payload)
    assert len(corpus) > 1000
    identical = lower = abstained = 0
    generic = {"টেস্ট", "টেস্টের"}
    for u in corpus:
        a, b = old.resolve(u), new.resolve(u, "bn")
        if a is None and b is None:
            identical += 1
        elif a is None or b is None:
            assert b is None, (u, "the new module must never ANSWER where the old one abstained")
            # DELIBERATE SPEC CHANGE, marked (2026-09-25, fourth live call): also an abstain when the old answer was NOT an
            # exact match and a second test fit within AMBIGUITY_MARGIN of it (agent/fast_path.py): the model's turn asks
            # the clinic API "which one?" instead of the fast path picking between two.
            # DELIBERATE SPEC CHANGE, marked (2026-09-25): also an abstain when the old answer was a DOCTOR's surname form that
            # belongs to more than one doctor ("রায়" is both Dr. Roy and Dr. Ray): the clinic API asks "which one?".
            if a.intent == "doctor_availability" and new.catalogue.doctor_owner_count("bn", a.matched_form) > 1:
                abstained += 1
                continue
            if not any(g in (a.matched_form or "").split() for g in generic):
                text = new._without_cues(_normalize_text(u), cues.table_for("bn"))
                best, _f, score = new.catalogue.match(text, "test", "bn", COMMIT_FLOOR)
                other, _f2, other_score = new.catalogue.match(text, "test", "bn", COMMIT_FLOOR, skip_name=best)
                assert best and score < 1.0 and other and score - other_score < AMBIGUITY_MARGIN, (u, a.matched_form)
            abstained += 1
        else:
            same_answer = (
                a.intent,
                a.slots["test_name"],
                a.slots["doctor_name"],
                a.slots["faq_topic"],
                a.direct_reply_bn,
            ) == (b.intent, b.slots["test_name"], b.slots["doctor_name"], b.slots["faq_topic"], b.direct_reply_bn)
            assert same_answer, (u, a.slots, b.slots)  # never a different answer
            if round(a.confidence, 6) == round(b.confidence, 6):
                identical += 1
            else:
                assert b.confidence < a.confidence and any(g in (a.matched_form or "").split() for g in generic)
                lower += 1
    assert identical > 1500 and (lower + abstained) < 0.05 * len(corpus), (identical, lower, abstained)
    assert new.stats["served"] > 300


# ---- found on the live pod: the generic word "test" is not evidence of WHICH test ------------------------------------


@pytest.mark.parametrize("garbled", ["হন আক এই টেস্ট দাম কত", "আক এই টেস্ট দাম কত"])
def test_a_garbled_turn_is_not_matched_to_a_test_on_the_word_test_alone(fp, garbled):
    """Logged on the pod: this matched the HIV test's alias "AIDS test" at 0.89 and only the recogniser's
    low-agreement gate stopped the price being quoted."""
    assert fp.resolve(garbled, "bn") is None


@pytest.mark.parametrize("text", ["সিন টেস্টের রেট কত", "রয় টেস্টের রেট কত", "ডাস টেস্টের রেট কত", "পাল টেস্টের রেট কত"])
def test_a_doctors_surname_followed_by_the_word_test_is_not_a_test(fp, catalogue_payload, text):
    """The frozen module answered these with the kidney, thyroid, AIDS and Widal tests."""
    old = legacy.FastPath(legacy.Catalogue(catalogue_payload), today=TODAY).resolve(text)
    assert old is not None and old.intent == "test_rate"  # the wrong answer the old code gave
    assert fp.resolve(text, "bn") is None


def test_a_specific_alias_is_not_confused_with_a_shorter_one_that_is_its_prefix(fp):
    r = fp.resolve("ডেঙ্গু আইজিজি টেস্টের রেট কত", "bn")
    assert r is not None and r.slots["test_name"] == "ডেঙ্গু আইজিজি"


def test_a_misspelt_generic_word_is_still_read_as_the_generic_word(fp):
    for text in ("লিভার টেশ্ট এর দাম কত", "লিভার টেস্ত এর দাম কত", "লিভার তেস্ট এর দাম কত"):
        r = fp.resolve(text, "bn")
        assert r is not None and r.slots["test_name"] == "লিভার টেস্ট", text


@pytest.mark.parametrize("text", ["ইউরিন টেস্টের দাম কত", "ওয়াইডাল টেস্টের রেট", "এইডস টেস্টের দাম কত"])
def test_naming_the_test_properly_is_still_served(fp, text):
    assert fp.resolve(text, "bn") is not None


def test_the_same_holds_in_english(fp):
    assert fp.resolve("what is the price of the test", "en") is None  # "test" alone names nothing
    assert fp.resolve("price of a widal test", "en") is not None


def test_the_default_language_is_bengali_so_existing_callers_are_unchanged(fp):
    assert _served(fp, "সিবিসি টেস্টের রেট কত", "bn") == (
        fp.resolve("সিবিসি টেস্টের রেট কত").intent,
        {"test_name": "সিবিসি"},
    )


# ====================================================================================== an unknown language abstains


@pytest.mark.parametrize("lang", ["unknown", "ta", "", None, "BN", "bengali"])
def test_a_language_with_no_cue_table_is_never_guessed_at(fp, lang):
    """Blueprint 4.4: an unidentified utterance is handed to the model; a Bengali table never fires on a Hindi one."""
    before = abstentions.snapshot().get("unknown_language", {})
    assert fp.resolve("সিবিসি টেস্টের রেট কত", lang) is None
    assert fp.resolve("what is the price of a lipid profile test", lang) is None
    after = abstentions.snapshot()["unknown_language"]
    assert sum(after.values()) >= sum(before.values()) + 2


def test_every_supported_language_is_asserted_and_every_unsupported_one_abstains(fp):
    cases = {"bn": "সিবিসি টেস্টের রেট কত", "hi": "सीबीसी की कीमत कितनी है", "en": "how much does the CBC cost"}
    for lang, text in cases.items():
        assert fp.resolve(text, lang) is not None, lang
    for lang in ("unknown", "ta", "fr"):
        assert all(fp.resolve(t, lang) is None for t in cases.values()), lang


def test_one_languages_table_does_not_fire_on_another_languages_sentence(fp):
    assert fp.resolve("सीबीसी की कीमत कितनी है", "bn") is None  # Hindi sentence, Bengali table
    assert fp.resolve("সিবিসি টেস্টের রেট কত", "hi") is None  # Bengali sentence, Hindi table
    assert fp.resolve("সিবিসি টেস্টের রেট কত", "en") is None
    assert fp.resolve("how much does the CBC cost", "bn") is None


# ============================================================================================ English and Hindi serve

ENGLISH = [
    ("what is the price of a lipid profile test", ("test_rate", {"test_name": "lipid profile"})),
    ("how much does the CBC cost", ("test_rate", {"test_name": "cbc"})),
    ("price of uric acid test", ("test_rate", {"test_name": "uric acid"})),
    ("when does doctor Sen sit today", ("doctor_availability", {"doctor_name": "sen", "date": "2026-09-25"})),
    (
        "is doctor Mukherjee available tomorrow",
        ("doctor_availability", {"doctor_name": "mukherjee", "date": "2026-09-26"}),
    ),
    (
        "is doctor Ghosh available the day after tomorrow",
        ("doctor_availability", {"doctor_name": "ghosh", "date": "2026-09-27"}),
    ),
    ("what should I do before a lipid profile test", ("test_prep", {"test_name": "lipid profile"})),
    ("do I need to fast before the TSH test", ("test_prep", {"test_name": "tsh"})),
    ("what are your opening hours", ("clinic_faq", {"faq_topic": "hours"})),
    ("what time do you open", ("clinic_faq", {"faq_topic": "hours"})),
    ("where is the clinic", ("clinic_faq", {"faq_topic": "location"})),
    ("do you have parking", ("clinic_faq", {"faq_topic": "parking"})),
    ("do you accept insurance", ("clinic_faq", {"faq_topic": "insurance"})),
    ("how can I collect my report", ("clinic_faq", {"faq_topic": "report_collection"})),
    ("hello", ("smalltalk", {})),
    ("thank you", ("smalltalk", {})),
]
HINDI = [
    ("लिपिड प्रोफाइल टेस्ट का रेट क्या है", ("test_rate", {"test_name": "लिपिड प्रोफाइल"})),
    ("सीबीसी की कीमत कितनी है", ("test_rate", {"test_name": "सीबीसी"})),
    ("डॉक्टर सेन आज कब बैठेंगे", ("doctor_availability", {"doctor_name": "सेन", "date": "2026-09-25"})),
    ("लिपिड प्रोफाइल से पहले क्या तैयारी करनी है", ("test_prep", {"test_name": "लिपिड प्रोफाइल"})),
    ("क्लिनिक कब खुलता है", ("clinic_faq", {"faq_topic": "hours"})),
    ("आपका पता क्या है", ("clinic_faq", {"faq_topic": "location"})),
    ("पार्किंग है क्या", ("clinic_faq", {"faq_topic": "parking"})),
    ("नमस्ते", ("smalltalk", {})),
    ("धन्यवाद", ("smalltalk", {})),
]


@pytest.mark.parametrize("text,expected", ENGLISH)
def test_english_routine_questions_are_served(fp, text, expected):
    assert _served(fp, text, "en") == expected


@pytest.mark.parametrize("text,expected", HINDI)
def test_hindi_routine_questions_are_served(fp, text, expected):
    assert _served(fp, text, "hi") == expected


# ================================================================================================= what must abstain


@pytest.mark.parametrize(
    "lang,text",
    [
        ("en", "I want to book an appointment"),  # booking is never the fast path
        ("en", "price of cbc and lipid profile"),  # two entities
        ("en", "what is the price of a table"),  # no such test
        ("en", "when does doctor Sen sit on Monday"),  # a weekday: not parsed here
        ("en", "when does doctor Sen sit tonight"),
        ("en", "when does doctor Sen sit next week"),
        ("en", "when does doctor Sen sit at 5"),  # a digit
        ("en", "when does doctor Nobody sit"),  # no such doctor
        ("en", "how much is the price of fasting sugar and when"),  # more than one cue set
        ("hi", "डॉक्टर सेन कल कब बैठेंगे"),  # "kal" is tomorrow AND yesterday
        ("hi", "डॉक्टर सेन सोमवार को कब बैठेंगे"),
        ("hi", "मुझे अपॉइंटमेंट बुक करना है"),
        ("hi", "सीबीसी और लिपिड प्रोफाइल का रेट"),
        ("hi", "डॉक्टर नोबडी कब बैठते हैं"),
    ],
)
def test_what_the_fast_path_cannot_be_sure_of_goes_to_the_model(fp, lang, text):
    assert fp.resolve(text, lang) is None, text


def test_a_short_surname_is_not_read_out_of_an_ordinary_word(catalogue_payload):
    """'days' scores 0.86 against the surname 'Das'; for English and Hindi a short form must match exactly."""
    fp = FastPath(Catalogue(catalogue_payload), today=TODAY)
    assert fp.resolve("when is the doctor available in two days", "en") is None
    assert _served(fp, "when is doctor das available today", "en") == (
        "doctor_availability",
        {"doctor_name": "das", "date": "2026-09-25"},
    )


def test_english_cues_match_whole_words_only(fp):
    """'rate' inside 'separate', 'book' inside 'notebook' and 'hi' inside 'this' must not fire."""
    assert fp.resolve("this is a separate question about the lipid profile", "en") is None
    assert fp.resolve("my notebook is at the counter", "en") is None
    assert fp.resolve("this", "en") is None


def test_faq_phrases_never_route_to_the_wrong_topic(fp):
    """A wrong FAQ answer is a wrong FACT, served with no model behind it. Each paraphrase goes to its topic or
    to the model, never to another topic."""
    cases = [
        ("en", "what time do you close", "hours"),
        ("en", "are you open on sunday", "hours"),
        ("en", "how do i get there", "location"),
        ("en", "can I pay by card", "payment_methods"),
        ("en", "do you take mediclaim", "insurance"),
        ("en", "where can I park my car", "parking"),
        ("en", "what is your phone number", "contact_number"),
        ("en", "can you collect a sample from home", "home_collection"),
        ("hi", "क्लिनिक कब बंद होता है", "hours"),
        ("hi", "क्लिनिक कहाँ है", "location"),
        ("hi", "घर से सैंपल", "home_collection"),
        ("hi", "फोन नंबर क्या है", "contact_number"),
    ]
    for lang, text, topic in cases:
        r = fp.resolve(text, lang)
        assert r is None or (r.intent == "clinic_faq" and r.slots["faq_topic"] == topic), (lang, text, r and r.slots)


# ================================================================== smalltalk replies pass the same rules as any reply


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_the_greeting_and_thanks_replies_pass_the_persona_and_spoken_text_rules(lang):
    from agent import spoken_text_lint
    from agent.lang_select import speakable
    from agent.persona import is_clean as persona_clean

    table = cues.table_for(lang)
    for reply in (table.greeting_reply, table.thanks_reply):
        assert speakable(reply, lang), reply  # the same gate main.py puts on a smalltalk reply
        assert persona_clean(reply, lang), reply
        assert not spoken_text_lint.find_artifacts(reply), reply


# ========================================================================================== serve rate per language


def test_serve_rate_is_counted_per_language(fp):
    for text, _e in ENGLISH:
        fp.resolve(text, "en")
    for text, _e in HINDI:
        fp.resolve(text, "hi")
    fp.resolve("সিবিসি টেস্টের রেট কত", "bn")
    fp.resolve("gibberish words", "en")
    by = fp.snapshot()["language_gap"]["by_language"]
    assert by["en"]["served"] == len(ENGLISH) and by["en"]["abstained"] == 1
    assert by["hi"]["served"] == len(HINDI)
    assert by["bn"]["served"] == 1


def test_a_gap_wider_than_the_margin_is_reported_as_a_defect():
    report = serve_rate_gap(
        {
            "bn": {"served": 60, "abstained": 40},
            "hi": {"served": 10, "abstained": 90},
            "en": {"served": 55, "abstained": 45},
        }
    )
    assert report["by_language"]["hi"]["serve_rate"] == 0.1
    assert report["widest"] == ["bn", "hi"] and report["gap"] == 0.5
    assert report["defect"] is True and report["margin"] == LANGUAGE_GAP_MARGIN


def test_a_gap_inside_the_margin_is_not_a_defect_and_a_thin_language_is_not_compared():
    ok = serve_rate_gap({"bn": {"served": 60, "abstained": 40}, "en": {"served": 50, "abstained": 50}})
    assert ok["gap"] == 0.1 and ok["defect"] is False
    thin = serve_rate_gap(
        {"bn": {"served": 60, "abstained": 40}, "hi": {"served": 0, "abstained": MIN_TURNS_FOR_GAP - 1}}
    )
    assert thin["gap"] is None and thin["defect"] is False  # too few Hindi turns: noise, not a finding


def test_the_parallel_synthetic_set_serves_hindi_and_english_no_worse_than_the_margin(fp):
    """The same eight questions in each language (price x2, availability, preparation x2, FAQ x2, greeting)."""
    parallel = {
        "en": [
            "what is the price of a lipid profile test",
            "how much does the CBC cost",
            "when does doctor Sen sit today",
            "what should I do before a lipid profile test",
            "do I need to fast before the TSH test",
            "what are your opening hours",
            "where is the clinic",
            "hello",
        ],
        "hi": [
            "लिपिड प्रोफाइल टेस्ट का रेट क्या है",
            "सीबीसी की कीमत कितनी है",
            "डॉक्टर सेन आज कब बैठेंगे",
            "लिपिड प्रोफाइल से पहले क्या तैयारी करनी है",
            "टीएसएच टेस्ट से पहले खाली पेट रहना है क्या",
            "क्लिनिक कब खुलता है",
            "आपका पता क्या है",
            "नमस्ते",
        ],
        "bn": [
            "লিপিড প্রোফাইল টেস্টের রেট কত",
            "সিবিসি টেস্টের দাম কত",
            "ডাক্তার সেন আজ আছেন",
            "লিপিড প্রোফাইল টেস্টের আগে কী প্রস্তুতি নিতে হবে",
            "টিএসএইচ টেস্টের আগে খালি পেটে থাকতে হবে",
            "ক্লিনিকের সময় কী",
            "আপনাদের ঠিকানা কোথায়",
            "নমস্কার",
        ],
    }
    for _ in range(MIN_TURNS_FOR_GAP // 8 + 1):
        for lang, texts in parallel.items():
            for t in texts:
                fp.resolve(t, lang)
    report = fp.snapshot()["language_gap"]
    rates = {lang: r["serve_rate"] for lang, r in report["by_language"].items()}
    assert report["defect"] is False, (rates, report["gap"])


# =============================================================================== adding a language is data, not code


def test_a_new_language_needs_a_cue_table_and_catalogue_columns_and_no_code():
    payload = {
        "tests": [{"name": "Lipid Profile", "aliases_ta": ["லிப்பிட் ப்ரொஃபைல்"]}],
        "doctors": [{"name": "Dr. A. Sen", "surname": "Sen", "aliases_ta": ["சென்"]}],
        "faq_topics": [{"topic": "parking", "keywords_ta": ["வாகன நிறுத்தம்"]}],
    }
    table = cues.CueTable(
        lang="ta",
        rate=("விலை",),
        avail=("எப்போது",),
        book=("முன்பதிவு",),
        prep=("தயாரிப்பு",),
        greeting=("வணக்கம்",),
        thanks=("நன்றி",),
        complexity=("மற்றும்",),
        greeting_reply="வணக்கம்",
        thanks_reply="நன்றி",
    )
    try:
        cues.register(table)
        fp = FastPath(Catalogue(payload), today=TODAY)
        r = fp.resolve("லிப்பிட் ப்ரொஃபைல் விலை", "ta")
        assert r is not None and r.intent == "test_rate"
        assert fp.resolve("வாகன நிறுத்தம்", "ta").slots["faq_topic"] == "parking"
        assert fp.resolve("வணக்கம்", "ta").intent == "smalltalk"
        assert "ta" in fp.snapshot()["language_gap"]["by_language"]
    finally:
        cues._TABLES.pop("ta", None)
    assert FastPath(Catalogue(payload), today=TODAY).resolve("லிப்பிட் ப்ரொஃபைல் விலை", "ta") is None


def test_the_registry_lists_the_supported_languages():
    assert set(cues.languages()) >= {"bn", "hi", "en"}
    assert cues.table_for("xx") is None and cues.table_for(None) is None


# ===================================================================================== speed, and the real catalogue


def test_a_bengali_turn_is_well_inside_the_150_ms_fast_path_budget(fp):
    """Blueprint 4.3 / KCD-092. Measured about 3 ms per turn after this change, 40 ms before (17-156 ms by turn)."""
    texts = [
        "ইউরিক অ্যাসিড টেস্টের রেট কত",
        "ডাক্তার সেন কবে চেম্বারে বসবেন",
        "লিপিড প্রোফাইল টেস্টের দাম কত",
        "সিবিসি টেস্ট এর জন্য কি খালি পেটে থাকতে হবে",
        "ক্লিনিক কখন খোলে",
    ]
    best_max = []
    for _attempt in range(3):
        worst = 0.0
        for t in texts:
            t0 = time.perf_counter()
            fp.resolve(t, "bn")
            worst = max(worst, (time.perf_counter() - t0) * 1000)
        best_max.append(worst)
    assert min(best_max) < 150.0, best_max


def test_a_catalogue_grown_to_five_thousand_rows_stays_inside_the_budget(catalogue_payload):
    import copy

    big = copy.deepcopy(catalogue_payload)
    for canonical, form in ev.synthetic_catalogue(5000):
        big["tests"].append({"name": canonical, "aliases_bn": [form], "aliases_hi": []})
    fp = FastPath(Catalogue(big), today=TODAY)
    assert fp.catalogue.table("test", "bn").indexed
    texts = ["ইউরিক অ্যাসিড টেস্টের রেট কত", "লিপিড প্রোফাইল টেস্টের দাম কত", "সিবিসি টেস্ট এর জন্য কি খালি পেটে থাকতে হবে"]
    worst = min(max(_timed(fp, t) for t in texts) for _ in range(3))
    assert worst < 150.0, worst
    assert _served(fp, texts[0], "bn")[0] == "test_rate"  # and it still answers, not merely quickly


def _timed(fp, text):
    t0 = time.perf_counter()
    fp.resolve(text, "bn")
    return (time.perf_counter() - t0) * 1000


def test_the_catalogue_payload_now_carries_hindi_and_english_faq_keywords(catalogue_payload):
    for row in catalogue_payload["faq_topics"]:
        assert row["keywords_bn"] and row["keywords_hi"] and row["keywords_en"], row["topic"]
