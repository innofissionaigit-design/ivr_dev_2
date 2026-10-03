"""Answer the common turn without waking the 7B model.

WHY THIS EXISTS
---------------
Measuring the semantic cache produced a finding that reaches further than
caching. Against bge-m3 on real Bengali clinic questions:

    same question, reworded / ASR-garbled     cosine 0.78 - 0.82
    DIFFERENT test, same sentence frame       cosine 0.7492
    same entity, character-level comparison   0.696 - 1.000
    different entity, character-level          0.231 - 0.381

The embedding separates "which test" by 0.03. Character overlap separates
it by 0.32 -- an order of magnitude better, because the embedding is
dominated by the sentence frame ("X টেস্টের রেট কত") while the part that
decides which price a patient is quoted is a handful of characters.

Follow that through and the conclusion is not "tune the cache". It is
that for a fixed 74-row catalogue, identifying the entity was never a
language-modelling problem. A caller asking "ইউরিক অ্যাসিড টেস্টের রেট কত"
needs two things recognised: an intent drawn from a set of four, and one
row out of 74. Both are decidable locally, in microseconds, with a wider
correctness margin than the 7B model's output was ever checked against.

So the LLM is demoted to what it is genuinely needed for: utterances this
module is NOT confident about.

Extended beyond price/availability to also cover test preparation
("test_prep") and a small fixed set of clinic FAQ topics ("clinic_faq" --
hours, location, payment, insurance, parking, report collection, contact
number, home collection): same reasoning, same commit-floor discipline,
same abstain-when-ambiguous default. See Catalogue's docstring for the
FAQ table's shape and its growth limit.

WHAT IT DELIBERATELY REFUSES
----------------------------
`book_appointment` always goes to the LLM. It needs a date, a time, a
patient name and a phone number pulled out of free speech -- open-ended
extraction with caller-specific data in it, which is exactly the case
where a pattern-matcher's failure mode is silent and wrong. The fast path
handles the questions with one entity and no PII, and hands over anything
else. Abstaining is a first-class result here, not a failure.

PER LANGUAGE (KCD-095)
----------------------
The cue words that say "this is a price question" live in agent/fast_path_cues.py, one table per language behind one
interface; `resolve(text, lang)` uses the table for `lang` and abstains for a language it has none for (including
"unknown"). Serve rate is counted per language and `snapshot()["language_gap"]` flags a gap wider than
LANGUAGE_GAP_MARGIN as a defect. Adding a language is a `fast_path_cues.register(...)` call plus catalogue columns,
not code.

FAST AT ANY CATALOGUE SIZE (KCD-096)
------------------------------------
Matching used to score every form against every word-window with difflib: 17-156 ms per turn at 74 rows and seconds
at thousands. It now goes through agent/form_index.py: an exact pruning step (a proven upper bound skips comparisons
that cannot reach the commit floor; results at or above the floor are unchanged) and, on large tables only, a bigram
shortlist whose only failure mode is an abstain. The commit gate stays a CHARACTER similarity on purpose: a
fast-path answer has no model behind it, and a sound-alike is a suggestion for the caller to confirm
(agent/gazetteer.py), never evidence to act on.
"""

from __future__ import annotations

import datetime
import logging
import re
import unicodedata
from collections import defaultdict

from agent import fast_path_cues as cues
from agent.enquiry_followup import is_reference_only
from agent.form_index import INDEX_MIN_FORMS, FormTable
from agent.outcome_metrics import abstentions, fast_path_served

logger = logging.getLogger("fast_path")

# Same floor as semantic_cache.ENTITY_MATCH_FLOOR, and for the same
# measured reason -- 0.696 was a real ASR variant of the right test, so a
# floor of 0.70 would have rejected it by four thousandths.
ENTITY_MATCH_FLOOR = 0.55

# A second, higher bar for committing WITHOUT the model. The cache could
# afford 0.55 because a wrong hit there still ran a live lookup against a
# name the LLM had produced; here nothing downstream re-checks the entity,
# so the margin has to carry the whole decision. 0.72 sits above every
# measured different-entity score (max 0.381) by a wide margin while
# staying under the worst same-entity score (0.696) -- anything between
# the two abstains to the LLM rather than guessing.
COMMIT_FLOOR = 0.72

# Same commit-without-a-model bar, applied to FAQ topic keyword phrases.
# REASONED, not measured -- there is no real-call FAQ-question corpus yet
# to calibrate against, unlike COMMIT_FLOOR above. Recalibrate against
# real Kolkata call audio before trusting this past the pilot; see
# agent/lid.py's module docstring for what "measured vs reasoned" means
# in this codebase.
FAQ_COMMIT_FLOOR = 0.72
# Two DIFFERENT tests that fit what was said nearly equally well ("blood sugar" fits both the fasting and the PP test)
# is not something to pick between: a price or a preparation rule for the wrong one is a confident wrong answer. When
# the best match is not exact and the runner-up is within this of it, the fast path abstains, and the model's turn
# reaches the clinic API, which answers "which one -- A or B?" (KCD-446). REASONED, not measured: the measured
# margins were 0.08-0.12 for the ambiguous phrasings and 0.24 or more for a name that is one test's.
AMBIGUITY_MARGIN = 0.2

# A gap in serve rate between two languages wider than this is a DEFECT in the cue data of the slower one (Blueprint
# 4.4: no language may be a second-class citizen). REASONED, not measured.
LANGUAGE_GAP_MARGIN = 0.25
# A language with fewer than this many turns is too thin to compare: its rate is noise.
MIN_TURNS_FOR_GAP = 20

_RE_WS = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[।?!,.;:'\"()\-]+")


def _normalize(text: str) -> str:
    """NFC first: Bengali conjuncts and vowel signs have multiple valid
    encodings, and two visually identical strings compare unequal if one
    is composed and the other is not. ASR output and seeded aliases come
    from different sources, so this is a live risk, not a theoretical one.
    The Devanagari nukta (U+093C) is dropped too: the Hindi recogniser
    writes a consonant with and without it for the same spoken word.
    """
    text = unicodedata.normalize("NFC", text).replace(chr(0x093C), "")
    return _RE_WS.sub(" ", _RE_PUNCT.sub(" ", text)).strip().lower()


# Words that say "a test" and nothing about WHICH test. Found on the live pod: the garbled Bengali turn "hon ak ei
# test dam koto" matched the HIV test's alias "AIDS test" at 0.89 -- on the generic word and one shared syllable --
# and only the recogniser's low-agreement gate stopped a wrong price being quoted. A match on a test form must now
# ALSO hold on the words that name the test (FormTable's `generic_words`): the forms themselves stay whole, so a
# specific alias is never confused with a shorter one that is merely its prefix. The same list, for the same
# reason, is in clinic-api/main.py (_GENERIC_WORDS).
_GENERIC_TEST_WORDS = frozenset(
    {
        "test",
        "tests",
        "টেস্ট",
        "টেস্টের",
        "টেস্টটা",
        "টেস্টটি",
        "টেস্টগুলো",
        "टेस्ट",
        "जांच",
        "जाँच",
    }
)


def _name_forms(name: str) -> list[str]:
    """A test called "Complete Blood Count (CBC)" is said as the whole name, without the bracket, or as the code."""
    forms = [name]
    if "(" in name and ")" in name:
        forms += [name.split("(")[0].strip(), name.split("(")[1].split(")")[0].strip()]
    return forms


def _dedupe(forms) -> list[str]:
    return list(dict.fromkeys(f for f in forms if f))


class Catalogue:
    """The 74 rows, with every spoken form that maps to each -- plus the
    FAQ topic table, which is the same shape (one canonical key, several
    spoken forms) even though its keys are topics ("hours") rather than
    tests or doctors. When this table outgrows a few dozen topics or needs
    ranked/partial matches, that is the signal to promote it to a real
    retrieval index (Epic E25) -- a flat keyword scan is the right tool
    only while it stays small and exhaustively enumerable, same trade-off
    this file's own docstring makes for the 74-row catalogue itself.

    One FormTable per (kind, language). The payload keys are `aliases_<lang>` (tests, doctors) and
    `keywords_<lang>` (FAQ); every language with a cue table gets a table, and a payload that carries another
    language's columns gets one for it too. The Bengali forms are exactly what they always were; Hindi and English
    also learn a test's own name, its bracket-less form and its code.
    """

    def __init__(self, payload: dict, index_min_forms: int = INDEX_MIN_FORMS):
        self._n_rows = (
            len(payload.get("tests", [])) + len(payload.get("doctors", [])) + len(payload.get("faq_topics", []))
        )
        langs = set(cues.languages()) | {"bn", "hi", "en"}
        for group in ("tests", "doctors"):
            for row in payload.get(group, []):
                langs |= {k[len("aliases_") :] for k in row if k.startswith("aliases_")}
        for row in payload.get("faq_topics", []):
            langs |= {k[len("keywords_") :] for k in row if k.startswith("keywords_")}
        # Which doctors (by position in the payload) a spoken form belongs to: "sen" belongs to every Sen. A form with
        # more than one owner is ambiguous, whatever its similarity score, and the fast path never answers on it.
        self._doctor_owners: dict[tuple[str, str], set[int]] = defaultdict(set)
        self._tables: dict[tuple[str, str], FormTable] = {}
        for lang in sorted(langs):
            for i, d in enumerate(payload.get("doctors", [])):
                for form in self._doctor_forms_of(d, lang):
                    self._doctor_owners[(lang, form)].add(i)
        for lang in sorted(langs):
            cue_table = cues.table_for(lang)
            exact_below = cue_table.exact_below_chars if cue_table else 5
            for kind, rows in (
                ("test", self._test_rows(payload, lang)),
                ("doctor", self._doctor_rows(payload, lang)),
                ("doctorfull", self._doctor_full_rows(payload, lang)),
                ("faq", self._faq_rows(payload, lang)),
            ):
                self._tables[(kind, lang)] = FormTable(
                    rows,
                    exact_below_chars=exact_below,
                    index_min_forms=index_min_forms,
                    generic_words=_GENERIC_TEST_WORDS if kind == "test" else frozenset(),
                )
        # the Bengali rows, under the names other code and tests have always read
        self.tests = self._tables[("test", "bn")].rows
        self.doctors = self._tables[("doctor", "bn")].rows
        self.faq_topics = self._tables[("faq", "bn")].rows

    @staticmethod
    def _test_rows(payload: dict, lang: str):
        rows = []
        for t in payload.get("tests", []):
            aliases = [_normalize(a) for a in t.get(f"aliases_{lang}", [])]
            names = [_normalize(t["name"])] if lang == "bn" else [_normalize(f) for f in _name_forms(t["name"])]
            rows.append((t["name"], _dedupe(aliases + names)))
        return rows

    @staticmethod
    def _doctor_forms_of(d: dict, lang: str) -> list[str]:
        forms = [_normalize(a) for a in d.get(f"aliases_{lang}", [])]
        forms.append(_normalize(d.get("surname") or d["name"].split()[-1]))
        whole = Catalogue._full_name_form(d, lang)
        if whole:
            forms.append(whole)  # owned by this doctor: a shared surname's owners are counted with it
        return _dedupe(forms)

    @staticmethod
    def _full_name_form(d: dict, lang: str) -> str:
        latin = " ".join(w for w in (d.get("full_name") or "").split() if w.strip(".").lower() not in ("dr", "doctor"))
        whole = {"bn": d.get("full_name_bn"), "hi": d.get("full_name_hi"), "en": latin}.get(lang)
        return _normalize(whole) if whole else ""

    @staticmethod
    def _doctor_full_rows(payload: dict, lang: str):
        """(doctor, [whole name]) for each doctor that has one: "পার্থ রায়" is one doctor's even where "রায়" is two's."""
        return [(d["name"], [f]) for d in payload.get("doctors", []) if (f := Catalogue._full_name_form(d, lang))]

    @staticmethod
    def _doctor_rows(payload: dict, lang: str):
        return [(d["name"], Catalogue._doctor_forms_of(d, lang)) for d in payload.get("doctors", [])]

    def doctor_owner_count(self, lang: str, form: str | None) -> int:
        """How many doctors the spoken form `form` belongs to (0 when unknown)."""
        return len(self._doctor_owners.get((lang, form), ())) if form else 0

    @staticmethod
    def _faq_rows(payload: dict, lang: str):
        return [
            (f["topic"], _dedupe(_normalize(k) for k in f.get(f"keywords_{lang}", [])))
            for f in payload.get("faq_topics", [])
        ]

    def __len__(self) -> int:
        return self._n_rows

    def languages(self) -> set[str]:
        return {lang for (_kind, lang) in self._tables}

    def table(self, kind: str, lang: str = "bn") -> FormTable | None:
        return self._tables.get((kind, lang))

    def match(
        self, text: str, kind: str, lang: str = "bn", floor: float = 0.0, skip_name: str | None = None
    ) -> tuple[str | None, str | None, float]:
        """-> (canonical_key, matched_spoken_form, score). `canonical_key`
        is a test/doctor name for kind in {"test", "doctor"}, or a FAQ
        topic key for kind="faq".

        floor=0 scores everything and returns the best score however low (what a caller that applies its own
        threshold, such as the history flow, has always seen). With a floor, comparisons that provably cannot
        reach it are skipped, which is what keeps this fast (agent/form_index.py); a best score at or above the
        floor is unchanged, and below it the answer is (None, None, 0.0)."""
        table = self._tables.get((kind, lang))
        if table is None:
            return None, None, 0.0
        return table.best(_normalize(text).split(), floor, skip_name)


class FastPathResult:
    __slots__ = ("intent", "slots", "direct_reply_bn", "confidence", "matched_form")

    def __init__(self, intent, slots, confidence, matched_form=None, direct_reply_bn=None):
        self.intent = intent
        self.slots = slots
        self.confidence = confidence
        self.matched_form = matched_form
        self.direct_reply_bn = direct_reply_bn

    def as_llm_shape(self) -> dict:
        """Same dict shape agent/llm.py returns, so callers cannot tell
        which path produced it and no downstream code needs a branch.
        (`direct_reply_bn` is the historical name: it holds the reply in the caller's language.)"""
        return {
            "intent": self.intent,
            "slots": self.slots,
            "direct_reply_bn": self.direct_reply_bn,
        }


def _empty_slots(**kw) -> dict:
    slots = {
        "test_name": None,
        "doctor_name": None,
        "date": None,
        "time_slot": None,
        "patient_name": None,
        "phone": None,
        "faq_topic": None,
    }
    slots.update(kw)
    return slots


def _complexity(text: str, table: cues.CueTable) -> bool:
    """Whole-word match, for cues where a substring hit would be a false positive.

    A real one this caught: the complexity guard rejected a textbook Bengali availability question ("Doctor Sen,
    when will he sit in his chamber") because the word for "will sit" contains the word for "all" as a substring.
    Bengali writes without internal word boundaries, so short function words appear inside longer unrelated words
    constantly. Every cue in a complexity list is a standalone word, so matching them as whole words is both
    correct and strictly safer -- for every language."""
    words = set(text.split())
    return any((cue in words) if " " not in cue else (cue in text) for cue in table.complexity)


def serve_rate_gap(
    by_lang: dict[str, dict[str, int]], margin: float = LANGUAGE_GAP_MARGIN, min_turns: int = MIN_TURNS_FOR_GAP
) -> dict:
    """Serve rate per language and the widest gap between two languages with enough turns to compare. A gap wider
    than `margin` is a defect in the cue data of the lower one (KCD-095)."""
    rates = {}
    for lang, c in by_lang.items():
        turns = c["served"] + c["abstained"]
        rates[lang] = {
            "served": c["served"],
            "abstained": c["abstained"],
            "turns": turns,
            "serve_rate": round(c["served"] / turns, 3) if turns else 0.0,
        }
    eligible = {lang: r["serve_rate"] for lang, r in rates.items() if r["turns"] >= min_turns}
    gap, widest = None, None
    if len(eligible) >= 2:
        top, bottom = max(eligible, key=lambda k: eligible[k]), min(eligible, key=lambda k: eligible[k])
        gap, widest = round(eligible[top] - eligible[bottom], 3), [top, bottom]
    return {
        "by_language": rates,
        "margin": margin,
        "min_turns": min_turns,
        "gap": gap,
        "widest": widest,
        "defect": gap is not None and gap > margin,
    }


class FastPath:
    def __init__(self, catalogue: Catalogue, today: datetime.date | None = None):
        self.catalogue = catalogue
        self._today = today
        self.stats = {"served": 0, "abstained": 0}
        self.by_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"served": 0, "abstained": 0})

    def _abstain(self, reason: str, intent: str = "unknown", lang: str = "") -> None:
        """KCD-457: abstention is a first-class, EXPECTED result here (see
        module docstring), but "why" still needs to be visible -- a shift
        in the reason distribution usually means an upstream change
        (catalogue growth, a new phrasing pattern), not that this module
        got worse. Recorded even though the caller ends up going to the
        LLM anyway, which is the whole point: this file's own decision is
        the thing being measured, not just the turn's eventual outcome."""
        self.stats["abstained"] += 1
        self.by_lang[lang or "?"]["abstained"] += 1
        abstentions.record(reason, intent, lang)

    def _serve(self, intent: str, lang: str) -> None:
        """KCD-460: serve rate published PER INTENT AND PER LANGUAGE --
        "a routine question is answered instantly, and their serve rate is
        published per intent and language" is this story's own wording."""
        self.stats["served"] += 1
        self.by_lang[lang]["served"] += 1
        fast_path_served.record(intent, "served", lang)

    def _resolve_date(self, text: str, table: cues.CueTable) -> tuple[str | None, bool]:
        """-> (iso_date_or_None, is_confident). Not confident means the
        utterance contains date-ish language this module will not try to
        parse, so the whole turn must go to the LLM."""
        today = self._today or datetime.date.today()
        if table.strict_dates and (re.search(r"\d", text) or table.any(text, table.date_words)):
            return None, False
        for word, offset in table.relative_days.items():
            if table.contains(text, word):
                return (today + datetime.timedelta(days=offset)).isoformat(), True
        # Any digit or weekday name means a date we are not handling here.
        if re.search(r"\d", text) or table.any(text, table.date_words):
            return None, False
        return None, True

    @staticmethod
    def _without_cues(text: str, table: cues.CueTable) -> str:
        """`text` with the rate and preparation cue phrases taken out: what is left is what NAMES something."""
        remaining = f" {text} "
        cues_longest_first = sorted(table.rate + table.prep, key=len, reverse=True)
        while True:  # taking one cue out can leave two words that form another ("how price much")
            before = remaining
            for cue in cues_longest_first:
                if table.match == "substring":
                    remaining = remaining.replace(cue, " ")
                else:
                    while f" {cue} " in remaining:
                        remaining = remaining.replace(f" {cue} ", " ")
            remaining = f" {' '.join(remaining.split())} "
            if remaining == before:
                break
        return " ".join(remaining.split())

    def _ambiguous_test(self, text: str, lang: str, table: cues.CueTable, score: float) -> bool:
        """Two different tests fit the words that NAME a test almost equally well. Judged on the text without the cue
        phrases: "খালি পেটে" is a cue for preparation, not part of the name of the test called "খালি পেটে সুগার", and
        letting it count made an ordinary "<test> ... খালি পেটে ..." look ambiguous."""
        if score >= 1.0:
            return False  # said in full: exactly one test's name
        body = self._without_cues(text, table)
        first, _form, first_score = self.catalogue.match(body, "test", lang, COMMIT_FLOOR)
        if first is None or first_score >= 1.0:
            return False
        other, _form2, other_score = self.catalogue.match(body, "test", lang, COMMIT_FLOOR, skip_name=first)
        return other is not None and first_score - other_score < AMBIGUITY_MARGIN

    def _names_nothing(self, text: str, table: cues.CueTable) -> bool:
        """True when, once the rate/preparation cue phrases are taken out, everything left is a little function word or
        a pointing word ("the same", "that", "test"): the caller asked about the topic, they did not name a test. One
        word this does not recognise -- possibly a test name -- makes it False, and the turn goes to the model."""
        known = set(table.function_words)
        return all(t in known or is_reference_only(t) for t in self._without_cues(text, table).split())

    def resolve(self, transcript: str, lang: str = "bn", topic_test: str | None = None) -> FastPathResult | None:
        """Returns None whenever it is not confident. None is the normal,
        expected outcome for anything non-routine -- the caller falls back
        to the semantic cache and then the LLM. A language with no cue table (including "unknown") always
        returns None: a table for one language never fires on another.

        `topic_test`: the test the last few turns were about (agent/enquiry_followup.recent_unique). A price or
        preparation question that names NO test -- nothing left in it but cue and function words, e.g. "do I need
        to fast?" straight after a price -- is answered about that test, with no model call. The reply always names the
        test, so a wrong assumption is heard and corrected. Without a topic, or with any word that could be a name,
        this abstains exactly as before."""
        table = cues.table_for(lang)
        if table is None:
            self._abstain("unknown_language", lang=str(lang))
            return None
        text = _normalize(transcript)
        if not text:
            self._abstain("empty_transcript", lang=lang)
            return None

        # Booking is never handled here: open-ended extraction with PII in
        # it. Check first, before any cue that might also appear in it.
        if table.any(text, table.book):
            self._abstain("booking_excluded", lang=lang)
            return None

        if _complexity(text, table):
            self._abstain("complexity_cue", lang=lang)
            return None

        wants_rate = table.any(text, table.rate)
        wants_avail = table.any(text, table.avail)
        wants_prep = table.any(text, table.prep)

        # More than one cue set firing means an utterance asking about more
        # than one thing (or genuinely ambiguous between them -- "what must I
        # observe" alone can read as prep, but combined with a rate/avail cue
        # it is not this module's call to make). Let the model decide.
        if sum((wants_rate, wants_avail, wants_prep)) > 1:
            self._abstain("ambiguous_multi_cue", lang=lang)
            return None

        if wants_rate or wants_prep:
            # A question with no test named in it -- only the cue and function words -- has no entity to look for. Matching
            # it anyway let the cue word itself ("fasting", "फास्टिंग") pick the test called "... Fasting" (measured on
            # the Hindi table: "do I need to fast" was answered for a blood-sugar test). It is about the recent topic if
            # there is one, and otherwise it is left to the model, which asks which test.
            intent = "test_rate" if wants_rate else "test_prep"
            if self._names_nothing(text, table):
                if topic_test:
                    self._serve(intent, lang)
                    logger.info("fast path: %s %r from the topic [%s] for %r", intent, topic_test, lang, transcript)
                    return FastPathResult(
                        intent, _empty_slots(test_name=topic_test), COMMIT_FLOOR, matched_form="topic"
                    )
                self._abstain("names_no_entity", intent, lang)
                return None

        if wants_rate:
            name, form, score = self.catalogue.match(text, "test", lang, COMMIT_FLOOR)
            if name and score >= COMMIT_FLOOR and self._ambiguous_test(text, lang, table, score):
                self._abstain("ambiguous_test", "test_rate", lang)
                return None
            if name and score >= COMMIT_FLOOR:
                self._serve("test_rate", lang)
                logger.info("fast path: test_rate %r (%.2f) [%s] from %r", name, score, lang, transcript)
                return FastPathResult("test_rate", _empty_slots(test_name=form or name), score, matched_form=form)
            self._abstain("below_commit_floor", "test_rate", lang)
            return None

        if wants_prep:
            name, form, score = self.catalogue.match(text, "test", lang, COMMIT_FLOOR)
            if name and score >= COMMIT_FLOOR and self._ambiguous_test(text, lang, table, score):
                self._abstain("ambiguous_test", "test_prep", lang)
                return None
            if name and score >= COMMIT_FLOOR:
                self._serve("test_prep", lang)
                logger.info("fast path: test_prep %r (%.2f) [%s] from %r", name, score, lang, transcript)
                return FastPathResult("test_prep", _empty_slots(test_name=form or name), score, matched_form=form)
            self._abstain("below_commit_floor", "test_prep", lang)
            return None

        if wants_avail:
            name, form, score = self.catalogue.match(text, "doctorfull", lang, 0.9)  # the whole name, if it was said
            if not name:
                name, form, score = self.catalogue.match(text, "doctor", lang, COMMIT_FLOOR)
            if name and score >= COMMIT_FLOOR and self.catalogue.doctor_owner_count(lang, form) > 1:
                # "Dr. A. Sen" when Ashok Sen and Abhishek Sen both sit here: the model's turn reaches the clinic API,
                # which answers "which one -- Ashok Sen or Abhishek Sen?" (by whole name, in the caller's script).
                self._abstain("ambiguous_doctor", "doctor_availability", lang)
                return None
            if name and score >= COMMIT_FLOOR:
                date_iso, confident = self._resolve_date(text, table)
                if not confident:
                    self._abstain("date_not_confident", "doctor_availability", lang)
                    return None
                self._serve("doctor_availability", lang)
                logger.info(
                    "fast path: doctor_availability %r (%.2f) date=%s [%s] from %r",
                    name,
                    score,
                    date_iso,
                    lang,
                    transcript,
                )
                return FastPathResult(
                    "doctor_availability",
                    _empty_slots(doctor_name=form or name, date=date_iso),
                    score,
                    matched_form=form,
                )
            if not table.faq_after_failed_avail:
                self._abstain("below_commit_floor", "doctor_availability", lang)
                return None
            # no doctor named: "what time do you open" is a clinic question, not a doctor's schedule

        # No rate/prep/availability cue at all -- try the FAQ topic table
        # before falling through to greeting/thanks/abstain. Deliberately
        # LAST among the "answerable" branches: every FAQ keyword phrase
        # is generic clinic language with no test/doctor cue word in it by
        # construction, so this cannot silently steal a
        # rate/prep/availability question -- those already returned above.
        faq_topic, faq_form, faq_score = self.catalogue.match(text, "faq", lang, FAQ_COMMIT_FLOOR)
        if faq_topic and faq_score >= FAQ_COMMIT_FLOOR:
            self._serve("clinic_faq", lang)
            logger.info("fast path: clinic_faq %r (%.2f) [%s] from %r", faq_topic, faq_score, lang, transcript)
            return FastPathResult("clinic_faq", _empty_slots(faq_topic=faq_topic), faq_score, matched_form=faq_form)

        # Pure greeting or thanks, with no entity and no question in it.
        if table.greeting_reply and table.any(text, table.greeting) and len(text.split()) <= 4:
            self._serve("smalltalk", lang)
            return FastPathResult("smalltalk", _empty_slots(), 1.0, direct_reply_bn=table.greeting_reply)
        if table.thanks_reply and table.any(text, table.thanks) and len(text.split()) <= 4:
            self._serve("smalltalk", lang)
            return FastPathResult("smalltalk", _empty_slots(), 1.0, direct_reply_bn=table.thanks_reply)

        self._abstain("no_cue_matched", lang=lang)
        return None

    def snapshot(self) -> dict:
        total = self.stats["served"] + self.stats["abstained"]
        return {
            **self.stats,
            "catalogue_rows": len(self.catalogue),
            "serve_rate": round(self.stats["served"] / total, 3) if total else 0.0,
            "language_gap": serve_rate_gap(dict(self.by_lang)),
            "abstention_reasons": abstentions.snapshot(),
            "served_by_intent": fast_path_served.snapshot(),
        }
