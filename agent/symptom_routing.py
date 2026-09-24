"""ADDED BY SOURAV -- "Caller describes symptoms and asks what is wrong"
story.

story title: Caller describes symptoms and asks what is wrong
user story: As a caller, I want to be routed to the right department
    without being diagnosed, so that the agent stays inside what a phone
    line can safely promise.
acceptance criteria: Symptom -> department routing uses a clinician-
    approved vocabulary and is deterministic/code-based, never something
    the model invents. The agent never states or implies a diagnosis. The
    spoken result is explicitly framed as administrative routing, not
    medical advice.

WHY THIS IS A SEPARATE, PRE-CLASSIFIER GUARD, STRUCTURED THE SAME WAY AS
agent/clinical_safety.py / agent/human_fast_path.py / agent/complaint_flow.py
/ agent/doctor_personal_request.py:

"deterministic/code-based rather than letting the LLM invent a department"
is the acceptance criterion, and a prompt instruction is not a guarantee --
it is advice a 7B model can and does ignore under the right phrasing
pressure (the exact reasoning every guard listed above already gives for
itself). So this module runs BEFORE the classifier -- checked in
main.py/main_pcm.py's _resolve_intent(), after every existing guard above
it, and BEFORE fast_path, the semantic cache, and Ollama. A department is
picked here or not at all; the model is never asked to name one, never
shown the vocabulary below, and never given the chance to fill a
"department" slot for this intent.

WHY THIS RUNS AFTER agent/clinical_safety.py, NOT BEFORE OR INSTEAD OF IT
(agent/clinical_safety.py ITSELF IS NOT MODIFIED BY THIS STORY):

A caller who names a symptom AND asks for a diagnosis or a danger judgement
("I have chest pain, am I having a heart attack?") must still be caught by
is_clinical_interpretation() -- that guard is checked FIRST in
_resolve_intent() and is left completely untouched here. This module only
ever runs for utterances that guard has already looked at and let through.
Two things keep that boundary honest even for phrasing
is_clinical_interpretation() itself does not happen to cover:

  1. Guard ORDER in main.py/main_pcm.py's _resolve_intent()/
     _continue_pending(): is_clinical_interpretation() is checked before
     this module is ever reached, full stop.
  2. _DIAGNOSIS_REQUEST_MARKERS below: a small, separate, self-contained
     list of diagnosis-seeking phrasings this module itself declines to
     act on, even when a known symptom phrase is ALSO present in the same
     utterance (e.g. "I have chest pain, what disease do I have?"). This
     is NOT a duplicate of agent/clinical_safety.py's phrase lists, and it
     does not try to answer or classify these utterances itself -- it only
     makes THIS module's own interception decision more conservative, so a
     genuine diagnosis request is never quietly answered with a department
     listing instead. An utterance excluded this way simply falls through
     to the ordinary classifier chain, the same as any other unmatched
     text -- which still has agent/clinical_safety.py's own (already-run)
     coverage behind it, plus extract_intent()'s "clinical_interpretation"
     intent as the same defense-in-depth second layer that guard's own
     module docstring describes for phrasing its literal phrase lists miss.

WHY THE VOCABULARY BELOW IS PROTOTYPE DATA, NOT A REAL POLICY:

The investigation for this story found no clinician-approved symptom ->
department vocabulary anywhere in this repository. SYMPTOM_DEPARTMENT_MAP
below is prototype/demo content, supplied to unblock engineering work on
the ROUTING MECHANISM, in the same spirit as clinic-api/seed.py's
WALKIN_POLICY/PRESCRIPTION_POLICY sample content -- it is explicitly
flagged here and must be replaced with the business's own clinically
reviewed mapping before this reaches a real caller. See that constant's
own docstring for specifics.

WHY THIS DOES NOT REUSE OR DUPLICATE clinic-api/match_band.py:

match_band.py lives in the clinic-api service and is reached by agent/
only indirectly, over HTTP, through agent/tools_client.py -- agent/ has
never imported clinic-api code directly (they are two separate
processes), and this guard must run synchronously, deterministically, and
before any network call at all, in the same tier as agent/clinical_safety.py
and agent/human_fast_path.py. It is also solving a different problem:
match_band.py ranks NEAR matches of a name the caller already gave (fuzzy
spelling of a department/doctor/test name); this module looks for a KNOWN
symptom phrase inside a longer sentence (containment, not edit-distance).
Once a department name is resolved here, the actual doctor lookup for
that department reuses the real thing -- clinic-api's own
_resolve_department()/match_band.py, via the pre-existing
agent/tools_client.py::get_doctors_by_department() -- so nothing about
department resolution or doctor lookup is duplicated, only "which
department does this symptom belong to" is new.

-------------------------------------------------------------------------
STORY 6 VALIDATION CLEANUP (this revision) -- three specific limitations
found during validation of the first cut of this module, fixed here:

1. DEPARTMENT ALIGNMENT. The first cut's prototype vocabulary carried
   three department names -- Ophthalmology, Gastroenterology, Neurology --
   that do not exist in clinic-api/seed.py's DEPARTMENTS at all. Rather
   than leave symptom phrases pointing at departments that can never
   resolve, or invent new departments/policy to justify keeping them (both
   explicitly out of scope), those three groups' phrases are folded into
   "General Medicine" below -- the one department this same table already
   used, before this change, as the generic/undifferentiated bucket
   ("fever", "weakness", "fatigue", "feeling unwell" -- symptoms with no
   single-organ specialist attached). Routing "no specialist for this in
   our department list" symptoms to the general physician is the ordinary
   front-desk default this clinic's own department list already implies
   by having a General Medicine department at all -- it is not a new
   clinical claim that, say, a headache belongs to General Medicine
   specifically, only that in the absence of a Neurology department this
   clinic can offer, a GP is the correct front-line contact. This is
   still prototype content requiring clinical sign-off (see the mapping's
   own docstring below) -- the fix removes an impossible destination, it
   does not manufacture a clinically-reviewed one.

2. BENGALI-SCRIPT COVERAGE. The first cut had English, Hindi/Hinglish and
   Bengali/Banglish (romanised) phrases only -- no Bengali script, unlike
   every other multilingual vocabulary in this codebase. Bengali-script
   phrases are now included for every department below, checked with the
   same word-boundary-safe helper (`_bn_bounded`, imported from
   agent/slot_parse.py) agent/clinical_safety.py's own Bengali phrase set
   already uses, and for the identical documented reason: Bengali vowel
   signs and the nukta are combining marks Python's `\b` does not treat as
   word characters, so a bare substring check could otherwise fire inside
   an unrelated longer word. Latin-script (English/Hinglish/Banglish)
   phrases keep the original plain substring check -- unchanged, and
   still appropriate for that script per this module's original reasoning
   below.

3. FALSE-POSITIVE MATCHING. The first cut's General Medicine list included
   a bare "cold" -- a common English word with no realistic connection to
   illness in most of the sentences it appears in ("cold water", "cold
   weather", "cold drinks"). That single entry (and its Hindi/Bengali
   equivalents, "sardi hai"/"sordi hoyeche", which carry the exact same
   ambiguity -- both languages use the same word for "a cold" the illness
   and "cold" the temperature) is replaced below with a set of specific,
   illness-context phrases ("have a cold", "having a cold", "had a cold",
   "mujhe sardi hai", "amar sordi hoyeche", etc.) that a bare weather or
   drink remark does not contain. No new matching logic was needed for
   this -- the existing substring-containment matcher is untouched; only
   the overly-broad phrase itself was replaced with safer, equally
   deterministic phrases, which is the smallest change that closes the
   false-positive gap without reaching for a classifier.
-------------------------------------------------------------------------
"""
from __future__ import annotations

from agent.slot_parse import _bn_bounded

COMMIT = "commit"
NONE = "none"


def _is_bengali_script(phrase: str) -> bool:
    """True if `phrase` contains at least one Bengali-script character (the
    Bengali Unicode block, U+0980-U+09FF). Used only to pick which of the
    two matching strategies below applies to a given vocabulary entry --
    see resolve_symptom_department()'s own docstring."""
    return any("ঀ" <= ch <= "৿" for ch in phrase)


# =============================================================================
# SYMPTOM_DEPARTMENT_MAP -- PROTOTYPE / DEMO DATA. REQUIRES CLINICAL REVIEW
# BEFORE PRODUCTION USE.
#
# Languages covered: English, Hindi/Hinglish, Bengali/Banglish (romanised),
# and Bengali script.
#
# Every key below is one of clinic-api/seed.py's own seeded DEPARTMENTS
# names, verbatim (General Medicine, Cardiology, ENT, Orthopaedics,
# Dermatology -- the five that have any symptom vocabulary at all;
# Gynaecology & Obstetrics, Paediatrics and Diabetology & Endocrinology are
# in that same seeded catalogue but carry no symptom phrases here, which is
# a coverage gap in this prototype vocabulary, not a bug in this module).
# tests/test_symptom_routing.py::TestDepartmentAlignment asserts this
# every-key-is-a-real-department property directly against seed.py's own
# DEPARTMENTS literal, so this cannot silently drift again.
#
# "Orthopedics" (as originally supplied) is keyed here by the clinic's own
# canonical spelling, "Orthopaedics" (clinic-api/seed.py's DEPARTMENTS
# key), rather than left as supplied -- both spellings already resolve to
# the same seeded department via clinic-api's DEPARTMENT_ALIASES
# ("orthopedics" is a listed English alias), but keying it by the
# canonical name here means this module's own resolution does not depend
# on that alias/fuzzy-match behaviour to get the right answer for
# something fully within this story's own control.
#
# WHAT WAS REMOVED AND WHY (see this module's own docstring, section "1.
# DEPARTMENT ALIGNMENT" above, for the full reasoning): the original
# prototype carried three additional groups -- "Ophthalmology" (eye pain,
# blurred vision, ...), "Gastroenterology" (stomach pain, acidity,
# vomiting, ...), "Neurology" (headache, migraine, dizziness, ...) -- none
# of which are departments this clinic's seeded catalogue has. Their
# phrases have been folded into "General Medicine" below (each group kept
# together and commented for traceability) rather than dropped, invented
# as new departments, or left pointing at a destination that can never
# resolve.
#
# KNOWN DATA-QUALITY LIMITATION, FLAGGED NOT FIXED: unlike
# agent/clinical_safety.py's deliberately multi-word-or-distinctive phrase
# design, a handful of entries below remain short, generic words in a
# caller's own language ("fever", "rash", "headache") that carry some
# residual false-positive risk in an unusual sentence, even though they
# are not the specific "cold weather"/"cold water" ambiguity this revision
# closes (see section "3. FALSE-POSITIVE MATCHING" above for exactly which
# entry that was and why "cold" specifically was singled out: it is a
# common English/Hindi/Bengali word for a temperature description AND an
# illness, which none of "fever", "rash", "headache" etc. are). Narrowing
# these further is a judgement call for whoever clinically reviews this
# table, not something this technical cleanup pass rewrites on its own
# authority.
# =============================================================================
SYMPTOM_DEPARTMENT_MAP: dict[str, list[str]] = {
    "Dermatology": [
        # English
        "skin rash", "rash", "itchy skin", "skin itching", "acne", "pimples",
        "skin allergy", "hair loss", "dandruff",
        # Hindi / Hinglish
        "skin pe rash hai", "skin mein khujli", "skin ki itching",
        "chehre pe pimples", "skin allergy hai", "baal jhad rahe hain",
        "dandruff hai",
        # Bengali / Banglish (romanised)
        "gaye rash hoyeche", "gaye chulkani", "skin e chulkani",
        "mukhe pimples", "skin allergy hoyeche", "chul pore jacche",
        "dandruff hoyeche",
        # Bengali script
        "গায়ে র‍্যাশ হয়েছে", "গায়ে চুলকানি", "স্কিনে চুলকানি",
        "মুখে ব্রণ", "স্কিন অ্যালার্জি হয়েছে", "চুল পড়ে যাচ্ছে",
        "খুশকি হয়েছে",
    ],

    "Cardiology": [
        # English
        "chest pain", "heart pain", "palpitations", "fast heartbeat",
        "irregular heartbeat", "shortness of breath",
        # Hindi / Hinglish
        "seene mein dard", "chest mein pain", "dil mein dard",
        "dil tez dhadak raha hai", "heartbeat fast hai",
        "dhadkan irregular hai", "saans lene mein dikkat",
        # Bengali / Banglish (romanised)
        "buk e betha", "buke byatha", "heart e betha", "buk dhorfor korche",
        "heart beat khub fast", "heartbeat irregular",
        "shash nite osubidha",
        # Bengali script
        "বুকে ব্যথা", "হৃদয়ে ব্যথা", "বুক ধড়ফড় করছে",
        "হার্টবিট খুব দ্রুত", "হার্টবিট অনিয়মিত",
        "শ্বাস নিতে অসুবিধা",
    ],

    "ENT": [
        # English
        "ear pain", "earache", "hearing problem", "sore throat",
        "throat pain", "blocked nose", "runny nose", "sinus problem",
        "voice problem",
        # Hindi / Hinglish
        "kaan mein dard", "sunai dene mein problem", "gale mein dard",
        "gala kharab hai", "naak band hai", "naak beh rahi hai",
        "sinus problem hai", "awaaz mein problem",
        # Bengali / Banglish (romanised)
        "kane betha", "shunte osubidha", "gola betha", "gola kharap",
        "nak bondho", "nak diye jol porche", "sinus er problem",
        "golar shor e problem",
        # Bengali script
        "কানে ব্যথা", "শুনতে অসুবিধা", "গলা ব্যথা", "গলা খারাপ",
        "নাক বন্ধ", "নাক দিয়ে জল পড়ছে", "সাইনাসের সমস্যা",
        "গলার স্বরে সমস্যা",
    ],

    # Canonical clinic-api spelling ("Orthopaedics") -- see this
    # constant's own docstring above for why this was renamed from the
    # supplied "Orthopedics".
    "Orthopaedics": [
        # English
        "back pain", "knee pain", "joint pain", "shoulder pain",
        "neck pain", "bone pain", "muscle pain", "ankle pain",
        # Hindi / Hinglish
        "kamar mein dard", "ghutne mein dard", "jodon mein dard",
        "kandhe mein dard", "gardan mein dard", "haddi mein dard",
        "muscle mein dard", "pair ke takhne mein dard",
        # Bengali / Banglish (romanised)
        "komor e betha", "hatu te betha", "joint e betha", "kandhe betha",
        "golar pechone betha", "har e betha", "muscle e betha",
        "gorali te betha",
        # Bengali script
        "কোমরে ব্যথা", "হাঁটুতে ব্যথা", "জয়েন্টে ব্যথা",
        "কাঁধে ব্যথা", "ঘাড়ে ব্যথা", "হাড়ে ব্যথা",
        "মাংসপেশিতে ব্যথা", "গোড়ালিতে ব্যথা",
    ],

    "General Medicine": [
        # -------------------------- original General Medicine vocabulary
        # English (bare "cold" removed -- see section 3 of this module's
        # docstring above; replaced by the illness-context phrases in the
        # "cold -- false-positive fix" block below).
        "fever", "weakness", "fatigue", "body pain",
        "general weakness", "feeling unwell", "not feeling well",
        # Hindi / Hinglish (bare "sardi hai" removed, same reason)
        "bukhar hai", "kamzori hai", "thakaan ho rahi hai",
        "poore badan mein dard", "bahut weakness hai",
        "tabiyat kharab hai", "theek nahi lag raha",
        # Bengali / Banglish, romanised (bare "sordi hoyeche" removed,
        # same reason)
        "jor hoyeche", "durbol lagche", "khub klanto lagche",
        "shorir e betha", "khub weakness lagche", "shorir kharap",
        "bhalo lagche na",
        # Bengali script
        "জ্বর হয়েছে", "দুর্বল লাগছে", "খুব ক্লান্ত লাগছে",
        "শরীরে ব্যথা", "শরীর খুব দুর্বল লাগছে", "শরীর খারাপ",
        "ভালো লাগছে না",

        # -------------------------- "cold" -- false-positive fix (section
        # 3 of this module's docstring above). Each phrase below requires
        # an illness-context word ("have"/"had"/"got"/"mujhe"/"amar" etc.)
        # around "cold" so a bare temperature or drink remark ("cold
        # weather", "cold water", "cold drinks") never matches, while an
        # actual "I have a cold" / "I've had a cold since yesterday" still
        # does.
        "have a cold", "having a cold", "have cold", "had a cold",
        "got a cold", "down with a cold", "bad cold", "cold since",
        "suffering from a cold",
        "mujhe sardi hai", "mujhe sardi ho gayi hai",
        "mujhe sardi lag gayi hai",
        "amar sordi hoyeche",
        "আমার সর্দি হয়েছে",

        # -------------------------- folded from the removed "Ophthalmology"
        # group (not a seeded department -- see this module's own docstring,
        # section "1. DEPARTMENT ALIGNMENT" above).
        "eye pain", "eye redness", "red eye", "blurred vision",
        "eye irritation", "watery eyes", "vision problem",
        "aankh mein dard", "aankh lal hai", "aankh se paani aa raha hai",
        "dhundhla dikh raha hai", "aankh mein jalan",
        "dekhne mein problem",
        "chokh e betha", "chokh lal hoyeche", "chokh diye jol porche",
        "jhaposha dekhchi", "chokh e jwala", "dekhte osubidha hocche",
        "চোখে ব্যথা", "চোখ লাল হয়েছে", "চোখ দিয়ে জল পড়ছে",
        "ঝাপসা দেখছি", "চোখে জ্বালা", "দেখতে অসুবিধা হচ্ছে",

        # -------------------------- folded from the removed
        # "Gastroenterology" group (not a seeded department -- see this
        # module's own docstring, section "1. DEPARTMENT ALIGNMENT" above).
        "stomach pain", "abdominal pain", "acidity", "heartburn",
        "indigestion", "vomiting", "diarrhea", "constipation",
        "gastric problem",
        "pet mein dard", "pet dard", "acidity hai", "seene mein jalan",
        "khana hazam nahi ho raha", "ulti ho rahi hai", "dast ho raha hai",
        "kabz hai", "gastric problem hai",
        "pete betha", "pet e byatha", "acidity hocche", "buk jwala",
        "khabar hojom hocche na", "bomi hocche", "patla paykhana hocche",
        "koshtokathinyo", "gastric er problem",
        "পেটে ব্যথা", "অ্যাসিডিটি হচ্ছে", "বুক জ্বালা",
        "খাবার হজম হচ্ছে না", "বমি হচ্ছে", "পাতলা পায়খানা হচ্ছে",
        "কষ্টকাঠিন্য", "গ্যাস্ট্রিকের সমস্যা",

        # -------------------------- folded from the removed "Neurology"
        # group (not a seeded department -- see this module's own docstring,
        # section "1. DEPARTMENT ALIGNMENT" above).
        "headache", "migraine", "dizziness", "numbness", "tingling",
        "memory problem", "trembling", "frequent headaches",
        "sar mein dard", "migraine hai", "chakkar aa raha hai",
        "haath pair sunn ho raha hai", "jhanjhanahat ho rahi hai",
        "memory problem hai", "haath kaanp raha hai",
        "baar baar sir dard",
        "matha betha", "migraine ache", "matha ghurchhe",
        "hat pa jhim jhim korche", "obosh lagche", "memory problem hocche",
        "hat kapche", "bar bar matha betha",
        "মাথা ব্যথা", "মাইগ্রেন আছে", "মাথা ঘুরছে",
        "হাত পা ঝিমঝিম করছে", "অবশ লাগছে", "মেমোরি সমস্যা হচ্ছে",
        "হাত কাঁপছে", "বার বার মাথা ব্যথা",
    ],
}


# ADDED BY SOURAV -- see this module's own docstring above ("2." in the
# boundary-with-clinical_safety explanation) for exactly why this exists
# and why it is deliberately NOT a duplicate of agent/clinical_safety.py's
# own phrase lists: this only makes THIS module decline to act, it never
# classifies or answers anything itself. Bengali-script entries were added
# in this revision alongside SYMPTOM_DEPARTMENT_MAP's own Bengali-script
# coverage, for the same reason: adding a language to the vocabulary this
# module matches against without adding it here too would quietly weaken
# this boundary for that language only.
_DIAGNOSIS_REQUEST_MARKERS_LATIN = (
    "what disease", "which disease", "what illness", "which illness",
    "what condition do i have", "which condition do i have",
    "diagnose me", "diagnose this",
    "kaunsi bimari", "kis bimari hai", "kaun sa rog", "ki rog hoyeche",
    "ki bemari hoyeche",
)
_DIAGNOSIS_REQUEST_MARKERS_BENGALI = (
    "কি রোগ হয়েছে", "কোন রোগ", "কী অসুখ", "আমার কি রোগ",
)


def resolve_symptom_department(text: str | None) -> tuple[str, str | None]:
    """-> (verdict, the one department name to route to, or None).

    COMMIT is returned only when the utterance contains a phrase from
    exactly ONE department's vocabulary above, and none of the diagnosis-
    request markers. Two or more different departments' phrases in the
    same utterance, a diagnosis-request marker anywhere in it, or no
    phrase at all, all return NONE, None -- this module never guesses
    between candidates and never overrides a caller who is asking to be
    diagnosed rather than routed. A NONE verdict is not a refusal of any
    kind; it simply means this deterministic guard has nothing safe to
    say, and the utterance falls through to whatever the ordinary
    classifier chain already does with it.

    Latin-script phrases (English/Hinglish/Banglish) are checked as a
    lowercased substring match, the same "already a multi-word phrase (or
    a distinctive single word) with no realistic false-positive risk"
    treatment agent/clinical_safety.py's own English/Latin-script phrase
    sets use. Bengali-script phrases are checked with `_bn_bounded`
    (shared from agent/slot_parse.py) instead -- the same word-boundary-
    safe helper agent/clinical_safety.py's own Bengali set already uses,
    and for the identical reason: Bengali vowel signs and the nukta are
    combining marks Python's `\\b` does not treat as word characters, so a
    bare substring check could otherwise fire inside an unrelated longer
    word. See this module's own docstring for the one place the "no
    realistic false-positive risk" reasoning was found to be wrong for
    this particular prototype vocabulary (the "cold" fix), and for a
    remaining, flagged-not-fixed limitation (a handful of other short,
    generic entries).
    """
    if not text or not text.strip():
        return NONE, None

    lowered = text.lower()

    for marker in _DIAGNOSIS_REQUEST_MARKERS_LATIN:
        if marker in lowered:
            return NONE, None
    for marker in _DIAGNOSIS_REQUEST_MARKERS_BENGALI:
        if _bn_bounded(marker, text):
            return NONE, None

    matched_departments: list[str] = []
    for department, phrases in SYMPTOM_DEPARTMENT_MAP.items():
        for phrase in phrases:
            if _is_bengali_script(phrase):
                hit = _bn_bounded(phrase, text)
            else:
                hit = phrase in lowered
            if hit:
                matched_departments.append(department)
                break  # one hit is enough to count this department as matched

    # dict.fromkeys() dedupes while keeping first-seen order -- irrelevant
    # for the COMMIT case (exactly one entry either way) but keeps the
    # NONE-vs-multiple-candidates check below simple and deterministic.
    unique_departments = list(dict.fromkeys(matched_departments))

    if len(unique_departments) == 1:
        return COMMIT, unique_departments[0]

    # Zero matches, or two-or-more different departments named in the same
    # utterance -- never guessed between. See this function's own
    # docstring above.
    return NONE, None
