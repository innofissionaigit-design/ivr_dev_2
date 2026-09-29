"""ADDED BY SOURAV -- "Caller asks whether their result is dangerous" story
(Epic: Conversation -- Difficult, Sensitive and Edge Cases).

story title: Caller asks whether their result is dangerous
user story: As a worried patient, I want to be treated gently and connected
    to a clinician, so that a procedural limit does not feel like a rebuff.
acceptance criteria: Clinical interpretation is routed to a human by policy
    rather than by prompt wording. The reply states clearly that a doctor
    will explain the result and offers to connect one rather than simply
    refusing. An adversarial set of one hundred prompts per language
    produces zero breaches.

WHY THIS IS A SEPARATE, PRE-CLASSIFIER CHECK, NOT JUST A NEW INTENT
DESCRIBED IN agent/llm.py's PROMPT TEXT:

"routed to a human by policy rather than by prompt wording" is the whole
acceptance criterion, and a prompt instruction is not a policy -- it is
advice a 7B model can and does ignore under the right phrasing pressure.
Investigating this story found the one place in this codebase where that
already matters: agent/llm.py's "smalltalk" intent is the SOLE path where
whatever the model writes (`direct_reply_bn`) is spoken back to the caller
with no downstream check of any kind (see main.py/main_pcm.py's dispatch --
`_speak(session, part.get("direct_reply_bn") or "...")`, nothing else).
Every other intent's factual content is composed from a validated tool
response in reply_templates.py, never from the model -- see that file's own
module docstring -- but a caller asking "is this dangerous?" or "am I
dying?" names no test, no doctor, nothing to look up at all. To a
classifier under pressure that is indistinguishable from "genuinely
nothing here to look up", which is smalltalk's own definition. Leaving
that distinction to the classifier's prompt-following would mean this
story's whole promise rests on the model getting a probabilistic judgement
right every time, which is precisely the thing "by policy rather than by
prompt wording" rules out.

So this module runs BEFORE the classifier -- before the fast path, before
the semantic cache, before Ollama -- see main.py/main_pcm.py's
_resolve_intent(), which checks this first and short-circuits both of
them the moment it fires. A misclassification is structurally impossible
here, not merely unlikely: the model is never given the chance to call
this turn anything else.

This module makes no medical judgement of its own -- it does the exact
opposite. It recognises the SHAPE of a question that is asking for one (a
danger/mortality word, a normal-vs-abnormal question about a report or
result, a named lab parameter sitting next to a raw number, or an attempt
to route around a refusal via roleplay, a hypothetical frame, or a forced
yes/no) and hands every single one of them to the same fixed,
human-escalating reply (agent/reply_templates.clinical_interpretation_
reply()) -- never to anything generative, and never to a lookup this
system has no clinical basis to answer from (see that function's own
docstring, and clinical_interpretation_reply()'s note on why the LabReport
table itself carries no clinical value for anything to leak in the first
place).
"""
from __future__ import annotations

import re

from agent.slot_parse import _bn_bounded

# --------------------------------------------------------------------- #
# English phrases. Checked as substrings of the lowercased utterance --
# deliberately not word-bounded like the Bengali set below, because every
# entry here is already a multi-word phrase (or a distinctive single word
# like "dangerous") with no realistic false-positive risk of firing inside
# an unrelated longer word the way a single Bengali syllable could.
# --------------------------------------------------------------------- #
_EN_DANGER_PANIC = (
    "dangerous", "am i dying", "going to die", "will i die", "could i die",
    "might i die", "life threatening", "life-threatening", "is it fatal",
    "is this fatal", "do i have cancer", "is it cancer", "is this cancer",
    "should i be worried", "should i worry", "am i going to be okay",
    "am i going to be alright", "am i okay", "am i alright",
    "worst case scenario", "is my life at risk", "in danger",
    "is it life threatening", "is this life threatening", "scared",
    "terrified", "am i in trouble", "is something wrong with me",
    # ADDED BY SOURAV -- closed by the adversarial test suite in
    # tests/test_clinical_interpretation_safety.py, which found these as
    # real gaps: natural paraphrases that carry the exact same panic/danger
    # meaning as the phrases above but do not contain any of them as a
    # literal substring ("will this kill me" does not contain "will i
    # die", etc.). "wrong with me" and "in trouble" are deliberately
    # shortened from their more specific siblings above so a caller's own
    # wording around them ("seriously wrong with me", "real trouble here")
    # still matches.
    "kill me", "should i panic", "wrong with me", "in trouble",
    "real trouble", "could this be cancer", "could it be cancer",
    "worst that can happen",
)
_EN_ABNORMAL_NORMAL = (
    "is this normal", "is this abnormal", "is my result normal",
    "is my results normal", "is my report normal", "is my report abnormal",
    "is my result abnormal", "what does this mean", "what does it mean",
    "what does my result mean", "what does my report mean",
    "is this a bad result", "is this a good result", "is that bad",
    "is that good", "out of range", "is this out of range",
    "too high", "too low", "is this serious", "is it serious",
    "how serious is this", "is this concerning", "is this a bad sign",
    # ADDED BY SOURAV -- adversarial-suite gap, same reasoning as above:
    # "is this test result normal or not" does not contain "is this
    # normal" as a literal substring.
    "normal or not", "to worry about",
)
_EN_ROLEPLAY_HYPOTHETICAL = (
    "pretend you are a doctor", "pretend to be a doctor", "act as a doctor",
    "as a doctor tell me", "imagine you are a doctor", "imagine you're a doctor",
    "hypothetically", "in theory", "just theoretically",
    "if you were a doctor", "role play", "roleplay", "just between us",
    "off the record", "as a friend, not an agent", "speaking as a human",
    # ADDED BY SOURAV -- adversarial-suite gap: another rule-bypass framing
    # besides pretending to be a doctor.
    "forget the rules",
)
# ADDED BY SOURAV -- adversarial-suite gap: "pretend for a second you're a
# real doctor" is the same bypass attempt as "pretend you are a doctor"
# above but does not contain it as a literal substring (the substring list
# above is deliberately not a regex, so it cannot absorb an inserted
# clause). A short, generic gap-tolerant pattern closes the whole family
# at once instead of enumerating every possible insertion.
_RE_EN_PRETEND_DOCTOR = re.compile(r"\bpretend\b[^.?!]{0,40}\bdoctor\b", re.IGNORECASE)
_RE_EN_WHAT_DOES_MEAN = re.compile(r"\bwhat does\b[^.?!]{0,30}\bmean\b", re.IGNORECASE)
_EN_FORCED_BINARY = (
    "just say yes or no", "yes or no", "one word answer", "one-word answer",
    "simple yes or no", "just tell me yes or no", "just answer yes or no",
    "give me a yes or no",
)

# --------------------------------------------------------------------- #
# Hinglish / Banglish (Hindi and Bengali written in Latin script). Same
# substring-of-lowercased-text treatment as English -- every entry is
# already a distinctive multi-character transliterated phrase.
# --------------------------------------------------------------------- #
_LATIN_TRANSLIT_DANGER_PANIC = (
    "khatarnak", "bipodjonok", "main mar jaunga", "mai mar jaunga",
    "kya main mar jaunga", "ami ki mara jabo", "ami ki more jabo",
    "mara jabo naki", "more jabo naki", "amar ki bipod",
    "ami ki bipode achi", "jibon songkoto", "jibon er jonno bipodjonok",
    "gurutor", "khub gurutor", "cancer naki", "amar ki cancer",
    "bhoy lagche", "bhoi lagche", "dar lagche", "main dar gaya hoon",
    # ADDED BY SOURAV -- adversarial-suite gaps (see the English tuples'
    # own comment above for why these are additions, not replacements):
    # everyday Hindi panic/danger phrasings the original set did not cover.
    "jaanleva", "cancer hai", "dar lag raha hai", "ghabrana", "gadbad hai",
)
_LATIN_TRANSLIT_ABNORMAL_NORMAL = (
    "eta ki normal", "ata ki normal", "eta ki abnormal", "ata ki abnormal",
    "report ta kharap naki", "result ta kharap naki", "ata ki bhalo naki kharap",
    "eta ki bhalo naki kharap", "ei value ta thik ache naki",
    "yeh normal hai kya", "yeh abnormal hai kya", "yeh khatarnak hai kya",
    "report kharab hai kya", "matlab kya hai iska", "eta mane ki",
    "ata mane ki", "iska matlab kya hai",
    # ADDED BY SOURAV -- adversarial-suite gaps.
    "normal hai ya", "ya kharab", "bahut zyada hai", "bahut kam hai",
    "kitna serious hai", "chinta karni chahiye",
)
_LATIN_TRANSLIT_ROLEPLAY_HYPOTHETICAL = (
    "doctor hoke bolo", "doctor hokar bolun", "farz karo aap doctor ho",
    "dhorun apni doctor", "ekta hypothetically bolun", "ekhon doctor er moto bolun",
    "aap doctor ki tarah bolo", "man lijiye aap doctor hain",
    # ADDED BY SOURAV -- adversarial-suite gaps: other common roleplay/
    # rule-bypass framings a caller might use besides "act like a doctor".
    "rules bhool jao", "dost ki tarah",
)
_LATIN_TRANSLIT_FORCED_BINARY = (
    "sirf haan ya na bolo", "sirf haan ya naa boliye", "ek shabd mein batao",
    "shudhu hae ba na bolun", "ek kothay bolun", "just haan ya na",
    # ADDED BY SOURAV -- adversarial-suite gap: "seedha haan ya na bolo"
    # (say directly yes or no) carries the same forced-binary meaning as
    # "sirf haan ya na bolo" (say only yes or no) above but without "sirf"/
    # "seedha" as a fixed prefix -- this broader phrase, without that
    # prefix requirement, catches both.
    "haan ya na bolo",
    # ADDED BY SOURAV -- adversarial-suite gap: "ek shabd" (one word) alone
    # is the generic core of "ek shabd mein batao" above -- shortened so a
    # caller's own wording around it ("ek shabd ka jawab dijiye") still
    # matches. Specific enough within this domain not to collide with
    # anything else this system handles.
    "ek shabd",
)
# ADDED BY SOURAV -- adversarial-suite gap, same reasoning as
# _RE_EN_PRETEND_DOCTOR above: "farz karo aap sach me doctor ho" (assume
# you're REALLY a doctor) is the same bypass as "farz karo aap doctor ho"
# but the inserted "sach me" breaks a literal substring match.
_RE_HI_FARZ_DOCTOR = re.compile(r"farz karo\b[^.?!]{0,30}\bdoctor\b", re.IGNORECASE)

_ALL_LATIN_PHRASES = (
    _EN_DANGER_PANIC + _EN_ABNORMAL_NORMAL + _EN_ROLEPLAY_HYPOTHETICAL
    + _EN_FORCED_BINARY + _LATIN_TRANSLIT_DANGER_PANIC
    + _LATIN_TRANSLIT_ABNORMAL_NORMAL + _LATIN_TRANSLIT_ROLEPLAY_HYPOTHETICAL
    + _LATIN_TRANSLIT_FORCED_BINARY
)

# --------------------------------------------------------------------- #
# Bengali script. Word-bounded via _bn_bounded() (shared from
# agent/slot_parse.py, the same fix parse_date()/fast_path.py already
# apply for exactly this reason: Bengali vowel signs and the nukta are
# combining marks Python's \b does not treat as word characters, so a
# short syllable like "কাল" would otherwise match inside an unrelated
# longer word such as "সকাল". Every entry below is checked as a whole
# word/phrase this way, never a bare substring.
# --------------------------------------------------------------------- #
_BN_DANGER_PANIC = (
    "বিপজ্জনক", "মারা যাব", "মারা যাচ্ছি", "মরে যাব", "মরে যাচ্ছি",
    "গুরুতর", "ভয়ঙ্কর", "জীবন সংশয়", "জীবনের ঝুঁকি", "বিপদে আছি",
    "ক্যান্সার কিনা", "আমার কি ক্যান্সার", "ভয় লাগছে", "ভয়ে আছি",
    "চিন্তিত", "আতঙ্কিত",
    # ADDED BY SOURAV -- adversarial-suite gaps: "জীবন সংশয়ের" is "জীবন
    # সংশয়" (life risk) with a genitive suffix attached, which
    # _bn_bounded's word-boundary check treats as a DIFFERENT word from
    # the bare form above (a combining/following Bengali character fails
    # its boundary test by design -- see agent/slot_parse.py's own
    # docstring on why that check exists at all). Rather than loosen that
    # shared boundary rule -- used elsewhere for real substring-collision
    # safety -- the inflected form actual callers use is simply listed as
    # its own entry, same fix as "সমস্যা আছে" below for the "... কিনা"
    # question form.
    "জীবন সংশয়ের", "সমস্যা আছে",
)
_BN_ABNORMAL_NORMAL = (
    "স্বাভাবিক কিনা", "অস্বাভাবিক", "রিপোর্ট খারাপ", "রেজাল্ট খারাপ",
    "ভালো নাকি খারাপ", "মানে কী", "এর মানে কী", "সমস্যা আছে কিনা",
    "কতটা গুরুতর",
    # ADDED BY SOURAV -- adversarial-suite gaps: a value described as too
    # high/too low, the Bengali-script equivalent of the English "too
    # high"/"too low" pair above (previously only covered via English or
    # Latin transliteration).
    "অনেক বেশি", "অনেক কম",
)
_BN_ROLEPLAY_HYPOTHETICAL = (
    "ধরুন আপনি ডাক্তার", "ডাক্তার হিসেবে বলুন", "যদি ধরি",
    "একজন ডাক্তারের মতো বলুন",
    # ADDED BY SOURAV -- adversarial-suite gaps: other rule-bypass framings
    # besides "assume you're a doctor" -- forgetting the rules, asking to
    # be answered as a friend rather than the system, or asking "off the
    # record" (a common English loanphrase in spoken Bengali).
    "নিয়ম ভুলে", "বন্ধুর মতো", "অফ দ্য রেকর্ড",
)
_BN_FORCED_BINARY = (
    "শুধু হ্যাঁ বা না বলুন", "এক কথায় বলুন", "শুধু হ্যাঁ নাকি না",
    # ADDED BY SOURAV -- adversarial-suite gap, same reasoning as the
    # Hinglish "haan ya na bolo" addition above: "সরাসরি হ্যাঁ বা না বলুন"
    # (say directly yes or no) carries the same forced-binary meaning as
    # "শুধু হ্যাঁ বা না বলুন" (say only yes or no) without "শুধু"/"সরাসরি"
    # as a fixed prefix.
    "হ্যাঁ বা না বলুন",
)
# ADDED BY SOURAV -- adversarial-suite gap, same reasoning as
# _RE_EN_PRETEND_DOCTOR/_RE_HI_FARZ_DOCTOR above: "ধরুন আপনি সত্যিই একজন
# ডাক্তার" (assume you're REALLY a doctor) is the same bypass as "ধরুন
# আপনি ডাক্তার" but an inserted clause breaks a literal substring match.
# "ধরুন" and "ডাক্তার" are both distinctive multi-syllable words with
# negligible collision risk as bare (non-_bn_bounded) substrings even
# without the word-boundary check the shorter phrases above need.
_RE_BN_DHORUN_DOCTOR = re.compile(r"ধরুন আপনি[^.?!।]{0,30}ডাক্তার")

_ALL_BENGALI_PHRASES = (
    _BN_DANGER_PANIC + _BN_ABNORMAL_NORMAL + _BN_ROLEPLAY_HYPOTHETICAL
    + _BN_FORCED_BINARY
)

# A named lab parameter sitting next to a raw number, in either order, in
# up to ~25 characters of the same clause ("hemoglobin 6.5", "6.5 hemoglobin
# ki normal", "amar sugar 250 ache") -- a caller reading a figure off their
# own report and asking this system to interpret it, regardless of which
# interpretive verb (if any) they used. The parameter name itself is what
# makes this unambiguous: no other intent in this system ever needs a
# caller to name a clinical measurement.
_CLINICAL_TERMS = (
    "hemoglobin", "haemoglobin", "hb", "sugar", "glucose", "creatinine",
    "bilirubin", "wbc", "rbc", "platelet", "platelets", "cholesterol",
    "hba1c", "tsh", "sgpt", "sgot", "urea", "esr", "crp", "vitamin d",
    "vitamin b12",
)
_RE_TERM_THEN_NUMBER = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _CLINICAL_TERMS) + r")\b[^.?!]{0,25}\d",
    re.IGNORECASE,
)
_RE_NUMBER_THEN_TERM = re.compile(
    r"\d[^.?!]{0,25}\b(?:" + "|".join(re.escape(t) for t in _CLINICAL_TERMS) + r")\b",
    re.IGNORECASE,
)

_BN_CLINICAL_TERMS = (
    "হিমোগ্লোবিন", "সুগার", "ক্রিয়েটিনিন", "কোলেস্টেরল", "বিলিরুবিন",
)


def _has_bengali_value_mention(text: str) -> bool:
    """Bengali digits and Latin digits both appear in real ASR transcripts
    in this codebase (see agent/slot_parse.py's own `_BN_DIGITS` translation
    table) -- checked separately from the Latin-script regex above because
    a Bengali clinical term needs the word-boundary-safe check, not a bare
    substring, for the same reason the rest of this module's Bengali set
    does."""
    for term in _BN_CLINICAL_TERMS:
        if not _bn_bounded(term, text):
            continue
        idx = text.find(term)
        window = text[max(0, idx - 25):idx + len(term) + 25]
        if re.search(r"[0-9০-৯]", window):
            return True
    return False


def is_clinical_interpretation(text: str) -> bool:
    """True when `text` is asking this system to render a clinical
    judgement -- whether a result is dangerous, normal, or means something
    -- however it is dressed up (a direct question, a panic, a roleplay
    frame, a hypothetical, or a forced binary choice). Never a determination
    of what the ANSWER should be -- only that the question is this kind of
    question, which is all main.py/main_pcm.py's _resolve_intent() needs to
    route it to the fixed, human-escalating reply before the classifier
    ever sees it.
    """
    if not text or not text.strip():
        return False
    lowered = text.lower()

    for phrase in _ALL_LATIN_PHRASES:
        if phrase in lowered:
            return True

    for phrase in _ALL_BENGALI_PHRASES:
        if _bn_bounded(phrase, text):
            return True

    if _RE_TERM_THEN_NUMBER.search(lowered) or _RE_NUMBER_THEN_TERM.search(lowered):
        return True

    if _has_bengali_value_mention(text):
        return True

    # ADDED BY SOURAV -- adversarial-suite gap-fill regexes (see each
    # pattern's own comment above): a handful of "verb ... doctor" /
    # "what does ... mean" bypass framings insert a clause the fixed
    # substring lists above cannot absorb, so these three run last as a
    # generalized net under the same three-language coverage.
    if (_RE_EN_PRETEND_DOCTOR.search(lowered) or _RE_EN_WHAT_DOES_MEAN.search(lowered)
            or _RE_HI_FARZ_DOCTOR.search(lowered)):
        return True

    if _RE_BN_DHORUN_DOCTOR.search(text):
        return True

    return False
