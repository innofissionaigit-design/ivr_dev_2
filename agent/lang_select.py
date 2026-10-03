"""Choosing the language of a turn when language ID alone is not enough.

Measured on the pod (docs/adr/0002 records the numbers): SpeechBrain
VoxLingua107 separates Bengali from Hindi and Hindi from Bengali well, but
Indian-accented ENGLISH is routinely labelled Hindi, at high confidence. A
confident-looking wrong answer is worse than an unsure one: the router
commits to it and the caller is answered in the wrong language, so the
decision cannot rest on LID's top label alone.

The pieces here are deliberately small and pure so they are testable
without a GPU:

  script_share   what fraction of a transcript is in a language's own script
  pick_candidate choose between two ASR results for the same audio
  speakable      can a language's TTS voice actually say this text
"""

from __future__ import annotations

from typing import Protocol

from agent.speech_norm import unspeakable_spans

_SCRIPT_RANGE = {"bn": ("ঀ", "৿"), "hi": ("ऀ", "ॿ")}
_WRONG_SCRIPT = {"bn": ("ऀ", "ॿ"), "hi": ("ঀ", "৿")}
_SHARED_PUNCTUATION = frozenset(chr(0x0964) + chr(0x0965))


def script_share(text: str, lang: str) -> float:
    """Fraction of the letters in `text` written in `lang`'s own script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if lang == "en":
        return sum(c.isascii() for c in letters) / len(letters)
    lo, hi = _SCRIPT_RANGE[lang]
    return sum(lo <= c <= hi for c in letters) / len(letters)


def speakable(text: str, lang: str) -> bool:
    """Can `lang`'s voice actually say `text`? Model-written smalltalk is the
    one place reply text is not from a template, so it is checked: a reply in
    the wrong script is dropped by the tokenizer and heard as silence."""
    if not text or unspeakable_spans(text, lang):
        return False
    wrong = _WRONG_SCRIPT.get(lang)
    # The danda (U+0964) and double danda (U+0965) sit in the Devanagari block but are the full stop of Bengali
    # too. Counting them as "wrong script" made every Bengali reply ending in "।" unspeakable, so it was
    # silently replaced by a default line.
    return not (wrong and any(wrong[0] <= c <= wrong[1] and c not in _SHARED_PUNCTUATION for c in text))


# ---------------------------------------------------------------- selection

# A second language must carry at least this much LID probability before the
# top label is treated as unsure. Measured on the pod: Indian-accented
# English is labelled Hindi at 0.90-0.91 with English at 0.09-0.36, while
# genuine Hindi and Bengali put 0.00 on English. So "any real mass on a
# second language" is the signal, not the top label's own confidence.
SECOND_LANGUAGE_FLOOR = 0.03


def languages_to_verify(language: str, scores: dict[str, float], active: tuple[str, ...]) -> list[str] | None:
    """Every active language, best LID score first, when LID is not decisive;
    None when it is.

    Not decisive means: LID found none of our languages ("unknown"), or a
    second language carries real probability. Verifying with ALL active ASRs
    rather than the top two is deliberate: measured, English speech came back
    as bn 0.83 / hi 0.16 / en 0.01, so English was not even in the top two,
    and only running its ASR could have found it."""
    ranked = [lang for lang, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if lang in active]
    for lang in active:
        if lang not in ranked:
            ranked.append(lang)
    if language == "unknown":
        return ranked
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
    return ranked if second >= SECOND_LANGUAGE_FLOOR else None


# A second Indic language is only worth running a recogniser for when language ID is genuinely torn between the two.
# Below this it cannot win anyway (pick_candidate: an Indic engine other than language ID's favourite needs SWITCH_MIN_LID
# to overrule it, and needs INDIC_MIN_LID just to be considered), so its ~0.6 s of GPU time buys nothing.
OTHER_INDIC_MIN_LID = 0.30


def engines_needed(language: str, scores: dict[str, float], active: tuple[str, ...]) -> list[str]:
    """The recognisers that can still change the answer, best language-ID score first (ties keep that order, so the
    LID-preferred language wins them). Measured on the pod: running all three took ~0.64 s; English took 0.21 s.

      * unknown language, or no scores: all of them, as before;
      * otherwise language ID's favourite, plus the OTHER Indic language only if LID is torn (>= OTHER_INDIC_MIN_LID);
      * English ALWAYS: it is cheap and concurrent, and language ID mislabels accented English as Bengali or Hindi
        (measured: English speech came back as bn 0.83 / hi 0.16 / en 0.01), so its probability cannot rule it out."""
    ranked = [lang for lang, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if lang in active]
    for lang in active:
        if lang not in ranked:
            ranked.append(lang)
    if language == "unknown" or not scores:
        return ranked
    need = []
    for lang in ranked:
        if lang == "en" or lang == language or scores.get(lang, 0.0) >= OTHER_INDIC_MIN_LID:
            need.append(lang)
    if language in active and language not in need:
        need.insert(0, language)
    return need


# English gate. Measured on synthetic clips (clean, 8 kHz, and through the live
# service), decoder agreement of the ENGLISH engine: 0.67-1.00 on English
# audio (n=7 runs), 0.00-0.43 on Bengali and Hindi audio (n=8). The Bengali engine cannot be used the same way: it
# transliterates English with agreement up to 1.00, so on English audio "the
# engine that agrees with itself most" picks Bengali. REASONED from a small
# synthetic sample, not calibrated on telephony -- recalibrate on real calls.
ENGLISH_MIN_AGREEMENT = 0.6


# An Indic (Bengali/Hindi) recogniser may only WIN on its own confidence if language ID gave that language at
# least this much probability. Found on the live pod: LID said Bengali 0.95 / Hindi 0.05, all three recognisers ran,
# and the Hindi one (which writes Bengali speech out in Devanagari, with a confident-looking 0.50) beat the Bengali
# one (0.20), so the caller was answered in Hindi. Recogniser confidences of different models are not comparable;
# language ID is the one signal that is. English is exempt: LID mislabels accented English, which is exactly why
# the English engine has its own decoder-agreement gate.
INDIC_MIN_LID = 0.10

# English may win outright on its own decoder agreement (LID mislabels accented English), but that gate alone let a
# Bengali sentence through: the English recogniser wrote it out in Latin letters with agreement 1.00 while language ID
# had given English 0.00, and the call was answered in English. So English must ALSO be real English: language ID
# gives it some probability, or most of what it wrote is words an English speaker actually says.
ENGLISH_MIN_LID = 0.05
ENGLISH_MIN_WORD_SHARE = 0.5
_ENGLISH_WORDS = frozenset(
    """
a an the and or but if so of to in on at for from with by about as is are was were be been am do does did have has had
i me my we our you your he she it they them this that these those there here what which who whom when where why how
can could will would shall should may might must not no yes ok okay please thanks thank hello hi sorry
price prices cost costs rate rates fee charge how much many test tests doctor doctors dr appointment appointments book
booking cancel reschedule time timing timings open close hours today tomorrow morning evening day week monday tuesday
wednesday thursday friday saturday sunday fasting fast empty stomach report reports result results sample blood urine
clinic address location parking insurance payment card cash number phone confirm confirmation need want tell give
know check see get take come go available sit sits sitting chamber counter staff person help
""".split()
)


def english_word_share(text: str) -> float:
    """Share of the tokens that are ordinary English words (a small fixed list). A Bengali or Hindi sentence written
    out in Latin letters ("sibisi test rate koto") scores low; a real English question scores high."""
    tokens = [t.strip(".,?!'\"").lower() for t in (text or "").split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    return sum(t in _ENGLISH_WORDS for t in tokens) / len(tokens)


# Leaving the language the call has been in needs language ID to believe the new one, not merely allow it.
SWITCH_MIN_LID = 0.50


class _Recognition(Protocol):
    """What the picker reads of one recogniser's result (agent/asr.ASRResult has both)."""

    text: str
    decoder_agreement: float


def pick_candidate(
    candidates: list[tuple[str, _Recognition]], lid_scores: dict[str, float] | None = None, prior: str | None = None
):
    """Several ASR engines ran on the same audio. Return (language, result)
    for the one that most plausibly heard its own language.

    1. English wins outright if the English engine's CTC and RNNT decoders
       agree (>= ENGLISH_MIN_AGREEMENT) and it produced Latin text -- the
       one engine whose agreement is a trustworthy "this was English" signal.
    2. Otherwise choose among the rest by transcript-in-own-script, then by
       decoder agreement (own-language 0.67-1.00 vs 0.00-0.67 for the wrong
       Indic engine, measured).
    Ties go to the earlier candidate, i.e. the LID-preferred one."""
    usable = [(lang, r) for lang, r in candidates if getattr(r, "text", "").strip()]
    if not usable:
        return candidates[0]
    for lang, r in usable:
        if lang == "en" and r.decoder_agreement >= ENGLISH_MIN_AGREEMENT and script_share(r.text, "en") >= 0.9:
            # With language ID scores in hand, "English" must also be English (see ENGLISH_MIN_WORD_SHARE).
            if (
                lid_scores is None
                or lid_scores.get("en", 0.0) >= ENGLISH_MIN_LID
                or english_word_share(r.text) >= ENGLISH_MIN_WORD_SHARE
            ):
                return lang, r
    rest = [(lang, r) for lang, r in usable if lang != "en"] or usable
    if lid_scores:
        believed = [(lang, r) for lang, r in rest if lid_scores.get(lang, 0.0) >= INDIC_MIN_LID]
        rest = believed or rest
    best = max(rest, key=lambda t: (round(script_share(t[1].text, t[0]), 1), t[1].decoder_agreement))
    if lid_scores and not prior:
        # First turn: no call language yet, so language ID's own favourite among these is the one to stay with.
        prior = max((lang for lang, _ in rest), key=lambda lang: lid_scores.get(lang, 0.0))
    if lid_scores and prior and best[0] != prior and lid_scores.get(best[0], 0.0) < SWITCH_MIN_LID:
        stay = [(lang, r) for lang, r in rest if lang == prior]
        if stay:
            return stay[0]  # no switch on a recogniser's say-so alone
    return best
