"""Per-language verbalization: rewrite a reply so its TTS can actually say it.

agent/bn_normalize.py documents the measured bug this exists for: the
AI4Bharat FastPitch tokenizers drop characters outside their own script, so
a Latin digit or Latin word is not mispronounced, it is silently ABSENT --
"rate 250 rupees" and "rate 987654 rupees" came out as identical audio.
Bengali is handled in bn_normalize.py; this module adds Hindi and English
with the same rules and dispatches by language.

Same shape as the Bengali pass, deliberately: numbers are spelled in Indian
grouping (thousand / lakh / crore), times use the idiomatic half/quarter
forms, dates omit the year (a booking is always for the near future), and
phone numbers and confirmation IDs are read character by character.

The Hindi number table is the irregular 0-99 list; it cannot be generated
by rule. tests/test_speech_norm.py pins spot values across the range.
"""

from __future__ import annotations

import re

from agent import bn_normalize, pronunciation
from agent.figures import id_groups, phone_groups, speak_grouped

# ----------------------------------------------------------------- shared

_RE_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RE_TIME_RANGE = re.compile(r"\b(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})\b")
_RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_RE_PHONE = re.compile(r"\b(\d{10,})\b")
_RE_CONF_ID = re.compile(r"\b([A-Z]{2,}[-]?\d[\d-]{2,})\b")
_RE_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_RE_INT = re.compile(r"\d+")
_RE_SPACES = re.compile(r"\s{2,}")


def _finish(text: str) -> str:
    return _RE_SPACES.sub(" ", text).strip()


# ------------------------------------------------------------------ hindi

_HI_0_99 = [
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पाँच",
    "छह",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "ग्यारह",
    "बारह",
    "तेरह",
    "चौदह",
    "पंद्रह",
    "सोलह",
    "सत्रह",
    "अठारह",
    "उन्नीस",
    "बीस",
    "इक्कीस",
    "बाईस",
    "तेईस",
    "चौबीस",
    "पच्चीस",
    "छब्बीस",
    "सत्ताईस",
    "अट्ठाईस",
    "उनतीस",
    "तीस",
    "इकतीस",
    "बत्तीस",
    "तैंतीस",
    "चौंतीस",
    "पैंतीस",
    "छत्तीस",
    "सैंतीस",
    "अड़तीस",
    "उनतालीस",
    "चालीस",
    "इकतालीस",
    "बयालीस",
    "तैंतालीस",
    "चवालीस",
    "पैंतालीस",
    "छियालीस",
    "सैंतालीस",
    "अड़तालीस",
    "उनचास",
    "पचास",
    "इक्यावन",
    "बावन",
    "तिरपन",
    "चौवन",
    "पचपन",
    "छप्पन",
    "सत्तावन",
    "अट्ठावन",
    "उनसठ",
    "साठ",
    "इकसठ",
    "बासठ",
    "तिरसठ",
    "चौंसठ",
    "पैंसठ",
    "छियासठ",
    "सड़सठ",
    "अड़सठ",
    "उनहत्तर",
    "सत्तर",
    "इकहत्तर",
    "बहत्तर",
    "तिहत्तर",
    "चौहत्तर",
    "पचहत्तर",
    "छिहत्तर",
    "सतहत्तर",
    "अठहत्तर",
    "उनासी",
    "अस्सी",
    "इक्यासी",
    "बयासी",
    "तिरासी",
    "चौरासी",
    "पचासी",
    "छियासी",
    "सत्तासी",
    "अट्ठासी",
    "नवासी",
    "नब्बे",
    "इक्यानवे",
    "बानवे",
    "तिरानवे",
    "चौरानवे",
    "पंचानवे",
    "छियानवे",
    "सत्तानवे",
    "अट्ठानवे",
    "निन्यानवे",
]


def number_to_hi_words(n: int) -> str:
    if n < 0:
        return "माइनस " + number_to_hi_words(-n)
    if n < 100:
        return _HI_0_99[n]
    if n < 1000:
        head, rest = divmod(n, 100)
        out = f"{_HI_0_99[head]} सौ"
        return out if rest == 0 else f"{out} {_HI_0_99[rest]}"
    for divisor, word in ((10_000_000, "करोड़"), (100_000, "लाख"), (1_000, "हज़ार")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_hi_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_hi_words(rest)}"
    return str(n)


_HI_LETTER = {
    "a": "ए",
    "b": "बी",
    "c": "सी",
    "d": "डी",
    "e": "ई",
    "f": "एफ",
    "g": "जी",
    "h": "एच",
    "i": "आई",
    "j": "जे",
    "k": "के",
    "l": "एल",
    "m": "एम",
    "n": "एन",
    "o": "ओ",
    "p": "पी",
    "q": "क्यू",
    "r": "आर",
    "s": "एस",
    "t": "टी",
    "u": "यू",
    "v": "वी",
    "w": "डब्ल्यू",
    "x": "एक्स",
    "y": "वाई",
    "z": "ज़ेड",
}

_HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]


def _hi_day_part(hour24: int) -> str:
    if 4 <= hour24 < 12:
        return "सुबह"
    if 12 <= hour24 < 16:
        return "दोपहर"
    if 16 <= hour24 < 19:
        return "शाम"
    return "रात"


def time_to_hi_words(hh: int, mm: int) -> str:
    part = _hi_day_part(hh)
    h12 = hh % 12 or 12
    if mm == 0:
        return f"{part} {_HI_0_99[h12]} बजे"
    if mm == 30:
        if h12 == 1:
            return f"{part} डेढ़ बजे"
        if h12 == 2:
            return f"{part} ढाई बजे"
        return f"{part} साढ़े {_HI_0_99[h12]} बजे"
    if mm == 15:
        return f"{part} सवा {_HI_0_99[h12]} बजे"
    if mm == 45:
        nxt = (h12 % 12) + 1
        return f"{_hi_day_part((hh + 1) % 24)} पौने {_HI_0_99[nxt]} बजे"
    return f"{part} {_HI_0_99[h12]} बजकर {_HI_0_99[mm]} मिनट"


def _hi_digits(s: str) -> str:
    return " ".join(_HI_0_99[int(c)] for c in s if c.isdigit())


def _hi_spell(s: str) -> str:
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(_HI_0_99[int(ch)])
        elif ch.isalpha() and ch.lower() in _HI_LETTER:
            out.append(_HI_LETTER[ch.lower()])
    return " ".join(out)


_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def verbalize_hi(text: str) -> str:
    if not text:
        return text
    text = text.translate(_DEVANAGARI_DIGITS)
    text = text.replace("₹", " रुपये ").replace("%", " प्रतिशत ")
    text = _RE_DATE.sub(
        lambda m: (
            f"{_HI_MONTHS[int(m.group(2)) - 1]} की {number_to_hi_words(int(m.group(3)))} तारीख"
            if 1 <= int(m.group(2)) <= 12
            else f"{number_to_hi_words(int(m.group(3)))} तारीख"
        ),
        text,
    )
    text = _RE_TIME_RANGE.sub(
        lambda m: (
            f"{time_to_hi_words(int(m.group(1)), int(m.group(2)))} से "
            f"{time_to_hi_words(int(m.group(3)), int(m.group(4)))} तक"
        ),
        text,
    )
    text = _RE_TIME.sub(lambda m: time_to_hi_words(int(m.group(1)), int(m.group(2))), text)
    text = _RE_CONF_ID.sub(lambda m: speak_grouped(id_groups(m.group(1)), _hi_spell), text)
    text = _RE_PHONE.sub(lambda m: speak_grouped(phone_groups(m.group(1)), _hi_digits), text)
    text = _RE_DECIMAL.sub(lambda m: f"{number_to_hi_words(int(m.group(1)))} दशमलव {_hi_digits(m.group(2))}", text)
    text = _RE_INT.sub(lambda m: number_to_hi_words(int(m.group(0))), text)
    # KCD-159: the explicit pronunciation path (agent/pronunciation.py): a curated
    # spoken form, an unlisted acronym spelled out, and anything else left Latin
    # for unspeakable_spans() to report and block (KCD-455), never guessed at.
    text = pronunciation.apply(text, "hi", _HI_LETTER)
    return _finish(text)


# ---------------------------------------------------------------- english

_EN_0_19 = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_en_words(n: int) -> str:
    """Indian grouping (lakh / crore), which is how an Indian caller and an
    Indian clinic price list say it: 150000 is "one lakh fifty thousand"."""
    if n < 0:
        return "minus " + number_to_en_words(-n)
    if n < 20:
        return _EN_0_19[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _EN_TENS[tens] + (f" {_EN_0_19[ones]}" if ones else "")
    if n < 1000:
        head, rest = divmod(n, 100)
        out = f"{_EN_0_19[head]} hundred"
        return out if rest == 0 else f"{out} {number_to_en_words(rest)}"
    for divisor, word in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_en_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_en_words(rest)}"
    return str(n)


_EN_ORDINAL = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    21: "twenty first",
    22: "twenty second",
    23: "twenty third",
    24: "twenty fourth",
    25: "twenty fifth",
    26: "twenty sixth",
    27: "twenty seventh",
    28: "twenty eighth",
    29: "twenty ninth",
    30: "thirtieth",
    31: "thirty first",
}
_EN_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_EN_LETTER = {
    "a": "ay",
    "b": "bee",
    "c": "see",
    "d": "dee",
    "e": "ee",
    "f": "eff",
    "g": "jee",
    "h": "aitch",
    "i": "eye",
    "j": "jay",
    "k": "kay",
    "l": "el",
    "m": "em",
    "n": "en",
    "o": "oh",
    "p": "pee",
    "q": "queue",
    "r": "ar",
    "s": "ess",
    "t": "tee",
    "u": "you",
    "v": "vee",
    "w": "double you",
    "x": "ex",
    "y": "why",
    "z": "zee",
}


def time_to_en_words(hh: int, mm: int) -> str:
    suffix = "AM" if hh < 12 else "PM"
    h12 = hh % 12 or 12
    if mm == 0:
        return f"{number_to_en_words(h12)} {suffix}"
    minutes = f"oh {number_to_en_words(mm)}" if mm < 10 else number_to_en_words(mm)
    return f"{number_to_en_words(h12)} {minutes} {suffix}"


def _en_digits(s: str) -> str:
    return " ".join(_EN_0_19[int(c)] for c in s if c.isdigit())


def _en_spell(s: str) -> str:
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(_EN_0_19[int(ch)])
        elif ch.isalpha() and ch.lower() in _EN_LETTER:
            out.append(_EN_LETTER[ch.lower()])
    return " ".join(out)


def verbalize_en(text: str) -> str:
    if not text:
        return text
    text = text.replace("₹", " rupees ").replace("%", " percent ")
    text = _RE_DATE.sub(
        lambda m: (
            f"{_EN_MONTHS[int(m.group(2)) - 1]} {_EN_ORDINAL.get(int(m.group(3)), m.group(3))}"
            if 1 <= int(m.group(2)) <= 12
            else m.group(0)
        ),
        text,
    )
    text = _RE_TIME_RANGE.sub(
        lambda m: (
            f"{time_to_en_words(int(m.group(1)), int(m.group(2)))} to "
            f"{time_to_en_words(int(m.group(3)), int(m.group(4)))}"
        ),
        text,
    )
    text = _RE_TIME.sub(lambda m: time_to_en_words(int(m.group(1)), int(m.group(2))), text)
    text = _RE_CONF_ID.sub(lambda m: speak_grouped(id_groups(m.group(1)), _en_spell), text)
    text = _RE_PHONE.sub(lambda m: speak_grouped(phone_groups(m.group(1)), _en_digits), text)
    text = _RE_DECIMAL.sub(lambda m: f"{number_to_en_words(int(m.group(1)))} point {_en_digits(m.group(2))}", text)
    text = _RE_INT.sub(lambda m: number_to_en_words(int(m.group(0))), text)
    return _finish(text)


# --------------------------------------------------------------- dispatch

_RE_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z .'-]*")
# Anything that is not Latin, punctuation or whitespace: the English
# voice will drop it the same way the Indic voices drop Latin.
_RE_NON_LATIN_RUN = re.compile(r"[^\x00-\x7f‘-”–—]+")


def verbalize(text: str, lang: str = "bn") -> str:
    if lang == "hi":
        return verbalize_hi(text)
    if lang == "en":
        return verbalize_en(text)
    return bn_normalize.verbalize(text)


_RE_PRICE = re.compile(r"\d[\d,]*\s*(টাকা|rupees|रुपये|₹)")


def contains_critical_figure(text: str) -> bool:
    """KCD-456: does this reply carry a price, phone number or reference
    ID -- something a caller needs to write down, not just hear. Checked
    against the RAW template text (before verbalize() spells the digits
    out), so the same phone/confirmation-ID/price patterns
    speech_norm.py already recognises decide this too, rather than a
    second, independent definition of "a figure" drifting from the one
    that actually gets spelled out."""
    return bool(_RE_PHONE.search(text) or _RE_CONF_ID.search(text) or _RE_PRICE.search(text))


def unspeakable_spans(text: str, lang: str = "bn") -> list[str]:
    """Spans the voice for `lang` will silently drop, so the gap shows up in
    the logs instead of only in a caller's ear."""
    if lang == "en":
        return [m.group(0).strip() for m in _RE_NON_LATIN_RUN.finditer(text) if m.group(0).strip()]
    return [m.group(0).strip() for m in _RE_LATIN_RUN.finditer(text) if len(m.group(0).strip()) > 1]
