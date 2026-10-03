"""Turning what a caller SAYS for the security questions into values the registry can compare.

The four questions (KCD-495) are a date of birth, a patient id, a full name and an address. Speech
recognition returns them as text -- digits or number words, Latin or Bengali or Devanagari script --
and this module does the deterministic part of reading them:

    parse_dob(text)          -> datetime.date | None        "12 May 1980", "১২/০৫/১৯৮০", "twelfth of May nineteen
                                                             eighty", "बारह मई उन्नीस सौ अस्सी", "বারো মে উনিশশো আশি"
    parse_patient_id(text)   -> "KCP100001" | None          "KCP-100001", "K C P one zero zero zero zero one"
    address_text(text)       -> str                          the caller's words, for the server to match

It parses; it never GUESSES. A date is returned only when day, month and a four-digit year are all
there, and an ambiguous reading is refused rather than resolved (a two-digit year is not accepted:
"eighty" could be 1980 or 2080). Anything not understood returns None and the agent asks again. The
comparison with the registry is not done here: the SERVER checks the answers, so nothing in the voice
agent can decide a caller is verified.

Indian date order is day-month-year, so "05/12/1980" is the 5th of December. A wrong reading only
ever makes the check FAIL (the registry compares exactly); it can never make a wrong person pass.

Number words are best-effort: an ASR that returns digits (the common case for dates) is handled
exactly, one that spells numbers out is handled for the forms below and returns None for the rest.
"""

from __future__ import annotations

import datetime
import re
import unicodedata

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_HI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

MONTHS = {
    1: ["january", "jan", "জানুয়ারি", "জানুয়ারী", "জানুআরি", "जनवरी"],
    2: ["february", "feb", "ফেব্রুয়ারি", "ফেব্রুয়ারী", "फ़रवरी", "फरवरी"],
    3: ["march", "mar", "মার্চ", "मार्च"],
    4: ["april", "apr", "এপ্রিল", "अप्रैल", "अप्रेल"],
    5: ["may", "মে", "मई"],
    6: ["june", "jun", "জুন", "जून"],
    7: ["july", "jul", "জুলাই", "जुलाई"],
    8: ["august", "aug", "আগস্ট", "অগাস্ট", "अगस्त"],
    9: ["september", "sep", "sept", "সেপ্টেম্বর", "सितंबर", "सितम्बर"],
    10: ["october", "oct", "অক্টোবর", "अक्टूबर", "अक्तूबर"],
    11: ["november", "nov", "নভেম্বর", "नवंबर", "नवम्बर"],
    12: ["december", "dec", "ডিসেম্বর", "दिसंबर", "दिसम्बर"],
}
_MONTH_LOOKUP = {name: n for n, names in MONTHS.items() for name in names}

_EN_UNITS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_EN_HUNDRED, _EN_THOUSAND = {"hundred"}, {"thousand"}

_BN_ONES = [
    "শূন্য",
    "এক",
    "দুই",
    "তিন",
    "চার",
    "পাঁচ",
    "ছয়",
    "সাত",
    "আট",
    "নয়",
    "দশ",
    "এগারো",
    "বারো",
    "তেরো",
    "চোদ্দো",
    "পনেরো",
    "ষোলো",
    "সতেরো",
    "আঠারো",
    "উনিশ",
    "কুড়ি",
    "একুশ",
    "বাইশ",
    "তেইশ",
    "চব্বিশ",
    "পঁচিশ",
    "ছাব্বিশ",
    "সাতাশ",
    "আটাশ",
    "ঊনত্রিশ",
    "ত্রিশ",
    "একত্রিশ",
    "বত্রিশ",
    "তেত্রিশ",
    "চৌত্রিশ",
    "পঁয়ত্রিশ",
    "ছত্রিশ",
    "সাঁইত্রিশ",
    "আটত্রিশ",
    "ঊনচল্লিশ",
    "চল্লিশ",
    "একচল্লিশ",
    "বিয়াল্লিশ",
    "তেতাল্লিশ",
    "চুয়াল্লিশ",
    "পঁয়তাল্লিশ",
    "ছেচল্লিশ",
    "সাতচল্লিশ",
    "আটচল্লিশ",
    "ঊনপঞ্চাশ",
    "পঞ্চাশ",
    "একান্ন",
    "বাহান্ন",
    "তিপ্পান্ন",
    "চুয়ান্ন",
    "পঞ্চান্ন",
    "ছাপ্পান্ন",
    "সাতান্ন",
    "আটান্ন",
    "ঊনষাট",
    "ষাট",
    "একষট্টি",
    "বাষট্টি",
    "তেষট্টি",
    "চৌষট্টি",
    "পঁয়ষট্টি",
    "ছেষট্টি",
    "সাতষট্টি",
    "আটষট্টি",
    "ঊনসত্তর",
    "সত্তর",
    "একাত্তর",
    "বাহাত্তর",
    "তিয়াত্তর",
    "চুয়াত্তর",
    "পঁচাত্তর",
    "ছিয়াত্তর",
    "সাতাত্তর",
    "আটাত্তর",
    "ঊনআশি",
    "আশি",
    "একাশি",
    "বিরাশি",
    "তিরাশি",
    "চুরাশি",
    "পঁচাশি",
    "ছিয়াশি",
    "সাতাশি",
    "অষ্টাশি",
    "ঊননব্বই",
    "নব্বই",
    "একানব্বই",
    "বিরানব্বই",
    "তিরানব্বই",
    "চুরানব্বই",
    "পঁচানব্বই",
    "ছিয়ানব্বই",
    "সাতানব্বই",
    "আটানব্বই",
    "নিরানব্বই",
]
_HI_ONES = [
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
    "चौवालीस",
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
    "उन्यासी",
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

_WORDS: dict[str, int] = dict(_EN_UNITS)
_WORDS.update({w: i for i, w in enumerate(_BN_ONES)})
_WORDS.update({w: i for i, w in enumerate(_HI_ONES)})
_WORDS.update({"पांच": 5, "छः": 6, "छ": 6, "अठ्ठाईस": 28, "पाँच": 5, "চোদ্দ": 14, "ঊনিশ": 19, "পাচ": 5})
# Bengali fuses "one nine hundred" into one word for 1900..: উনিশশো. Others are hundreds words.
_HUNDRED_WORDS = {"hundred", "সৌ", "শো", "সौ", "सौ", "শ"}
_FUSED_HUNDREDS = {
    "উনিশশো": 1900,
    "একশো": 100,
    "দুইশো": 200,
    "তিনশো": 300,
    "চারশো": 400,
    "পাঁচশো": 500,
    "ছয়শো": 600,
    "সাতশো": 700,
    "আটশো": 800,
    "নয়শো": 900,
}
_THOUSAND_WORDS = {"thousand", "হাজার", "हज़ार", "हजार"}
_ORDINAL_SUFFIX = re.compile(r"(?<=\d)(st|nd|rd|th|ই|লা|ला|वीं|वां)\b", re.I)

_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "শূন্য": "0",
    "এক": "1",
    "দুই": "2",
    "তিন": "3",
    "চার": "4",
    "পাঁচ": "5",
    "ছয়": "6",
    "সাত": "7",
    "আট": "8",
    "নয়": "9",
    "शून्य": "0",
    "ज़ीरो": "0",
    "एक": "1",
    "दो": "2",
    "तीन": "3",
    "चार": "4",
    "पाँच": "5",
    "पांच": "5",
    "छह": "6",
    "छः": "6",
    "सात": "7",
    "आठ": "8",
    "नौ": "9",
}

MIN_YEAR, MAX_YEAR = 1900, None


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").translate(_BN_DIGITS).translate(_HI_DIGITS).lower()
    return re.sub(r"[,;।]", " ", t)


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", _norm(text)) if t]


def _value_of(token: str) -> int | None:
    t = token.strip(".")
    if t.isdigit():
        return int(t)
    if t in _WORDS:
        return _WORDS[t]
    if len(t) > 1 and t[:-1] in _WORDS and t[-1] in "ই":  # বারোই
        return _WORDS[t[:-1]]
    return None


def _to_numbers(tokens: list[str]) -> list:
    """Replace runs of number words with integers; leave everything else as text. Composition:
    tens + units ("eighty five" -> 85), "<n> hundred" and "<n> thousand", and a year said as two
    numbers ("nineteen eighty five" -> 1985)."""
    out: list = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in _FUSED_HUNDREDS:
            value, i = _FUSED_HUNDREDS[t], i + 1
            if i < len(tokens) and _value_of(tokens[i]) is not None:
                value += _value_of(tokens[i])
                i += 1
                if (
                    i < len(tokens)
                    and _value_of(tokens[i]) is not None
                    and value % 10 == 0
                    and _value_of(tokens[i]) < 10
                ):
                    value += _value_of(tokens[i])
                    i += 1
            out.append(value)
            continue
        v = _value_of(t)
        if v is None:
            out.append(t)
            i += 1
            continue
        run = []
        while i < len(tokens):
            tok = tokens[i]
            if tok in _HUNDRED_WORDS or tok in _THOUSAND_WORDS:
                run.append(tok)
                i += 1
                continue
            vv = _value_of(tok)
            if vv is None or tok.isdigit() and run and not isinstance(run[-1], str):
                break
            run.append(vv)
            i += 1
        out.extend(_compose(run))
    return out


def _compose(run: list) -> list:
    """Numbers and the words hundred/thousand -> a list of integers (one per number said)."""
    numbers: list[int] = []
    cur = 0
    total = 0
    pending = False
    for item in run:
        if item in _HUNDRED_WORDS:
            cur = (cur or 1) * 100
            pending = True
        elif item in _THOUSAND_WORDS:
            total += (cur or 1) * 1000
            cur = 0
            pending = True
        else:
            if pending and cur and item < 100 and cur % 100 == 0:
                cur += item
            elif cur and cur % 10 == 0 and cur >= 20 and item < 10 and cur < 100:
                cur += item  # eighty + five
            elif cur and not pending:
                numbers.append(total + cur)
                total = 0
                cur = item  # a new number begins
            else:
                cur = cur + item if pending else item
    numbers.append(total + cur)
    # "nineteen" "eighty five" said as two numbers, a year
    if len(numbers) == 2 and numbers[0] in (19, 20) and 0 <= numbers[1] < 100:
        return [numbers[0] * 100 + numbers[1]]
    return numbers


def parse_dob(text: str, today: datetime.date | None = None) -> datetime.date | None:
    """A date of birth from speech, or None. Requires day, month and a FOUR-digit year."""
    today = today or datetime.date.today()
    t = _ORDINAL_SUFFIX.sub("", _norm(text))
    tokens = [x for x in re.split(r"[\s]+", t) if x]
    # numeric forms first: 12/05/1980, 12-5-1980, 12.05.1980
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})", t)
    if m:
        return _make(int(m.group(3)), int(m.group(2)), int(m.group(1)), today)
    nums = _to_numbers(tokens)
    month = next((_MONTH_LOOKUP[x] for x in nums if isinstance(x, str) and x in _MONTH_LOOKUP), None)
    if month is None:
        # "12 5 1980" as three bare numbers, day month year
        ints = [x for x in nums if isinstance(x, int)]
        if len(ints) == 3 and ints[2] >= 1000 and 1 <= ints[0] <= 31 and 1 <= ints[1] <= 12:
            return _make(ints[2], ints[1], ints[0], today)
        return None
    ints = [x for x in nums if isinstance(x, int)]
    years = [x for x in ints if x >= 1000]
    days = [x for x in ints if 1 <= x <= 31]
    if len(years) != 1 or not days:
        return None
    return _make(years[0], month, days[0], today)


def _make(year: int, month: int, day: int, today: datetime.date) -> datetime.date | None:
    if not (MIN_YEAR <= year <= today.year):
        return None
    try:
        d = datetime.date(year, month, day)
    except ValueError:
        return None
    return d if d <= today else None


_ID = re.compile(r"\b([a-z]{2,4})[\s\-]*((?:\d[\s\-]*){4,10})\b", re.I)


def parse_patient_id(text: str) -> str | None:
    """A patient id like KCP-100001, from digits or spoken digits, letters said one by one or as a
    word. Returned normalised (letters + digits, no separators) or None."""
    t = _norm(text)
    words = []
    for tok in t.split():
        tok = tok.strip(".-")
        words.append(_DIGIT_WORDS.get(tok, tok))
    joined = " ".join(words)
    # letters said singly ("k c p") collapse to a word
    joined = re.sub(r"\b([a-z])\s+(?=[a-z]\b)", r"\1", joined)
    m = _ID.search(joined)
    if m and m.group(1).lower() not in _MONTH_LOOKUP:  # "may 1980" is a month and a year, not an id
        return (m.group(1) + re.sub(r"\D", "", m.group(2))).upper()
    digits = re.sub(r"\D", "", joined)
    return (
        digits
        if 5 <= len(digits) <= 10 and digits == re.sub(r"[^\d]", "", joined) and not re.search(r"[a-z]", joined)
        else None
    )


_NAME_LEAD = re.compile(r"^\s*(my name is|this is|i am|i'm|name is|it is|it's|আমার নাম|আমি|নাম|मेरा नाम|मैं|नाम)\s+", re.I)


def clean_name(text: str) -> str:
    """The name part of an answer. The server matches every part of the registered name against the
    words said, so a lead-in is harmless; this only tidies what is logged and shown."""
    return _NAME_LEAD.sub("", _norm(text)).strip()


def address_text(text: str) -> str:
    """The caller's words for the address, digits normalised. The server compares distinctive words
    and the pincode; nothing is guessed here."""
    words = []
    for tok in _norm(text).split():
        words.append(_DIGIT_WORDS.get(tok, tok) if tok in _DIGIT_WORDS and len(text.split()) > 1 else tok)
    return " ".join(words).strip()
