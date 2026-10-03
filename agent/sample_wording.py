"""What sample a test needs, as a sentence a caller can follow (found on the live pod: "সৈম্পল: Blood").

The price reply used to append the catalogue's `sample_type` as a label and a value: "Sample: Blood" in English and
Hindi, "স্যাম্পল: {sample}" in Bengali. Three things were wrong with that when spoken:

  * a label, a colon and a value is a field read out, not a sentence (KCD-454);
  * the value is a Latin word ("Blood") inside Bengali or Hindi speech, which the voice drops or mangles;
  * Bengali said whatever the column held, so a scan's category ("Imaging", "Cardiac") or "Sample (Cervical)" was
    read out as if it were a specimen.

A specimen is a sample a person gives; anything else says nothing about a sample here, exactly as the Hindi and
English replies already did. The names below are everyday words in each language and are for a native reviewer to
confirm; the sentence structure is one clause with no colon.
"""

from __future__ import annotations

_SPECIMENS = {
    #            bengali (genitive, before "নমুনা")   hindi                 english
    "blood": ("রক্তের", "ब्लड", "blood"),
    "urine": ("প্রস্রাবের", "यूरिन", "urine"),
    "stool": ("মলের", "स्टूल", "stool"),
    "serum": ("সিরামের", "सीरम", "serum"),
    "saliva": ("লালার", "लार", "saliva"),
    "swab": ("সোয়াবের", "स्वैब", "swab"),
    "plasma": ("প্লাজমার", "प्लाज़्मा", "plasma"),
}


def is_specimen(sample: str | None) -> bool:
    return (sample or "").strip().lower() in _SPECIMENS


def sample_sentence(sample: str | None, lang: str) -> str:
    """One sentence saying which sample is taken, or "" when `sample` is not a specimen."""
    entry = _SPECIMENS.get((sample or "").strip().lower())
    if entry is None:
        return ""
    bn, hi, en = entry
    if lang == "hi":
        return f"इसके लिए {hi} का सैंपल लिया जाता है।"
    if lang == "en":
        return f"A {en} sample is needed."
    return f"এর জন্য {bn} নমুনা নেওয়া হয়।"
