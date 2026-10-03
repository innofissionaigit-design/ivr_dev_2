"""agent/phonetic_match.py (KCD-436): a mispronounced doctor's name
should still match, and a nonsense name must never match a real doctor --
the exact "Doctor Nobody" regression CLAUDE.md documents. Pure and
offline, no database.

    python -m pytest tests/test_phonetic_match.py -v
"""

from agent.phonetic_match import phonetic_key, phonetic_match


def test_mispronounced_surname_still_matches():
    # A common mishearing: vowel drift and doubled-letter spelling
    # variants (both real alternate spellings of the same surname), but
    # the consonant skeleton is identical.
    assert phonetic_match("Bhattacharya", "Bhattacharyya")
    assert phonetic_match("Mukherjee", "Mukharji")


def test_bengali_sibilant_confusion_matches():
    # শ/ষ/স are the single most common ASR sibilant confusion on this line.
    assert phonetic_match("সেন", "ষেন")


def test_devanagari_aspirated_confusion_matches():
    assert phonetic_match("भट्टाचार्य", "भट्टाचार्यय")
    assert phonetic_match("मुखर्जी", "मुखरजी")


def test_cross_script_same_surname_matches():
    # The whole reason this module exists: fold to a comparable skeleton
    # regardless of which script ASR happened to emit.
    assert phonetic_match("Sen", "সেন")
    assert phonetic_match("Sen", "सेन")


def test_unrelated_short_names_do_not_collide():
    # The regression case itself: a nonsense name must not fold onto a
    # real, unrelated surname just because both are short.
    assert not phonetic_match("Xyz", "Roy")
    assert not phonetic_match("Nobody", "Roy")
    assert not phonetic_match("Kar", "Ray")  # different consonant skeleton (K vs R start)


def test_empty_or_no_letters_never_matches_anything():
    assert phonetic_key("") == ""
    assert phonetic_key("123") == ""
    assert not phonetic_match("", "Sen")
    assert not phonetic_match("Sen", "")


def test_bengali_flap_consonant_matches_despite_nfc_composition_exclusion():
    # CodeRabbit-flagged, real bug: unicodedata.normalize("NFC", ...)
    # DECOMPOSES the Bengali flap consonants (they are in Unicode's NFC
    # composition exclusion list) instead of composing them, so a name
    # folded from the precomposed codepoint and one folded from an
    # already-decomposed base+nukta pair must land on the identical key.
    # Built with chr()/explicit codepoints, not typed glyphs -- a typed
    # or pasted glyph is whatever normalization form the editor happened
    # to save, which would silently defeat this exact test.
    precomposed = "ব" + "ড়" + "ুয়া"  # BA + precomposed DDA+nukta (U+09DC)
    decomposed = "ব" + "ড়" + "ুয়া"  # BA + DDA (U+09A1) + nukta (U+09BC)
    assert precomposed != decomposed, "test fixture must actually differ at the codepoint level"
    assert phonetic_key(precomposed) == phonetic_key(decomposed)
    assert phonetic_match(precomposed, "Barua")
    assert phonetic_match(decomposed, "Barua")


def test_devanagari_flap_consonant_matches_despite_nfc_composition_exclusion():
    precomposed = "ब" + "ड़" + "ुआ"  # BA + precomposed DDA+nukta (U+095C)
    decomposed = "ब" + "ड़" + "ुआ"  # BA + DDA (U+0921) + nukta (U+093C)
    assert precomposed != decomposed, "test fixture must actually differ at the codepoint level"
    assert phonetic_key(precomposed) == phonetic_key(decomposed)
    assert phonetic_match(precomposed, "Barua")
    assert phonetic_match(decomposed, "Barua")
