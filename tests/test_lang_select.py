"""lang_select: the LID-unsure path, pinned to the numbers measured on the pod."""

import dataclasses

from agent.lang_select import languages_to_verify, pick_candidate, speakable


@dataclasses.dataclass
class R:
    text: str
    decoder_agreement: float = 1.0


ACTIVE = ("bn", "hi", "en")


def test_measured_indian_english_scores_trigger_verification_of_every_language():
    # LID said hi:0.91 for English speech; en:0.09 is the tell.
    assert languages_to_verify("hi", {"bn": 0.00, "hi": 0.91, "en": 0.09}, ACTIVE) == ["hi", "en", "bn"]
    assert languages_to_verify("hi", {"bn": 0.01, "hi": 0.62, "en": 0.36}, ACTIVE) == ["hi", "en", "bn"]


def test_english_that_lid_ranks_third_is_still_tried():
    # measured on the pod: English speech came back bn 0.83 / hi 0.16 / en 0.01
    got = languages_to_verify("unknown", {"bn": 0.83, "hi": 0.16, "en": 0.01}, ACTIVE)
    assert got == ["bn", "hi", "en"] and "en" in got


def test_decisive_lid_skips_extra_asr():
    assert languages_to_verify("bn", {"bn": 1.00, "hi": 0.00, "en": 0.00}, ACTIVE) is None
    assert languages_to_verify("hi", {"bn": 0.00, "hi": 1.00, "en": 0.00}, ACTIVE) is None


def test_bengali_hindi_confusion_is_also_verified():
    assert languages_to_verify("bn", {"bn": 0.51, "hi": 0.49, "en": 0.00}, ACTIVE)[:2] == ["bn", "hi"]


def test_only_active_languages_are_verified():
    assert languages_to_verify("hi", {"bn": 0.1, "hi": 0.8, "en": 0.1}, ("bn", "hi")) == ["hi", "bn"]


def test_unknown_language_with_no_scores_still_tries_everything():
    assert languages_to_verify("unknown", {}, ACTIVE) == ["bn", "hi", "en"]


def test_english_audio_wins_even_when_the_bengali_engine_agrees_with_itself_perfectly():
    # measured: "What is the price of the uric acid test?" (clean)
    #   bn ASR agreement 1.00 (transliterated), hi 0.60, en 0.78
    bn = R("হোয়াট ইজ দ্য প্রাইজ অফ দ্য উইক অ্যাসিড টেস্ট", 1.00)
    hi = R("वॉट इज़ द प्राइस ऑफ द वीक एसिड टेस्ट", 0.60)
    en = R("what is the price of the rick acid test", 0.78)
    assert pick_candidate([("bn", bn), ("hi", hi), ("en", en)])[0] == "en"


def test_bengali_audio_is_not_mistaken_for_english():
    # measured: bn 1.00 / hi 0.50 / en 0.00 on Bengali audio
    bn = R("আমাদের ক্লিনিক প্রতিদিন সকাল আটটা", 1.00)
    hi = R("आमादर क्लिनिक प्रतिदिन सकाल आठ टा", 0.50)
    en = R("ahmad kenic protyin shokhal", 0.00)
    assert pick_candidate([("bn", bn), ("hi", hi), ("en", en)])[0] == "bn"


def test_hindi_audio_is_not_mistaken_for_english_or_bengali():
    # measured: bn 0.40 / hi 0.75 / en 0.38 on "यूरिक एसिड टेस्ट की कीमत क्या है"
    bn = R("ইউরিক অ্যাসিড টেস্ট কি কীমত ক্যা হ্য", 0.40)
    hi = R("यूरिक एसिड टेस्ट की क़ीमत क्या है", 0.75)
    en = R("euric as it restkee kemah", 0.38)
    assert pick_candidate([("hi", hi), ("en", en), ("bn", bn)])[0] == "hi"


def test_english_engine_below_the_gate_does_not_win():
    assert pick_candidate([("hi", R("हाँ", 0.5)), ("en", R("yes", 0.59))])[0] == "hi"
    assert pick_candidate([("hi", R("हाँ", 0.5)), ("en", R("yes", 0.60))])[0] == "en"


def test_english_at_the_lowest_observed_agreement_still_wins():
    # live pod: LID hi 0.63 / en 0.36, agreements hi 0.78, en 0.67, bn 0.60
    hi = R("वॉट इज़ द प्राइस ऑफ़ द वीक एसिड टेस्ट", 0.78)
    en = R("what is the price of the rick acid test", 0.67)
    bn = R("হোয়াট ইজ দ্য প্রাইস", 0.60)
    assert pick_candidate([("hi", hi), ("en", en), ("bn", bn)])[0] == "en"


def test_english_gate_needs_latin_text():
    assert pick_candidate([("hi", R("हाँ", 0.5)), ("en", R("हाँ", 1.0))])[0] == "hi"


def test_bengali_engine_on_hindi_audio_loses_on_measured_agreement():
    # measured: bn ASR on Hindi audio 0.40-0.67; hi ASR on Hindi audio 0.75-1.00
    bn_on_hi = R("হম ক্লিনিক হরদিন সুবহ আট বজে", 0.67)
    hi_on_hi = R("हमारा क्लिनिक हर दिन सुबह आठ बजे", 1.00)
    assert pick_candidate([("bn", bn_on_hi), ("hi", hi_on_hi)])[0] == "hi"


def test_ties_go_to_the_lid_preferred_candidate():
    a, b = R("হ্যাঁ", 0.8), R("हाँ", 0.8)
    assert pick_candidate([("bn", a), ("hi", b)])[0] == "bn"


def test_empty_transcripts_are_never_chosen():
    assert pick_candidate([("hi", R("", 1.0)), ("en", R("yes", 0.9))])[0] == "en"
    assert pick_candidate([("hi", R("", 1.0)), ("en", R("", 1.0))])[0] == "hi"


def test_speakable_rejects_wrong_script_and_untalkable_spans():
    assert speakable("नमस्कार, मैं क्या मदद कर सकती हूँ?", "hi")
    assert not speakable("নমস্কার", "hi")
    assert not speakable("hello there", "hi")
    assert not speakable("नमस्कार", "en")
    assert speakable("Hello, how can I help?", "en")
    assert not speakable("", "bn")


def test_a_bengali_reply_ending_in_a_danda_is_speakable_and_devanagari_letters_still_are_not():
    """The danda is the full stop of Bengali too; it lives in the Devanagari block and used to make every Bengali
    reply that ended in one 'the wrong script'."""
    assert speakable("ধন্যবাদ" + chr(0x0964) + " আর কিছু জানতে চান?", "bn")
    assert speakable("ধন্যবাদ" + chr(0x0965), "bn")
    assert not speakable("नमस्कार, मैं क्या मदद कर सकती हूँ?", "bn")  # real Devanagari letters: still rejected
    assert speakable("नमस्कार" + chr(0x0964), "hi")


# ---- found on the live pod (call c7eb4b15): Bengali answered in Hindi -------------------------------------------


def test_the_live_pod_case_language_id_said_bengali_95_and_the_hindi_engine_must_not_win_on_its_own_confidence():
    """Logged: LID bn 0.95 / en 0.00 / hi 0.05; recogniser agreement bn 0.20, hi 0.50, en 0.33. The Hindi engine
    wrote the Bengali speech out in Devanagari and, being the more 'confident', won; the caller was answered in Hindi."""
    bn = R("এইচবিএ ওয়ান সি টেস্টের দাম কত", 0.20)
    hi = R("एइबी वन एसी ए टेस्टर्ड दाम को तो", 0.50)
    en = R("hb a one c test her dam co", 0.33)
    scores = {"bn": 0.95, "en": 0.0, "hi": 0.05}
    assert pick_candidate([("bn", bn), ("hi", hi), ("en", en)])[0] == "hi"  # what it did before
    assert pick_candidate([("bn", bn), ("hi", hi), ("en", en)], scores, "bn")[0] == "bn"  # what it does now
    assert pick_candidate([("bn", bn), ("hi", hi), ("en", en)], scores, None)[0] == "bn"


def test_a_call_that_has_been_in_bengali_does_not_switch_to_hindi_unless_language_id_believes_it():
    bn = R("বাংলা কথা", 0.30)
    hi = R("हिंदी बात", 0.80)
    assert pick_candidate([("bn", bn), ("hi", hi)], {"bn": 0.60, "hi": 0.40}, "bn")[0] == "bn"  # 0.40 < 0.50
    assert pick_candidate([("bn", bn), ("hi", hi)], {"bn": 0.20, "hi": 0.80}, "bn")[0] == "hi"  # LID believes Hindi


def test_a_genuine_hindi_caller_on_the_first_turn_is_still_recognised():
    bn = R("हिंदी", 0.10)
    hi = R("मुझे सीबीसी का रेट बताइए", 0.90)
    assert pick_candidate([("bn", bn), ("hi", hi)], {"bn": 0.02, "hi": 0.98}, None)[0] == "hi"


def test_english_still_wins_through_its_own_gate_whatever_language_id_said():
    """LID mislabels accented English (bn 0.83 / hi 0.16 / en 0.01 was measured), so it cannot veto the English gate."""
    bn = R("হোয়াট ইজ দ্য প্রাইজ", 1.00)
    en = R("what is the price of the uric acid test", 0.78)
    assert pick_candidate([("bn", bn), ("en", en)], {"bn": 0.83, "hi": 0.16, "en": 0.01}, "bn")[0] == "en"


def test_without_language_id_scores_the_old_behaviour_is_unchanged():
    bn = R("বাংলা কথা", 0.30)
    hi = R("हिंदी बात", 0.80)
    assert pick_candidate([("bn", bn), ("hi", hi)])[0] in ("bn", "hi")


# ---- which recognisers must run (latency: language ID and recognition overlap; Hindi is not run for Bengali) -------

from agent.lang_select import OTHER_INDIC_MIN_LID, engines_needed  # noqa: E402


def test_the_live_pod_bengali_turn_needs_bengali_and_english_only_not_hindi():
    """LID bn 0.95 / hi 0.05: Hindi cannot win (needs 0.5 to overrule), so its ~0.6 s of GPU time is not spent."""
    assert engines_needed("bn", {"bn": 0.95, "en": 0.0, "hi": 0.05}, ACTIVE) == ["bn", "en"]
    assert engines_needed("bn", {"bn": 0.84, "en": 0.0, "hi": 0.15}, ACTIVE) == ["bn", "en"]


def test_english_is_always_run_because_language_id_cannot_rule_it_out():
    """Measured: English speech came back as bn 0.83 / hi 0.16 / en 0.01."""
    assert "en" in engines_needed("bn", {"bn": 0.83, "hi": 0.16, "en": 0.01}, ACTIVE)
    assert "en" in engines_needed("hi", {"bn": 0.0, "hi": 1.0, "en": 0.0}, ACTIVE)


def test_when_language_id_is_torn_between_the_two_indic_languages_both_run():
    assert engines_needed("bn", {"bn": 0.55, "hi": 0.45, "en": 0.0}, ACTIVE) == ["bn", "hi", "en"]
    assert OTHER_INDIC_MIN_LID <= 0.5 < 0.55


def test_an_unknown_language_or_no_scores_runs_everything_as_before():
    assert sorted(engines_needed("unknown", {"bn": 0.4, "hi": 0.4, "en": 0.2}, ACTIVE)) == ["bn", "en", "hi"]
    assert engines_needed("unknown", {}, ACTIVE) == ["bn", "hi", "en"]


def test_the_language_id_favourite_leads_so_it_wins_ties():
    assert engines_needed("hi", {"bn": 0.02, "hi": 0.90, "en": 0.08}, ACTIVE)[0] == "hi"


def test_a_language_that_is_not_active_is_never_planned():
    assert engines_needed("hi", {"bn": 0.1, "hi": 0.9}, ("bn", "en")) == ["bn", "en"] or "hi" not in engines_needed(
        "hi", {"bn": 0.1, "hi": 0.9}, ("bn", "en")
    )


# ---- found while measuring latency: a Bengali sentence was routed to English ------------------------------------------

from agent.lang_select import english_word_share  # noqa: E402


def test_bengali_speech_written_in_latin_letters_by_the_english_engine_is_not_english():
    """Seen on the pod: LID bn 0.89 / hi 0.11 / en 0.00, the English engine transliterated the Bengali with agreement
    1.00, won the English gate, and the call was answered in English (with the disclosure)."""
    bn = R("সিবিসি টেস্টের রেট কত", 1.00)
    hi = R("x", 0.0)
    en = R("sibisi tester rate koto", 1.00)
    scores = {"bn": 0.89, "hi": 0.11, "en": 0.0}
    assert pick_candidate([("bn", bn), ("hi", hi), ("en", en)], scores, "bn")[0] == "bn"


def test_real_english_still_wins_when_language_id_gave_it_almost_nothing():
    """The measured case: English audio came back bn 0.83 / hi 0.16 / en 0.01 -- but what the English engine wrote is
    English words."""
    bn = R("হোয়াট ইজ দ্য প্রাইজ অফ দ্য ইউরিক অ্যাসিড টেস্ট", 1.00)
    en = R("what is the price of the uric acid test", 0.78)
    assert pick_candidate([("bn", bn), ("en", en)], {"bn": 0.83, "hi": 0.16, "en": 0.01}, "bn")[0] == "en"


def test_english_with_some_language_id_support_wins_on_the_agreement_gate_alone():
    bn = R("বাংলা", 0.3)
    en = R("kolkata care diagnostics", 0.9)  # not common words, but LID gives English real probability
    assert pick_candidate([("bn", bn), ("en", en)], {"bn": 0.6, "hi": 0.05, "en": 0.35}, "bn")[0] == "en"


def test_the_english_word_share_separates_english_from_transliteration():
    assert english_word_share("what is the price of a lipid profile test") > 0.7
    assert english_word_share("sibisi tester rate koto") < 0.4
    assert english_word_share("") == 0.0


def test_without_language_id_scores_the_english_gate_is_the_old_one():
    bn = R("হোয়াট ইজ দ্য প্রাইজ", 1.00)
    en = R("sibisi tester rate koto", 1.00)
    assert pick_candidate([("bn", bn), ("en", en)])[0] == "en"  # no scores: unchanged behaviour
