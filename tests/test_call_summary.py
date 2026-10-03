"""agent/call_summary.py: a factual, never-invented note for the human who reads a callback/prescription record.

python -m pytest tests/test_call_summary.py -v
"""

from hypothesis import given
from hypothesis import strategies as st

from agent.call_summary import MAX_TRANSCRIPT_CHARS, build_call_summary


def test_every_given_fact_appears_in_the_summary_verbatim():
    s = build_call_summary(
        "lay_term_blood",
        lang="bn",
        transcript="আমার রক্ত পরীক্ষা করাতে হবে",
        phone="9830012345",
        patient_name="Ravi Das",
        call_id="c1",
    )
    assert "reason: lay_term_blood" in s
    assert "language: bn" in s
    assert '"আমার রক্ত পরীক্ষা করাতে হবে"' in s
    assert "phone: 9830012345" in s
    assert "name: Ravi Das" in s
    assert "call_id: c1" in s


def test_omitted_optional_facts_are_not_mentioned_at_all():
    s = build_call_summary("test_not_found", lang="en")
    assert "phone:" not in s and "name:" not in s and "call_id:" not in s and "caller said:" not in s
    assert s == "reason: test_not_found | language: en"


def test_a_long_transcript_is_clipped_not_silently_truncated_without_a_marker():
    long_text = "a" * (MAX_TRANSCRIPT_CHARS + 50)
    s = build_call_summary("lay_term_head", lang="en", transcript=long_text)
    assert len(s) < len(long_text) + 50
    assert s.count('"a') and s.rstrip('"').endswith("...")


def test_nothing_is_invented_the_reason_is_exactly_what_was_passed_never_reinterpreted():
    s = build_call_summary("handoff:decoders_disagree", lang="hi", transcript="???")
    assert "reason: handoff:decoders_disagree" in s
    assert '"???"' in s  # the caller's own words, quoted -- not paraphrased into something that sounds clearer


@given(
    st.text(max_size=50),
    st.text(max_size=500),
    st.one_of(st.none(), st.text(max_size=20)),
    st.one_of(st.none(), st.text(max_size=20)),
)
def test_build_call_summary_never_raises(reason, transcript, phone, name):
    s = build_call_summary(reason, lang="bn", transcript=transcript, phone=phone, patient_name=name)
    assert isinstance(s, str)
