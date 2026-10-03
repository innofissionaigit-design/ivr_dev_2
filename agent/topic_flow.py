"""A caller who changes the subject mid-booking keeps the booking (KCD-104).

THE PROBLEM. Cross-turn slot memory and corrections already existed (agent/booking_flow.py: a later value
overwrites an earlier one, `correction_acknowledgement` says what changed). What did not exist:

  * A question asked in the middle of a booking ("what does the CBC cost?") was answered, but nothing brought the
    caller back, so the booking quietly stalled.
  * At the last step, "shall I confirm this?", EVERY reply was treated as yes or no. A question, or a correction
    ("make it 11 o'clock"), was answered by reading the confirmation out again: never understood, never answered.
  * Starting a DIFFERENT task (say, cancelling another appointment) while one was half-collected replaced the
    booking state, and the caller's captured details were gone.

WHAT THIS MODULE IS. The pure decisions, with no model, network or audio in them:

    classify(booking, intent)     what this turn is, relative to the task in progress
    worth_suspending(booking)     is there anything a caller would be annoyed to lose
    resume_line / confirm_resume_line / offer_resume   what the agent says to bring the caller back

and the wording, in Bengali, Hindi and English. The orchestrator (main.py) does the I/O: it answers the question,
keeps or suspends the state, and speaks the resume line. The resume line is a question ("...Shall I confirm it?" or
the next missing detail), so it is skipped when the reply it would follow already asks one, and when the caller-state
policy allows no question at all.

DECIDED, NOT GUESSED. Whether a turn is a topic change is decided from the intent the extractor returned, which is
already validated against the closed intent set (agent/llm.py); this module never reads the words. A yes or a no is
still decided by agent/booking_flow.classify_yes_no, deterministically, and anything else at the confirmation step
is treated as a fresh turn rather than a guess at yes or no.
"""

from __future__ import annotations

from agent.booking_flow import BookingState
from agent.enquiry_followup import ENQUIRY_INTENTS

BOOKING_ACTIONS = (
    "book_appointment",
    "book_test",
    "reschedule_appointment",
    "cancel_appointment",
    "add_test_booking",
)

# What a turn is, relative to the task in progress.
NO_TASK = "none"  # nothing in progress
CONTINUE = "continue"  # the same task, a correction, or a bare answer to what was asked
TOPIC_CHANGE = "topic_change"  # an enquiry or small talk: answer it, keep the task
NEW_TASK = "new_task"  # a different booking action: suspend the task, start the new one


def classify(booking: BookingState | None, intent: str) -> str:
    if booking is None or booking.stage == "done" or booking.is_stale():
        return NO_TASK
    if intent == booking.action or intent == "unclear":
        return CONTINUE
    if intent in BOOKING_ACTIONS:
        return NEW_TASK
    if intent in ENQUIRY_INTENTS or intent in (
        "smalltalk",
        "lookup_booking",
        "resend_confirmation",
    ):
        return TOPIC_CHANGE
    return CONTINUE  # an intent this module does not know: leave the task alone, decide nothing


def worth_suspending(booking: BookingState | None) -> bool:
    """A task with nothing captured is not worth a resume offer; one with any detail, a hold, or a pending
    confirmation is."""
    if booking is None or booking.stage == "done" or booking.is_stale():
        return False
    return (
        bool(booking.slots)
        or bool(booking.test_names)
        or booking.hold_token is not None
        or booking.stage in ("confirming", "awaiting_charge_confirm")
    )


def resume_allowed(questions_per_turn: int, reply_so_far: str) -> bool:
    """A resume line is a question. Not when the policy allows none, and not on top of a reply that already asks."""
    return questions_per_turn >= 1 and "?" not in reply_so_far and "؟" not in reply_so_far


# ---------------------------------------------------------------------------------------------------- wording

_BACK = {
    "en": "Coming back to your booking.",
    "hi": "अब आपकी बुकिंग पर आते हैं।",
    "bn": "এবার আপনার বুকিংয়ের কথায় আসি।",
}
_CONFIRM_Q = {
    "en": "Shall I confirm it?",
    "hi": "क्या मैं इसे कन्फ़र्म कर दूँ?",
    "bn": "কনফার্ম করে দেব?",
}
_WHAT = {
    "book_appointment": {
        "en": "booking an appointment",
        "hi": "अपॉइंटमेंट बुक करने के बारे में",
        "bn": "অ্যাপয়েন্টমেন্ট বুক করা নিয়ে",
    },
    "book_test": {
        "en": "booking a test",
        "hi": "टेस्ट बुक करने के बारे में",
        "bn": "টেস্ট বুক করা নিয়ে",
    },
    "reschedule_appointment": {
        "en": "changing an appointment",
        "hi": "अपॉइंटमेंट का समय बदलने के बारे में",
        "bn": "অ্যাপয়েন্টমেন্টের সময় বদল নিয়ে",
    },
    "cancel_appointment": {
        "en": "cancelling an appointment",
        "hi": "अपॉइंटमेंट रद्द करने के बारे में",
        "bn": "অ্যাপয়েন্টমেন্ট বাতিল করা নিয়ে",
    },
    "add_test_booking": {
        "en": "adding a test",
        "hi": "टेस्ट जोड़ने के बारे में",
        "bn": "টেস্ট যোগ করা নিয়ে",
    },
}
# Each phrase is a noun phrase that reads correctly after "talking about" in its language (an earlier draft used
# "you were doing <gerund>", which came out as "doing changing an appointment").
_OFFER = {
    "en": "Earlier we were talking about {what}. Shall I go back to it?",
    "hi": "पहले हम {what} बात कर रहे थे। क्या मैं उसी पर वापस चलूँ?",
    "bn": "আগে আমরা {what} কথা বলছিলাম। সেটায় ফিরে যাব?",
}


def resume_line(next_question: str, lang: str) -> str:
    """After answering an aside in the middle of collecting details: back to the booking, with the next question."""
    return f"{_BACK.get(lang, _BACK['en'])} {next_question}"


def confirm_resume_line(lang: str) -> str:
    """After answering an aside at the confirmation step: the pending confirmation is asked again, briefly."""
    return f"{_BACK.get(lang, _BACK['en'])} {_CONFIRM_Q.get(lang, _CONFIRM_Q['en'])}"


def offer_resume(action: str, lang: str) -> str:
    """After a NEW task is finished: offer, never assume, to return to the one that was suspended."""
    what = _WHAT.get(action, _WHAT["book_appointment"])
    return _OFFER.get(lang, _OFFER["en"]).format(what=what.get(lang, what["en"]))
