"""KCD-353, text channel: every outbound message says it comes from an automated
assistant. Kept in its own tiny module because clinic-api is a separate service
that does not import the voice agent's `agent/` package; the version string is
asserted equal to agent/disclosure.DISCLOSURE_VERSION by
tests/test_disclosure_and_human_request.py, so the two cannot drift silently.

The wording is DRAFT and pending clinical and legal review (see agent/disclosure.py).
"""

DISCLOSURE_VERSION = "2.1-draft"

TEXT_NOTICE = "Sent by Sonoscan Vaani, an automated assistant. To speak to a person, call the counter."


def with_notice(message: str) -> str:
    """`message` with the notice appended once (idempotent)."""
    if TEXT_NOTICE in message:
        return message
    return f"{message.rstrip()} {TEXT_NOTICE}"
