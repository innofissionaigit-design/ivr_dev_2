"""agent/identity.py: what we know about who is on the call -- claim, verification,
revocation and expiry. Pure state; the clock is injected.

    python -m pytest tests/test_identity.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.identity import CLAIMED, NONE, VERIFICATION_TTL_S, VERIFIED, IdentityState


class Clock:
    def __init__(self):
        self.t = 5000.0

    def __call__(self):
        return self.t


def _state():
    c = Clock()
    return IdentityState(clock=c), c


def test_a_new_call_knows_nothing_and_discloses_nothing():
    s, _ = _state()
    assert s.level == NONE and not s.is_verified and not s.may_disclose_for("p1")


def test_a_claim_is_not_a_verification():
    s, _ = _state()
    s.claim("p1")
    assert s.level == CLAIMED and not s.is_verified
    assert not s.may_disclose_for("p1")  # saying who you are discloses nothing


def test_a_claim_never_upgrades_a_verified_level_or_replaces_the_reference():
    s, _ = _state()
    s.verify("otp", "p1")
    s.claim("someone-else")
    assert s.level == VERIFIED and s.patient_ref == "p1"


def test_verification_allows_disclosure_only_for_the_verified_patient():
    s, _ = _state()
    s.verify("otp", "p1")
    assert s.may_disclose_for("p1")
    assert not s.may_disclose_for("p2")  # verified as p1 says nothing about p2
    assert not s.may_disclose_for(None)


def test_revocation_drops_to_claimed_and_stops_disclosure():
    s, _ = _state()
    s.verify("dob_confirmed", "p1")
    assert s.revoke("speaker_changed") is True
    assert s.level == CLAIMED and not s.may_disclose_for("p1")
    assert s.patient_ref == "p1"  # the booking context is kept; only the trust is lost
    assert s.revocations == ["speaker_changed"]


def test_revoking_something_never_verified_is_a_no_op():
    s, _ = _state()
    assert s.revoke("speaker_changed") is False and s.revocations == []


def test_a_verification_expires():
    s, c = _state()
    s.verify("otp", "p1")
    c.t += VERIFICATION_TTL_S - 1
    assert s.is_verified
    c.t += 2
    assert not s.is_verified and s.revocations == ["expired"] and s.level == CLAIMED


def test_reverification_after_revocation_restores_disclosure():
    s, c = _state()
    s.verify("otp", "p1")
    s.revoke("speaker_changed")
    s.verify("otp", "p1")
    assert s.may_disclose_for("p1")


def test_the_log_form_never_contains_the_patient_reference():
    s, _ = _state()
    s.verify("otp", "patient-12345")
    assert "patient-12345" not in str(s.to_log_dict())
