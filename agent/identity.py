"""Who is on this call, and how sure are we? One small state object per call.

Three different things were being called "the phone" in this codebase, and they
must not be collapsed (a review of this repository made the point sharply):

  * a number the caller SAID          -- a claim, never proof of ownership;
  * the network's caller ID (ANI)     -- useful, spoofable, not in this repo yet;
  * a person who has PASSED a check   -- a one-time password, a confirmed date of
                                         birth for a proxy, whatever the clinic
                                         decides counts.

`IdentityState` records which of those we actually have, and every disclosure
decision reads it rather than a phone string. Levels:

    none      nothing known
    claimed   the caller stated a number or a name; NOTHING personal may be spoken
    verified  a check passed (method recorded); personal history may be spoken

A verification can be REVOKED -- by a detected change of speaker
(agent/speaker_change.py, KCD-054), by an explicit "that was not me", by timeout
-- and revocation drops the call back to `claimed`: the caller keeps their booking
context (nothing is lost) but no further disclosure happens until they verify
again. The mechanism that VERIFIES (an OTP to the registered number, KCD-203) is
external to this module and to this repository; it plugs in by calling verify().

Pure Python, no I/O, clock injected -- testable off-pod.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

NONE = "none"
CLAIMED = "claimed"
VERIFIED = "verified"

# How long a verification stays good without renewal (REASONED; the clinic's
# security policy decides the real value).
VERIFICATION_TTL_S = 15 * 60


@dataclasses.dataclass
class IdentityState:
    clock: Callable[[], float] = time.time
    level: str = NONE
    method: str | None = None  # "otp" | "dob_confirmed" | ... (whatever verify() was told)
    patient_ref: str | None = None  # an opaque reference, never a phone number or a name
    phone: str | None = None  # the registered number the record was found by; never logged
    verified_at: float | None = None
    revocations: list[str] = dataclasses.field(default_factory=list)
    ttl_s: float = VERIFICATION_TTL_S

    # ------------------------------------------------------------ transitions
    def claim(self, patient_ref: str | None = None, phone: str | None = None) -> None:
        """The caller stated who they are. Never raises the level above `claimed`."""
        if self.level == NONE:
            self.level = CLAIMED
        if patient_ref and self.patient_ref is None:
            self.patient_ref = patient_ref
        if phone and self.phone is None:
            self.phone = phone

    def verify(self, method: str, patient_ref: str, phone: str | None = None) -> None:
        self.level, self.method, self.patient_ref, self.verified_at = VERIFIED, method, patient_ref, self.clock()
        if phone:
            self.phone = phone

    def revoke(self, reason: str) -> bool:
        """Drop back to `claimed`. Returns True if a verification was actually lost."""
        was = self.is_verified
        if was:
            self.level, self.method, self.verified_at = CLAIMED, None, None
            self.revocations.append(reason)
        return was

    # ---------------------------------------------------------------- queries
    @property
    def is_verified(self) -> bool:
        if self.level != VERIFIED or self.verified_at is None:
            return False
        if self.clock() - self.verified_at > self.ttl_s:
            self.level, self.method, self.verified_at = CLAIMED, None, None
            self.revocations.append("expired")
            return False
        return True

    def may_disclose_for(self, patient_ref: str | None) -> bool:
        """May something personal about `patient_ref` be spoken? Only when a check
        passed AND it was for that same patient."""
        return self.is_verified and patient_ref is not None and patient_ref == self.patient_ref

    def to_log_dict(self) -> dict:
        """Safe for logs: levels and methods, never the reference itself."""
        return {"level": self.level, "method": self.method, "revocations": list(self.revocations)}
