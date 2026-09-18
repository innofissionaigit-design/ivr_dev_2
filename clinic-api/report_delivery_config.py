"""ADDED BY SOURAV -- KCD-384 ("Caller asks for their report to be
sent") -- Report Link Delivery: the second, previously-missing config
point this story's AC needs.

WHY THIS IS A SEPARATE FILE FROM otp_messaging_config.py
======================================================================
That file's send_otp_via_provider() sends the OTP CODE itself -- the
first leg of this story (RULE 4-9). Once the caller speaks that code
back correctly, clinic-api/main.py's verify_report_otp() mints a SECOND,
different credential: a one-time, expiring signed link (RULE 11/12) that
actually carries the report. Sending that link is a distinct delivery
step with its own payload shape and its own failure mode, so it gets its
own clearly-marked config point -- exactly the same "one file, one job"
separation otp_messaging_config.py already established, not a new
convention.

Before this file existed, the token was minted and stored (and
/api/v1/reports/link/{token} correctly validated it), but nothing ever
constructed the actual URL or pushed it anywhere -- the voice agent told
the caller "your report has been securely sent" while nothing was
actually sent. This file, plus the two call sites in
verify_report_otp() below, closes that gap.

======================================================================
WHAT TO DO
======================================================================
Two settings, both environment variables (see deploy/env.sh, the same
place CLINIC_API_BASE/TTS_URL/OTP_MESSAGING_WEBHOOK_URL already live):

1. REPORT_LINK_BASE_URL -- the PUBLIC base URL a patient's phone/e-mail
   client can actually reach to open the link (NOT clinic-api's internal
   CLINIC_API_BASE, which is typically localhost and unreachable from
   outside the pod). Point this at whatever domain actually fronts
   clinic-api for real patients -- a reverse proxy, a public port, a CDN
   in front of it.
2. REPORT_DELIVERY_WEBHOOK_URL -- your own SMS/WhatsApp/e-mail provider's
   endpoint, the same shape as OTP_MESSAGING_WEBHOOK_URL. Leave unset to
   keep running with no real provider connected (see "IF NO URL IS SET"
   below).

That is the ONLY change needed. clinic-api/main.py's verify_report_otp()
already calls build_report_link_url() and send_report_link_via_provider()
below every time OTP verification succeeds, regardless of whether a real
URL is set.

======================================================================
WHAT GETS SENT
======================================================================
A plain JSON POST to REPORT_DELIVERY_WEBHOOK_URL:

    {"phone": "<patient's phone number, digits only>",
     "report_link_url": "<the full, one-time, expiring signed URL>"}

Shape it into whatever your own provider/automation flow expects on your
side of that URL, same as OTP_MESSAGING_WEBHOOK_URL -- that translation
is entirely your side's responsibility, not this file's.

Deliberately its OWN setting rather than forced to reuse
OTP_MESSAGING_WEBHOOK_URL: a real deployment may want the OTP code on
one channel (e.g. SMS, since it must be typed back over the phone) and
the report link on another (e.g. e-mail, since it is clicked, not
spoken). Point both at the same URL if your provider handles both
shapes on one endpoint -- nothing here forces them apart.

======================================================================
IF NO URL IS SET (the default, out of the box)
======================================================================
send_report_link_via_provider() is a safe no-op: it logs (without ever
logging the link itself -- see the security note below) that no provider
is configured, and returns False. The signed_link_token row is still
minted and stored exactly as before, so this system stays fully
runnable and testable with no real provider connected -- whoever needs
the link for a test call reads signed_link_token straight off the
ReportDelivery row instead of it arriving by SMS/e-mail, the same
fallback otp_messaging_config.py's own docstring describes for the OTP
code itself.

======================================================================
WHY A PROVIDER FAILURE NEVER BLOCKS THE OTP-VERIFY RESPONSE
======================================================================
The ReportDelivery row (RULE 11/12's expiring token, and the audit trail
this story's AC names) is already committed to the database BEFORE this
is ever called -- see verify_report_otp(). A network error, timeout, or
non-2xx response from your provider only means this one push attempt
failed; it must never turn a successful OTP verification into a failure
response, the same "fail safe, not fail open" posture
otp_messaging_config.py already holds itself to for the OTP code.
send_report_link_via_provider() therefore always degrades to returning
False on any error instead of raising, and verify_report_otp() records
the true outcome in the delivery row's own audit_note (the "verification
path" part of this story's audit-trail requirement) rather than in the
response the voice agent speaks from -- a caller is never told a link
failed to send just because a provider hiccupped; that distinction lives
in the audit trail for staff, not in the call.

======================================================================
SECURITY NOTE -- never let the signed link itself leak into a log line
======================================================================
The link's token is a bearer credential (RULE 12: whoever holds it can
open the report with no further login) -- same posture as
otp_messaging_config.py's own RULE 9 note on the OTP code. Nothing in
this file ever logs the constructed URL or the raw token, only the last
4 digits of the phone number, purely for a human operator to correlate a
log line with a support ticket.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("clinic_api.report_delivery")

# <<< A DEPLOYING COMPANY PUTS THEIR PUBLIC-FACING DOMAIN HERE >>>
# Leave this as the localhost default while developing -- the link will
# still be minted and stored correctly, it just won't be reachable from
# outside this machine. Prefer setting the REPORT_LINK_BASE_URL
# environment variable instead of editing this literal -- it always
# takes priority.
REPORT_LINK_BASE_URL = os.environ.get("REPORT_LINK_BASE_URL", "http://localhost:8080")

# <<< A DEPLOYING COMPANY PUTS THEIR REPORT-DELIVERY PROVIDER'S URL HERE >>>
# Leave this exactly as "" to keep running without a real provider
# connected (see "IF NO URL IS SET" above). Prefer setting the
# REPORT_DELIVERY_WEBHOOK_URL environment variable instead of editing
# this literal -- it always takes priority.
REPORT_DELIVERY_WEBHOOK_URL = os.environ.get("REPORT_DELIVERY_WEBHOOK_URL", "")

REPORT_DELIVERY_TIMEOUT_S = 5.0


def _masked_phone_last4(phone: str) -> str:
    digits = "".join(c for c in phone if c.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits


def build_report_link_url(token: str) -> str:
    """-> the full, public URL a patient's phone/e-mail client opens to
    reach clinic-api's own GET /api/v1/reports/link/{token} (RULE 11/12
    -- that endpoint alone decides whether the token is actually valid;
    this function only builds the string, it never re-validates it)."""
    return f"{REPORT_LINK_BASE_URL.rstrip('/')}/api/v1/reports/link/{token}"


def send_report_link_via_provider(phone: str, report_link_url: str) -> bool:
    """POSTs {"phone": phone, "report_link_url": report_link_url} to
    REPORT_DELIVERY_WEBHOOK_URL. Returns True only on a real 2xx response
    from that URL; False whenever no URL is configured or the call fails
    for any reason. Never raises -- see this file's own module docstring
    ("WHY A PROVIDER FAILURE NEVER BLOCKS THE OTP-VERIFY RESPONSE")."""
    if not REPORT_DELIVERY_WEBHOOK_URL:
        logger.info(
            "REPORT_DELIVERY_WEBHOOK_URL is not set -- no real provider connected; "
            "skipping report-link send for phone ending in %s (see "
            "clinic-api/report_delivery_config.py).",
            _masked_phone_last4(phone),
        )
        return False
    try:
        r = httpx.post(
            REPORT_DELIVERY_WEBHOOK_URL,
            json={"phone": phone, "report_link_url": report_link_url},
            timeout=REPORT_DELIVERY_TIMEOUT_S,
        )
        r.raise_for_status()
        return True
    except httpx.HTTPError as e:
        # Security note above: report_link_url is deliberately never
        # interpolated into this (or any) log line -- same discipline as
        # otp_messaging_config.py's own send_otp_via_provider().
        logger.error(
            "Report delivery provider call failed for phone ending in %s: %s",
            _masked_phone_last4(phone), e,
        )
        return False
