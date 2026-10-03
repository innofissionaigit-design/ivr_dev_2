"""REST client for the clinic's Java (Spring Boot) + PostgreSQL service.

This service owns the actual facts -- prices, schedules, slot availability
-- and is the only thing allowed to state them. The contract below is what
that service needs to implement; nothing here assumes it exists yet.

Every method returns a plain dict and NEVER raises for a normal "not
found" / "unavailable" outcome -- those are valid, expected answers a
caller can be told. It only raises ToolCallError for actual infrastructure
failure (timeout, connection refused, 5xx), which main.py maps to a
distinct "I couldn't check that right now" reply instead of a false
"not found".
"""

from __future__ import annotations

import os
import uuid

import httpx
from pydantic import BaseModel

from agent.api_models import (
    ConflictAnswer,
    DoctorAnswer,
    FaqAnswer,
    FindAnswer,
    FoundAnswer,
    HomeCollectionEligibilityMultiAnswer,
    HomeCollectionPaymentPolicyAnswer,
    HomeCollectionQuoteAnswer,
    HomeCollectionSlotsAnswer,
    IdentifyAnswer,
    LookupAnswer,
    PrepAnswer,
    PrepMergeAnswer,
    RouteAnswer,
    SuccessAnswer,
    TestAnswer,
    VerifyAnswer,
    validated,
)

DEFAULT_TIMEOUT_S = 4.0  # a phone caller will not wait much longer than this per lookup


class ToolCallError(Exception):
    """The backing service itself failed -- distinct from a normal
    not-found/unavailable result, which is not an error."""


class ClinicToolsClient:
    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        # The clinic API's service token (clinic-api/main.py), read from the environment, never logged.
        token = os.environ.get("CLINIC_API_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s, headers=headers)

    async def aclose(self):
        await self._client.aclose()

    # ---- Tool 1: GET /api/v1/tests/search?name=... ----
    # Expected response shape:
    #   found=true:  {"found": true, "test_name": "...", "rate_inr": 650,
    #                 "sample_type": "Blood", "report_time_hours": 24}
    #   found=false: {"found": false, "query": "...", "did_you_mean": ["..."]}
    async def get_test_rate(self, test_name: str) -> dict:
        try:
            r = await self._client.get("/api/v1/tests/search", params={"name": test_name})
            r.raise_for_status()
            return validated(TestAnswer, r.json(), "get_test_rate")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"get_test_rate({test_name!r}): {e}") from e

    # ---- Tool 2: GET /api/v1/doctors/availability?name=...&date=YYYY-MM-DD ----
    # date is OPTIONAL -- omit it to ask "when is this doctor next available".
    # Expected response shape:
    #   found=true:  {"found": true, "doctor_name": "...", "date": "...",
    #                 "available": true, "chamber_hours": "18:00-20:00",
    #                 "next_available_date": null}
    #                or, if not available that date:
    #                {"found": true, ..., "available": false,
    #                 "next_available_date": "2026-08-27"}
    #   found=false: {"found": false, "query": "..."}
    async def get_doctor_availability(self, doctor_name: str, date: str | None) -> dict:
        params = {"name": doctor_name}
        if date:
            params["date"] = date
        try:
            r = await self._client.get("/api/v1/doctors/availability", params=params)
            r.raise_for_status()
            return validated(DoctorAnswer, r.json(), "get_doctor_availability")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"get_doctor_availability({doctor_name!r}, {date!r}): {e}") from e

    # ---- Tool 4: GET /api/v1/tests/prep?name=... ----
    # Expected response shape:
    #   found=true:  {"found": true, "test_name": "...", "test_name_bn": "...",
    #                 "fasting_required": true, "prep_instructions": "..."}
    #   found=false: {"found": false, "query": "...", "did_you_mean": ["..."]}
    async def get_test_prep(self, test_name: str, lang: str = "bn") -> dict:
        try:
            r = await self._client.get("/api/v1/tests/prep", params={"name": test_name, "lang": lang})
            r.raise_for_status()
            return validated(PrepAnswer, r.json(), "get_test_prep")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"get_test_prep({test_name!r}): {e}") from e

    # ---- KCD-382: POST /api/v1/tests/prep/merge (story "Caller has several tests with conflicting preparation")
    # Resolution and the strictest-constraint merge both happen server-side
    # (clinic-api/main.py::prep_merge_endpoint -> enquiry_service.merge_prep_instructions) -- this client sends the
    # names the model copied verbatim from the caller and only validates the shape of what comes back, exactly like
    # every other tool call here. `lang` is accepted for the same signature shape as get_test_prep even though the
    # merge endpoint itself is language-neutral (the spoken sentence is composed afterwards, in reply_templates).
    # Expected response shape:
    #   found=true:  {"found": true, "test_names": [...], "fasting_hours": 11,
    #                 "water_allowed_while_fasting": true, "prescription_required": false,
    #                 "escalate_to_human": false, "not_found": []}
    #                 -- "escalate_to_human" is also true, with "incomplete_data": [...], when a resolved test is
    #                 known to need fasting but has no structured hour count yet (never guessed, see the merge
    #                 function's own docstring).
    #   found=false: {"found": false}  (none of the requested names resolved)
    async def merge_test_prep(self, test_names: list[str], lang: str = "bn") -> dict:
        try:
            r = await self._client.post("/api/v1/tests/prep/merge", json={"test_names": test_names})
            r.raise_for_status()
            return validated(PrepMergeAnswer, r.json(), "merge_test_prep")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"merge_test_prep({test_names!r}): {e}") from e

    # ---- Tool 5: GET /api/v1/faq?topic=... ----
    # `topic` is the STABLE KEY from /api/v1/catalogue's faq_topics, already
    # resolved by agent/fast_path.py's FAQCatalogue -- never free caller text.
    # Expected response shape:
    #   found=true:  {"found": true, "topic": "hours", "answer": "..."}
    #   found=false: {"found": false, "topic": "..."}
    async def get_faq(self, topic: str, lang: str = "bn") -> dict:
        try:
            r = await self._client.get("/api/v1/faq", params={"topic": topic, "lang": lang})
            r.raise_for_status()
            return validated(FaqAnswer, r.json(), "get_faq")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"get_faq({topic!r}): {e}") from e

    # ---- Tool 3: POST /api/v1/appointments ----
    # Body: {"doctor_name", "date", "time_slot", "patient_name", "phone"}
    # Expected response shape:
    #   success=true:  {"success": true, "confirmation_id": "KCD-20260824-0031",
    #                    "doctor_name": "...", "date": "...", "time_slot": "..."}
    #   success=false: {"success": false, "reason": "slot_taken" | "missing_field" | "doctor_not_found",
    #                    "alternative_slots": ["17:30", "18:15"]}
    async def book_appointment(
        self, doctor_name: str, date: str, time_slot: str, patient_name: str, phone: str
    ) -> dict:
        body = {
            "doctor_name": doctor_name,
            "date": date,
            "time_slot": time_slot,
            "patient_name": patient_name,
            "phone": phone,
        }
        try:
            r = await self._client.post("/api/v1/appointments", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "book_appointment")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"book_appointment({body!r}): {e}") from e

    # =========================================================================
    # Epic E26: booking, rescheduling and cancellation. Two-phase (hold, then
    # confirm) for the same reason /api/v1/bookings/hold itself is two-phase
    # -- see clinic-api/booking_service.py and models.SlotLock's docstrings
    # for the concurrency reasoning (KCD-376).
    # =========================================================================
    async def hold_slot(self, doctor_name: str, date: str, time_slot: str) -> dict:
        body = {"doctor_name": doctor_name, "date": date, "time_slot": time_slot}
        try:
            r = await self._client.post("/api/v1/bookings/hold", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "hold_slot")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"hold_slot({body!r}): {e}") from e

    async def confirm_booking(
        self,
        hold_token: str,
        doctor_id: int,
        date: str,
        time_slot: str,
        patient_name: str,
        phone: str,
        caller_phone: str,
        patient_age: int | None = None,
        relationship: str = "self",
    ) -> dict:
        body = {
            "hold_token": hold_token,
            "doctor_id": doctor_id,
            "date": date,
            "time_slot": time_slot,
            "patient_name": patient_name,
            "phone": phone,
            "caller_phone": caller_phone,
            "patient_age": patient_age,
            "relationship": relationship,
        }
        try:
            r = await self._client.post("/api/v1/bookings/confirm", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "confirm_booking")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"confirm_booking({body!r}): {e}") from e

    async def reschedule_appointment(self, confirmation_id: str, new_date: str, new_time_slot: str) -> dict:
        body = {"confirmation_id": confirmation_id, "new_date": new_date, "new_time_slot": new_time_slot}
        try:
            r = await self._client.post("/api/v1/bookings/reschedule", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "reschedule_appointment")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"reschedule_appointment({body!r}): {e}") from e

    async def cancel_appointment(self, confirmation_id: str, confirm_charge: bool = False) -> dict:
        body = {"confirmation_id": confirmation_id, "confirm_charge": confirm_charge}
        try:
            r = await self._client.post("/api/v1/bookings/cancel", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "cancel_appointment")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"cancel_appointment({body!r}): {e}") from e

    async def lookup_bookings(
        self, phone: str | None = None, confirmation_id: str | None = None, name: str | None = None
    ) -> dict:
        params = {k: v for k, v in {"phone": phone, "confirmation_id": confirmation_id, "name": name}.items() if v}
        try:
            r = await self._client.get("/api/v1/bookings/lookup", params=params)
            r.raise_for_status()
            return validated(LookupAnswer, r.json(), "lookup_bookings")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"lookup_bookings({params!r}): {e}") from e

    # KCD-084: senior mode persisted against the patient (a boolean only).
    async def set_patient_senior(self, phone: str, senior: bool = True, caller_phone: str | None = None) -> dict:
        try:
            r = await self._client.post(
                "/api/v1/patients/senior", json={"phone": phone, "senior": senior, "caller_phone": caller_phone}
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON
            raise ToolCallError(f"set_patient_senior: {e}") from e

    async def get_patient_senior(self, phone: str, caller_phone: str | None = None) -> bool:
        try:
            r = await self._client.get(
                "/api/v1/patients/senior",
                params={"phone": phone, **({"caller_phone": caller_phone} if caller_phone else {})},
            )
            r.raise_for_status()
            return bool(r.json().get("senior"))
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON
            raise ToolCallError(f"get_patient_senior: {e}") from e

    async def booking_conflict(self, phone: str, date: str, time_slot: str) -> dict:
        params = {"phone": phone, "date": date, "time_slot": time_slot}
        try:
            r = await self._client.get("/api/v1/bookings/conflict", params=params)
            r.raise_for_status()
            return validated(ConflictAnswer, r.json(), "booking_conflict")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"booking_conflict({params!r}): {e}") from e

    async def book_tests(
        self,
        test_names: list[str],
        date: str,
        patient_name: str,
        phone: str,
        caller_phone: str,
        patient_age: int | None = None,
        relationship: str = "self",
    ) -> dict:
        body = {
            "test_names": test_names,
            "date": date,
            "patient_name": patient_name,
            "phone": phone,
            "caller_phone": caller_phone,
            "patient_age": patient_age,
            "relationship": relationship,
        }
        try:
            r = await self._client.post("/api/v1/bookings/tests", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "book_tests")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"book_tests({body!r}): {e}") from e

    async def add_test_to_booking(self, confirmation_id: str, test_name: str) -> dict:
        body = {"confirmation_id": confirmation_id, "test_name": test_name}
        try:
            r = await self._client.post("/api/v1/bookings/add-test", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "add_test_to_booking")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"add_test_to_booking({body!r}): {e}") from e

    async def doctor_earliest(self, doctor_name: str) -> dict:
        try:
            r = await self._client.get("/api/v1/doctors/earliest", params={"name": doctor_name})
            r.raise_for_status()
            return validated(FoundAnswer, r.json(), "doctor_earliest")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"doctor_earliest({doctor_name!r}): {e}") from e

    async def route_department(self, query_text: str, lang: str = "bn") -> dict:
        params = {"query": query_text, "lang": lang}
        try:
            r = await self._client.get("/api/v1/departments/route", params=params)
            r.raise_for_status()
            return validated(RouteAnswer, r.json(), "route_department")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"route_department({params!r}): {e}") from e

    async def resend_confirmation(self, confirmation_id: str) -> dict:
        try:
            r = await self._client.post(
                "/api/v1/bookings/resend",
                json={"confirmation_id": confirmation_id},
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "resend_confirmation")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"resend_confirmation({confirmation_id!r}): {e}") from e

    async def request_callback(
        self, phone: str, call_id: str, requested_window: str, reason: str = "", call_summary: str = ""
    ) -> dict:
        """Epic E27's callback queue (clinic-api/enquiry_service.request_callback): the durable half of "a human
        colleague will call you back" -- there is no outbound-calling system, so this only records the promise, never
        claims a call was placed. `call_summary` (agent/call_summary.py) rides along so the human has the call's
        context next to the phone number."""
        body = {
            "phone": phone,
            "call_id": call_id,
            "requested_window": requested_window,
            "reason": reason,
            "call_summary": call_summary,
        }
        try:
            r = await self._client.post("/api/v1/callbacks", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "request_callback")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"request_callback({body!r}): {e}") from e

    async def list_departments(self) -> dict:
        """Every department and how many doctors sit in each.

        Ported story "Caller asks which doctors are available". Distinct from
        route_department() below, which maps a SYMPTOM to one department and
        cannot answer "what departments do you have".
        """
        try:
            r = await self._client.get("/api/v1/departments")
            r.raise_for_status()
            return validated(FoundAnswer, r.json(), "list_departments")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"list_departments(): {e}") from e

    async def list_doctors(self) -> dict:
        """Every doctor, with the department each sits in.

        How many to actually SAY is a presentation decision and belongs to
        agent/reply_templates.py, not here: this returns the facts.
        """
        try:
            r = await self._client.get("/api/v1/doctors")
            r.raise_for_status()
            return validated(FoundAnswer, r.json(), "list_doctors")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"list_doctors(): {e}") from e

    async def doctors_by_department(self, department: str) -> dict:
        """The doctors in one named department.

        Resolution follows the same no-guessing discipline as get_test_rate and
        get_doctor_availability: a written-form match resolves, several matches
        come back `ambiguous` with the names offered, and a near match is a
        `did_you_mean` suggestion the agent must ask about -- never a silent best
        guess at which department the caller meant.
        """
        try:
            r = await self._client.get("/api/v1/doctors/by-department", params={"department": department})
            r.raise_for_status()
            return validated(FoundAnswer, r.json(), "doctors_by_department")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"doctors_by_department({department!r}): {e}") from e

    async def submit_complaint(
        self, call_id: str, text: str, detected_by: str = "", lang: str = "", phone: str | None = None
    ) -> dict:
        """Ported story "Caller wants to make a complaint" (agent/complaint.py): the durable half of "I have written
        your complaint down". The caller's words go to clinic-api's `complaints` table UNEDITED -- acceptance
        criterion 3 -- which is also the only place in this system they are kept: the escalation ledger and the call
        record stay text-free by design (agent/call_record.py). Nothing here summarises or re-words the complaint; a
        complaint paraphrased by a model is no longer the complaint that was made.

        Deduplicated server side by (call_id, text), like request_callback() above, so a retried call is one
        complaint. No Idempotency-Key header: the endpoint is deliberately not @idempotent (see its own docstring in
        clinic-api/main.py for why that decorator cannot be used on this file's endpoints today)."""
        body = {"call_id": call_id, "text": text, "detected_by": detected_by, "lang": lang, "phone": phone}
        try:
            r = await self._client.post("/api/v1/complaints", json=body)
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "submit_complaint")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"submit_complaint(call_id={call_id!r}): {e}") from e

    async def send_payment_link(self, confirmation_id: str) -> dict:
        """Only a confirmed DOCTOR appointment ever gets a link (clinic-api/booking_service.send_payment_link) -- a
        lab test's confirmation_id belongs to a different table there and is never found, by design: a lab test is
        paid and booked at the counter, never over the call."""
        try:
            r = await self._client.post(
                "/api/v1/payments/send-link",
                json={"confirmation_id": confirmation_id},
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
            r.raise_for_status()
            return validated(SuccessAnswer, r.json(), "send_payment_link")
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"send_payment_link({confirmation_id!r}): {e}") from e

    async def save_draft_booking(self, caller_phone: str, call_id: str, slots_json: str) -> dict:
        body = {"caller_phone": caller_phone, "call_id": call_id, "slots_json": slots_json}
        try:
            r = await self._client.post("/api/v1/bookings/draft", json=body)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON
            raise ToolCallError(f"save_draft_booking({body!r}): {e}") from e

    async def get_draft_booking(self, caller_phone: str) -> dict:
        try:
            r = await self._client.get("/api/v1/bookings/draft", params={"phone": caller_phone})
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON
            raise ToolCallError(f"get_draft_booking({caller_phone!r}): {e}") from e

    # ---- Epic E33: patient context and history (clinic-api/patient_context.py) ----
    async def _get(self, path: str, params: dict, what: str, model: type[BaseModel] | None = None) -> dict:
        try:
            r = await self._client.get(path, params={k: v for k, v in params.items() if v is not None})
            r.raise_for_status()
            return validated(model, r.json(), what) if model else r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"{what}: {e}") from e

    async def _post(self, path: str, body: dict, what: str, model: type[BaseModel] | None = None) -> dict:
        try:
            r = await self._client.post(path, json=body, headers={"Idempotency-Key": uuid.uuid4().hex})
            r.raise_for_status()
            return validated(model, r.json(), what) if model else r.json()
        except (httpx.HTTPError, ValueError) as e:  # ValueError: not JSON, or not the expected shape
            raise ToolCallError(f"{what}: {e}") from e

    async def identify_patient(self, phone: str) -> dict:
        """KCD-494: {"status": new|single|ambiguous, "record_exists", "count"} -- never a name."""
        return await self._get("/api/v1/patients/identify", {"phone": phone}, "identify_patient", IdentifyAnswer)

    async def resolve_patient(self, phone: str, name: str, age: int | None = None) -> dict:
        return await self._post(
            "/api/v1/patients/resolve", {"phone": phone, "name": name, "age": age}, "resolve_patient"
        )

    async def patient_timeline(self, patient_ref: int, caller_phone: str, call_id: str) -> dict:
        return await self._get(
            f"/api/v1/patients/{patient_ref}/timeline",
            {"caller_phone": caller_phone, "call_id": call_id},
            "patient_timeline",
        )

    async def patient_test_status(self, patient_ref: int, test_name: str, caller_phone: str, call_id: str) -> dict:
        return await self._get(
            f"/api/v1/patients/{patient_ref}/test-status",
            {"test_name": test_name, "caller_phone": caller_phone, "call_id": call_id},
            "patient_test_status",
        )

    async def get_preferences(self, patient_ref: int, caller_phone: str) -> dict:
        return await self._get(
            f"/api/v1/patients/{patient_ref}/preferences", {"caller_phone": caller_phone}, "get_preferences"
        )

    async def set_preferences(self, patient_ref: int, caller_phone: str, **fields) -> dict:
        return await self._post(
            f"/api/v1/patients/{patient_ref}/preferences", {"caller_phone": caller_phone, **fields}, "set_preferences"
        )

    async def continuity(self, caller_phone: str, call_id: str | None = None, patient_ref: int | None = None) -> dict:
        """KCD-496: unfinished work, the last interactions, and the last calls cached for one day."""
        return await self._get(
            "/api/v1/continuity",
            {"caller_phone": caller_phone, "call_id": call_id, "patient_ref": patient_ref},
            "continuity",
        )

    async def verify_patient(
        self, call_id: str, patient_ref: int, answers: dict, caller_phone: str | None = None
    ) -> dict:
        """KCD-495: the SERVER checks the security answers. {"verified", "attempts_left", "locked",
        and after a pass "age_years", "is_senior"} -- never which answer was wrong."""
        return await self._post(
            "/api/v1/patients/verify",
            {"call_id": call_id, "patient_ref": patient_ref, "caller_phone": caller_phone, **answers},
            "verify_patient",
            VerifyAnswer,
        )

    async def find_patient(self, call_id: str, answers: dict) -> dict:
        """KCD-497: find a patient from a patient id, or a date of birth with a name. Opaque reference only."""
        return await self._post("/api/v1/patients/find", {"call_id": call_id, **answers}, "find_patient", FindAnswer)

    async def agent_messages(self, lang: str | None = None) -> dict:
        """KCD-353/500/513: the operator-editable wording (agent/messages.py)."""
        return await self._get("/api/v1/agent/messages", {"lang": lang}, "agent_messages")

    async def search_bookings(self, caller_phone: str, **criteria) -> dict:
        """KCD-497: any combination of phone, name, approx_date, test_name, branch."""
        return await self._post(
            "/api/v1/bookings/search", {"caller_phone": caller_phone, **criteria}, "search_bookings"
        )

    async def booking_test_names(self, confirmation_id: str) -> dict:
        """KCD-382b: every real confirmed test name in one booking group -- search_bookings()
        above only returns one representative test per confirmation_id, never the full set, so
        the registered/booked-patient preparation path reads it separately, once a single
        confirmation_id has already been resolved."""
        return await self._get(
            f"/api/v1/bookings/{confirmation_id}/tests", {}, "booking_test_names"
        )

    # =========================================================================
    # ADDED BY SOURAV: home-collection full flow (KCD-387 full flow). Previously this client had
    # NO method at all calling the home-collection backend -- the eligibility endpoint existed in
    # clinic-api but nothing in the agent ever reached it (the architecture investigation's own
    # central finding). These methods are the fix for that specific gap, plus the new slot/quote/
    # payment-policy/booking endpoints the full-flow story needs. Hold-then-confirm mirrors
    # hold_slot/confirm_booking's own two-phase shape above (KCD-376), applied to a different
    # resource (a home-collection slot, not a doctor's time).
    # =========================================================================
    async def home_collection_eligibility(self, test_name: str, postal_code: str) -> dict:
        # ADDED BY SOURAV: call the existing home-collection eligibility service instead of
        # duplicating eligibility rules in the agent.
        return await self._get(
            "/api/v1/home-collection/eligibility",
            {"test_name": test_name, "postal_code": postal_code},
            "home_collection_eligibility",
            FoundAnswer,
        )

    async def home_collection_eligibility_multi(self, test_names: list[str], postal_code: str) -> dict:
        # ADDED BY SOURAV: CASE 3 of the story -- every requested/booked test's eligibility,
        # evaluated and returned individually, never merged or silently reduced to one.
        try:
            r = await self._client.get(
                "/api/v1/home-collection/eligibility-multi",
                params={"test_names": test_names, "postal_code": postal_code},
            )
            r.raise_for_status()
            return validated(HomeCollectionEligibilityMultiAnswer, r.json(), "home_collection_eligibility_multi")
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"home_collection_eligibility_multi({test_names!r}): {e}") from e

    async def home_collection_slots(self, postal_code: str, date: str) -> dict:
        # ADDED BY SOURAV: real DB-backed slot availability for a caller's date question --
        # never a calculated/estimated window.
        return await self._get(
            "/api/v1/home-collection/slots",
            {"postal_code": postal_code, "date": date},
            "home_collection_slots",
            HomeCollectionSlotsAnswer,
        )

    async def home_collection_quote(self, test_names: list[str], postal_code: str) -> dict:
        # ADDED BY SOURAV: the deterministic test-price + home-collection-charge + total
        # breakdown -- the agent only ever formats these numbers, never computes them.
        try:
            r = await self._client.get(
                "/api/v1/home-collection/quote", params={"test_names": test_names, "postal_code": postal_code}
            )
            r.raise_for_status()
            return validated(HomeCollectionQuoteAnswer, r.json(), "home_collection_quote")
        except (httpx.HTTPError, ValueError) as e:
            raise ToolCallError(f"home_collection_quote({test_names!r}): {e}") from e

    async def home_collection_payment_policy(self, lang: str = "bn") -> dict:
        # ADDED BY SOURAV: the real, versioned payment policy -- never a string the agent invents.
        return await self._get(
            "/api/v1/home-collection/payment-policy",
            {"lang": lang},
            "home_collection_payment_policy",
            HomeCollectionPaymentPolicyAnswer,
        )

    async def home_collection_hold(
        self, slot_id: int, caller_phone: str | None = None, call_id: str | None = None
    ) -> dict:
        # ADDED BY SOURAV: claim one unit of real capacity before asking the caller to confirm,
        # so two simultaneous callers can never both book the same capacity (KCD-376-style hold).
        return await self._post(
            "/api/v1/home-collection/hold",
            {"slot_id": slot_id, "caller_phone": caller_phone, "call_id": call_id},
            "home_collection_hold",
            SuccessAnswer,
        )

    async def home_collection_release_hold(self, hold_token: str) -> dict:
        # ADDED BY SOURAV: the caller changed their mind before confirming -- free the capacity
        # immediately rather than waiting out the hold's TTL.
        return await self._post(
            "/api/v1/home-collection/release-hold", {"hold_token": hold_token}, "home_collection_release_hold",
            SuccessAnswer,
        )

    async def home_collection_book(
        self,
        hold_token: str,
        test_names: list[str],
        postal_code: str,
        patient_name: str,
        phone: str,
        caller_phone: str | None,
        address_line: str,
        locality: str = "",
        city: str = "",
        state: str = "",
        landmark: str = "",
        patient_id: int | None = None,
    ) -> dict:
        # ADDED BY SOURAV: the actual, persistent home-collection booking transaction -- never
        # created before this call, and never faked in session memory only.
        body = {
            "hold_token": hold_token,
            "test_names": test_names,
            "postal_code": postal_code,
            "patient_name": patient_name,
            "phone": phone,
            "caller_phone": caller_phone,
            "address_line": address_line,
            "locality": locality,
            "city": city,
            "state": state,
            "landmark": landmark,
            "patient_id": patient_id,
        }
        return await self._post("/api/v1/home-collection/book", body, "home_collection_book", SuccessAnswer)

    async def home_collection_booking_status(self, booking_reference: str) -> dict:
        return await self._get(
            f"/api/v1/home-collection/booking/{booking_reference}", {}, "home_collection_booking_status", FoundAnswer
        )

    async def home_collection_cancel(self, booking_reference: str) -> dict:
        return await self._post(
            "/api/v1/home-collection/cancel", {"booking_reference": booking_reference}, "home_collection_cancel",
            SuccessAnswer,
        )

    async def write_call_event(
        self, call_id: str, seq: int, kind: str, payload: dict | None = None, caller_phone: str | None = None
    ) -> dict:
        """KCD-501: idempotent by (call_id, seq)."""
        return await self._post(
            f"/api/v1/calls/{call_id}/events",
            {"seq": seq, "kind": kind, "payload": payload or {}, "caller_phone": caller_phone},
            "write_call_event",
        )
