"""Hindi and English reply templates.

Same discipline as reply_templates.py (which owns Bengali and dispatches
here): every fact in a spoken sentence is a substitution from a live API
response, never model text. Only the sentence around the fact changes with
the language.

Spoken-name rule, same as the Bengali `_spoken_test_name`: the voice for a
language cannot pronounce another language's script, so the name spoken is
the one in the caller's own script -- Devanagari for Hindi (from the API's
seeded `*_hi` alias), the catalogue's English label for English -- and
never a name in a script the TTS would silently drop.
"""

from __future__ import annotations

import re

from agent.home_collection_flow import MAX_SLOTS_READ_ALOUD
from agent.sample_wording import sample_sentence

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

_HI_MISSING = {
    ("test_rate", "test_name"): "किस टेस्ट की कीमत जाननी है, ज़रा बताइए?",
    ("doctor_availability", "doctor_name"): "किन डॉक्टर के बारे में पूछ रहे हैं?",
    ("book_appointment", "doctor_name"): "किस डॉक्टर के साथ अपॉइंटमेंट लेना चाहते हैं?",
    ("book_appointment", "date"): "किस दिन के लिए अपॉइंटमेंट चाहिए?",
    ("book_appointment", "time_slot"): "किस समय के लिए अपॉइंटमेंट चाहिए?",
    ("book_appointment", "patient_name"): "मरीज़ का नाम बताइए?",
    ("book_appointment", "phone"): "एक फ़ोन नंबर दीजिए, ताकि मैं कन्फ़र्मेशन भेज सकूँ?",
    ("test_prep", "test_name"): "किस टेस्ट की तैयारी जाननी है, ज़रा बताइए?",
    ("clinic_faq", "faq_topic"): "माफ़ कीजिए, ठीक से समझ नहीं पाई। क्या आप फिर से बताएँगे?",
    ("book_test", "test_names"): "कौन से टेस्ट बुक करना चाहते हैं, ज़रा बताइए?",
    ("book_test", "date"): "किस दिन टेस्ट करवाना चाहते हैं?",
    ("book_test", "patient_name"): "मरीज़ का नाम बताइए?",
    ("book_test", "phone"): "एक फ़ोन नंबर दीजिए, ताकि मैं कन्फ़र्मेशन भेज सकूँ?",
    ("reschedule_appointment", "confirmation_id"): "आपका कन्फ़र्मेशन नंबर या जिस नंबर से बुक किया था वह बताइए?",
    ("reschedule_appointment", "new_date"): "किस दिन ले जाना चाहते हैं?",
    ("reschedule_appointment", "new_time_slot"): "किस समय ले जाना चाहते हैं?",
    ("cancel_appointment", "confirmation_id"): "आपका कन्फ़र्मेशन नंबर या जिस नंबर से बुक किया था वह बताइए?",
    ("add_test_booking", "confirmation_id"): "आपका कन्फ़र्मेशन नंबर बताइए?",
    ("add_test_booking", "test_name"): "कौन सा टेस्ट जोड़ना चाहते हैं?",
    ("lookup_booking", "phone"): "जिस नंबर से बुक किया था वह बताइए?",
    # ADDED BY SOURAV: KCD-387 full flow -- see reply_templates.py's own Bengali entries.
    ("home_collection", "test_name"): "किस टेस्ट के लिए घर पर सैंपल देने की बात पूछ रहे हैं, ज़रा बताइए?",
    ("home_collection", "postal_code"): "आपके इलाके का 6 अंकों का पिनकोड बताइए?",
    ("home_collection", "date"): "किस दिन घर पर सैंपल लेने आना होगा?",
    ("home_collection", "address_line"): "पूरा पता बताइए -- घर नंबर, सड़क, इलाका, शहर सहित?",
    ("home_collection", "patient_name"): "मरीज़ का नाम बताइए?",
    ("home_collection", "phone"): "एक फ़ोन नंबर दीजिए, ताकि बुकिंग की जानकारी भेज सकूँ?",
}
_HI_MISSING_DEFAULT = "माफ़ कीजिए, क्या आप थोड़ा साफ़ बोलेंगे?"

_EN_MISSING = {
    ("test_rate", "test_name"): "Which test would you like the price of?",
    ("doctor_availability", "doctor_name"): "Which doctor are you asking about?",
    ("book_appointment", "doctor_name"): "Which doctor would you like to book with?",
    ("book_appointment", "date"): "Which day would you like the appointment for?",
    ("book_appointment", "time_slot"): "What time would you like the appointment?",
    ("book_appointment", "patient_name"): "May I have the patient's name?",
    ("book_appointment", "phone"): "Could you give me a phone number so I can send the confirmation?",
    ("test_prep", "test_name"): "Which test would you like the preparation for?",
    ("clinic_faq", "faq_topic"): "Sorry, I didn't quite get that. Could you say it again?",
    ("book_test", "test_names"): "Which tests would you like to book?",
    ("book_test", "date"): "Which day would you like the tests for?",
    ("book_test", "patient_name"): "May I have the patient's name?",
    ("book_test", "phone"): "Could you give me a phone number so I can send the confirmation?",
    (
        "reschedule_appointment",
        "confirmation_id",
    ): "Could you give me your confirmation number, or the number you booked from?",
    ("reschedule_appointment", "new_date"): "Which day would you like to move it to?",
    ("reschedule_appointment", "new_time_slot"): "Which time would you like to move it to?",
    (
        "cancel_appointment",
        "confirmation_id",
    ): "Could you give me your confirmation number, or the number you booked from?",
    ("add_test_booking", "confirmation_id"): "Could you give me your confirmation number?",
    ("add_test_booking", "test_name"): "Which test would you like to add?",
    ("lookup_booking", "phone"): "Could you give me the number you booked from?",
    # ADDED BY SOURAV: KCD-387 full flow -- see reply_templates.py's own Bengali entries.
    ("home_collection", "test_name"): "Which test are you asking about for home collection?",
    ("home_collection", "postal_code"): "Could you give me your area's 6-digit pincode?",
    ("home_collection", "date"): "Which day would you like the home visit for?",
    ("home_collection", "address_line"): "Could you give me the full address -- house number, street, area and city?",
    ("home_collection", "patient_name"): "May I have the patient's name?",
    ("home_collection", "phone"): "Could you give me a phone number so I can send the booking details?",
}
_EN_MISSING_DEFAULT = "Sorry, could you say that a little more clearly?"

# Sample types that are a specimen a person gives. "Imaging"/"Cardiac" are
# test categories, not samples -- "sample: imaging" would be nonsense.
_SPECIMENS = {"blood", "urine", "stool", "serum", "saliva", "swab", "plasma"}


def insufficient_information_reply(lang: str) -> str:
    if lang == "hi":
        return "माफ़ कीजिए, ठीक से सुन नहीं पाई। क्या आप फिर से थोड़ा साफ़ बोलेंगे?"
    return "Sorry, I didn't catch that clearly enough. Could you say it again?"


def missing_slot_prompt(intent: str, missing: str, lang: str) -> str:
    if lang == "hi":
        return _HI_MISSING.get((intent, missing), _HI_MISSING_DEFAULT)
    return _EN_MISSING.get((intent, missing), _EN_MISSING_DEFAULT)


def _script_ok(text: str, lang: str) -> bool:
    if lang == "hi":
        return bool(_DEVANAGARI.search(text))
    return bool(_LATIN.search(text)) and not _DEVANAGARI.search(text)


def _suggestions(result: dict, lang: str) -> list[str]:
    """did_you_mean mixes English names and aliases in every script. Keep
    only the ones this language's voice can say, or the caller hears a
    list with the names missing."""
    scripted = result.get(f"did_you_mean_{lang}") or []
    if scripted:
        return list(scripted)
    return [s for s in (result.get("did_you_mean") or []) if _script_ok(s, lang)]


def _test_name(slots: dict, result: dict, lang: str) -> str:
    if lang == "hi":
        heard = slots.get("test_name") or ""
        return result.get("test_name_hi") or (heard if _DEVANAGARI.search(heard) else "") or "यह"
    return result.get("test_name") or slots.get("test_name") or "the test"


def _doctor_name(slots: dict, result: dict, lang: str) -> str:
    if lang == "hi":
        alias = result.get("doctor_name_hi")
        if alias:
            return f"डॉक्टर {alias}"
        heard = slots.get("doctor_name") or ""
        return heard if _DEVANAGARI.search(heard) else "डॉक्टर"
    full = result.get("doctor_name") or slots.get("doctor_name") or ""
    surname = full.replace("Dr.", "").replace("Dr", "").split()[-1:] if full else []
    return f"Doctor {surname[0]}" if surname else "the doctor"


def _query(slots: dict, key: str, lang: str) -> str:
    q = slots.get(key) or ""
    return q if _script_ok(q, lang) else ""


def test_rate_reply(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("found"):
        return _not_found_test(slots, result, lang)
    name = _test_name(slots, result, lang)
    rate = result["rate_inr"]
    hours = result.get("report_time_hours")
    sample = (result.get("sample_type") or "").strip()
    if hi:
        reply = f"{name} टेस्ट की कीमत {rate} रुपये है।"
        sentence = sample_sentence(sample, "hi")
        if sentence:
            reply += f" {sentence}"
        if hours:
            reply += f" रिपोर्ट {hours} घंटे में मिल जाएगी।"
        return reply
    reply = f"The price of the {name} test is {rate} rupees."
    sentence = sample_sentence(sample, "en")
    if sentence:
        reply += f" {sentence}"
    if hours:
        reply += f" You will get the report within {hours} hours."
    return reply


def _not_found_test(slots: dict, result: dict, lang: str) -> str:
    q = _query(slots, "test_name", lang)
    sugg = _suggestions(result, lang)
    if result.get("ambiguous") and sugg:
        # KCD-446: exists, but matched more than one row equally well --
        # distinct framing from "not found" below, which would be false.
        return (
            f"एक से ज़्यादा टेस्ट मिले -- कौन सा: {' या '.join(sugg)}?"
            if lang == "hi"
            else f"I found more than one test -- which one did you mean: {' or '.join(sugg)}?"
        )
    # KCD-lay-terms: see reply_templates.test_rate_reply's Bengali counterpart for why this invitation is here.
    if lang == "hi":
        if sugg:
            return (
                f"'{q}' नाम का टेस्ट नहीं मिला। क्या आप {', '.join(sugg)} कहना चाह रहे हैं?"
                if q
                else f"वह टेस्ट नहीं मिला। क्या आप {', '.join(sugg)} कहना चाह रहे हैं?"
            )
        base = f"माफ़ कीजिए, '{q}' नाम का कोई टेस्ट हमारी सूची में नहीं है।" if q else "माफ़ कीजिए, वह टेस्ट हमारी सूची में नहीं है।"
        return f"{base} कोई और टेस्ट या डॉक्टर की अपॉइंटमेंट बुक करनी है?"
    if sugg:
        return (
            f"I couldn't find a test called '{q}'. Did you mean: {', '.join(sugg)}?"
            if q
            else f"I couldn't find that test. Did you mean: {', '.join(sugg)}?"
        )
    base = f"Sorry, there is no test called '{q}' on our list." if q else "Sorry, that test is not on our list."
    return f"{base} Is there any other test or doctor's appointment you'd like to book?"


def _not_found_doctor(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    q = _query(slots, "doctor_name", lang)
    sugg = _suggestions(result, lang)
    if result.get("ambiguous") and sugg:
        # Doctor-side counterpart of test_rate_reply's ambiguous framing.
        return (
            f"एक से ज़्यादा डॉक्टर मिले -- किनकी बात कर रहे हैं, {' या '.join(sugg)}?"
            if hi
            else f"I found more than one doctor -- did you mean {' or '.join(sugg)}?"
        )
    # KCD-lay-terms: see reply_templates.doctor_availability_reply's Bengali counterpart for why this invitation
    # is here, and why it needs no new state -- department_query already handles the caller's next turn.
    if hi:
        if sugg:
            return (
                f"'{q}' नाम के डॉक्टर नहीं मिले। क्या आप {', '.join(sugg)} कहना चाह रहे हैं?"
                if q
                else f"वह डॉक्टर नहीं मिले। क्या आप {', '.join(sugg)} कहना चाह रहे हैं?"
            )
        base = (
            f"माफ़ कीजिए, '{q}' नाम के कोई डॉक्टर हमारे यहाँ नहीं हैं।"
            if q
            else "माफ़ कीजिए, इस नाम के कोई डॉक्टर हमारे यहाँ नहीं हैं।"
        )
        return f"{base} आपको क्या तकलीफ़ है बताइए, तो मैं सही विभाग के डॉक्टर बता सकूँगी।"
    if sugg:
        return (
            f"I couldn't find a doctor named '{q}'. Did you mean {', '.join(sugg)}?"
            if q
            else f"I couldn't find that doctor. Did you mean {', '.join(sugg)}?"
        )
    base = f"Sorry, we don't have a doctor named '{q}'." if q else "Sorry, we don't have a doctor by that name."
    return f"{base} Tell me what's troubling you, and I can suggest the right doctor."


def doctor_availability_reply(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("found"):
        return _not_found_doctor(slots, result, lang)

    name = _doctor_name(slots, result, lang)
    if result.get("available"):
        hours = result.get("chamber_hours", "")
        date = result.get("date")
        if hi:
            when = f" {date} को" if date else " आज"
            return f"जी हाँ,{when} {name} चेंबर में रहेंगे। समय: {hours}।"
        when = f" on {date}" if date else " today"
        return f"Yes, {name} will be in the chamber{when}. Timing: {hours}."

    next_date = result.get("next_available_date")
    if next_date:
        if hi:
            return f"{name} उस दिन नहीं बैठेंगे। अगला उपलब्ध दिन: {next_date}।"
        return f"{name} does not sit on that day. The next available day is {next_date}."
    if hi:
        return f"{name} अभी किसी तय दिन नहीं बैठ रहे हैं। आप हमारे काउंटर पर पूछ सकते हैं।"
    return f"{name} is not sitting on any fixed day right now. You can ask at our counter."


def test_prep_reply(slots: dict, result: dict, lang: str) -> str:
    if not result.get("found"):
        return _not_found_test(slots, result, lang)
    name = _test_name(slots, result, lang)
    # The lab-test table decides: its preparation text if it has one; else its fasting_required column.
    if result.get("fasting_required"):
        default = (
            "इस टेस्ट के लिए खाली पेट रहना होगा। कितने घंटे, यह काउंटर पर पूछ लें।"
            if lang == "hi"
            else "Fasting is required for this test. Please ask the counter how many hours."
        )
    else:
        default = (
            "इस टेस्ट के लिए किसी ख़ास तैयारी की ज़रूरत नहीं है।"
            if lang == "hi"
            else "No special preparation is needed for this test."
        )
    instructions = result.get("prep_instructions") or default
    return f"{name} टेस्ट के लिए: {instructions}" if lang == "hi" else f"For the {name} test: {instructions}"


def merged_test_prep_reply(test_names: list[str], result: dict, lang: str) -> str:
    """Hindi/English counterpart of reply_templates.merged_test_prep_reply -- see that function's docstring
    (KCD-382, "several tests with conflicting preparation"). Same rule as everywhere else in this file: every
    fact is a substitution from `result`, never composed here."""
    hi = lang == "hi"
    not_found = result.get("not_found") or []
    if not result.get("found") or not_found:
        missing = ", ".join(not_found) if not_found else ", ".join(test_names)
        if hi:
            return f"माफ़ कीजिए, {missing} -- इस नाम का टेस्ट हमारी सूची में नहीं मिला। क्या आप नाम फिर से बताएँगे?"
        return f"Sorry, I couldn't find {missing} on our list. Could you say the test names again?"

    names = ", ".join(result.get("test_names") or test_names)
    hours = result.get("fasting_hours")
    if hours:
        if hi:
            water = "सिर्फ़ पानी पी सकते हैं।" if result.get("water_allowed_while_fasting") else "पानी भी नहीं पी सकते।"
            instruction = f"सबसे सख़्त नियम माननी होगी -- कम से कम {hours} घंटे खाली पेट रहना होगा, {water}"
        else:
            water = "Only water is allowed." if result.get("water_allowed_while_fasting") else "Not even water is allowed."
            instruction = f"The strictest rule applies -- fast for at least {hours} hours. {water}"
    else:
        instruction = (
            "इन टेस्ट के लिए अलग से खाली पेट रहने की ज़रूरत नहीं है।"
            if hi
            else "No fasting is needed for these tests."
        )
    if result.get("prescription_required"):
        instruction += (
            " इनमें से कम से कम एक टेस्ट के लिए डॉक्टर का पर्चा ज़रूरी है।"
            if hi
            else " At least one of these tests needs a doctor's prescription."
        )
    return f"{names} टेस्ट के लिए, {instruction}" if hi else f"For {names}, {instruction}"


def clinic_faq_reply(slots: dict, result: dict, lang: str) -> str:
    fallback = (
        "माफ़ कीजिए, इस बारे में अभी सही जानकारी नहीं दे सकती। कृपया काउंटर पर संपर्क करें।"
        if lang == "hi"
        else "Sorry, I can't give you accurate information on that right now. Please contact the counter."
    )
    if not result.get("found"):
        return fallback
    return result.get("answer") or fallback


def booking_reply(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("success"):
        doctor = _doctor_name(slots, result, lang)
        if hi:
            return (
                f"आपकी अपॉइंटमेंट कन्फ़र्म हो गई है। {doctor}, {result['date']}, "
                f"समय {result['time_slot']}। कन्फ़र्मेशन नंबर: {result['confirmation_id']}।"
            )
        return (
            f"Your appointment is confirmed. {doctor}, {result['date']}, "
            f"at {result['time_slot']}. Your confirmation number is {result['confirmation_id']}."
        )

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return (
                f"वह समय बुक हो चुका है। ये समय खाली हैं: {', '.join(alts)}। कौन सा चाहिए?"
                if hi
                else f"That time is already booked. These times are free: {', '.join(alts)}. Which one would you like?"
            )
        return (
            "वह समय बुक हो चुका है, और आसपास कोई समय खाली नहीं है।"
            if hi
            else "That time is already booked, and nothing nearby is free."
        )
    if reason == "doctor_ambiguous":
        names = result.get("did_you_mean_hi" if hi else "did_you_mean") or result.get("did_you_mean") or []
        if hi:
            return (
                f"एक से ज़्यादा डॉक्टर मिले -- आप किसके बारे में पूछ रहे हैं, {' या '.join(names)}?"
                if names
                else "एक से ज़्यादा डॉक्टर मिले। आप किस डॉक्टर के बारे में पूछ रहे हैं, पूरा नाम बताएँगे?"
            )
        return (
            f"I found more than one doctor -- which do you mean, {' or '.join(names)}?"
            if names
            else "I found more than one doctor. Which one do you mean? Please say the full name."
        )
    if reason == "doctor_not_found":
        q = _query(slots, "doctor_name", lang)
        if hi:
            return f"माफ़ कीजिए, '{q}' नाम के कोई डॉक्टर नहीं मिले।" if q else "माफ़ कीजिए, इस नाम के कोई डॉक्टर नहीं मिले।"
        return f"Sorry, I couldn't find a doctor named '{q}'." if q else "Sorry, I couldn't find a doctor by that name."
    if reason == "doctor_not_available_that_day":
        return (
            "माफ़ कीजिए, उस दिन डॉक्टर नहीं बैठते। कोई और दिन बताएँगे?"
            if hi
            else "Sorry, the doctor doesn't sit that day. Could you say another day?"
        )
    if reason == "invalid_slot":
        return (
            "माफ़ कीजिए, वह समय ठीक नहीं है। चेंबर के समय के भीतर कोई समय बताएँगे?"
            if hi
            else "Sorry, that time isn't valid. Could you give a time within the chamber hours?"
        )
    if reason == "date_in_past":
        return (
            "वह तारीख तो निकल चुकी है। अगले हफ़्ते उसी दिन के लिए बुक कर दूँ?"
            if hi
            else "That date has already passed. Shall I book the same weekday next week instead?"
        )
    if reason == "hold_expired":
        return (
            "माफ़ कीजिए, समय रोका नहीं जा सका, थोड़ी देर हो गई। फिर कोशिश करें?"
            if hi
            else "Sorry, that slot couldn't be held in time. Shall we try again?"
        )
    return (
        "माफ़ कीजिए, अपॉइंटमेंट बुक नहीं हो सका। थोड़ी देर बाद फिर कोशिश करें, या काउंटर पर संपर्क करें।"
        if hi
        else "Sorry, I couldn't book the appointment. Please try again shortly, or contact the counter."
    )


def booking_confirmation_readback(slots: dict, action: str, lang: str) -> str:
    hi = lang == "hi"
    if action == "book_appointment":
        if hi:
            return (
                f"तो {slots.get('doctor_name')} डॉक्टर के पास {slots.get('date')} को, "
                f"समय {slots.get('time_slot')} पर, मरीज़ का नाम {slots.get('patient_name')}, "
                f"फ़ोन नंबर {slots.get('phone')} -- यह अपॉइंटमेंट कन्फ़र्म कर दूँ?"
            )
        return (
            f"So, with Dr. {slots.get('doctor_name')} on {slots.get('date')} at {slots.get('time_slot')}, "
            f"for {slots.get('patient_name')}, phone number {slots.get('phone')} -- shall I confirm this?"
        )
    if action == "book_test":
        tests = ", ".join(slots.get("_test_names_display", [])) or ("टेस्ट" if hi else "the tests")
        if hi:
            return (
                f"तो {slots.get('date')} को {tests} -- मरीज़ का नाम {slots.get('patient_name')}, "
                f"फ़ोन नंबर {slots.get('phone')} -- यह बुकिंग कन्फ़र्म कर दूँ?"
            )
        return (
            f"So {tests} on {slots.get('date')}, for {slots.get('patient_name')}, "
            f"phone number {slots.get('phone')} -- shall I confirm this booking?"
        )
    if action == "reschedule_appointment":
        return (
            f"तो अपॉइंटमेंट {slots.get('new_date')} को, समय {slots.get('new_time_slot')} पर ले जाऊँ?"
            if hi
            else f"So I'll move the appointment to {slots.get('new_date')} at {slots.get('new_time_slot')} -- confirm?"
        )
    if action == "cancel_appointment":
        return "आपकी अपॉइंटमेंट रद्द कर दूँ?" if hi else "Shall I cancel your appointment?"
    if action == "add_test_booking":
        return (
            f"तो आपकी बुकिंग में {slots.get('test_name')} टेस्ट जोड़ दूँ?"
            if hi
            else f"Shall I add the {slots.get('test_name')} test to your booking?"
        )
    return "क्या यह कन्फ़र्म कर दूँ?" if hi else "Shall I confirm this?"


def reschedule_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("success"):
        if hi:
            return (
                f"आपकी अपॉइंटमेंट {result['date']} को, समय {result['time_slot']} पर बदल दी गई है। "
                f"नया कन्फ़र्मेशन नंबर: {result['confirmation_id']}।"
            )
        return (
            f"Your appointment has been moved to {result['date']} at {result['time_slot']}. "
            f"New confirmation number: {result['confirmation_id']}."
        )
    reason = result.get("reason")
    if reason == "not_found":
        return (
            "माफ़ कीजिए, इस कन्फ़र्मेशन नंबर पर कोई अपॉइंटमेंट नहीं मिली।"
            if hi
            else "Sorry, I couldn't find an appointment with that confirmation number."
        )
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return (
                f"वह समय बुक हो चुका है। ये समय खाली हैं: {', '.join(alts)}। कौन सा चाहिए?"
                if hi
                else f"That time is already booked. These times are free: {', '.join(alts)}. Which one would you like?"
            )
        return (
            "वह समय बुक हो चुका है। आपकी पुरानी अपॉइंटमेंट वैसी ही है।"
            if hi
            else "That time is already booked. Your original appointment is unchanged."
        )
    return (
        "माफ़ कीजिए, अपॉइंटमेंट बदली नहीं जा सकी। आपकी पुरानी अपॉइंटमेंट वैसी ही है।"
        if hi
        else "Sorry, I couldn't move the appointment. Your original appointment is unchanged."
    )


def cancel_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    reason = result.get("reason")
    if reason == "charge_confirmation_required":
        charge = result["charge_inr"]
        return (
            f"इस समय रद्द करने पर {charge} रुपये कटेंगे। फिर भी रद्द कर दूँ?"
            if hi
            else f"Cancelling now means a charge of {charge} rupees. Shall I still cancel it?"
        )
    if result.get("success"):
        charge = result.get("charge_inr") or 0
        if charge:
            return (
                f"आपकी अपॉइंटमेंट रद्द कर दी गई है। {charge} रुपये काटे गए हैं।"
                if hi
                else f"Your appointment has been cancelled. A charge of {charge} rupees has been applied."
            )
        return (
            "आपकी अपॉइंटमेंट बिना किसी शुल्क के रद्द कर दी गई है।"
            if hi
            else "Your appointment has been cancelled with no charge."
        )
    if reason == "not_found":
        return (
            "माफ़ कीजिए, इस कन्फ़र्मेशन नंबर पर कोई अपॉइंटमेंट नहीं मिली।"
            if hi
            else "Sorry, I couldn't find an appointment with that confirmation number."
        )
    return (
        "माफ़ कीजिए, रद्द नहीं हो सकी। काउंटर से संपर्क करें।"
        if hi
        else "Sorry, I couldn't cancel it. Please contact the counter."
    )


def lookup_reply(bookings: list[dict], lang: str) -> str:
    hi = lang == "hi"
    if not bookings:
        return (
            "माफ़ कीजिए, आपके नाम कोई आने वाली अपॉइंटमेंट नहीं मिली।"
            if hi
            else "Sorry, I couldn't find any upcoming appointment for you."
        )
    b = bookings[0]
    if hi:
        return (
            f"आपकी अगली अपॉइंटमेंट: {b.get('doctor_name')} डॉक्टर के पास, {b['date']} को, "
            f"समय {b['time_slot']} पर। कन्फ़र्मेशन नंबर: {b['confirmation_id']}। "
            f"चाहें तो लिखित कन्फ़र्मेशन फिर से भेज सकती हूँ।"
        )
    return (
        f"Your next appointment: with {b.get('doctor_name')}, on {b['date']} at {b['time_slot']}. "
        f"Confirmation number: {b['confirmation_id']}. I can resend the written confirmation if you'd like."
    )


def multiple_bookings_reply(bookings: list[dict], lang: str) -> str:
    hi = lang == "hi"
    parts = (
        [f"{b.get('doctor_name')} डॉक्टर के पास {b['date']} की" for b in bookings[:3]]
        if hi
        else [f"with {b.get('doctor_name')} on {b['date']}" for b in bookings[:3]]
    )
    if hi:
        return f"आपके नाम कई बुकिंग हैं -- {', '.join(parts)}। कौन सी बुकिंग, कन्फ़र्मेशन नंबर बताइए?"
    return (
        f"You have more than one booking -- {', '.join(parts)}. Which one -- could you give me the confirmation number?"
    )


def multiple_test_bookings_reply(bookings: list[dict], lang: str) -> str:
    """KCD-382b: see agent.reply_templates.multiple_test_bookings_reply's own docstring (the bn
    primary this function is the hi/en fallback for)."""
    hi = lang == "hi"
    parts = (
        [f"{b.get('test_name')}, {b['date']} की" for b in bookings[:3]]
        if hi
        else [f"{b.get('test_name')} on {b['date']}" for b in bookings[:3]]
    )
    if hi:
        return f"आपके नाम कई टेस्ट बुकिंग हैं -- {', '.join(parts)}। कौन सी बुकिंग, कन्फ़र्मेशन नंबर बताइए?"
    return (
        f"You have more than one test booking -- {', '.join(parts)}. "
        f"Which one -- could you give me the confirmation number?"
    )


def multi_test_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("success"):
        return (
            "माफ़ कीजिए, टेस्ट बुक नहीं हो सके। थोड़ी देर बाद फिर कोशिश करें।"
            if hi
            else "Sorry, I couldn't book the tests. Please try again shortly."
        )
    names = ", ".join(result["test_names"])
    if hi:
        reply = (
            f"{names} -- ये टेस्ट {result['date']} को बुक कर दिए गए हैं। "
            f"कुल खर्च {result['total_rate_inr']} रुपये। कन्फ़र्मेशन नंबर: {result['confirmation_id']}।"
        )
        if result.get("combined_prep"):
            reply += f" तैयारी: {result['combined_prep']}"
        return reply
    reply = (
        f"{names} have been booked for {result['date']}. Total cost {result['total_rate_inr']} rupees. "
        f"Confirmation number: {result['confirmation_id']}."
    )
    if result.get("combined_prep"):
        reply += f" Preparation: {result['combined_prep']}"
    return reply


def add_test_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("success"):
        return (
            f"{result['test_name']} टेस्ट आपकी {result['date']} की बुकिंग में जोड़ दिया गया है।"
            if hi
            else f"The {result['test_name']} test has been added to your booking for {result['date']}."
        )
    reason = result.get("reason")
    if reason == "not_found":
        return (
            "माफ़ कीजिए, इस कन्फ़र्मेशन नंबर पर कोई टेस्ट बुकिंग नहीं मिली।"
            if hi
            else "Sorry, I couldn't find a test booking with that confirmation number."
        )
    if reason == "test_not_found":
        return "माफ़ कीजिए, यह टेस्ट हमारी सूची में नहीं है।" if hi else "Sorry, that test is not on our list."
    if reason == "already_booked":
        return "यह टेस्ट तो पहले से ही आपकी बुकिंग में है।" if hi else "That test is already part of your booking."
    return "माफ़ कीजिए, टेस्ट जोड़ा नहीं जा सका।" if hi else "Sorry, I couldn't add that test."


def resend_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("success"):
        return (
            f"कन्फ़र्मेशन दोबारा भेज दिया गया है, नंबर खत्म होता है {result['sent_to_last4']} से।"
            if hi
            else f"The confirmation has been resent, to the number ending in {result['sent_to_last4']}."
        )
    if result.get("reason") == "rate_limited":
        return (
            "अभी थोड़ी देर पहले ही भेजा गया था। थोड़ी देर बाद फिर कहिए।"
            if hi
            else "It was already sent a moment ago. Please ask again shortly."
        )
    if result.get("reason") == "no_phone_on_file":
        return (
            "माफ़ कीजिए, इस बुकिंग के लिए कोई फ़ोन नंबर दर्ज नहीं है, इसलिए भेज नहीं पा रही।"
            if hi
            else "Sorry, there's no phone number on file for this booking, so I can't send it."
        )
    return (
        "माफ़ कीजिए, इस कन्फ़र्मेशन नंबर पर कोई बुकिंग नहीं मिली।"
        if hi
        else "Sorry, I couldn't find a booking with that confirmation number."
    )


_NAMES_PER_GROUP = 5


def blood_test_list_reply(spoken_names: list[str], lang: str) -> str:
    """`spoken_names` are already the short codes (reply_templates.blood_test_list_reply calls
    agent.catalogue_forms.lab_test_code before delegating here).

    Grouped into short runs, each ending in a sentence break -- same fix, same reason, as the Bengali version:
    one long comma-separated run with no internal sentence break became one over-long TTS clause on a live call
    (2026-09-28) and was never actually spoken."""
    hi = lang == "hi"
    if not spoken_names:
        return (
            "माफ़ कीजिए, अभी हमारी सूची में कोई ब्लड टेस्ट नहीं है। काउंटर पर पूछ सकते हैं।"
            if hi
            else "Sorry, we don't have any blood tests listed right now. You could ask at the counter."
        )
    groups = [spoken_names[i : i + _NAMES_PER_GROUP] for i in range(0, len(spoken_names), _NAMES_PER_GROUP)]
    if hi:
        listed = "। ".join(", ".join(g) for g in groups)
        return f"हमारे यहाँ ये ब्लड टेस्ट होते हैं -- {listed}। इनमें से कौन सा करवाना है?"
    listed = ". ".join(", ".join(g) for g in groups)
    return f"These are the blood tests we do -- {listed}. Which one would you like?"


def department_route_reply(result: dict, symptom: str, lang: str) -> str:
    hi = lang == "hi"
    if result.get("matched"):
        doctors = result.get("doctors") or []
        who = (
            (f" हमारे यहाँ {', '.join(doctors)} हैं।" if hi else f" We have {', '.join(doctors)} here.") if doctors else ""
        )
        base = (
            f"इसके लिए {result['department_name']} विभाग में दिखाना ठीक रहेगा।"
            if hi
            else f"For this, it would be best to see the {result['department_name']} department."
        )
        ask = " डॉक्टर की अपॉइंटमेंट बना दूँ?" if hi else " Shall I book a doctor's appointment?"
        return base + who + ask
    if result.get("ambiguous"):
        by_dept = result.get("doctors_by_department") or {}

        def _with_doctors(name: str) -> str:
            docs = by_dept.get(name) or []
            return f"{name} ({', '.join(docs)})" if docs else name

        parts = [_with_doctors(n) for n in result["candidates"]]
        return (
            f"यह {' या '.join(parts)} -- इनमें से कोई एक विभाग हो सकता है। कौन सा बताएँगे?"
            if hi
            else f"This could be either {' or '.join(parts)}. Which would you like?"
        )
    return (
        "माफ़ कीजिए, ठीक से समझ नहीं पाई किस विभाग में दिखाना होगा। काउंटर पर पूछ सकते हैं।"
        if hi
        else "Sorry, I couldn't quite tell which department this needs. You could ask at the counter."
    )


def conflict_reply(conflict: dict, lang: str) -> str:
    hi = lang == "hi"
    if hi:
        return (
            f"उस समय आपकी पहले से एक अपॉइंटमेंट है -- {conflict.get('doctor_name')} डॉक्टर के पास, "
            f"{conflict['date']} को, समय {conflict['time_slot']} पर। पुरानी रखूँ, बदलूँ, या नई जोड़ दूँ?"
        )
    return (
        f"You already have an appointment at that time -- with {conflict.get('doctor_name')}, "
        f"on {conflict['date']} at {conflict['time_slot']}. Shall I keep it, move it, or add a new one?"
    )


def earliest_available_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("found"):
        q = result.get("query")
        return f"माफ़ कीजिए, '{q}' नाम के कोई डॉक्टर नहीं मिले।" if hi else f"Sorry, I couldn't find a doctor named '{q}'."
    if not result.get("available"):
        return (
            f"माफ़ कीजिए, {result.get('doctor_name')} डॉक्टर के पास नज़दीकी समय में कोई जगह खाली नहीं है। "
            f"बाद में फिर फ़ोन करें तो बता पाऊँगी।"
            if hi
            else f"Sorry, {result.get('doctor_name')} has nothing free in the near term. "
            f"Please call again later and I can check for you."
        )
    if hi:
        reply = f"सबसे जल्दी खाली है {result['date']} को, समय {result['time_slot']} पर।"
        alts = result.get("alternatives") or []
        if alts:
            reply += f" इसके अलावा: {', '.join(alts)}।"
        return reply
    reply = f"The earliest opening is on {result['date']} at {result['time_slot']}."
    alts = result.get("alternatives") or []
    if alts:
        reply += f" Also available: {', '.join(alts)}."
    return reply


def draft_resume_reply(draft: dict, lang: str) -> str:
    hi = lang == "hi"
    return (
        "पिछली बार कॉल कट गई थी, आपकी बुकिंग पूरी नहीं हुई थी। जहाँ से छोड़ा था वहीं से जारी रखूँ, या नए सिरे से शुरू करूँ?"
        if hi
        else "Your call dropped last time before the booking finished. Shall I continue where we left off, or start fresh?"
    )


def spelling_prompt(lang: str) -> str:
    hi = lang == "hi"
    return "नाम एक-एक अक्षर करके बताएँगे?" if hi else "Could you spell the name out for me, letter by letter?"


def spelling_readback(spelled: str, lang: str) -> str:
    hi = lang == "hi"
    return f"मैंने सुना {spelled.upper()} -- क्या यह ठीक है?" if hi else f"I heard {spelled.upper()} -- is that right?"


# ============================================== ported story: "Caller asks which doctors are available"
# Same discipline as the Bengali originals in reply_templates.py: every name is
# a column from the verified API response, in the script this language's voice
# can say, and a long list is grouped into short runs so TTS gets a sentence
# break (one long comma run was never actually spoken on a live call).


def doctor_list_reply(result: dict, lang: str) -> str:
    from agent.reply_templates import MAX_DOCTORS_SPOKEN, _grouped, _spoken_doctor

    hi = lang == "hi"
    names = [n for n in (_spoken_doctor(d, lang) for d in (result.get("doctors") or [])) if n]
    if not result.get("found") or not names:
        return (
            "माफ़ कीजिए, अभी डॉक्टरों की सूची नहीं देख पा रही। काउंटर पर पूछ सकते हैं।"
            if hi
            else "Sorry, I cannot see the list of doctors just now. You could ask at the counter."
        )
    if len(names) > MAX_DOCTORS_SPOKEN:
        departments = sorted({d.get("department") for d in result["doctors"] if d.get("department")})
        if hi:
            listed = "। ".join(", ".join(g) for g in _grouped(departments))
            return f"हमारे यहाँ इन विभागों में डॉक्टर बैठते हैं -- {listed}। आप किस विभाग के बारे में पूछ रहे हैं?"
        listed = ". ".join(", ".join(g) for g in _grouped(departments))
        return f"We have doctors in these departments -- {listed}. Which department did you mean?"
    if hi:
        listed = "। ".join(", ".join(g) for g in _grouped(names))
        return f"हमारे यहाँ ये डॉक्टर बैठते हैं -- {listed}। किनके साथ अपॉइंटमेंट लेना चाहेंगे?"
    listed = ". ".join(", ".join(g) for g in _grouped(names))
    return f"These are the doctors we have -- {listed}. Who would you like an appointment with?"


def doctors_by_department_reply(result: dict, lang: str) -> str:
    from agent.reply_templates import _grouped, _spoken_doctor

    hi = lang == "hi"
    if not result.get("found"):
        offered = result.get("did_you_mean") or []
        query = result.get("query") or ""
        if result.get("ambiguous") and offered:
            return (
                f"कई विभाग मिले -- आप {' या '.join(offered)}, किसकी बात कर रहे हैं?"
                if hi
                else f"I found more than one department -- did you mean {' or '.join(offered)}?"
            )
        if offered:
            return (
                f"'{query}' नाम का विभाग नहीं मिला। क्या आपका मतलब {', '.join(offered)} है?"
                if hi
                else f"I could not find a department called '{query}'. Did you mean {', '.join(offered)}?"
            )
        return (
            f"माफ़ कीजिए, '{query}' नाम का कोई विभाग हमारी सूची में नहीं है। किसी और विभाग या डॉक्टर का नाम बताएँगे?"
            if hi
            else f"Sorry, there is no department called '{query}' on our list. "
            "Could you give me another department or a doctor's name?"
        )
    names = [n for n in (_spoken_doctor(d, lang) for d in (result.get("doctors") or [])) if n]
    department = result.get("department") or ""
    if not names:
        return (
            f"{department} विभाग में अभी कोई डॉक्टर सूची में नहीं है। किसी और विभाग के बारे में देखूँ?"
            if hi
            else f"There are no doctors listed in {department} just now. Shall I look at another department?"
        )
    if hi:
        listed = "। ".join(", ".join(g) for g in _grouped(names))
        return f"{department} विभाग में ये डॉक्टर बैठते हैं -- {listed}। किनके साथ अपॉइंटमेंट लेना चाहेंगे?"
    listed = ". ".join(", ".join(g) for g in _grouped(names))
    return f"These are the doctors in {department} -- {listed}. Who would you like an appointment with?"


# ============================================== ported story: "Caller asks to compare two options"
# Prices only -- see the Bengali original in reply_templates.py for why nothing
# here ranks the two options on anything but arithmetic.


def compare_options_reply(cmp: dict, name_a: str, name_b: str, lang: str) -> str:
    hi = lang == "hi"
    if not cmp.get("a_found") or not cmp.get("b_found"):
        missing = name_a if not cmp.get("a_found") else name_b
        return (
            f"'{missing}' हमारी सूची में नहीं मिला, इसलिए तुलना नहीं कर पा रही। दोनों नाम फिर से बताएँगे?"
            if hi
            else f"I could not find '{missing}' on our list, so I cannot compare them. Could you give me both names again?"
        )
    if cmp.get("price_delta") is None:
        return (
            f"{name_a} और {name_b} -- दोनों की कीमत अभी नहीं देख पा रही। काउंटर पर पूछ सकते हैं।"
            if hi
            else f"I cannot see the price of {name_a} and {name_b} just now. You could ask at the counter."
        )
    delta = cmp["price_delta"]
    if cmp.get("cheaper") is None:
        return (
            f"{name_a} और {name_b} -- दोनों की कीमत एक ही है। कौन सा करवाना है?"
            if hi
            else f"{name_a} and {name_b} cost the same. Which would you like?"
        )
    cheaper = name_a if cmp["cheaper"] == "a" else name_b
    dearer = name_b if cmp["cheaper"] == "a" else name_a
    if hi:
        line = f"{cheaper} {dearer} से {delta} रुपये सस्ता है। "
        if cmp.get("both_packages") and cmp.get("more_tests_side"):
            more = name_a if cmp["more_tests_side"] == "a" else name_b
            line += f"{more} में {cmp['test_count_delta']} टेस्ट ज़्यादा हैं। "
        return line + "कौन सा करवाना है?"
    line = f"{cheaper} is {delta} rupees cheaper than {dearer}. "
    if cmp.get("both_packages") and cmp.get("more_tests_side"):
        more = name_a if cmp["more_tests_side"] == "a" else name_b
        line += f"{more} includes {cmp['test_count_delta']} more tests. "
    return line + "Which would you like?"


# ============================================== KCD-387 full flow: "Caller asks whether a test can
# be collected at home" -- Hindi/English counterparts of reply_templates.py's Bengali originals.
# Same discipline: every number here is a straight substitution from a real clinic-api response.


def home_collection_eligibility_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    rows = result.get("results") or []
    not_found = result.get("not_found") or []
    eligible = [r for r in rows if r.get("found") and r.get("eligible")]
    ineligible = [r for r in rows if r.get("found") and not r.get("eligible")]
    parts = []
    if eligible:
        names = ", ".join(r.get("test_name") or "" for r in eligible)
        parts.append(
            f"{names} -- इन टेस्ट(ओं) के लिए घर पर सैंपल लिया जा सकता है।"
            if hi
            else f"{names} -- home collection is available for this/these test(s)."
        )
    for r in ineligible:
        name = r.get("test_name") or ""
        if r.get("reason") == "area_not_covered":
            parts.append(
                f"{name} के लिए आपके इलाके में अभी घर पर सैंपल लेने की सुविधा नहीं है।"
                if hi
                else f"Home collection is not yet available in your area for {name}."
            )
        else:
            parts.append(
                f"{name} टेस्ट घर पर नहीं किया जा सकता, लैब में ही आना होगा।"
                if hi
                else f"{name} cannot be collected at home -- it needs a lab visit."
            )
    for name in not_found:
        parts.append(f"'{name}' नाम का कोई टेस्ट नहीं मिला।" if hi else f"I could not find a test called '{name}'.")
    if not parts:
        parts.append(
            "माफ़ कीजिए, अभी इस टेस्ट की जानकारी नहीं दे पा रही।" if hi else "Sorry, I can't check that test right now."
        )
    return " ".join(parts)


def home_collection_slots_reply(slots: list[dict], lang: str) -> str:
    hi = lang == "hi"
    if not slots:
        return "माफ़ कीजिए, उस दिन कोई स्लॉट खाली नहीं है। कोई और दिन बताएँगे?" if hi else "Sorry, there's no slot free that day. Could you try another day?"
    shown = slots[:MAX_SLOTS_READ_ALOUD]
    if len(shown) == 1:
        s = shown[0]
        return (
            f"{s['date']} को {s['start_time']} से {s['end_time']} के बीच स्लॉट खाली है। यही रखूँ?"
            if hi
            else f"There's a slot free on {s['date']} between {s['start_time']} and {s['end_time']}. Shall I hold that one?"
        )
    windows = ", ".join(f"{s['start_time']} से {s['end_time']}" if hi else f"{s['start_time']} to {s['end_time']}" for s in shown)
    return (
        f"{shown[0]['date']} को ये समय खाली हैं -- {windows}। कौन सा चाहिए?"
        if hi
        else f"These windows are free on {shown[0]['date']} -- {windows}. Which would you like?"
    )


def home_collection_hold_failed_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("reason") == "slot_full":
        return "माफ़ कीजिए, वह समय अभी भर गया। कोई और समय बताएँगे?" if hi else "Sorry, that window just filled up. Could you pick another?"
    return "माफ़ कीजिए, समय रोक नहीं पाई। फिर कोशिश करूँ?" if hi else "Sorry, I couldn't hold that slot. Shall I try again?"


def home_collection_quote_reply(quote: dict, payment_policy_text: str | None, lang: str) -> str:
    hi = lang == "hi"
    if not quote.get("quote_available"):
        return (
            "माफ़ कीजिए, इस टेस्ट की कीमत अभी नहीं बता पा रही।" if hi else "Sorry, I can't work out the price for this right now."
        )
    lines = [f"{r.get('test_name')} {r.get('rate_inr')}" for r in (quote.get("per_test") or []) if r.get("eligible")]
    breakdown = ", ".join(lines)
    charge = quote.get("home_collection_charge_inr") or 0
    total = quote.get("total_inr") or 0
    if hi:
        reply = f"{breakdown} रुपये, और घर पर सैंपल लेने के लिए {charge} रुपये -- कुल मिलाकर {total} रुपये।"
    else:
        reply = f"{breakdown} rupees, plus {charge} rupees for home collection -- {total} rupees in total."
    if payment_policy_text:
        reply += f" {payment_policy_text}"
    return reply + (" आगे बढ़ूँ?" if hi else " Shall I go ahead?")


def home_collection_stored_address_offer(address: str, lang: str) -> str:
    hi = lang == "hi"
    return (
        f"पिछली बार दिया गया यही पता भेजूँ -- {address}? या नया पता देंगे?"
        if hi
        else f"Shall I use the address you gave last time -- {address}? Or would you like to give a new one?"
    )


def home_collection_address_pincode_mismatch_reply(address_pin: str, given_pin: str, lang: str) -> str:
    hi = lang == "hi"
    if hi:
        return f"आपके पते में {address_pin} पिनकोड सुना, लेकिन पहले आपने {given_pin} बताया था -- दोनों अलग हैं। पता फिर से बताएँगे?"
    return f"I heard pincode {address_pin} in your address, but you earlier gave {given_pin} -- those are different. Could you give me the address again?"


def home_collection_address_confirm_reply(address: str, lang: str) -> str:
    hi = lang == "hi"
    return f"पता सुना -- {address}। यह सही है?" if hi else f"I heard the address as -- {address}. Is that correct?"


def home_collection_final_confirm_reply(summary: dict, lang: str) -> str:
    # ADDED BY SOURAV: the hi readback is split across 4 short sentences, not 1 long run-on --
    # persona.violations()'s own per-language MAX_SENTENCE_WORDS cap (hi: 18) exists so a caller
    # hears this read-back in digestible pieces, not a single breathless clause.
    hi = lang == "hi"
    tests = ", ".join(summary.get("eligible_tests") or [])
    if hi:
        return (
            f"तो {tests} -- {summary.get('date')} को, समय {summary.get('start_time')} से "
            f"{summary.get('end_time')} के बीच घर पर सैंपल लिया जाएगा। "
            f"पता -- {summary.get('address')}। "
            f"कुल {summary.get('total_inr')} रुपये। यह कन्फ़र्म कर दूँ?"
        )
    return (
        f"So {tests} -- on {summary.get('date')}, between {summary.get('start_time')} and "
        f"{summary.get('end_time')}, at {summary.get('address')} -- home collection, total "
        f"{summary.get('total_inr')} rupees. Shall I confirm this?"
    )


def home_collection_booking_result_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("success"):
        reason = result.get("reason")
        if reason == "hold_expired":
            return (
                "माफ़ कीजिए, समय रोक नहीं पाई, थोड़ी देर हो गई। फिर कोशिश करूँ?"
                if hi
                else "Sorry, the slot couldn't be held in time. Shall I try again?"
            )
        if reason in ("no_eligible_test", "test_not_found"):
            return (
                "माफ़ कीजिए, इस टेस्ट के लिए घर पर बुकिंग नहीं हो पाई।" if hi else "Sorry, I couldn't book home collection for this test."
            )
        return (
            "माफ़ कीजिए, बुकिंग नहीं हो पाई। थोड़ी देर बाद फिर कोशिश करें, या काउंटर पर संपर्क करें।"
            if hi
            else "Sorry, the booking couldn't go through. Please try again shortly, or contact the counter."
        )
    collector = result.get("collector_name")
    if hi:
        assignment_line = f" {collector} आएँगी/आएँगे।" if collector else " कौन आएगा, यह हमारा स्टाफ़ बाद में कन्फ़र्म करेगा।"
        return (
            f"आपकी घर पर सैंपल लेने की बुकिंग कन्फ़र्म हो गई है। "
            f"{result.get('date')} को, समय {result.get('start_time')} से {result.get('end_time')} के बीच, "
            f"पता {result.get('address_line')} पर।{assignment_line} "
            f"कुल {result.get('total_inr')} रुपये। कन्फ़र्मेशन नंबर {result.get('booking_reference')}।"
        )
    assignment_line = f" {collector} will come." if collector else " Our staff will confirm who will come."
    return (
        f"Your home collection booking is confirmed. On {result.get('date')}, between "
        f"{result.get('start_time')} and {result.get('end_time')}, at {result.get('address_line')}."
        f"{assignment_line} Total {result.get('total_inr')} rupees. Confirmation number "
        f"{result.get('booking_reference')}."
    )


def home_collection_cancelled_reply(lang: str) -> str:
    hi = lang == "hi"
    return "ठीक है, बुकिंग नहीं की गई। कुछ और मदद चाहिए?" if hi else "Okay, no booking was made. Is there anything else I can help with?"
