"""Composes the spoken Bengali reply from TOOL DATA, never from the LLM's
own words, for any intent where a fact (a price, a date, a confirmation
ID) is at stake.

This is the same discipline voicerx/gate.py already applies to drug names
("the SLM proposes, the gazetteer decides") ported to this domain: the LLM
may decide WHAT the caller wants and WHICH slots it heard, but the actual
number in the caller's ear always comes from the Spring Boot response,
substituted into a fixed template. The model never gets a chance to
misremember or round a price it was merely shown a moment ago.

Only "smalltalk" skips this file entirely and uses the LLM's own
direct_reply_bn -- there is no fact to get wrong in "নমস্কার" or "ধন্যবাদ".
"""

from __future__ import annotations

from agent import reply_templates_i18n as _i18n
from agent.home_collection_flow import MAX_SLOTS_READ_ALOUD
from agent.sample_wording import sample_sentence


def _spoken_test_name(slots: dict, result: dict) -> str:
    """What the caller HEARS as the test's name.

    Order matters. The API's `test_name` is the catalogue's English label
    ("Uric Acid") and the Bengali TTS tokenizer drops Latin script
    outright, so putting it in a spoken sentence removes the name from the
    reply entirely -- the caller hears a price attached to nothing. Prefer
    the seeded Bengali alias; failing that, echo the caller's own words
    back, which is what a person at the counter would do anyway.
    """
    return result.get("test_name_bn") or slots.get("test_name") or result.get("test_name") or "টেস্ট"


def _spoken_doctor_name(slots: dict, result: dict) -> str:
    """Same problem, same order. Aliases are seeded as surnames ("সেন"),
    so this adds the honorific the English label already carried."""
    alias = result.get("doctor_name_bn")
    if alias:
        return f"ডাঃ {alias}"
    return slots.get("doctor_name") or result.get("doctor_name") or "ডাক্তার"


def missing_slot_prompt(intent: str, missing: str, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.missing_slot_prompt(intent, missing, lang)
    prompts = {
        ("test_rate", "test_name"): "কোন টেস্টের রেট জানতে চান, একটু বলবেন?",
        ("doctor_availability", "doctor_name"): "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?",
        ("book_appointment", "doctor_name"): "কোন ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট করতে চান?",
        ("book_appointment", "date"): "কোন দিনের জন্য অ্যাপয়েন্টমেন্ট চাই?",
        ("book_appointment", "time_slot"): "কোন সময়ের জন্য অ্যাপয়েন্টমেন্ট চাই?",
        ("book_appointment", "patient_name"): "রোগীর নামটা বলবেন?",
        ("book_appointment", "phone"): "একটা ফোন নম্বর দেবেন, যাতে কনফার্মেশন পাঠাতে পারি?",
        ("test_prep", "test_name"): "কোন টেস্টের প্রস্তুতি জানতে চান, একটু বলবেন?",
        ("clinic_faq", "faq_topic"): "দুঃখিত, ঠিক বুঝতে পারলাম না। আর একটু বলবেন?",
        ("book_test", "test_names"): "কোন টেস্টগুলো বুক করতে চান, একটু বলবেন?",
        ("book_test", "date"): "কোন দিনের জন্য টেস্ট করাতে চান?",
        ("book_test", "patient_name"): "রোগীর নামটা বলবেন?",
        ("book_test", "phone"): "একটা ফোন নম্বর দেবেন, যাতে কনফার্মেশন পাঠাতে পারি?",
        ("reschedule_appointment", "confirmation_id"): "আপনার কনফার্মেশন নম্বরটা বা যে নম্বর থেকে বুক করেছিলেন সেটা বলবেন?",
        ("reschedule_appointment", "new_date"): "কোন দিনে নিয়ে যেতে চান?",
        ("reschedule_appointment", "new_time_slot"): "কোন সময়ে নিয়ে যেতে চান?",
        ("cancel_appointment", "confirmation_id"): "আপনার কনফার্মেশন নম্বরটা বা যে নম্বর থেকে বুক করেছিলেন সেটা বলবেন?",
        ("add_test_booking", "confirmation_id"): "আপনার কনফার্মেশন নম্বরটা বলবেন?",
        ("add_test_booking", "test_name"): "কোন টেস্টটা যোগ করতে চান?",
        ("lookup_booking", "phone"): "যে নম্বর থেকে বুক করেছিলেন সেটা বলবেন?",
        # ADDED BY SOURAV: KCD-387 full flow -- one prompt per still-missing fact in the home
        # collection flow, same "one question per missing slot" shape as every entry above.
        ("home_collection", "test_name"): "কোন টেস্টটা বাড়িতে এসে করানোর কথা জিজ্ঞেস করছেন, একটু বলবেন?",
        ("home_collection", "postal_code"): "আপনার এলাকার ৬ সংখ্যার পিনকোডটা বলবেন?",
        ("home_collection", "date"): "কোন দিন বাড়িতে এসে স্যাম্পল নিতে হবে?",
        ("home_collection", "address_line"): "পুরো ঠিকানাটা বলবেন -- বাড়ি নম্বর, রাস্তা, এলাকা, শহর সমেত?",
        ("home_collection", "patient_name"): "রোগীর নামটা বলবেন?",
        ("home_collection", "phone"): "একটা ফোন নম্বর দেবেন, যাতে বুকিং-এর তথ্য পাঠাতে পারি?",
    }
    return prompts.get((intent, missing), "দুঃখিত, একটু স্পষ্ট করে বলবেন?")


def insufficient_information_reply(lang: str = "bn") -> str:
    """KCD-442: a distinct THIRD outcome, neither "this does not exist"
    (KCD-443's not-found path) nor "the system is unreachable"
    (phrase("tool_failure", lang)) -- the turn itself was heard too
    unclearly (agent/confidence_gate.py) to trust running a lookup on
    what was extracted from it at all. Saying so plainly beats a
    confident answer about the wrong test."""
    if lang != "bn":
        return _i18n.insufficient_information_reply(lang)
    return "দুঃখিত, ঠিক শুনতে পাইনি। আপনি কি আবার একটু স্পষ্ট করে বলবেন?"


def test_rate_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.test_rate_reply(slots, result, lang)
    if not result.get("found"):
        suggestions = result.get("did_you_mean_bn") or result.get("did_you_mean") or []
        if result.get("ambiguous") and suggestions:
            # KCD-446: this test EXISTS -- several rows matched equally
            # well -- distinct from the not-found framing below, which
            # would tell the caller something untrue.
            return f"একাধিক টেস্ট পেলাম -- কোনটার কথা বলছেন: {' নাকি '.join(suggestions)}?"
        if suggestions:
            return f"'{slots.get('test_name')}' নামে টেস্ট খুঁজে পাইনি। আপনি কি {', '.join(suggestions)} বলতে চাইছেন?"
        # KCD-lay-terms: nothing at all matched -- a flat "no such test" tells the caller nothing useful next.
        # Asking what else they need, once, costs nothing and might save a second call. No package/checkup offer
        # here -- there is no such catalogue entry or flow to point them to, so it never invites one.
        return (
            f"দুঃখিত, '{slots.get('test_name')}' নামে কোনো টেস্ট আমাদের তালিকায় নেই। "
            "অন্য কোনো নির্দিষ্ট টেস্ট বা ডাক্তারের অ্যাপয়েন্টমেন্ট বুক করতে চান?"
        )

    rate = result["rate_inr"]
    name = _spoken_test_name(slots, result)
    sample = result.get("sample_type")
    hours = result.get("report_time_hours")
    reply = f"{name} টেস্টের রেট {rate} টাকা।"
    sentence = sample_sentence(sample, "bn")
    if sentence:
        reply += f" {sentence}"
    if hours:
        reply += f" রিপোর্ট {hours} ঘণ্টার মধ্যে পাবেন।"
    return reply


def doctor_availability_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.doctor_availability_reply(slots, result, lang)
    if not result.get("found"):
        suggestions = result.get("did_you_mean_bn") or result.get("did_you_mean") or []
        if result.get("ambiguous") and suggestions:
            # Doctor-side counterpart of KCD-446's test-ambiguity framing.
            return f"একাধিক ডাক্তার পেলাম -- কার কথা বলছেন, {' নাকি '.join(suggestions)}?"
        if suggestions:
            return f"'{slots.get('doctor_name')}' নামে ডাক্তার খুঁজে পাইনি। আপনি কি {', '.join(suggestions)} বলতে চাইছেন?"
        # KCD-lay-terms: no name matched at all -- inviting the caller to describe the PROBLEM instead composes with
        # the existing department_query intent on their very next turn (no new state needed: the model already
        # classifies a symptom description as department_query, which already returns BOTH departments when a
        # symptom is genuinely ambiguous between them -- clinic-api/booking_service.route_department).
        return (
            f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার আমাদের এখানে নেই। "
            "আপনার কী সমস্যা হচ্ছে বলুন, তাহলে ঠিক বিভাগের ডাক্তার বলতে পারব।"
        )

    name = _spoken_doctor_name(slots, result)
    if result.get("available"):
        hours = result.get("chamber_hours", "")
        date_txt = f" {result.get('date')} তারিখে" if result.get("date") else " আজ"
        return f"হ্যাঁ,{date_txt} {name} চেম্বারে থাকবেন। সময়: {hours}।"

    next_date = result.get("next_available_date")
    if next_date:
        return f"{name} ওই দিন বসবেন না। পরবর্তী উপলব্ধ দিন: {next_date}।"
    return f"{name} এখন কোনো নির্দিষ্ট দিন বসছেন না। আমাদের কাউন্টারে খোঁজ নিতে পারেন।"


def test_prep_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.test_prep_reply(slots, result, lang)
    if not result.get("found"):
        suggestions = result.get("did_you_mean_bn") or result.get("did_you_mean") or []
        if result.get("ambiguous") and suggestions:
            return f"একাধিক টেস্ট পেলাম -- কোনটার কথা বলছেন: {' নাকি '.join(suggestions)}?"
        if suggestions:
            return f"'{slots.get('test_name')}' নামে টেস্ট খুঁজে পাইনি। আপনি কি {', '.join(suggestions)} বলতে চাইছেন?"
        return f"দুঃখিত, '{slots.get('test_name')}' নামে কোনো টেস্ট আমাদের তালিকায় নেই।"

    name = _spoken_test_name(slots, result)
    # The lab-test table decides: its preparation text if it has one; else its fasting_required column. A test the table
    # marks as needing fasting is never told "no special preparation" because the text column was left empty.
    instructions = result.get("prep_instructions") or (
        "এই টেস্টের জন্য উপবাস থাকতে হবে। কত ঘণ্টা, তা কাউন্টারে জেনে নিন।"
        if result.get("fasting_required")
        else "এই টেস্টের জন্য বিশেষ কোনো প্রস্তুতির প্রয়োজন নেই।"
    )
    return f"{name} টেস্টের জন্য: {instructions}"


def merged_test_prep_reply(test_names: list[str], result: dict, lang: str = "bn") -> str:
    """KCD-382 spoken: caller story "Caller has several tests with
    conflicting preparation" (AC2 -- ONE coherent instruction, never a
    separate paragraph per test). `result` is exactly what
    clinic-api/enquiry_service.py::merge_prep_instructions() returned,
    forwarded verbatim through agent.tools_client.merge_test_prep() --
    this function only picks words around numbers it did not compute
    (CLAUDE.md rule 1). The caller (main.py) is responsible for checking
    `result["escalate_to_human"]` BEFORE calling this function; it is
    never reached for that case, so there is nothing here resembling a
    guessed number for an unresolved conflict."""
    if lang != "bn":
        return _i18n.merged_test_prep_reply(test_names, result, lang)
    not_found = result.get("not_found") or []
    if not result.get("found") or not_found:
        missing = ", ".join(not_found) if not_found else ", ".join(test_names)
        return f"দুঃখিত, {missing} -- এই নামে টেস্ট আমাদের তালিকায় খুঁজে পাইনি। নামগুলো একটু আবার বলবেন?"

    names = ", ".join(result.get("test_names") or test_names)
    hours = result.get("fasting_hours")
    if hours:
        water = "শুধু জল খেতে পারবেন।" if result.get("water_allowed_while_fasting") else "জলও খাওয়া যাবে না।"
        instruction = f"সবচেয়ে কড়া নিয়মটা মানতে হবে -- অন্তত {hours} ঘণ্টা উপবাস থাকতে হবে, {water}"
    else:
        instruction = "এই টেস্টগুলোর জন্য আলাদা করে উপবাসের প্রয়োজন নেই।"
    if result.get("prescription_required"):
        instruction += " এর মধ্যে অন্তত একটা টেস্টের জন্য ডাক্তারের প্রেসক্রিপশন লাগবে।"
    return f"{names} -- এই টেস্টগুলোর জন্য, {instruction}"


def clinic_faq_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.clinic_faq_reply(slots, result, lang)
    if not result.get("found"):
        return "দুঃখিত, এই বিষয়ে এখন সঠিক তথ্য দিতে পারছি না। কাউন্টারে যোগাযোগ করুন।"
    return result.get("answer") or "দুঃখিত, এই বিষয়ে এখন সঠিক তথ্য দিতে পারছি না। কাউন্টারে যোগাযোগ করুন।"


def booking_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.booking_reply(slots, result, lang)
    if result.get("success"):
        return (
            f"আপনার অ্যাপয়েন্টমেন্ট কনফার্ম হয়েছে। "
            f"{_spoken_doctor_name(slots, result)}, {result['date']}, সময় {result['time_slot']}। "
            f"কনফার্মেশন নম্বর: {result['confirmation_id']}।"
        )

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return f"ওই সময়টা বুক হয়ে গেছে। এই সময়গুলো ফাঁকা আছে: {', '.join(alts)}। কোনটা চান?"
        return "ওই সময়টা বুক হয়ে গেছে, এবং কাছাকাছি কোনো সময় ফাঁকা নেই।"
    if reason == "doctor_ambiguous":
        # Two doctors fit the name (two Sens): never picked -- asked, by whole name.
        names = result.get("did_you_mean_bn") or result.get("did_you_mean") or []
        return (
            f"একাধিক ডাক্তার পেলাম -- কার কথা বলছেন, {' নাকি '.join(names)}?"
            if names
            else "একাধিক ডাক্তার পেলাম। আপনি কোন ডাক্তারের কথা বলছেন, পুরো নামটা বলবেন?"
        )
    if reason == "doctor_not_found":
        return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার খুঁজে পেলাম না।"
    if reason == "doctor_not_available_that_day":
        return "দুঃখিত, ওই দিন ডাক্তার বসেন না। অন্য কোনো দিন বলবেন?"
    if reason == "invalid_slot":
        return "দুঃখিত, ওই সময়টা ঠিক নেই। চেম্বারের সময়ের মধ্যে একটা সময় বলবেন?"
    if reason == "date_in_past":
        # KCD-363: never silently rolled forward -- stated plainly, and
        # the caller is offered the same weekday next week.
        return "ওই তারিখটা তো চলে গেছে। আগামী সপ্তাহে ওই দিনের জন্য বুক করব?"
    if reason == "hold_expired":
        return "দুঃখিত, সময়টা ধরে রাখা যায়নি, একটু দেরি হয়ে গেছে। আবার চেষ্টা করি?"
    return "দুঃখিত, অ্যাপয়েন্টমেন্ট বুক করা গেল না। একটু পরে আবার চেষ্টা করুন, অথবা কাউন্টারে যোগাযোগ করুন।"


def booking_confirmation_readback(slots: dict, action: str, lang: str = "bn") -> str:
    """Read back what is about to be written BEFORE it is written -- the
    step KCD-367/369 depend on: a caller who spots a mistake here corrects
    it before anything is committed, not after."""
    if lang != "bn":
        return _i18n.booking_confirmation_readback(slots, action, lang)
    if action == "book_appointment":
        # KCD-448: read back the number the confirmation actually goes to
        # (contact_phone wins over phone, same precedence as
        # booking_flow.effective_phone) -- never the raw `phone` slot,
        # which is None whenever the caller gave a different contact
        # number or declined one, both of which would otherwise be read
        # back as the literal word "None". A declined phone has already
        # been announced separately (phrase("no_confirmation_number", lang)
        # in main.py, before this readback runs) so it is omitted here
        # rather than repeated.
        phone = slots.get("contact_phone") or slots.get("phone")
        phone_clause = f", ফোন নম্বর {phone} " if phone else " "
        return (
            f"তাহলে {slots.get('doctor_name')} ডাক্তারের কাছে {slots.get('date')} তারিখে, "
            f"সময় {slots.get('time_slot')}-এ, রোগীর নাম {slots.get('patient_name')}"
            f"{phone_clause}-- এই অ্যাপয়েন্টমেন্টটা কনফার্ম করব?"
        )
    if action == "book_test":
        # ", " not the ideographic "、" -- a stray full-width character
        # from an earlier edit, inconsistent with every other list-join
        # in this file and in reply_templates_i18n.py.
        tests = ", ".join(slots.get("_test_names_display", [])) or "টেস্ট"
        phone = slots.get("contact_phone") or slots.get("phone")
        phone_clause = f", ফোন নম্বর {phone} " if phone else " "
        return (
            f"তাহলে {slots.get('date')} তারিখে {tests} -- রোগীর নাম {slots.get('patient_name')}"
            f"{phone_clause}-- এই বুকিংটা কনফার্ম করব?"
        )
    if action == "reschedule_appointment":
        return f"তাহলে অ্যাপয়েন্টমেন্টটা {slots.get('new_date')} তারিখে, সময় {slots.get('new_time_slot')}-এ নিয়ে যাব?"
    if action == "cancel_appointment":
        return "আপনার অ্যাপয়েন্টমেন্টটা বাতিল করে দেব?"
    if action == "add_test_booking":
        return f"তাহলে আপনার বুকিং-এ {slots.get('test_name')} টেস্টটা যোগ করে দেব?"
    return "এটা কনফার্ম করব?"


def reschedule_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.reschedule_reply(result, lang)
    if result.get("success"):
        return (
            f"আপনার অ্যাপয়েন্টমেন্টটা {result['date']} তারিখে, সময় {result['time_slot']}-এ "
            f"পাল্টে দেওয়া হয়েছে। নতুন কনফার্মেশন নম্বর: {result['confirmation_id']}।"
        )
    reason = result.get("reason")
    if reason == "not_found":
        return "দুঃখিত, এই কনফার্মেশন নম্বরে কোনো অ্যাপয়েন্টমেন্ট খুঁজে পেলাম না।"
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return f"ওই সময়টা বুক হয়ে গেছে। এই সময়গুলো ফাঁকা আছে: {', '.join(alts)}। কোনটা চান?"
        return "ওই সময়টা বুক হয়ে গেছে। আপনার আগের অ্যাপয়েন্টমেন্টটা ঠিক আগের মতোই আছে।"
    return "দুঃখিত, অ্যাপয়েন্টমেন্টটা পাল্টানো গেল না। আপনার আগের অ্যাপয়েন্টমেন্টটা ঠিক আগের মতোই আছে।"


def cancel_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.cancel_reply(result, lang)
    reason = result.get("reason")
    if reason == "charge_confirmation_required":
        charge = result["charge_inr"]
        return f"এই সময়ে বাতিল করলে {charge} টাকা কাটা যাবে। তাও কি বাতিল করব?"
    if result.get("success"):
        charge = result.get("charge_inr") or 0
        if charge:
            return f"আপনার অ্যাপয়েন্টমেন্টটা বাতিল করা হয়েছে। {charge} টাকা কাটা হয়েছে।"
        return "আপনার অ্যাপয়েন্টমেন্টটা কোনো চার্জ ছাড়াই বাতিল করা হয়েছে।"
    if reason == "not_found":
        return "দুঃখিত, এই কনফার্মেশন নম্বরে কোনো অ্যাপয়েন্টমেন্ট খুঁজে পেলাম না।"
    return "দুঃখিত, বাতিল করা গেল না। কাউন্টারে যোগাযোগ করুন।"


def lookup_reply(bookings: list[dict], lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.lookup_reply(bookings, lang)
    if not bookings:
        return "দুঃখিত, আপনার নামে কোনো আসন্ন অ্যাপয়েন্টমেন্ট খুঁজে পেলাম না।"
    b = bookings[0]
    return (
        f"আপনার পরবর্তী অ্যাপয়েন্টমেন্ট: {b.get('doctor_name')} ডাক্তারের কাছে, "
        f"{b['date']} তারিখে, সময় {b['time_slot']}-এ। কনফার্মেশন নম্বর: {b['confirmation_id']}। "
        f"লিখিত কনফার্মেশনটা আবার পাঠিয়ে দিতে পারি, চাইলে বলবেন।"
    )


def multi_test_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.multi_test_reply(result, lang)
    if not result.get("success"):
        return "দুঃখিত, টেস্টগুলো বুক করা গেল না। একটু পরে আবার চেষ্টা করুন।"
    names = ", ".join(result["test_names"])
    reply = (
        f"{names} -- এই টেস্টগুলো {result['date']} তারিখে বুক করা হয়েছে। "
        f"মোট খরচ {result['total_rate_inr']} টাকা। কনফার্মেশন নম্বর {result['confirmation_id']}।"
    )
    if result.get("combined_prep"):
        reply += f" প্রস্তুতি: {result['combined_prep']}"
    return reply


def add_test_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.add_test_reply(result, lang)
    if result.get("success"):
        return f"{result['test_name']} টেস্টটা আপনার {result['date']} তারিখের বুকিং-এ যোগ করা হয়েছে।"
    reason = result.get("reason")
    if reason == "not_found":
        return "দুঃখিত, এই কনফার্মেশন নম্বরে কোনো টেস্ট বুকিং খুঁজে পেলাম না।"
    if reason == "test_not_found":
        return "দুঃখিত, এই নামে কোনো টেস্ট আমাদের তালিকায় নেই।"
    if reason == "already_booked":
        return "এই টেস্টটা তো আগে থেকেই আপনার বুকিং-এ আছে।"
    return "দুঃখিত, টেস্টটা যোগ করা গেল না।"


# A live call (2026-09-28): the whole test list, joined into one comma-separated run with only a single sentence
# break at the end, came back as one TTS clause (agent/clause_split.py only splits at "।.!?") -- 25+ items in one
# synthesis call, and the caller never heard it. Grouped into short runs, each ending in "।", so
# split_into_clauses turns this into several short clauses instead of one very long one, same fix in spirit as
# KCD-462's own reason for splitting a reply into clauses at all.
_NAMES_PER_GROUP = 5


def _grouped(items: list[str], size: int = _NAMES_PER_GROUP) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def blood_test_list_reply(names: list[str], lang: str = "bn") -> str:
    """ "রক্ত পরীক্ষা" ("a blood test") on its own names nothing: the REAL list of blood tests, off the live catalogue
    (main.py's lay-term handling, agent/lay_terms.py), read as their short codes (agent/catalogue_forms.lab_test_code)
    rather than full clinical names -- "CBC" is what a caller says back, not "Complete Blood Count"."""
    from agent.catalogue_forms import lab_test_code

    spoken = [lab_test_code(n) for n in names]
    if lang != "bn":
        return _i18n.blood_test_list_reply(spoken, lang)
    if not spoken:
        return "দুঃখিত, এই মুহূর্তে আমাদের তালিকায় কোনো রক্ত পরীক্ষা নেই। কাউন্টারে খোঁজ নিতে পারেন।"
    listed = "। ".join(", ".join(g) for g in _grouped(spoken))
    return f"আমাদের এখানে রক্তের যে পরীক্ষাগুলো হয় তা হল {listed}। এর মধ্যে কোনটা করাতে চান?"


def resend_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.resend_reply(result, lang)
    if result.get("success"):
        return f"কনফার্মেশনটা আবার পাঠিয়ে দেওয়া হয়েছে, নম্বর শেষ হচ্ছে {result['sent_to_last4']} দিয়ে।"
    reason = result.get("reason")
    if reason == "rate_limited":
        return "একটু আগেই পাঠানো হয়েছে। একটু অপেক্ষা করে আবার বলবেন।"
    if reason == "no_phone_on_file":
        # Distinct from "booking not found" (below) -- the booking DOES
        # exist, there is simply no number on file to send anything to,
        # since the caller declined to give one at booking time.
        return "দুঃখিত, এই বুকিং-এর জন্য কোনো ফোন নম্বর রাখা নেই, তাই পাঠাতে পারছি না।"
    return "দুঃখিত, এই কনফার্মেশন নম্বরে কোনো বুকিং খুঁজে পেলাম না।"


def department_route_reply(result: dict, symptom: str, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.department_route_reply(result, symptom, lang)
    if result.get("matched"):
        # Administrative routing only, never a diagnosis. The doctor names are a live query result
        # (clinic-api's _department_doctor_names), never invented -- an empty list just omits the sentence.
        doctors = result.get("doctors") or []
        who = f" আমাদের এখানে {' ও '.join(doctors)} আছেন।" if doctors else ""
        return f"এই সমস্যার জন্য {result['department_name']} বিভাগে দেখানো ভালো হবে।{who} ডাক্তারের অ্যাপয়েন্টমেন্ট করে দেব?"
    if result.get("ambiguous"):
        by_dept = result.get("doctors_by_department") or {}
        parts = []
        for name in result["candidates"]:
            docs = by_dept.get(name) or []
            parts.append(f"{name} ({', '.join(docs)})" if docs else name)
        return f"এটা {' অথবা '.join(parts)} -- দুটোর যেকোনো একটা বিভাগ হতে পারে। কোনটা বলবেন?"
    return "দুঃখিত, ঠিক কোন বিভাগে দেখাবেন বুঝতে পারলাম না। কাউন্টারে জিজ্ঞেস করে নিতে পারেন।"


def conflict_reply(conflict: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.conflict_reply(conflict, lang)
    return (
        f"আপনার তো ওই সময়ে আগে থেকেই একটা অ্যাপয়েন্টমেন্ট আছে -- {conflict.get('doctor_name')} ডাক্তারের কাছে, "
        f"{conflict['date']} তারিখে, সময় {conflict['time_slot']}-এ। আগেরটা রাখব, পাল্টাব, নাকি নতুন করে যোগ করব?"
    )


def earliest_available_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.earliest_available_reply(result, lang)
    if not result.get("found"):
        return f"দুঃখিত, '{result.get('query')}' নামে কোনো ডাক্তার খুঁজে পেলাম না।"
    if not result.get("available"):
        return f"দুঃখিত, {result.get('doctor_name')} ডাক্তারের কাছাকাছি সময়ে কোনো ফাঁকা সময় নেই। পরে আবার ফোন করলে জানাতে পারব।"
    reply = f"সবচেয়ে তাড়াতাড়ি ফাঁকা আছে {result['date']} তারিখে, সময় {result['time_slot']}-এ।"
    alts = result.get("alternatives") or []
    if alts:
        reply += f" এছাড়াও: {', '.join(alts)}।"
    return reply


def draft_resume_reply(draft: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.draft_resume_reply(draft, lang)
    return "গত বার কল কেটে গিয়েছিল, আপনার বুকিং শেষ হয়নি। যেখানে ছিলাম সেখান থেকে চালিয়ে যাব, নাকি নতুন করে শুরু করব?"


def multiple_bookings_reply(bookings: list[dict], lang: str = "bn") -> str:
    """KCD/CodeRabpit-flagged: resolving a reschedule/cancel/resend target
    by phone alone used to silently take bookings[0] with no
    disambiguation and no statement of which one -- a caller with
    several bookings (their own, or a proxy's) could have a "yes" act on
    the WRONG one. Lists up to three by doctor+date+time (same cap as
    KCD-446's near-match offer) and asks for the confirmation number,
    the same deterministic bearer-token identifier every other lookup
    path in this codebase already uses to pick exactly one booking."""
    if lang != "bn":
        return _i18n.multiple_bookings_reply(bookings, lang)
    parts = [f"{b.get('doctor_name')} ডাক্তারের {b['date']} তারিখের" for b in bookings[:3]]
    return f"আপনার নামে একাধিক বুকিং আছে -- {', '.join(parts)}। কোনটার কথা বলছেন, কনফার্মেশন নম্বরটা বলবেন?"


def multiple_test_bookings_reply(bookings: list[dict], lang: str = "bn") -> str:
    """KCD-382b: the same never-take-bookings[0] rule multiple_bookings_reply above already
    applies to doctor appointments, for TEST bookings found by phone during the registered/
    booked-patient preparation flow (agent.tools_client.search_bookings's "several" status, kind
    "test_booking"). Lists up to three by test+date and asks for the confirmation number -- the
    same bearer-token identifier lookup_bookings/resend_confirmation already use to pick exactly
    one booking, reused here rather than inventing a second disambiguation scheme."""
    if lang != "bn":
        return _i18n.multiple_test_bookings_reply(bookings, lang)
    parts = [f"{b.get('test_name')}, {b['date']} তারিখের" for b in bookings[:3]]
    return f"আপনার নামে একাধিক টেস্ট বুকিং আছে -- {', '.join(parts)}। কোনটার কথা বলছেন, কনফার্মেশন নম্বরটা বলবেন?"


def spelling_prompt(lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.spelling_prompt(lang)
    return "নামটা একটু বানান করে বলবেন, এক একটা অক্ষর করে?"


def spelling_readback(spelled: str, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.spelling_readback(spelled, lang)
    return f"আমি শুনলাম {spelled.upper()} -- ঠিক আছে?"


# ============================================== ported story: "Caller asks which doctors are available"
#
# Both replies below are pure template substitution from a verified
# /api/v1/doctors or /api/v1/doctors/by-department response. No name is ever
# composed, abbreviated or guessed here -- `_spoken_doctor` picks one of the
# columns the API sent, in the script the caller's own voice can actually SAY
# (a Bengali or Hindi synthesiser silently drops Latin text), and falls back to
# the Latin name only when no alias was seeded.

# The clinic seeds 32 doctors across 8 departments. Reading 32 names down a
# phone line is not an answer, it is a recital nobody can hold in their head --
# so a bare "which doctors do you have" names the DEPARTMENTS and asks which
# one, and only a named department lists its own (4, which fits in one breath).
MAX_DOCTORS_SPOKEN = 8


def _spoken_doctor(doc: dict, lang: str = "bn") -> str:
    """The doctor's name in a script this language's voice can pronounce."""
    if lang == "bn":
        return doc.get("full_name_bn") or doc.get("alias_bn") or doc.get("full_name") or doc.get("name") or ""
    if lang == "hi":
        return doc.get("full_name_hi") or doc.get("alias_hi") or doc.get("full_name") or doc.get("name") or ""
    return doc.get("full_name") or doc.get("name") or ""


def _spoken_doctors(result: dict, lang: str = "bn") -> list[str]:
    return [n for n in (_spoken_doctor(d, lang) for d in (result.get("doctors") or [])) if n]


def doctor_list_reply(result: dict, lang: str = "bn") -> str:
    """"Which doctors do you have?" -- no doctor and no department named."""
    if lang != "bn":
        return _i18n.doctor_list_reply(result, lang)
    names = _spoken_doctors(result, "bn")
    if not result.get("found") or not names:
        return "দুঃখিত, এই মুহূর্তে ডাক্তারদের তালিকা দেখতে পারছি না। কাউন্টারে জিজ্ঞেস করে নিতে পারেন।"
    if len(names) > MAX_DOCTORS_SPOKEN:
        departments = sorted({d.get("department") for d in result["doctors"] if d.get("department")})
        listed = "। ".join(", ".join(g) for g in _grouped(departments))
        return f"আমাদের এই বিভাগগুলোতে ডাক্তার বসেন -- {listed}। কোন বিভাগের কথা বলছেন?"
    listed = "। ".join(", ".join(g) for g in _grouped(names))
    return f"আমাদের এখানে এই ডাক্তাররা বসেন -- {listed}। কার কাছে অ্যাপয়েন্টমেন্ট নিতে চান?"


def doctors_by_department_reply(result: dict, lang: str = "bn") -> str:
    """"Which doctors are there in cardiology?" -- a department was named.

    A department that did not resolve is ASKED about, never guessed at: the same
    discipline `test_rate_reply` and `doctor_availability_reply` already hold to,
    and for the same reason ("Doctor Nobody" once fuzzy-matched to a real
    doctor's real schedule).
    """
    if lang != "bn":
        return _i18n.doctors_by_department_reply(result, lang)
    if not result.get("found"):
        offered = result.get("did_you_mean") or []
        if result.get("ambiguous") and offered:
            return f"একাধিক বিভাগ পেলাম -- আপনি {' নাকি '.join(offered)}, কোনটার কথা বলছেন?"
        if offered:
            return f"'{result.get('query')}' নামে বিভাগ পাইনি। আপনি কি {', '.join(offered)} বলতে চাইছেন?"
        return (
            f"দুঃখিত, '{result.get('query')}' নামে কোনো বিভাগ আমাদের তালিকায় নেই। "
            "অন্য কোনো বিভাগ বা ডাক্তারের নাম বলবেন?"
        )
    names = _spoken_doctors(result, "bn")
    department = result.get("department") or ""
    if not names:
        return f"{department} বিভাগে এই মুহূর্তে কোনো ডাক্তার তালিকায় নেই। অন্য কোনো বিভাগের কথা বলব?"
    listed = "। ".join(", ".join(g) for g in _grouped(names))
    return f"{department} বিভাগে এই ডাক্তাররা বসেন -- {listed}। কার কাছে অ্যাপয়েন্টমেন্ট নিতে চান?"


# ============================================== ported story: "Caller asks to compare two options"
#
# States PRICES and nothing else. agent/compare_flow.py's dict carries no
# "which is better" field by construction, and nothing is added here: neither
# catalogue exposes a clinical or suitability axis, so a recommendation would be
# invented rather than retrieved. A caller who literally asks "which is better"
# gets the two prices and the difference, which is the only honest answer this
# system has.
#
# Both numbers and the difference come from the live lookups; the difference is
# computed in Decimal (agent/compare_flow._to_decimal) so a money value is never
# altered by one digit.


def compare_options_reply(cmp: dict, name_a: str, name_b: str, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.compare_options_reply(cmp, name_a, name_b, lang)
    if not cmp.get("a_found") or not cmp.get("b_found"):
        missing = name_a if not cmp.get("a_found") else name_b
        return f"'{missing}' আমাদের তালিকায় পাইনি, তাই তুলনা করতে পারছি না। নাম দুটো আবার বলবেন?"
    if cmp.get("price_delta") is None:
        return f"{name_a} আর {name_b} -- দুটোরই দাম এই মুহূর্তে দেখতে পারছি না। কাউন্টারে জিজ্ঞেস করে নিতে পারেন।"
    delta = cmp["price_delta"]
    if cmp.get("cheaper") is None:
        return f"{name_a} আর {name_b} -- দুটোরই দাম একই। কোনটা করাতে চান?"
    cheaper = name_a if cmp["cheaper"] == "a" else name_b
    dearer = name_b if cmp["cheaper"] == "a" else name_a
    line = f"{cheaper} {dearer}-এর চেয়ে {delta} টাকা কম। "
    if cmp.get("both_packages") and cmp.get("more_tests_side"):
        more = name_a if cmp["more_tests_side"] == "a" else name_b
        line += f"{more}-এ {cmp['test_count_delta']}টা টেস্ট বেশি আছে। "
    return line + "কোনটা করাতে চান?"


# ============================================== KCD-387 full flow: "Caller asks whether a test can
# be collected at home" -- see agent/home_collection_flow.py's own module docstring for why this
# story carries its own cross-turn state instead of reusing BookingState. Every function below only
# formats a fact a real clinic-api response already carries (a price, a slot time, a booking
# reference, a collector's name) -- never invents, estimates, or calculates one; see
# home_collection_service.py's own docstring for where each of those numbers actually comes from.


def home_collection_eligibility_reply(result: dict, lang: str = "bn") -> str:
    """`result` is GET .../home-collection/eligibility-multi's own shape. EVERY requested test is
    reported -- eligible ones named together, then each ineligible/not-covered/not-found one with
    its own reason -- never merged into a single yes/no (section 8's "never silently choose one
    test" rule, the same one multi_test_eligibility() itself is built around)."""
    if lang != "bn":
        return _i18n.home_collection_eligibility_reply(result, lang)
    rows = result.get("results") or []
    not_found = result.get("not_found") or []
    eligible = [r for r in rows if r.get("found") and r.get("eligible")]
    ineligible = [r for r in rows if r.get("found") and not r.get("eligible")]
    parts = []
    if eligible:
        names = ", ".join(r.get("test_name") or "" for r in eligible)
        parts.append(f"{names} -- এই টেস্ট(গুলো) বাড়িতে এসে করানো যাবে।")
    for r in ineligible:
        name = r.get("test_name") or ""
        if r.get("reason") == "area_not_covered":
            parts.append(f"{name} টেস্টের জন্য আপনার এলাকায় এখনো বাড়িতে এসে স্যাম্পল নেওয়ার সুবিধা নেই।")
        else:
            parts.append(f"{name} টেস্টটা বাড়িতে এসে করানো যায় না, ল্যাবেই আসতে হবে।")
    for name in not_found:
        parts.append(f"'{name}' নামে কোনো টেস্ট আমাদের তালিকায় পেলাম না।")
    if not parts:
        parts.append("দুঃখিত, এই টেস্ট(গুলো)-র তথ্য এই মুহূর্তে দেখতে পারছি না।")
    return " ".join(parts)


def home_collection_slots_reply(slots: list[dict], lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.home_collection_slots_reply(slots, lang)
    if not slots:
        return "দুঃখিত, ওই দিনে কোনো স্লট ফাঁকা নেই। অন্য কোনো দিন বলবেন?"
    shown = slots[:MAX_SLOTS_READ_ALOUD]
    if len(shown) == 1:
        s = shown[0]
        return f"{s['date']} তারিখে {s['start_time']} থেকে {s['end_time']}-এর মধ্যে স্লট ফাঁকা আছে। এটা রাখব?"
    windows = ", ".join(f"{s['start_time']} থেকে {s['end_time']}" for s in shown)
    return f"{shown[0]['date']} তারিখে এই সময়গুলো ফাঁকা আছে -- {windows}। কোনটা চান?"


def home_collection_hold_failed_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.home_collection_hold_failed_reply(result, lang)
    if result.get("reason") == "slot_full":
        return "দুঃখিত, ওই সময়টা এর মধ্যে ভর্তি হয়ে গেছে। অন্য একটা সময় বলবেন?"
    return "দুঃখিত, সময়টা ধরে রাখা গেল না। আবার চেষ্টা করি?"


def home_collection_quote_reply(quote: dict, payment_policy_text: str | None, lang: str = "bn") -> str:
    """The deterministic Test1/Test2/.../home-collection-charge/Total breakdown (section 17),
    followed by the real payment policy (section 18) and the "shall I proceed?" gate (section 19) --
    every number here is a straight substitution from `quote` (clinic-api's
    home_collection_service.quote_home_collection, never recomputed here)."""
    if lang != "bn":
        return _i18n.home_collection_quote_reply(quote, payment_policy_text, lang)
    if not quote.get("quote_available"):
        return "দুঃখিত, এই টেস্ট(গুলো)-র জন্য এখন দাম হিসেব করা যাচ্ছে না।"
    lines = [f"{r.get('test_name')} {r.get('rate_inr')} টাকা" for r in (quote.get("per_test") or []) if r.get("eligible")]
    breakdown = ", ".join(lines)
    charge = quote.get("home_collection_charge_inr") or 0
    total = quote.get("total_inr") or 0
    reply = f"{breakdown}, এবং বাড়িতে এসে নেওয়ার জন্য {charge} টাকা -- সব মিলিয়ে মোট {total} টাকা।"
    if payment_policy_text:
        reply += f" {payment_policy_text}"
    return reply + " এগিয়ে যাব?"


def home_collection_stored_address_offer(address: str, lang: str = "bn") -> str:
    """Section 21/KCD-499: a stored address is OFFERED, never silently applied -- same
    confirm-before-use discipline as main.py's own _offer_preferences."""
    if lang != "bn":
        return _i18n.home_collection_stored_address_offer(address, lang)
    return f"আগের বার দেওয়া এই ঠিকানাতেই পাঠাব -- {address}? নাকি নতুন ঠিকানা দেবেন?"


def home_collection_address_pincode_mismatch_reply(address_pin: str, given_pin: str, lang: str = "bn") -> str:
    """Section 22: never guess which of the two pincodes is right -- say what was heard and ask
    again, the same honest-conflict discipline as agent/reply_templates.booking_reply's own
    "slot_taken"/"doctor_ambiguous" branches."""
    if lang != "bn":
        return _i18n.home_collection_address_pincode_mismatch_reply(address_pin, given_pin, lang)
    return (
        f"আপনার ঠিকানায় {address_pin} পিনকোড শুনলাম, কিন্তু আগে আপনি {given_pin} বলেছিলেন -- দুটো আলাদা। "
        "ঠিকানাটা আর একবার বলবেন?"
    )


def home_collection_address_confirm_reply(address: str, lang: str = "bn") -> str:
    """Section 23: read the address back BEFORE anything is written, same "readback before
    commit" discipline as booking_confirmation_readback above."""
    if lang != "bn":
        return _i18n.home_collection_address_confirm_reply(address, lang)
    return f"ঠিকানাটা শুনলাম -- {address}। এটা ঠিক আছে তো?"


def home_collection_final_confirm_reply(summary: dict, lang: str = "bn") -> str:
    """Section 25's last gate before the real booking write: the full summary, read back once
    more with the slot, address and total all together, so a caller who has been answering one
    question at a time hears the whole picture before committing."""
    if lang != "bn":
        return _i18n.home_collection_final_confirm_reply(summary, lang)
    # ADDED BY SOURAV: split across 4 short sentences, not 1 long run-on -- persona.violations()'s
    # own per-language MAX_SENTENCE_WORDS cap (bn: 15) exists so a caller hears this read-back in
    # digestible pieces, not a single breathless clause.
    tests = ", ".join(summary.get("eligible_tests") or [])
    return (
        f"তাহলে {tests} -- {summary.get('date')} তারিখে, সময় {summary.get('start_time')} থেকে "
        f"{summary.get('end_time')}-এর মধ্যে বাড়িতে এসে স্যাম্পল নেওয়া হবে। "
        f"ঠিকানা {summary.get('address')}। "
        f"মোট {summary.get('total_inr')} টাকা। এটা কনফার্ম করব?"
    )


def home_collection_booking_result_reply(result: dict, lang: str = "bn") -> str:
    """Section 25/26/27: a real, persisted booking only -- `result` is clinic-api's own
    create_home_collection_booking()/_booking_dict() shape. A collector's name is spoken ONLY when
    `collector_name` is actually set (a real assignment exists); otherwise the caller is told
    staff will confirm who comes, exactly section 26's own "never invent a collector name" rule --
    never "Rahul will come tomorrow" on a booking nobody has actually assigned."""
    if lang != "bn":
        return _i18n.home_collection_booking_result_reply(result, lang)
    if not result.get("success"):
        reason = result.get("reason")
        if reason == "hold_expired":
            return "দুঃখিত, সময়টা ধরে রাখা যায়নি, একটু দেরি হয়ে গেছে। আবার চেষ্টা করি?"
        if reason in ("no_eligible_test", "test_not_found"):
            return "দুঃখিত, এই টেস্ট(গুলো)-র জন্য বাড়িতে এসে বুকিং করা গেল না।"
        return "দুঃখিত, বুকিং করা গেল না। একটু পরে আবার চেষ্টা করুন, অথবা কাউন্টারে যোগাযোগ করুন।"
    collector = result.get("collector_name")
    assignment_line = (
        f" {collector} আসবেন।" if collector else " কে আসবেন, সেটা আমাদের স্টাফ পরে কনফার্ম করে জানিয়ে দেবে।"
    )
    return (
        f"আপনার বাড়িতে এসে স্যাম্পল নেওয়ার বুকিং কনফার্ম হয়েছে। "
        f"{result.get('date')} তারিখে, সময় {result.get('start_time')} থেকে {result.get('end_time')}-এর মধ্যে, "
        f"ঠিকানা {result.get('address_line')}-এ।{assignment_line} "
        f"মোট {result.get('total_inr')} টাকা। কনফার্মেশন নম্বর {result.get('booking_reference')}।"
    )


def home_collection_cancelled_reply(lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.home_collection_cancelled_reply(lang)
    return "ঠিক আছে, বুকিং করা হলো না। আর কিছু সাহায্য করতে পারি?"
