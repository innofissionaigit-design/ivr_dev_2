"""Fixed, non-factual phrases the agent says, per language.

Everything here is a sentence with no fact in it -- a greeting, an apology,
a hand-off notice. Facts (prices, times, instructions) never live in this
file; they are template substitutions from a live API response
(agent/reply_templates.py).

Kept as one table so the set of things the agent can say on a fixed path
exists in exactly one place: the same strings are spoken, pre-synthesized
at startup (TTSClient.prewarm) and pinned by tests/test_i18n_phrases.py to
be identical across languages in coverage.
"""

from __future__ import annotations

PHRASES: dict[str, dict[str, str]] = {
    "bn": {
        "greeting": "নমস্কার। বলুন, কীভাবে সাহায্য করতে পারি?",
        "asr_empty": "দুঃখিত। আমি শুনতে পাইনি। আবার বলবেন?",
        "unclear": "দুঃখিত। আমি বুঝতে পারিনি। আবার বলবেন?",
        "llm_failure": "একটু সমস্যা হচ্ছে, একটু ধরুন।",
        "tool_failure": "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
        "idle_close": "লাইনে কোনো সাড়া পাচ্ছি না, কল শেষ করছি। ধন্যবাদ।",
        "smalltalk_default": "নমস্কার, কী সাহায্য করতে পারি?",
        "handoff": "আমি আপনাকে আমাদের স্টাফের সাথে যুক্ত করছি, দয়া করে লাইনে থাকুন।",
        "cancel_aborted": "ঠিক আছে, আপনার অ্যাপয়েন্টমেন্টটা যেমন ছিল তেমনই আছে।",
        "no_confirmation_number": "ঠিক আছে, তাহলে বুকিংটা হয়ে যাবে, কিন্তু কোনো লিখিত কনফার্মেশন পাঠাতে পারব না।",
        "language_switched": "ঠিক আছে, এখন থেকে বাংলায় বলছি।",
        # KCD-461: spoken only when the turn is STILL unanswered FILLER_THRESHOLD_S (0.7 s) after the caller stopped
        # (main.py's _await_with_filler); an answer that arrives sooner is given with no filler at all.
        # OWNER'S WORDING, 2026-09-25: "আচ্ছা বলছি" ("okay, I'm telling you"), in all three languages -- pre-warmed
        # and cached like every other line here, so it costs no synthesis time when the system is already slow.
        "please_wait": "আচ্ছা, বলছি।",
        # The caller has said nothing for SILENCE_PROMPT_S after the agent finished speaking: ask once whether there is
        # more (one question), and say that the call ends if not. If still silent, idle_close ends the call.
        "silence_prompt": "আর কিছু জানতে চান? না চাইলে আমি কল শেষ করে দেব।",
        "silence_go_on": "জি, বলুন।",
        "silence_goodbye": "ঠিক আছে, ধন্যবাদ। ভালো থাকবেন।",
        # Abuse from the caller (agent/abuse.py): calm, no imitation, no lecture; the third time the call is closed.
        "abuse_first": "আমি আপনাকে সাহায্য করতে এখানে আছি। ভদ্রভাবে বললে আমি আরও ভালো সাহায্য করতে পারব। বলুন, কী জানতে চান?",
        "abuse_second": "ভদ্রভাবে কথা বললেই আমি সাহায্য করতে পারব। আপনি কি কোনো তথ্য জানতে চান?",
        "abuse_final": "আমি এভাবে কথা চালিয়ে যেতে পারছি না। দরকার হলে আবার ফোন করবেন। ধন্যবাদ।",
        # Ported story "Caller wants to make a complaint" (agent/complaint.py). Fixed text, never the model:
        # acceptance criterion 5 forbids defending the clinic or explaining the complaint away, and a sentence
        # the model composes is a sentence that can do both. It confirms three things and promises nothing
        # further -- heard, written down as said, passed to a person -- and claims no live transfer this stack
        # cannot make (there is no SIP transfer here; see main._handoff_to_human).
        "complaint_acknowledged": "বুঝতে পেরেছি, এবং আপনার এই অসুবিধার জন্য আমি দুঃখিত। আপনি যেভাবে বলেছেন ঠিক সেভাবেই আমি আপনার অভিযোগটি লিখে নিয়েছি। এটি আমাদের টিমের কাছে পাঠিয়ে দেওয়া হবে।",
        # Ported story "Caller asks whether their result is dangerous" (agent/clinical_safety.py). The acceptance
        # criterion is explicit that this must OFFER a clinician, not merely decline: a frightened caller met with
        # a refusal, or with the generic greeting this turn used to fall through to, experiences the procedural
        # limit as a rebuff. So it says who WILL explain the result, and asks to connect them -- one question, and
        # no reassurance of any kind, which agent/persona.py would reject anyway.
        "clinical_interpretation": "এটা একজন ডাক্তারের ব্যাখ্যা করার বিষয়। আমি আপনাকে আমাদের চিকিৎসকের সাথে যুক্ত করে দিতে পারি। করে দেব?",
        # Ported story "Caller wants to speak to a doctor personally"
        # (agent/doctor_personal_request.py). Three facts and one offer, in that order: no direct connection is
        # possible on this call, no NAMED doctor can be promised a callback, and an appointment is the real route
        # to a doctor. It never names a doctor and never repeats one the caller named -- there is no clinician on
        # this line and no way to reach one from it, so anything warmer would be a promise this stack cannot keep.
        "doctor_personal_request": "এই কলে আমি কোনো ডাক্তারের সাথে যুক্ত করে দিতে পারি না। কোনো নির্দিষ্ট ডাক্তার ফোন করবেন, এমন কথাও দিতে পারি না। আমি আপনার জন্য একটা অ্যাপয়েন্টমেন্ট করে দিতে পারি। করে দেব?",
        # Ported story "Caller states something the agent cannot verify" (agent/unverifiable_claim.py), for the one
        # category nothing in this schema can ever check: a doctor having "already approved" something. It declines
        # the claim without echoing it back as though it were true, and offers the staff who CAN check -- never a
        # live transfer, which this stack cannot make. A claim about an appointment or a payment never reaches this
        # line: both are checkable here, so they go to the real lookup instead.
        "cannot_confirm_claim": "এটা আমি এখান থেকে নিশ্চিত করতে পারছি না। আমাদের স্টাফ এটা ভালোভাবে দেখে নিতে পারবেন। যুক্ত করে দেব?",
        # Ported story "Caller asks to be called back" (agent/callback_request.py). There is NO outbound calling in
        # this stack, so none of these three lines may imply the agent itself will ring: the request is written
        # down for a colleague, and that is what is said. No line promises a TIME either -- this repository has no
        # structured opening hours to judge one from (only localised free text), and inferring one from prose is
        # exactly the kind of invented fact CLAUDE.md rule 1 forbids.
        "callback_noted": "আপনার নম্বরটা লিখে নিয়েছি, আমাদের একজন সহকর্মী আপনাকে ফোন করবেন। ঠিক কখন, সেটা বলতে পারছি না।",
        "callback_need_number": "আমাদের একজন সহকর্মী আপনাকে ফোন করতে পারবেন। কোন নম্বরে করবেন?",
        "callback_unavailable": "এই লাইনে আমি কলব্যাকের ব্যবস্থা করতে পারছি না। বলুন, আর কীভাবে সাহায্য করতে পারি।",
        # Ported story "Caller asks to compare two options" (agent/compare_flow.py): the compare cue was heard but
        # only one of the two things could be resolved against the catalogue. Asking is the only honest move --
        # comparing against a guessed second item would state a difference that was never computed from real data.
        "compare_need_two": "কোন দুটোর তুলনা করব, একটু দুটো নামই বলবেন?",
        # KCD-486: spoken instead of a confirmation number when the write
        # could not be verified in the system of record -- never a false
        # confirmation, and never the generic tool_failure apology either,
        # since the write likely DID happen and a retry could double-book.
        "booking_hold_for_verification": "আপনার বুকিংটা প্রসেস হচ্ছে, একটু দেখে নিশ্চিত করছি।",
        # KCD-382, story "Caller has several tests with conflicting preparation", AC3: spoken BEFORE
        # main._handoff_to_human, the same two-step shape as "booking_hold_for_verification" above, for the
        # one case merge_prep_instructions() cannot safely resolve on its own (a mutually exclusive SOP, or a
        # resolved test whose structured fasting data is still incomplete -- see that function's docstring).
        # Never says WHY in clinical terms -- that would be the model, or this file, inventing a reason.
        "test_prep_conflict_handoff": "এই টেস্টগুলোর প্রস্তুতি একসাথে নিশ্চিতভাবে বলতে পারছি না। আমি আপনাকে আমাদের স্টাফের সাথে যুক্ত করছি।",
        # Empathetic re-asks, spoken BEFORE any hand-off when the caller could
        # not be heard or understood (agent/reask_policy.py). The wording
        # takes the blame ("that's on me"), never the caller's -- a person who
        # is faint, mumbling or in a noisy room is usually already
        # embarrassed, and being told they are the problem makes it worse.
        # One question each, short sentences: they must satisfy the senior
        # policy (agent/speech_policy.py) too. Wording pending clinical-lead
        # review, like every acknowledgement in this codebase.
        "reask_low_volume": "দুঃখিত। আপনার গলা একটু আস্তে শোনাচ্ছে। একটু জোরে বলবেন, প্লিজ?",
        "reask_noisy": "দুঃখিত। লাইনে একটু শব্দ আছে। শান্ত জায়গা থেকে আবার বলবেন?",
        "reask_crosstalk": "দুঃখিত। পাশে আরও কারও গলা শোনা যাচ্ছে। আপনি আবার বলবেন?",
        "reask_mumbled": "দুঃখিত। আমি ঠিক বুঝতে পারিনি। কোনও তাড়া নেই, একটু ধীরে বলবেন?",
        "reask_generic": "দুঃখিত। আমি ঠিক শুনতে পাইনি। আবার বলবেন?",
        "reask_final": "দুঃখিত। আমি আপনাকে বুঝতে পারলাম না।",
    },
    "hi": {
        "greeting": "नमस्कार। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
        "asr_empty": "माफ़ कीजिए। मैं सुन नहीं पाई। क्या आप फिर बोलेंगे?",
        "unclear": "माफ़ कीजिए। मैं समझ नहीं पाई। क्या आप फिर बताएँगे?",
        "llm_failure": "थोड़ी दिक्कत आ रही है, कृपया एक पल रुकिए।",
        "tool_failure": "अभी मैं यह देख नहीं पा रही हूँ। कृपया काउंटर पर संपर्क करें।",
        "idle_close": "लाइन पर कोई आवाज़ नहीं आ रही, कॉल समाप्त कर रही हूँ। धन्यवाद।",
        "smalltalk_default": "नमस्कार, मैं आपकी क्या मदद कर सकती हूँ?",
        "handoff": "मैं आपको हमारे स्टाफ़ से जोड़ रही हूँ, कृपया लाइन पर बने रहिए।",
        "cancel_aborted": "ठीक है, आपकी अपॉइंटमेंट जैसी थी वैसी ही है।",
        "no_confirmation_number": "ठीक है, बुकिंग हो जाएगी, लेकिन मैं कोई लिखित कन्फ़र्मेशन नहीं भेज पाऊँगी।",
        "language_switched": "ठीक है, अब हिंदी में बात करती हूँ।",
        "please_wait": "अच्छा, बताती हूँ।",
        "silence_prompt": "क्या आप कुछ और जानना चाहते हैं? नहीं तो मैं कॉल समाप्त कर दूँगी।",
        "silence_go_on": "जी, बताइए।",
        "silence_goodbye": "ठीक है, धन्यवाद। अपना ख़याल रखिए।",
        "abuse_first": "मैं आपकी मदद के लिए यहाँ हूँ। सम्मान से बात करेंगे तो मैं और अच्छी मदद कर पाऊँगी। बताइए, आप क्या जानना चाहते हैं?",
        "abuse_second": "सम्मान से बात करने पर ही मैं मदद कर पाऊँगी। क्या आप कोई जानकारी चाहते हैं?",
        "abuse_final": "मैं इस तरह बात जारी नहीं रख सकती। ज़रूरत हो तो दोबारा फ़ोन करें। धन्यवाद।",
        # Ported story "Caller wants to make a complaint" (agent/complaint.py). Fixed text, never the model:
        # acceptance criterion 5 forbids defending the clinic or explaining the complaint away, and a sentence
        # the model composes is a sentence that can do both. It confirms three things and promises nothing
        # further -- heard, written down as said, passed to a person -- and claims no live transfer this stack
        # cannot make (there is no SIP transfer here; see main._handoff_to_human).
        "complaint_acknowledged": "मैं समझ गई, और आपको जो असुविधा हुई उसके लिए मुझे खेद है। आपने जैसा बताया है ठीक वैसे ही मैंने आपकी शिकायत लिख ली है। इसे हमारी टीम तक पहुँचा दिया जाएगा।",
        # Ported story "Caller asks whether their result is dangerous" (agent/clinical_safety.py). The acceptance
        # criterion is explicit that this must OFFER a clinician, not merely decline: a frightened caller met with
        # a refusal, or with the generic greeting this turn used to fall through to, experiences the procedural
        # limit as a rebuff. So it says who WILL explain the result, and asks to connect them -- one question, and
        # no reassurance of any kind, which agent/persona.py would reject anyway.
        "clinical_interpretation": "यह बात डॉक्टर ही ठीक से समझा सकते हैं। मैं आपको हमारे चिकित्सक से जोड़ सकती हूँ। जोड़ दूँ?",
        # Ported story "Caller wants to speak to a doctor personally"
        # (agent/doctor_personal_request.py). Three facts and one offer, in that order: no direct connection is
        # possible on this call, no NAMED doctor can be promised a callback, and an appointment is the real route
        # to a doctor. It never names a doctor and never repeats one the caller named -- there is no clinician on
        # this line and no way to reach one from it, so anything warmer would be a promise this stack cannot keep.
        "doctor_personal_request": "इस कॉल पर मैं आपको किसी डॉक्टर से नहीं जोड़ सकती। यह भी नहीं कह सकती कि कोई खास डॉक्टर आपको फ़ोन करेंगे। मैं आपके लिए अपॉइंटमेंट बुक कर सकती हूँ। कर दूँ?",
        # Ported story "Caller states something the agent cannot verify" (agent/unverifiable_claim.py), for the one
        # category nothing in this schema can ever check: a doctor having "already approved" something. It declines
        # the claim without echoing it back as though it were true, and offers the staff who CAN check -- never a
        # live transfer, which this stack cannot make. A claim about an appointment or a payment never reaches this
        # line: both are checkable here, so they go to the real lookup instead.
        "cannot_confirm_claim": "यह मैं यहाँ से पुष्टि नहीं कर सकती। हमारे स्टाफ़ इसे ठीक से देख सकते हैं। जोड़ दूँ?",
        # Ported story "Caller asks to be called back" (agent/callback_request.py). There is NO outbound calling in
        # this stack, so none of these three lines may imply the agent itself will ring: the request is written
        # down for a colleague, and that is what is said. No line promises a TIME either -- this repository has no
        # structured opening hours to judge one from (only localised free text), and inferring one from prose is
        # exactly the kind of invented fact CLAUDE.md rule 1 forbids.
        "callback_noted": "आपका नंबर मैंने लिख लिया है, हमारे एक सहकर्मी आपको फ़ोन करेंगे। ठीक कब, यह नहीं कह सकती।",
        "callback_need_number": "हमारे एक सहकर्मी आपको फ़ोन कर सकते हैं। किस नंबर पर करें?",
        "callback_unavailable": "इस लाइन पर मैं कॉलबैक की व्यवस्था नहीं कर सकती। बताइए, और किस तरह मदद कर सकती हूँ।",
        # Ported story "Caller asks to compare two options" (agent/compare_flow.py): the compare cue was heard but
        # only one of the two things could be resolved against the catalogue. Asking is the only honest move --
        # comparing against a guessed second item would state a difference that was never computed from real data.
        "compare_need_two": "किन दो चीज़ों की तुलना करनी है, दोनों नाम बताएँगे?",
        "booking_hold_for_verification": "आपकी बुकिंग प्रोसेस हो रही है, मैं अभी पुष्टि करके बताती हूँ।",
        "test_prep_conflict_handoff": "मैं इन टेस्ट की तैयारी को पक्के तौर पर एक साथ नहीं बता पा रही। मैं आपको हमारे स्टाफ़ से जोड़ रही हूँ।",
        "reask_low_volume": "माफ़ कीजिए। आपकी आवाज़ थोड़ी धीमी है। क्या आप थोड़ा ज़ोर से बोलेंगे?",
        "reask_noisy": "माफ़ कीजिए। लाइन पर थोड़ा शोर है। क्या आप शांत जगह से फिर बोलेंगे?",
        "reask_crosstalk": "माफ़ कीजिए। पास से किसी और की आवाज़ आ रही है। क्या आप फिर बोलेंगे?",
        "reask_mumbled": "माफ़ कीजिए। मैं ठीक से समझ नहीं पाई। कोई जल्दी नहीं, क्या आप धीरे बोलेंगे?",
        "reask_generic": "माफ़ कीजिए। मैं ठीक से सुन नहीं पाई। क्या आप फिर बोलेंगे?",
        "reask_final": "माफ़ कीजिए। मैं आपको समझ नहीं पाई।",
    },
    "en": {
        "greeting": "Hello. How can I help you?",
        "asr_empty": "Sorry. I did not catch that. Could you say it again?",
        "unclear": "Sorry. I did not understand. Could you say that again?",
        "llm_failure": "We're having a small problem, please hold for a moment.",
        "tool_failure": "I can't check that right now. Please contact the counter.",
        "idle_close": "I can't hear anything on the line, so I'm ending the call. Thank you.",
        "smalltalk_default": "Hello, how can I help you?",
        "handoff": "Let me connect you to our staff, please stay on the line.",
        "cancel_aborted": "Okay, your appointment is unchanged.",
        "no_confirmation_number": "Okay, I'll go ahead with the booking, but I won't be able to send you a written confirmation.",
        "language_switched": "Okay, switching to English now.",
        "please_wait": "Alright, let me tell you.",
        "silence_prompt": "Is there anything else you would like to know? If not, I will end the call.",
        "silence_go_on": "Yes, please go ahead.",
        "silence_goodbye": "Alright, thank you. Take care.",
        "abuse_first": "I am here to help you. I can help you better if we speak politely. What would you like to know?",
        "abuse_second": "I can only help if we speak politely. Is there some information you need?",
        "abuse_final": "I cannot continue the conversation like this. Please call again whenever you need. Thank you.",
        # Ported story "Caller wants to make a complaint" (agent/complaint.py). Fixed text, never the model:
        # acceptance criterion 5 forbids defending the clinic or explaining the complaint away, and a sentence
        # the model composes is a sentence that can do both. It confirms three things and promises nothing
        # further -- heard, written down as said, passed to a person -- and claims no live transfer this stack
        # cannot make (there is no SIP transfer here; see main._handoff_to_human).
        "complaint_acknowledged": "I understand, and I am sorry this happened. I have written your complaint down exactly as you described it, and it will go straight to our team.",
        # Ported story "Caller asks whether their result is dangerous" (agent/clinical_safety.py). The acceptance
        # criterion is explicit that this must OFFER a clinician, not merely decline: a frightened caller met with
        # a refusal, or with the generic greeting this turn used to fall through to, experiences the procedural
        # limit as a rebuff. So it says who WILL explain the result, and asks to connect them -- one question, and
        # no reassurance of any kind, which agent/persona.py would reject anyway.
        "clinical_interpretation": "That is for a doctor to explain, and you should hear it from one. I can connect you with our clinician. Shall I?",
        # Ported story "Caller wants to speak to a doctor personally"
        # (agent/doctor_personal_request.py). Three facts and one offer, in that order: no direct connection is
        # possible on this call, no NAMED doctor can be promised a callback, and an appointment is the real route
        # to a doctor. It never names a doctor and never repeats one the caller named -- there is no clinician on
        # this line and no way to reach one from it, so anything warmer would be a promise this stack cannot keep.
        "doctor_personal_request": "I cannot connect you to a doctor directly on this call. I also cannot promise that a particular doctor will ring you back. What I can do is book you an appointment to see one. Shall I?",
        # Ported story "Caller states something the agent cannot verify" (agent/unverifiable_claim.py), for the one
        # category nothing in this schema can ever check: a doctor having "already approved" something. It declines
        # the claim without echoing it back as though it were true, and offers the staff who CAN check -- never a
        # live transfer, which this stack cannot make. A claim about an appointment or a payment never reaches this
        # line: both are checkable here, so they go to the real lookup instead.
        "cannot_confirm_claim": "I cannot confirm that from here. Our staff can check it properly for you. Shall I connect you?",
        # Ported story "Caller asks to be called back" (agent/callback_request.py). There is NO outbound calling in
        # this stack, so none of these three lines may imply the agent itself will ring: the request is written
        # down for a colleague, and that is what is said. No line promises a TIME either -- this repository has no
        # structured opening hours to judge one from (only localised free text), and inferring one from prose is
        # exactly the kind of invented fact CLAUDE.md rule 1 forbids.
        "callback_noted": "I have noted your number, and a colleague will call you back. I am not able to say exactly when.",
        "callback_need_number": "A colleague can call you back. What number should they use?",
        "callback_unavailable": "I am not able to arrange a callback on this line. Tell me what you need and I will help where I can.",
        # Ported story "Caller asks to compare two options" (agent/compare_flow.py): the compare cue was heard but
        # only one of the two things could be resolved against the catalogue. Asking is the only honest move --
        # comparing against a guessed second item would state a difference that was never computed from real data.
        "compare_need_two": "Which two should I compare? Please give me both names.",
        "booking_hold_for_verification": "Your booking is being processed, let me confirm it for you.",
        "test_prep_conflict_handoff": "I'm not able to safely combine the preparation for these tests into one instruction. Let me connect you with our staff.",
        "reask_low_volume": "Sorry. Your voice is a little soft. Could you please speak a little louder?",
        "reask_noisy": "Sorry. There is some noise on the line. Could you try somewhere quieter?",
        "reask_crosstalk": "Sorry. I can hear someone else near you. Could you say that again?",
        "reask_mumbled": "Sorry. I did not understand. There is no rush. Could you say it slowly?",
        "reask_generic": "Sorry. I did not catch that. Could you say it again?",
        "reask_final": "Sorry. I could not understand you.",
    },
}

DEFAULT_LANGUAGE = "bn"

# KCD-353: the greeting says, before anything else, WHO the caller is speaking with and that it is an
# automated assistant, and that a person is available. It is composed at call time from parts the
# operator can change in the database (agent/messages.py): the welcome, the disclosure
# (agent/disclosure.py, versioned), and a one-line pointer to 112 for emergencies. The text kept in
# PHRASES["greeting"] is the built-in composition, used for the start-up pre-synthesis cache.
from agent import messages as _messages  # noqa: E402  (deliberately after the tables above: see the comment)

EMERGENCY_HINT = {
    "bn": "জরুরি অবস্থায় সরাসরি ১১২ নম্বরে ফোন করবেন।",
    "hi": "आपात स्थिति में कृपया सीधे 112 पर कॉल करें।",
    "en": "In an emergency, please call 112 directly.",
}


# OWNER'S INSTRUCTION, 2026-09-25 (after the first live call): the first thing a caller hears is only
#   "Namaskar. You are speaking with Sonoscan Vaani. Tell me, how can I help you?"
# so the spoken greeting is the welcome, this one identity sentence, and the question. It replaces the longer
# KCD-353 opening (that it is an automated assistant, that staff are available at any time, and the pointer to 112).
# The full disclosure text still exists (the text-message notice, and the once-only re-statement when a caller
# switches language) and both removed parts can be brought back with no deploy: change the `greeting_identity`
# row to the longer text, or add an `emergency_hint` row (the pointer is spoken only when such a row exists).
GREETING_IDENTITY = {
    "bn": "আপনি সোনোস্ক্যান বাণীর সঙ্গে কথা বলছেন।",
    "hi": "आप सोनोस्कैन वाणी से बात कर रहे हैं।",
    "en": "You are speaking with Sonoscan Vaani.",
}


def greeting_text(lang: str) -> str:
    """welcome. who you are speaking with. question -- the welcome and the question are the two sentences of the
    `greeting` phrase; the identity sentence (and, only if the operator has set one, the 112 pointer) go between."""
    import re

    base = _messages.text("greeting", lang, _BASE_GREETING.get(lang) or _BASE_GREETING["bn"])
    parts = [x for x in re.split(r"(?<=[।.!?])\s+", base.strip()) if x]
    middle = [_messages.text("greeting_identity", lang, GREETING_IDENTITY.get(lang) or GREETING_IDENTITY["bn"])]
    if _messages.has("emergency_hint", lang):
        middle.append(_messages.text("emergency_hint", lang, EMERGENCY_HINT.get(lang) or EMERGENCY_HINT["bn"]))
    if len(parts) < 2:
        return " ".join(parts + middle)
    return " ".join([parts[0]] + middle + parts[1:])


_BASE_GREETING = {lang: table["greeting"] for lang, table in PHRASES.items()}
for _lang, _table in PHRASES.items():
    _table["greeting"] = greeting_text(_lang)

# KCD-054: said when the voice on the line changes after the caller was verified.
# Deliberately says nothing about WHY (the agent cannot know who is speaking, only
# that the voice differs) and does not accuse anyone.
PHRASES["bn"]["reverify_notice"] = "নিরাপত্তার জন্য ব্যক্তিগত তথ্য বলার আগে আমাকে আবার পরিচয় যাচাই করতে হবে।"
PHRASES["hi"]["reverify_notice"] = "सुरक्षा के लिए, निजी जानकारी बताने से पहले मुझे दोबारा पहचान की पुष्टि करनी होगी।"
# A possible emergency (agent/emergency.py). DRAFT: the wording, and the number it names, need
# clinical sign-off before this is relied on. It states no condition and gives no advice beyond
# where to call; a person joins the call straight after.
PHRASES["bn"]["emergency_notice"] = "এটা জরুরি অবস্থা হতে পারে। এখনই ১১২ নম্বরে ফোন করুন, অথবা কাছের হাসপাতালে যান।"
PHRASES["hi"]["emergency_notice"] = "यह आपात स्थिति हो सकती है। कृपया अभी 112 पर कॉल करें, या नज़दीकी अस्पताल जाएँ।"
PHRASES["en"]["emergency_notice"] = "This may be an emergency. Please call 112 now, or go to the nearest hospital."
# The turn was heard but the NAME in it was not trusted (the recognisers disagreed): say so and ask for just the name.
PHRASES["bn"]["name_not_caught"] = "দুঃখিত, নামটা ঠিক শুনতে পাইনি।"
PHRASES["hi"]["name_not_caught"] = "माफ़ कीजिए, नाम ठीक से सुन नहीं पाई।"
PHRASES["en"]["name_not_caught"] = "Sorry, I did not catch the name."
PHRASES["en"]["reverify_notice"] = (
    "For your security, I need to verify who I am speaking with again before I share any personal details."
)

# ------------------------------------------------------------------------------------------- lay terms / prescription
# Said whenever the caller is asked to send a prescription photo over WhatsApp so a human colleague can call back
# (agent/lay_terms.py's HEAD category, and both blood/abdomen flows in main.py). The number is fixed and spoken
# plainly -- digits, not a link, since this is a voice call.
PHRASES["bn"]["send_prescription_whatsapp"] = (
    "প্রেসক্রিপশনের একটা ছবি এই হোয়াটসঅ্যাপ নম্বরে পাঠিয়ে দিন, ৯৬৩৫৫৮৮৩০৬। আমাদের একজন সহকর্মী দেখে আপনাকে কল করবেন।"
)
PHRASES["hi"]["send_prescription_whatsapp"] = (
    "पर्चे की एक फोटो इस व्हाट्सऐप नंबर पर भेज दीजिए, 9635588306। हमारा एक सहकर्मी देखकर आपको कॉल करेगा।"
)
PHRASES["en"]["send_prescription_whatsapp"] = (
    "Please send a photo of the prescription to this WhatsApp number, 9635588306. A colleague will look at it "
    "and call you back."
)

# The catalogue has no head/brain imaging test today (agent/lay_terms.py's module docstring); this says so plainly
# instead of naming a service the clinic does not offer, then hands off to send_prescription_whatsapp above.
PHRASES["bn"]["head_imaging_not_listed"] = (
    "মাথার স্ক্যান বা এমআরআই এখন আমাদের তালিকায় নেই। ডাক্তার প্রেসক্রিপশনে ঠিক কী বলেছেন সেটা দেখে বলা ভালো।"
)
PHRASES["hi"]["head_imaging_not_listed"] = (
    "सिर का स्कैन या एमआरआई अभी हमारी सूची में नहीं है। पर्चे में डॉक्टर ने ठीक क्या लिखा है, वह देखकर बताना बेहतर होगा।"
)
PHRASES["en"]["head_imaging_not_listed"] = (
    "A head scan or MRI is not on our list right now. It's best a colleague checks exactly what the doctor's "
    "prescription says."
)

# ------------------------------------------------------------------------------------------------------- lab tests
# A lab test is paid and booked at the counter, never over the call (main.py's book_test/add_test_booking intents).
PHRASES["bn"]["lab_test_counter_only"] = (
    "টেস্টের বুকিং আর পেমেন্ট আমাদের কাউন্টারে গিয়ে করতে হয়, ফোনে হয় না। প্রেসক্রিপশন নিয়ে কাউন্টারে চলে আসুন।"
)
PHRASES["hi"]["lab_test_counter_only"] = (
    "टेस्ट की बुकिंग और पेमेंट काउंटर पर आकर करनी होती है, फ़ोन पर नहीं होती। पर्चा लेकर काउंटर पर आ जाइए।"
)
PHRASES["en"]["lab_test_counter_only"] = (
    "A lab test has to be booked and paid for at the counter, not over the phone. Please come to the counter with "
    "your prescription."
)

# ---------------------------------------------------------------------------------------------------- payment link
# Said as the closing line after a doctor's appointment is confirmed: the payment link has been sent, and the call
# ends here (main.py's _end_call) -- the caller pays on their own phone, never a number spoken or taken on the call.
PHRASES["bn"]["payment_link_sent"] = (
    "আপনার অ্যাপয়েন্টমেন্ট বুক হয়েছে। পেমেন্ট লিংক আপনার মোবাইলে পাঠানো হয়েছে, ওখান থেকে পেমেন্ট করে বুকিং নিশ্চিত করুন। ধন্যবাদ।"
)
PHRASES["hi"]["payment_link_sent"] = (
    "आपकी अपॉइंटमेंट बुक हो गई है। पेमेंट लिंक आपके मोबाइल पर भेज दिया गया है, वहाँ से पेमेंट करके बुकिंग पक्की कर लीजिए। धन्यवाद।"
)
PHRASES["en"]["payment_link_sent"] = (
    "Your appointment is booked. A payment link has been sent to your phone -- please pay there to confirm it. "
    "Thank you."
)

# ---------------------------------------------------------------------------- KCD-382b: registered/booked prep
# Asked once per call, only when a test_prep turn names no test at all (main.py's test_prep dispatch branch):
# whether the caller is already registered / has a booking, so a DB-backed caller is never made to repeat test
# names the booking record already has. Answered deterministically (agent/booking_flow.classify_yes_no), never
# by the model -- same reasoning as every other yes/no decision point in this file.
PHRASES["bn"]["test_prep_ask_registered"] = "আপনি কি আগে থেকেই রেজিস্টার্ড পেশেন্ট, নাকি এই টেস্টটা বুক করা আছে?"
PHRASES["hi"]["test_prep_ask_registered"] = "क्या आप पहले से रजिस्टर्ड मरीज़ हैं, या आपने यह टेस्ट बुक करवाया है?"
PHRASES["en"]["test_prep_ask_registered"] = "Are you already a registered patient, or have you booked this test with us?"

# Asked right after a "yes" to the question above -- the existing identity key this repository's booking lookups
# already use (the registered/caller phone number), never a second, invented identity field.
PHRASES["bn"]["test_prep_ask_booking_phone"] = "যে নম্বর থেকে বুকিং করা হয়েছে, সেটা বলবেন?"
PHRASES["hi"]["test_prep_ask_booking_phone"] = "आपकी बुकिंग जिस नंबर से की गई है, वह बता देंगे?"
PHRASES["en"]["test_prep_ask_booking_phone"] = "Could you tell me the phone number your booking is under?"

# No TEST booking matched that number (wrong number, no booking at all, or only an unrelated doctor
# appointment) -- never invented, never silently merged with something else; the caller is then asked for
# the test name directly, exactly PATH A's own generic path.
PHRASES["bn"]["test_prep_no_booking_found"] = "ওই নম্বরে কোনো টেস্ট বুকিং পাইনি।"
PHRASES["hi"]["test_prep_no_booking_found"] = "उस नंबर से मुझे कोई टेस्ट बुकिंग नहीं मिली।"
PHRASES["en"]["test_prep_no_booking_found"] = "I couldn't find a test booking with that number."

# Spoken when a call is turned away before the caller has said a word, so
# there is no language to choose: all three, shortest first is not worth the
# complexity -- they are pre-synthesized and cached, so this costs no GPU
# time at exactly the moment the system is overloaded.
HANDOFF_ALL_LANGUAGES = ("bn", "hi", "en")


def phrase(key: str, lang: str = DEFAULT_LANGUAGE) -> str:
    table = PHRASES.get(lang) or PHRASES[DEFAULT_LANGUAGE]
    if key == "greeting":
        return greeting_text(lang if lang in PHRASES else DEFAULT_LANGUAGE)
    # the operator can change any phrase from the database (agent/messages.py); the built-in text
    # is what is spoken whenever the database has no row or cannot be reached
    return _messages.text(key, lang if lang in PHRASES else DEFAULT_LANGUAGE, table[key])


def prewarm_lines() -> dict[str, list[str]]:
    return {lang: list(table.values()) for lang, table in PHRASES.items()}
