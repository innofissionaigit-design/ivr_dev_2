# Emergency handling: samples for clinical and native-speaker approval

**Status: APPROVED by the project owner (in chat, 2026-09-24): the notice as wording A, the greeting pointer, and the phrase samples exactly as listed.** No named clinician or native reviewer is recorded in the table below; if the clinic requires one, that is a separate sign-off and the wording can still be changed from the database.

The code is `agent/emergency.py` (what is detected) and `agent/phrases.py` (`emergency_notice`, and the
greeting pointer `EMERGENCY_HINT`). Approved wording is changed in one place each; a notice change needs
no code change, only a database row (`emergency_notice`, `emergency_hint` in `agent_messages`).

## What happens

1. On EVERY turn, in every language, at any speech-recognition confidence, before any model is consulted,
   the transcript is checked for the situations below.
2. If one is found the agent stops whatever it was doing (a booking, a question), says the **notice**, and
   a person joins the call.
3. It says no condition, gives no advice, and asks no question.

Known limit, accepted for the pilot: it runs on a **finished transcript**, one turn at a time. A caller
who is mid-sentence is not caught until they pause. That is why the greeting also tells every caller, in
their language, to call 112 directly in an emergency.

## The notice (spoken when an emergency phrase is found)

| Language | Wording |
|---|---|
| Bengali | এটা জরুরি অবস্থা হতে পারে। এখনই ১১২ নম্বরে ফোন করুন, অথবা কাছের হাসপাতালে যান। |
| Hindi | यह आपात स्थिति हो सकती है। कृपया अभी 112 पर कॉल करें, या नज़दीकी अस्पताल जाएँ। |
| English | This may be an emergency. Please call 112 now, or go to the nearest hospital. |

Then the usual hand-over line: "I am connecting you with our staff, please stay on the line."

**Alternative wordings for the clinician to choose from** (say which, or give your own):

| | A (current) | B (staff first) | C (shortest) |
|---|---|---|---|
| English | This may be an emergency. Please call 112 now, or go to the nearest hospital. | This may be an emergency. I am connecting you to our staff now. If someone needs help right away, please call 112. | Please call 112 now. |
| Hindi | यह आपात स्थिति हो सकती है। कृपया अभी 112 पर कॉल करें, या नज़दीकी अस्पताल जाएँ। | यह आपात स्थिति हो सकती है। मैं आपको अभी स्टाफ़ से जोड़ रही हूँ। तुरंत मदद चाहिए तो 112 पर कॉल करें। | कृपया अभी 112 पर कॉल करें। |
| Bengali | এটা জরুরি অবস্থা হতে পারে। এখনই ১১২ নম্বরে ফোন করুন, অথবা কাছের হাসপাতালে যান। | এটা জরুরি অবস্থা হতে পারে। আমি আপনাকে এখনই স্টাফের সঙ্গে যুক্ত করছি। এখনই সাহায্য লাগলে ১১২ নম্বরে ফোন করুন। | এখনই ১১২ নম্বরে ফোন করুন। |

**Questions for the clinical lead:**
1. Is 112 the right number to name, or should the clinic's own emergency line be said as well?
2. Should the agent also offer a transfer to the clinic's own team (B), or only name 112 (A, C)?
3. Is "may be an emergency" acceptable, or should it be firmer ("This is an emergency")?

> **CHANGED 2026-09-25 (owner's instruction after the first live call): the greeting no longer speaks this pointer.**
> The spoken greeting is now only the welcome, "you are speaking with Sonoscan Vaani", and the question. The
> pointer below is spoken again only if an operator adds an `emergency_hint` row (no deploy). The accepted limit
> earlier in this document ("that is why the greeting also tells every caller to call 112") no longer holds until
> then: a caller who is mid-sentence in an emergency is not caught until they pause, and is no longer told in
> advance to call 112 themselves. The emergency notice on a detected phrase is unchanged.

## The greeting pointer (spoken to every caller at the start of the call) -- NOT SPOKEN BY DEFAULT since 2026-09-25

| Language | Wording |
|---|---|
| Bengali | জরুরি অবস্থায় সরাসরি ১১২ নম্বরে ফোন করবেন। |
| Hindi | आपात स्थिति में कृपया सीधे 112 पर कॉल करें। |
| English | In an emergency, please call 112 directly. |

## Sample phrases that trigger it

These are the kinds of thing a caller might say. The list is deliberately generous (a false alarm costs
one short notice and a person joining; a miss costs far more) and it does NOT understand negation, so
"no chest pain" also triggers it. **Please add, remove or correct**, especially spellings as a speech
recogniser actually writes them.

### English
- I have chest pain / there is pain in my chest
- he is not breathing / she can't breathe / difficulty breathing / short of breath
- he fainted / she is unconscious / he collapsed / not responding
- there is heavy bleeding / the bleeding won't stop
- he is having a seizure / a fit / convulsions
- my child swallowed poison / an overdose / swallowed pills
- I think it is a heart attack / a stroke / a snake bite
- there was a road accident / a serious accident / badly hurt
- I want to kill myself / I want to die
- send an ambulance / I need an ambulance
- this is an emergency / a medical emergency

### Bengali
- আমার বুকে ব্যথা করছে / বুকে চাপ লাগছে
- শ্বাসকষ্ট হচ্ছে / শ্বাস নিতে পারছি না / দম আটকে আসছে
- বাবা অজ্ঞান হয়ে গেছে / জ্ঞান নেই / সাড়া দিচ্ছে না
- প্রচুর রক্ত পড়ছে / রক্ত বন্ধ হচ্ছে না
- খিঁচুনি হচ্ছে
- বিষ খেয়েছে / ওভারডোজ
- হার্ট অ্যাটাক মনে হচ্ছে / স্ট্রোক / সাপে কামড়েছে
- দুর্ঘটনা হয়েছে / অ্যাক্সিডেন্ট
- আত্মহত্যা / মরে যেতে চাই
- অ্যাম্বুলেন্স পাঠান
- এটা ইমার্জেন্সি / জরুরি অবস্থা

### Hindi
- मेरे सीने में दर्द है / सीने में भारीपन
- सांस नहीं ले पा रहे / सांस लेने में तकलीफ़ / दम घुट रहा है
- वह बेहोश हो गया / होश नहीं है / जवाब नहीं दे रहे
- खून बहुत बह रहा है / खून नहीं रुक रहा
- दौरा पड़ा है / मिर्गी / झटके आ रहे हैं
- ज़हर खा लिया / ओवरडोज़
- दिल का दौरा पड़ा / हार्ट अटैक / स्ट्रोक / साँप ने काट लिया
- दुर्घटना हो गई / एक्सीडेंट
- आत्महत्या / मर जाना चाहता हूँ
- एम्बुलेंस भेजिए
- यह इमरजेंसी है / आपातकाल

## What a native speaker should check

- Would a caller in a real emergency actually say these words, in these forms, on a phone?
- Is anything here a phrase people say in ordinary calls that would set it off by mistake (for example a
  common word that also means "accident" or "emergency")?
- Does each notice sound calm, respectful and clear, with the respectful "you" (আপনি / आप)?

## Sign-off

| Language / role | Name | Decision (approved / changes needed) | Date |
|---|---|---|---|
| Clinical lead (situations, notice, number, hand-over) | | | |
| Bengali native reviewer (phrases and wording) | | | |
| Hindi native reviewer (phrases and wording) | | | |
| English reviewer (phrases and wording) | | | |

`agent/emergency.py`'s `REVIEW_STATUS` is now `approved_by_project_owner`. It changes again only if a clinician or reviewer asks for different wording.
