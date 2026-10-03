# Reminders

Nothing here is in the code. This file is only a reminder so the ideas are not forgotten.

- Items 1 and 2: build ONLY IF THE CLIENT ASKS.
- Item 3: the owner wants this built, and built well (rural callers). Start from the design and the fallback algorithm below.

---

## 1. Prescription OCR over WhatsApp

**The idea.** A caller sends a photo of a prescription on WhatsApp. The bot reads it (OCR), and answers correctly from what it
says: for example which tests were prescribed, so it can quote prices, preparation and offer to book them.

**To build if asked:**
- A WhatsApp inbound channel that receives the image (and links it to the call or the caller's phone number).
- An OCR step for handwritten and printed prescriptions, in English and Bengali or Hindi where the doctor wrote them.
- Match what was read to the clinic's own test list. A name that does not match confidently is asked back to the caller,
  never guessed (the same rule as "Doctor Nobody").

**Rules that must hold (from CLAUDE.md):**
- OCR output is a *reading of an image*, not a fact. The bot may say what it read ("I can see CBC and Lipid Profile on the
  prescription, is that right?") and must get a yes before acting on it. It never interprets medicines, doses or diagnoses.
- Prices, preparation and timings still come from the clinic API, never from the model.
- A blurred or unreadable photo gets a polite "please send a clearer photo" or a hand-off to staff.

**Also decide first:** consent and retention (how long the image is kept, who can see it; India's DPDP Act), where it is stored,
and whether a staff member reviews unclear ones.

---

## 2. Payment link on WhatsApp, and checking that the booking is paid

**The idea.** After a booking, the bot sends a payment link on WhatsApp. Later it can tell whether the booking has been paid.

**To build if asked:**
- A payment provider (for example UPI or a card gateway) that creates a payment link for a booking's amount.
- Send the link on WhatsApp to the number on the booking.
- The paid / not paid status comes from the provider's confirmation (a webhook), **not from the caller saying "I paid"**.
  The bot may only say a booking is paid when the provider has said so.
- A payment status on the booking record, and a reply when the caller asks "did my payment go through?".

**Things to plan for:**
- Retries and duplicates: use the idempotent-write support already in the clinic API (`Idempotency-Key`) so a repeated
  request never creates a second payment link or a second booking.
- What "unpaid" means for the booking: hold the slot for how long, when it is released, and how that fits with the existing
  cancellation charge rules.
- Refunds and failed payments, and what the bot says in each case.
- No card, UPI PIN or bank details are ever spoken or typed into the call. The link takes the caller to the provider's page.


---

## 3. Rural callers say everyday words, not test names ("পেটের ছবি", "মাথার ছবি", "রক্ত পরীক্ষা")

**The problem.** Rural callers rarely say "abdominal ultrasound" or "brain MRI". They say what they see or feel: *peter chhobi*
(belly picture), *mathar chhobi* (head picture), *rokto porikhha* (blood test), and in their own accent. The bot must understand
these and answer with the EXACT test names on the clinic's list, by asking back, never by guessing.

**What the bot should do (examples, wording to be confirmed by a doctor and a native speaker):**

| Caller says | Bot asks back (exact names from the list) |
|---|---|
| *peter chhobi* (পেটের ছবি) | "পেটের আলট্রাসাউন্ড (USG Whole Abdomen) নাকি পেটের এক্স-রে?" |
| *mathar chhobi* (মাথার ছবি) | "মাথার সিটি স্ক্যান নাকি এমআরআই?" |
| *rokto porikhha* (রক্ত পরীক্ষা) | "কোন রক্ত পরীক্ষা? যেমন সিবিসি, সুগার, লিপিড..." (a short list of the commonest, then ask) |
| *buker chhobi* (বুকের ছবি) | "বুকের এক্স-রে (Chest X-Ray)?" |

Today the test list has USG Whole Abdomen and Chest X-Ray, and **no** brain CT, brain MRI or abdominal X-ray. Before this is built, the clinic
must say which of these it really offers; for one it does not list, the bot says so and offers a staff callback.

**Design (deterministic first, the model last, in the same spirit as the current gazetteer and fast path):**
1. **Lay-term table (a "hot-word gazetteer").** Data, not code: lay phrase -> the list of exact tests it can mean, per language and per spelling
   (Bengali script, Latin spellings such as *chhobi / chobi / cobi*, Hindi/Devanagari), with a body-part word + a word for "picture / test / check" as
   the pattern (*peter, pete, pet* + *chhobi, chobi, photo*). Owned by the clinic; edited without a deploy, like the other messages.
2. **Match with the existing machinery:** the phonetic fold and typo-tolerant matching already used for test and doctor names
   (`agent/gazetteer.py`, `agent/form_index.py`), so accent and spelling variants land on the same entry. A suggestion is never a resolution.
3. **One phrase, several tests = ask, always.** The reply names the candidates, in the caller's script, and takes a yes / the name / "the first one".
   One phrase that maps to exactly one test is still read back once ("আপনি কি ... বলছেন?") before any price or booking.
4. **Nothing else changes:** prices, preparation and slots still come from the clinic API; the lay table only decides WHICH test to ask about.
5. **Measure it:** the share of lay phrases resolved in one turn, in two turns, or handed to staff; the phrases nobody matched (logged as counts of
   the words, reviewed weekly and added to the table).

**If it still does not understand the diction or accent after tuning (the fallback algorithm):**
1. **Use the n-best, not the top guess.** Keep the recogniser's top 3-5 transcripts per turn (and across the Bengali / Hindi / English recognisers) and
   run each through the lay table and the gazetteer; take the entry that several hypotheses agree on.
2. **Bias the recogniser toward the clinic's vocabulary** (contextual biasing / hot-word boosting: give it the list of test names and lay phrases at
   decode time). Research shows this improves rare and domain words without retraining. Needs a recogniser that supports it.
3. **Score confidence in three levels.** High -> read back once. Medium (two candidates close) -> "A or B?". Low / nothing -> do not guess.
4. **Rephrase by category, not by word.** If nothing matches: ask a broad, easy question first ("রক্ত, প্রস্রাব, নাকি ছবি (স্ক্যান/এক্স-রে)?"), then narrow
   ("কোন অংশের ছবি? পেট, বুক, মাথা?"). Callers answer a category question much more reliably than they repeat a hard word.
5. **Offer a second channel.** Ask them to say the name of the doctor's advice, or to send a photo of the prescription on WhatsApp (see item 1), or
   offer a staff callback. A person is always one step away.
6. **Learn from real calls.** Keep (with consent) the unmatched utterances, have a person label them, add the lay phrases to the table, and only then
   consider fine-tuning the recogniser on that labelled rural speech. Order matters: the table and the questions fix most of it cheaply; retraining is last.

**How others solve this (from a short search, not a survey):**
- **Lay-language vocabularies.** Health-literacy work shows patients understand everyday terms far better than jargon, and voice assistants recognise
  simplified terms much better ([npj Digital Medicine](https://www.nature.com/articles/s41746-019-0133-x)); mapping lay to clinical terms is a known
  resource problem ([JMIR/PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11522659/)).
- **Contextual biasing.** Recognisers are steered to domain terms at decode time, without fine-tuning ([Whisper contextual biasing](https://arxiv.org/html/2410.18363v1),
  [Soniox](https://soniox.com/wiki/context-biasing), [WCTC keyword spotting](https://arxiv.org/pdf/2506.01263)).
- **Rural, accent-diverse speech.** Indian projects collect dialect speech and note that rural dialects such as Bhojpuri lose accuracy
  ([VAANI dataset](https://arxiv.org/html/2603.28714v3), [Bhojpuri women ASR](https://arxiv.org/pdf/2506.09653), [Gram Vaani / Mobile Vaani](https://arxiv.org/pdf/2104.07901),
  [voice Q&A for low-income users in India](https://dl.acm.org/doi/fullHtml/10.1145/3460112.3471946)). Their common lesson: keep the interface simple, use people in the loop, and collect real speech.
- **Clarification dialogue.** Standard practice for an ambiguous or misheard entity is to detect ambiguity from the n-best list and ask "did you mean A or B?"
  ([entity disambiguation](https://ceur-ws.org/Vol-1556/paper5.pdf), [ambiguity detection in an assistant](https://aclanthology.org/2024.emnlp-industry.28.pdf)).

**Also decide first:** who owns the lay-term table (a doctor to approve which lay phrase can mean which test), and which languages and dialect areas to start with.
