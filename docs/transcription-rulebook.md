# Transcription rulebook and evaluation kit (before any fine-tuning)

Everything here works **without a single real recording**, and is ready for the day the first consented call is saved.
The code: `agent/transcript_rules.py`, `agent/asr_metrics.py`, `agent/asr_manifest.py`, `agent/catalogue_forms.py`,
`tools/asr_eval.py`. Tests: `tests/test_asr_evaluation.py`.

## 1. What a transcriber writes: the VERBATIM transcript

| Rule | Example |
|---|---|
| Write **what was said**, not what was meant. No correcting, no completing, no fixing grammar. | "ami aaj ashbo na" stays as said |
| Write each word **in the script it was said in**: Bengali speech in Bengali script, Hindi in Devanagari, an English word in Latin. **Never romanise** Bengali or Hindi. | `আমার CBC টেস্ট করাতে হবে` |
| A test said letter by letter is written **as heard**; it is not turned into the code by hand. | `সি বি সি`, `সিবিসি` and `CBC` are all valid verbatim |
| Numbers: write them the way they were said (digits or words); the tool reads both. | `পঁচিশ` or `২৫` or `twenty five` |
| Hesitations, repeats and false starts stay in. | `আমি আমি ডাক্তার সেন` |
| Words you cannot make out: `<unk>`. Two people at once: `<overlap>`. Non-speech: `<noise>`. Hold music or silence: no line at all. | |
| No punctuation is required; case is ignored. | |
| One utterance = one speaker's turn. Agent speech is **not** transcribed here (it is known text). | |

## 2. What the tool derives: the NORMALIZED transcript (never typed by hand)

Applied identically to the human transcript and to the recogniser's output, so a difference in spelling that means the
same thing is not counted as an error, and a difference that changes the meaning is.

1. Unicode NFC; the Bengali and Hindi flap letters folded; the Devanagari nukta dropped; lower case; punctuation removed
   (the same comparison form the live gazetteer uses: `agent/gazetteer.py`).
2. Bengali and Devanagari digits become ASCII digits.
3. Every doctor or test the caller named becomes **one token** naming the catalogue entry: `TEST:CBC`,
   `TEST:Blood_Sugar_PP`, `DOCTOR:Dr._A._Sen`. Longest form wins (so the `TSH` inside "T3 T4 TSH" is not a second test).
   The matching is **exact on the written form**: it never guesses from sound.
4. A name that belongs to **more than one entry** is left as said and reported as *ambiguous*. The Bengali "রায়" is both
   Dr. N. Roy and Dr. P. Ray; the tool does not choose, exactly as the live agent does not.

Codes: a test's code is the one in its brackets ("Complete Blood Count (CBC)" → `CBC`), or its first word if that is a
code ("HIV Test (ELISA)" → `HIV`); if two tests would share a code, both use their whole name.

## 3. Standard spellings

| Kind | Canonical form | Where it comes from |
|---|---|---|
| Test | its name in the clinic catalogue; code as above | `/api/v1/catalogue` (`tests`) |
| Doctor | the catalogue's short name, e.g. `Dr. A. Sen` | `/api/v1/catalogue` (`doctors`) |
| CBC, CRP, HbA1c, LFT, KFT, TSH, ECG, ESR, HBsAg, HCV | as written here (upper case, `HbA1c`, `HBsAg` mixed) | the catalogue names |

The rulebook has **no list of its own**: it reads the catalogue, so it cannot drift from what the agent can book. Every
alias a caller might say (Bengali, Hindi, the whole doctor name) is a form of the same entry. When the real clinic's
catalogue replaces the seeded one, run `tools/asr_eval.py --catalogue <saved /api/v1/catalogue>` and
`tools/confusable_pairs.py` again.

## 4. What is measured

| Figure | Meaning |
|---|---|
| **WER**, **CER** | total edits ÷ total reference words (or characters, spaces ignored) over the whole set, not a mean of per-utterance rates. |
| **95% interval** | resamples **callers**, not utterances (one caller's utterances are not independent). One caller gives no interval, and the tool says so. |
| **Normalized WER** | the same on the normalized transcripts: forgives how a name is spelled, not which name it is. |
| **Entities** (doctor, test, number, yes/no) | *said* · *correct* · *missed* (said, not heard) · **WRONG** (heard, not said: the wrong test or doctor) · *ambiguous* (heard as a name that belongs to several entries: the agent must ask). A yes/no is scored only for a reference of ≤ 4 words. |
| **By difficulty tag** | `rural_accent`, `strong_dialect`, `noise`, `speakerphone`, `low_volume`, `overlap`, `clipped`, `code_switch`, `telephone_8k`, `elderly`, `child`, so a rural failure is never averaged into an urban figure. |

**Not yet measured:** dates and times (only the numbers in them count); they need their own constrained grammar.

## 5. The manifest (one JSON object per utterance) and its three checks

Fields: `id`, `audio` (path of the **original** telephone audio, not a cleaned copy), `speaker` (an opaque id, never a
name or number), `split` (`train`/`dev`/`test`), `source` (`real`/`synthetic`), `language` (`bn`/`hi`/`en`/`mixed`),
`verbatim`, `tags`, `locked`, `consent`, optional `original_sample_rate`, `duration_s`, `entities`.

1. **Split by speaker.** A caller in two splits is refused.
2. **Locked test set.** `locked` rows are real, in the test split, and fixed by a digest (`--lock file`). If the set
   changes the tool refuses to score, because a before/after comparison on a different set is not a comparison.
3. **Real and synthetic never mix.** A real row needs consent on record; a synthetic row cannot be locked and is
   reported in its own group, labelled *not a benchmark*.

## 6. How to run it

```bash
python tools/asr_eval.py --manifest data/asr/manifest.jsonl --hyp out/hyp.jsonl --lock data/asr/locked.sha256
python tools/confusable_pairs.py --markdown docs/confusable-pairs-report.md
```

`hyp.jsonl` is one `{"id": ..., "hyp": "..."}` per utterance from any recogniser. Producing it from the model (a loop over
`TurnASR.transcribe_utterance_sync`) needs the pod and is not written yet.

## 7. Honest limits

* **No real data yet.** The tests use invented sentences. Nothing here says how accurate the recogniser is.
* **Test size.** The 5-10 hour locked set that is often quoted is more than a first pilot will have. A locked one or two
  hours from real callers is a start; with few callers the interval will be wide, and the tool shows it.
* **Consent.** Saving a patient's call needs the caller to be told and to agree. The manifest refuses a real row without
  it; what the disclosure says is a decision for the clinic and is not changed here.
