"""Compare ways of asking the intent model, on the pod, before any of them is made the default.

    python tools/intent_ab.py                                   # classic vs fast, 3 repeats each
    python tools/intent_ab.py --variants classic,fast --num-ctx 0,4096 --repeats 5

For every variant (and context size) it runs a fixed set of caller sentences through the REAL model (agent/llm.py over
Ollama) and prints, side by side:

  * accuracy: the intent, and the slots that matter, against the expected answer (a slot must be non-empty and contain the
    expected text). A variant that is faster but less accurate is not an improvement;
  * time: median and 95th percentile per call, and where it went (reading the prompt vs writing the answer), from the
    timings Ollama reports.

The sentences below are a SMALL hand-written set covering every intent in the three languages -- a smoke test for a change
of prompt, not the golden set the backlog calls for (Epic E24). Add real transcripts to it as they are collected.
Needs the model running on this machine; nothing here starts it.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import llm  # noqa: E402

# (language, sentence, expected intent, {slot: text the slot must contain})
CASES: list[tuple[str, str, str, dict[str, str]]] = [
    ("bn", "সিবিসি টেস্টের দাম কত", "test_rate", {"test_name": "সিবিসি"}),
    ("bn", "ইউরিক অ্যাসিড টেস্টের রেট কত", "test_rate", {"test_name": "ইউরিক"}),
    ("bn", "ডাক্তার সেন কবে বসবেন", "doctor_availability", {"doctor_name": "সেন"}),
    ("bn", "ডাক্তার মুখার্জী কাল বসবেন", "doctor_availability", {"doctor_name": "মুখার্জী"}),
    ("bn", "সুগার টেস্টের আগে খালি পেটে থাকতে হবে", "test_prep", {}),
    ("bn", "ক্লিনিক কখন খোলে", "clinic_faq", {"faq_topic": "hours"}),
    ("bn", "আপনাদের পার্কিং আছে", "clinic_faq", {"faq_topic": "parking"}),
    ("bn", "আমি ডাক্তার সেনের অ্যাপয়েন্টমেন্ট বুক করতে চাই", "book_appointment", {"doctor_name": "সেন"}),
    ("bn", "আমার অ্যাপয়েন্টমেন্ট বাতিল করতে চাই", "cancel_appointment", {}),
    ("bn", "আমার বুকিং কোথায়", "lookup_booking", {}),
    ("bn", "বুকের ব্যথা হচ্ছে কোন ডাক্তার দেখাব", "department_query", {"symptom_description": "ব্যথা"}),
    ("bn", "নমস্কার", "smalltalk", {}),
    ("hi", "सीबीसी टेस्ट का रेट कितना है", "test_rate", {"test_name": "सीबीसी"}),
    ("hi", "डॉक्टर सेन कब बैठते हैं", "doctor_availability", {"doctor_name": "सेन"}),
    ("hi", "क्लिनिक कब खुलता है", "clinic_faq", {"faq_topic": "hours"}),
    ("hi", "मुझे डॉक्टर घोष से अपॉइंटमेंट बुक करना है", "book_appointment", {"doctor_name": "घोष"}),
    ("hi", "मेरी अपॉइंटमेंट कैंसल करनी है", "cancel_appointment", {}),
    ("hi", "धन्यवाद", "smalltalk", {}),
    ("en", "how much is the CBC test", "test_rate", {"test_name": "CBC"}),
    ("en", "when does Dr Sen sit", "doctor_availability", {"doctor_name": "Sen"}),
    ("en", "do I need to fast for the lipid profile", "test_prep", {"test_name": "lipid"}),
    ("en", "what are your opening hours", "clinic_faq", {"faq_topic": "hours"}),
    ("en", "I want to book an appointment with Dr Ghosh", "book_appointment", {"doctor_name": "Ghosh"}),
    ("en", "I want to book a CBC test", "book_test", {}),
    ("en", "please resend my confirmation message", "resend_confirmation", {}),
    ("en", "I have a skin rash which department should I go to", "department_query", {"symptom_description": "rash"}),
    ("en", "thank you", "smalltalk", {}),
    ("en", "asdf qwer", "unclear", {}),
]


def score_case(expected_intent: str, expected_slots: dict[str, str], data: dict) -> tuple[bool, bool]:
    """(intent right, slots right). A slot is right when the model filled it and it contains the expected text (case-blind);
    an exact-key slot (faq_topic) is compared whole."""
    intent_ok = data.get("intent") == expected_intent
    slots = data.get("slots") or {}
    slots_ok = all(
        isinstance(slots.get(k), str) and want.lower() in slots[k].lower() if k != "faq_topic" else slots.get(k) == want
        for k, want in expected_slots.items()
    )
    return intent_ok, slots_ok


def run_variant(variant: str, num_ctx: int, repeats: int, cases=CASES) -> dict:
    llm.INTENT_PROMPT_VARIANT, llm.OLLAMA_NUM_CTX = variant, num_ctx
    intent_ok = slots_ok = total = failures = 0
    latencies: list[float] = []
    prefill: list[int] = []
    decode: list[int] = []
    out_tokens: list[int] = []
    for _ in range(repeats):
        for lang, text, want_intent, want_slots in cases:
            t0 = time.perf_counter()
            try:
                data, diag = llm.extract_intent(text, lang=lang)
            except llm.ExtractionError:
                failures += 1
                total += 1
                continue
            latencies.append((time.perf_counter() - t0) * 1000)
            timing = diag.get("model_timing") or {}
            prefill.append(timing.get("prefill_ms", 0))
            decode.append(timing.get("decode_ms", 0))
            out_tokens.append(timing.get("output_tokens", 0))
            i_ok, s_ok = score_case(want_intent, want_slots, data)
            intent_ok += i_ok
            slots_ok += s_ok
            total += 1

    def pct(values: list[float], q: float) -> float:
        return sorted(values)[min(len(values) - 1, int(q * len(values)))] if values else 0.0

    med = lambda v: statistics.median(v) if v else 0  # noqa: E731
    return {
        "variant": variant,
        "num_ctx": num_ctx or "default",
        "calls": total,
        "failed": failures,
        "intent_acc": round(intent_ok / total, 3) if total else 0.0,
        "slot_acc": round(slots_ok / total, 3) if total else 0.0,
        "p50_ms": round(pct(latencies, 0.5)),
        "p95_ms": round(pct(latencies, 0.95)),
        "prefill_ms": med(prefill),
        "decode_ms": med(decode),
        "out_tokens": med(out_tokens),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", default="classic,fast")
    ap.add_argument("--num-ctx", default="0", help="comma-separated context sizes; 0 = the model's default")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    rows = [run_variant(v, int(c), args.repeats) for v in args.variants.split(",") for c in args.num_ctx.split(",")]
    cols = list(rows[0])
    print("  ".join(f"{c:>10}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]!s:>10}" for c in cols))
    base = rows[0]
    for r in rows[1:]:
        verdict = (
            "no less accurate"
            if r["intent_acc"] >= base["intent_acc"] and r["slot_acc"] >= base["slot_acc"]
            else "LESS ACCURATE"
        )
        print(
            f"{r['variant']}/ctx {r['num_ctx']} vs {base['variant']}/ctx {base['num_ctx']}: p50 {r['p50_ms'] - base['p50_ms']:+d} ms, {verdict}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
