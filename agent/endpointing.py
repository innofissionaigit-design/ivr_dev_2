"""Turn-end decision logic, separated from the VAD model that feeds it.

agent/vad_stream.py used to hold both "where is the speech?" (Silero, needs
torch and a model) and "has the caller finished?" (arithmetic on the spans).
Only the second is a decision, and it is the part three stories are about, so
it lives here as pure functions that run anywhere and are tested on synthetic
audio:

  KCD-046  false-turn-end guards: a minimum speech duration rejects a cough or
           click, a tail guard stops decode lag being read as silence, and a
           maximum duration force-cuts a runaway utterance.
  KCD-048  semantic endpointing: the silence needed to end a turn depends on
           whether the partial transcript reads as finished. A completed
           question ends sooner; an unfinished one ("my number is nine
           eight...") is given longer than the baseline, never less.
  KCD-049  audio-driven scheduling: a decision also says WHEN the answer could
           change (commit_after_s), so the caller of poll() wakes exactly then
           instead of on a fixed timer whose average tax is half its period.

THRESHOLD STATUS: every number below is REASONED, not measured on real
telephone speech (there is none locally). KCD-047 exists to replace them with
measured ones; tools/calibrate_endpointing.py is the instrument, and
THRESHOLD_STATUS is flipped to "MEASURED" only by loading a calibration file it
wrote (load_calibrated_config). Until then any report must say REASONED.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Sequence

import numpy as np

THRESHOLD_STATUS = "REASONED"


@dataclasses.dataclass(frozen=True)
class EndpointConfig:
    # Baseline trailing quiet needed to commit. Raised from 0.8 by feel after
    # callers were cut off mid-sentence (REASONED).
    silence_confirm_s: float = 1.0
    # The freshest slice of the buffer is never trusted as confirmed silence:
    # decode lag or a late chunk would otherwise read as a pause (REASONED).
    tail_guard_s: float = 0.3
    # Shorter than this cannot be a finished utterance: a cough, click or pop.
    min_speech_s: float = 0.35
    # Runaway utterance: force a cut so one talker cannot stall the pipeline.
    max_utterance_s: float = 20.0
    # KCD-048. The transcript reads as finished -> less silence is needed; as
    # unfinished -> more. `complete` is floored well above ordinary
    # intra-sentence pauses; `incomplete` is never below the baseline.
    complete_confirm_s: float = 0.45
    incomplete_confirm_s: float = 1.5
    # Speculative transcription is attempted once the trailing quiet reaches
    # this, so the completeness verdict exists by the time `complete_confirm_s`
    # could be satisfied.
    speculate_after_s: float = 0.30

    def confirm_for(self, completeness: str | None) -> float:
        if completeness == "complete":
            return self.complete_confirm_s
        if completeness == "incomplete":
            return max(self.incomplete_confirm_s, self.silence_confirm_s)
        return self.silence_confirm_s


@dataclasses.dataclass
class TurnDecision:
    utterance_end_s: float | None  # relative to the slice; None => not yet
    had_any_speech: bool
    reason: str
    commit_after_s: float | None = None  # buffer seconds until this could flip, if silence continues
    speculate: bool = False  # ask for a partial transcript now (KCD-048)
    trailing_silence_s: float = 0.0
    confirm_s: float = 0.0
    speech_end_s: float | None = None  # where the last speech ended, slice-relative


def decide(
    spans: Sequence[dict],
    duration_s: float,
    cfg: EndpointConfig = EndpointConfig(),
    completeness: str | None = None,
    semantic: bool = True,
) -> TurnDecision:
    """spans: [{"start": s, "end": s}] in seconds from the start of the slice."""
    if duration_s < 0.2:
        return TurnDecision(None, False, "too_little_audio")
    if not spans:
        return TurnDecision(None, False, "no_speech")

    first_start = float(spans[0]["start"])
    last_end = float(spans[-1]["end"])

    # Guard 3: a runaway utterance is cut where the last speech ended.
    if duration_s >= cfg.max_utterance_s:
        return TurnDecision(last_end, True, "max_utterance", speech_end_s=last_end)

    # Guard 1: a transient is not a turn.
    if (last_end - first_start) < cfg.min_speech_s:
        return TurnDecision(None, True, "transient", speech_end_s=last_end)

    # Guard 2: audio inside the tail guard is still arriving, not yet silent.
    confirmed = max(0.0, duration_s - cfg.tail_guard_s)
    trailing = confirmed - last_end
    confirm = cfg.confirm_for(completeness if semantic else None)

    if trailing >= confirm:
        return TurnDecision(
            last_end, True, "silence_confirmed", trailing_silence_s=trailing, confirm_s=confirm, speech_end_s=last_end
        )

    # Not yet. Say when it could flip, and whether a transcript would help.
    wait = confirm - trailing
    speculate = semantic and completeness is None and trailing >= cfg.speculate_after_s
    # Before the speculation point the next interesting moment is reaching it.
    if semantic and completeness is None and trailing < cfg.speculate_after_s:
        wait = min(wait, cfg.speculate_after_s - trailing)
    return TurnDecision(
        None,
        True,
        "waiting_silence",
        commit_after_s=max(wait, 0.0),
        speculate=speculate,
        trailing_silence_s=trailing,
        confirm_s=confirm,
        speech_end_s=last_end,
    )


# --------------------------------------------------------------- completeness

_TERMINAL = re.compile(r"[।.?!॥]\s*$")
_DIGIT_WORDS = {
    "en": "zero one two three four five six seven eight nine oh".split(),
    "hi": "शून्य ज़ीरो एक दो तीन चार पाँच पांच छह छः सात आठ नौ".split(),
    "bn": "শূন্য এক দুই দুই তিন চার পাঁচ ছয় সাত আট নয়".split(),
}
_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_HINDI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Words a speaker does not stop on. Conservative on purpose: a wrongly
# "incomplete" verdict costs a little latency, a wrongly "complete" one cuts a
# caller off.
_INCOMPLETE_TAIL = {
    "en": {
        "and",
        "or",
        "but",
        "the",
        "a",
        "an",
        "to",
        "for",
        "with",
        "of",
        "in",
        "on",
        "at",
        "my",
        "is",
        "are",
        "was",
        "that",
        "because",
        "if",
        "so",
        "um",
        "uh",
        "er",
        "hmm",
        "like",
        "want",
        "need",
        "would",
        "can",
        "could",
        "i",
        "it",
        "this",
        "about",
        "from",
    },
    "hi": {
        "और",
        "या",
        "लेकिन",
        "कि",
        "अगर",
        "तो",
        "का",
        "की",
        "के",
        "में",
        "से",
        "को",
        "पर",
        "मेरा",
        "मेरी",
        "मुझे",
        "क्योंकि",
        "मतलब",
        "वो",
        "यह",
        "ये",
        "एक",
        "उस",
        "इस",
        "अं",
        "उम्म",
    },
    "bn": {
        "এবং",
        "আর",
        "কিন্তু",
        "বা",
        "অথবা",
        "যদি",
        "যে",
        "কারণ",
        "মানে",
        "আমার",
        "আমি",
        "একটা",
        "ওই",
        "এই",
        "সেই",
        "তো",
        "অ্যাঁ",
        "উম",
    },
}
# A finished utterance ends here. Verb-final Bengali/Hindi make this list the
# strongest positive signal available without a parser.
_COMPLETE_TAIL = {
    "en": {
        "please",
        "thanks",
        "thank",
        "you",
        "today",
        "tomorrow",
        "yes",
        "no",
        "okay",
        "ok",
        "done",
        "morning",
        "evening",
        "afternoon",
        "available",
        "there",
        "tests",
        "test",
        "doctor",
        "hello",
    },
    "hi": {
        "है",
        "हैं",
        "हूँ",
        "हूं",
        "था",
        "थी",
        "हो",
        "चाहिए",
        "चाहता",
        "चाहती",
        "करें",
        "कीजिए",
        "दीजिए",
        "बताइए",
        "बताएँ",
        "बताईए",
        "नहीं",
        "हाँ",
        "जी",
        "ठीक",
        "धन्यवाद",
        "शुक्रिया",
        "सकता",
        "सकती",
        "सकते",
        "है।",
        "कल",
        "आज",
        "परसों",
    },
    "bn": {
        "আছে",
        "আছেন",
        "নেই",
        "চাই",
        "করুন",
        "দিন",
        "বলুন",
        "হবে",
        "হয়",
        "পারি",
        "পারেন",
        "পারবেন",
        "যাবে",
        "হ্যাঁ",
        "না",
        "ঠিক",
        "ধন্যবাদ",
        "কি",
        "কী",
        "কখন",
        "কোথায়",
        "কত",
        "আজ",
        "কাল",
        "পরশু",
        "চাইছি",
        "করব",
        "করছি",
        "দেখাতে",
        "হলো",
        "হল",
    },
}
_YES_NO = {"yes", "no", "ok", "okay", "হ্যাঁ", "না", "ঠিক আছে", "हाँ", "नहीं", "जी", "ठीक है", "जी हाँ", "जी नहीं"}


def _norm_digits(text: str) -> str:
    return text.translate(_BENGALI_DIGITS).translate(_HINDI_DIGITS)


def _digit_count(tokens: Iterable[str], lang: str) -> tuple[int, bool]:
    """(digits spoken, whether the WHOLE utterance is digits/number words)."""
    words = set(_DIGIT_WORDS.get(lang, [])) | set(_DIGIT_WORDS["en"])
    count, only_digits = 0, True
    for t in tokens:
        t = _norm_digits(t.strip(",.-"))
        if t.isdigit():
            count += len(t)
        elif t.lower() in words:
            count += 1
        else:
            only_digits = False
    return count, only_digits


def classify_completeness(text: str, lang: str = "en") -> str:
    """'complete' | 'incomplete' | 'unknown' for a partial transcript.

    No parser, no model: three cheap, independently sensible signals.
      * a run that is only digits is finished at ten of them (a phone number)
        and unfinished before, because callers pause between digit groups --
        this is the commonest way a fixed silence timer cuts a caller off;
      * a trailing conjunction, preposition, determiner or filler means the
        thought has not landed;
      * terminal punctuation, a sentence-final verb/particle, or a bare
        yes/no means it has.
    Anything else is 'unknown', which keeps the baseline threshold."""
    raw = (text or "").strip()
    if not raw:
        return "unknown"
    tokens = raw.split()
    last = tokens[-1].strip(",;:").lower()
    last_bare = last.rstrip("।.?!॥")

    n_digits, only_digits = _digit_count(tokens, lang)
    if only_digits and n_digits:
        return "complete" if n_digits >= 10 else "incomplete"
    if raw.lower().rstrip("।.?!॥ ") in _YES_NO:
        return "complete"

    if last_bare in _INCOMPLETE_TAIL.get(lang, set()) | _INCOMPLETE_TAIL["en"] and not _TERMINAL.search(raw):
        return "incomplete"
    if raw.endswith(("...", "…", "-", ",")):
        return "incomplete"
    if _TERMINAL.search(raw):
        return "complete"
    if last_bare in _COMPLETE_TAIL.get(lang, set()):
        return "complete"
    return "unknown"


# ------------------------------------------------------------- energy spans


def energy_spans(
    samples: np.ndarray, sr: int = 16000, frame_s: float = 0.02, margin_db: float = 12.0, hang_s: float = 0.12
) -> list[dict]:
    """A dependency-free stand-in for Silero, for tests and for the calibration
    tool when torch is absent. Speech = frames within `margin_db` of the loud
    end, with a short hangover so a stop consonant is not a gap. It is NOT what
    runs on a call; it only has to be a consistent yardstick."""
    x = np.asarray(samples, dtype=np.float32)
    n = int(frame_s * sr)
    if x.size < n:
        return []
    count = x.size // n
    e = 10 * np.log10(np.maximum(np.mean(x[: count * n].reshape(count, n) ** 2, axis=1), 1e-12))
    thresh = max(float(np.percentile(e, 95)) - margin_db, -70.0)
    active = e >= thresh
    hang = int(hang_s / frame_s)
    spans, start, quiet = [], None, 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet > hang:
                spans.append({"start": start * frame_s, "end": (i - quiet + 1) * frame_s})
                start, quiet = None, 0
    if start is not None:
        spans.append({"start": start * frame_s, "end": (count - quiet) * frame_s})
    return spans


# ------------------------------------------------------------- calibration


def config_from_json(path: str, base: EndpointConfig = EndpointConfig()) -> EndpointConfig:
    """Load a calibration file written by tools/calibrate_endpointing.py. Only
    the fields it contains change; THRESHOLD_STATUS is reported by the caller
    from the file's own `status`, never assumed here."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    fields = {f.name for f in dataclasses.fields(EndpointConfig)}
    return dataclasses.replace(base, **{k: v for k, v in data.get("config", {}).items() if k in fields})
