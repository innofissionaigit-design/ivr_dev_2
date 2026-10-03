"""Score a recogniser's transcripts against human transcripts: WER, CER, and the entities that must not be wrong.

    python tools/asr_eval.py --manifest data/asr/manifest.jsonl --hyp out/hyp.jsonl
    python tools/asr_eval.py --manifest m.jsonl --hyp h.jsonl --lock data/asr/locked.sha256 --json

Inputs (JSON Lines):
  manifest   one row per utterance (agent/asr_manifest.py: id, speaker, split, source, language, verbatim, tags, locked ...)
  hyp        {"id": ..., "hyp": "the recogniser's transcript"} per utterance -- from any recogniser
The catalogue (doctors, tests and their aliases) defaults to the seeded clinic data; `--catalogue cat.json` uses a saved
/api/v1/catalogue answer, and the real clinic's catalogue should be used before go-live.

What it prints, for the test split unless `--split` says otherwise:
  * WER and CER with a 95% interval that resamples CALLERS, overall and by language and by difficulty tag;
  * WER on the normalized transcripts too (each doctor and test collapsed to one token), which forgives a spelling
    difference in a name but not a wrong name;
  * for doctors, tests, numbers and yes/no: said, correct, missed, WRONG (heard but not said), ambiguous.
Real and synthetic utterances are scored in separate groups. A synthetic group is a regression check, never a benchmark.

It refuses to run (exit 2) when a caller is in two splits, a hypothesis is missing, or a recorded lock digest no longer
matches the locked test rows. Nothing here needs a GPU or the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from agent.asr_manifest import (  # noqa: E402
    ManifestRow,
    locked_digest,
    locked_problems,
    parse_manifest,
    speaker_split_problems,
)
from agent.asr_metrics import (  # noqa: E402
    CorpusScore,
    Edits,
    EntityScore,
    bootstrap_interval,
    char_edits,
    score_entities,
    word_edits,
)
from agent.transcript_rules import KINDS, AliasTable, Entity, normalize_text  # noqa: E402


def _reference_entities(row: ManifestRow, table: AliasTable) -> list[Entity]:
    if row.entities is not None:
        return [Entity(e["kind"], e["value"], e["value"]) for e in row.entities]
    lang = "en" if row.language == "en" else "hi" if row.language == "hi" else "bn"
    return table.entities(row.verbatim, lang, reference=True)


def evaluate(rows: list[ManifestRow], hyps: dict[str, str], table: AliasTable) -> dict[str, Any]:
    """Score `rows` against `hyps`. Returns {group: {"overall": CorpusScore, "by_language": {...}, "by_tag": {...},
    "normalized": Edits}} where group is "real" or "synthetic"."""
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        hyp = hyps[row.id]
        g = groups.setdefault(
            row.source,
            {
                "overall": CorpusScore(),
                "by_language": {},
                "by_tag": {},
                "normalized": Edits(),
                "normalized_by_caller": {},
            },
        )
        lang = "en" if row.language == "en" else "hi" if row.language == "hi" else "bn"
        ref_text, hyp_text = normalize_text(row.verbatim), normalize_text(hyp)
        words, chars = word_edits(ref_text, hyp_text), char_edits(ref_text, hyp_text)
        entities = score_entities(_reference_entities(row, table), table.entities(hyp, lang))
        norm = word_edits(table.normalize(row.verbatim), table.normalize(hyp))
        g["normalized"] += norm
        g["normalized_by_caller"][row.speaker] = g["normalized_by_caller"].get(row.speaker, Edits()) + norm
        g["overall"].add(row.speaker, words, chars, entities)
        g["by_language"].setdefault(row.language, CorpusScore()).add(row.speaker, words, chars, entities)
        for tag in row.tags:
            g["by_tag"].setdefault(tag, CorpusScore()).add(row.speaker, words, chars, entities)
    return groups


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def _line(label: str, s: CorpusScore) -> str:
    lo, hi = bootstrap_interval(s.by_caller)
    callers = len(s.by_caller)
    return (
        f"  {label:<14} utts {s.utterances:>4}  callers {callers:>3}  WER {_pct(s.words.rate)} [{_pct(lo)} .. {_pct(hi)}]"
        f"  CER {_pct(s.chars.rate)}" + ("   (one caller: no interval)" if callers < 2 else "")
    )


def _entity_table(entities: dict[str, EntityScore]) -> list[str]:
    out = ["    entity      said  correct  missed  WRONG  ambiguous  accuracy"]
    for kind in KINDS:
        e = entities[kind]
        out.append(
            f"    {kind:<10} {e.said:>5} {e.correct:>8} {e.missed:>7} {e.wrong:>6} {e.ambiguous:>10}  {_pct(e.accuracy)}"
        )
    return out


def render(groups: dict[str, dict[str, Any]], split: str) -> str:
    out: list[str] = []
    for source in ("real", "synthetic"):
        g = groups.get(source)
        if g is None:
            continue
        head = f"{source.upper()} utterances, {split} split"
        out += ["", head + ("" if source == "real" else "  -- SYNTHETIC: a regression check, NOT a benchmark")]
        out.append(_line("overall", g["overall"]))
        lo, hi = bootstrap_interval(g["normalized_by_caller"])
        out.append(
            f"  {'normalized':<14} WER {_pct(g['normalized'].rate)} [{_pct(lo)} .. {_pct(hi)}]  (entities collapsed)"
        )
        out += _entity_table(g["overall"].entities)
        for label, key in (("language", "by_language"), ("difficulty tag", "by_tag")):
            if g[key]:
                out.append(f"  by {label}:")
                out += [_line(k, v) for k, v in sorted(g[key].items())]
    return "\n".join(out)


def _load_catalogue(path: str | None) -> dict[str, Any]:
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    sys.path.append(os.path.join(ROOT, "tests"))
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        return client.get("/api/v1/catalogue").json()


def _read_hyps(path: str) -> dict[str, str]:
    hyps: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                hyps[str(obj["id"])] = str(obj["hyp"])
    return hyps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--catalogue")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--lock", help="file holding the locked test set's digest; written on first use, checked after")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    with open(args.manifest, encoding="utf-8") as f:
        rows, problems = parse_manifest(f)
    problems += speaker_split_problems(rows)
    digest_seen = None
    if args.lock and os.path.exists(args.lock):
        with open(args.lock, encoding="utf-8") as f:
            digest_seen = f.read().strip() or None
    if args.split == "test":
        problems += locked_problems(rows, digest_seen)
    if problems:
        print("REFUSING TO SCORE:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    if args.lock and digest_seen is None:
        with open(args.lock, "w", encoding="utf-8", newline="\n") as f:
            f.write(locked_digest(rows) + "\n")
        print(f"recorded the locked test set digest in {args.lock}", file=sys.stderr)

    hyps = _read_hyps(args.hyp)
    # the test split is the locked set (real callers) plus any synthetic rows, which are scored apart from it
    chosen = [
        r for r in rows if r.split == args.split and (r.locked or args.split != "test" or r.source == "synthetic")
    ]
    missing = sorted(r.id for r in chosen if r.id not in hyps)
    if missing:
        print(f"REFUSING TO SCORE: no hypothesis for {len(missing)} utterance(s), e.g. {missing[:5]}", file=sys.stderr)
        return 2

    groups = evaluate(chosen, hyps, AliasTable(_load_catalogue(args.catalogue)))
    if args.json:
        summary: dict[str, Any] = {}
        for source, g in groups.items():
            s = g["overall"]
            summary[source] = {
                "utterances": s.utterances,
                "wer": s.words.rate,
                "cer": s.chars.rate,
                "normalized_wer": g["normalized"].rate,
                "entities": {k: vars(v) for k, v in s.entities.items()},
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render(groups, args.split))
    return 0


if __name__ == "__main__":
    sys.exit(main())
