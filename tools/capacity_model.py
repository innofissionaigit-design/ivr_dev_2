"""GPU capacity sizing from MEASURED throughput (KCD-019).

The Blueprint's Appendix H gives indicative GPU counts per concurrency tier and says to
"size against your own bake-off throughput numbers". This is the model that does it: it turns

    recogniser real-time factor          (GPU-seconds per audio-second decoded)
    synthesis throughput                 (audio-seconds produced per GPU-second)
    language-model tokens per second     (aggregate, per device, at the tested concurrency)

into GPUs per data centre for a target concurrency, states the headroom that leaves, and states
what happens at capacity. The arithmetic is here; the NUMBERS are not. Nothing in this repository
has measured them on the target hardware, so:

  * every input is tagged with where it came from -- MEASURED (you ran it), STATED (given by the
    client/ADR), or REASONED (an assumption you can and should replace);
  * a MEASURED throughput left blank makes the corresponding GPU figure UNDETERMINED, and the
    report says so instead of substituting a plausible number. A sizing built on an unmeasured
    throughput is a guess with a decimal point;
  * Appendix H's row for the target tier is always printed beside the result as INDICATIVE, so
    the measured answer can be compared with the starting point it replaces.

What it does not model: network, SBC capacity, disk, or cross-stage contention on a shared GPU
(the per-stage demands are ADDED, which is conservative when stages do not peak together and
optimistic when they contend -- the load test, KCD-236, is what settles that). Erlang-B assumes
Poisson arrivals and blocked calls cleared, which is what admission control does (a refused call
goes to a person; it is not queued -- agent/admission.py).

    python tools/capacity_model.py                       # workload analysis + what is missing
    python tools/capacity_model.py --measured M.json     # full sizing; see docs/capacity-measurements.template.json
    python tools/capacity_model.py --measured M.json --json
"""

import argparse
import dataclasses
import json
import math
import sys

MEASURED, STATED, REASONED = "MEASURED", "STATED", "REASONED"

# Blueprint Appendix H, "indicative only".
APPENDIX_H = [
    (30, "1 x 7-8B LLM; ASR ~1-2 GB; TTS/embeddings shared", "1-2 x 24 GB"),
    (100, "2-3 x 7-8B or 1 x larger; 1 dedicated TTS", "3-4 x 24-48 GB"),
    (300, "model-server pool with autoscale; 2 dedicated TTS", "8-12 x 48 GB, 2 nodes"),
    (10**9, "sharded model-server cluster; pooled ASR/TTS", "capacity-plan from measured tokens/s"),
]


@dataclasses.dataclass
class Input:
    value: float | None
    source: str
    note: str = ""


def default_inputs() -> dict[str, Input]:
    return {
        # ---- workload (ADR 0001 section 5)
        "calls_per_day": Input(7000, STATED, "upper end of the stated 6,000-7,000/day"),
        "window_hours": Input(10, STATED, "operating window"),
        "peak_hour_factor": Input(
            1.5, REASONED, "busiest hour / average hour; replace with the client's hourly profile"
        ),
        "aht_min": Input(2.5, REASONED, "ADR 0001 section 5 range 2.0-3.0 min; average AI call duration"),
        "target_utilisation": Input(0.70, REASONED, "keep each GPU under this share of its measured throughput"),
        "configured_cap": Input(
            28, STATED, "ADMISSION_MAX_CALLS default in main.py / agent/admission.py (ADR 0001 section 5)"
        ),
        "max_blocking": Input(0.02, REASONED, "share of calls that may be refused to a person at peak"),
        # ---- call mix
        "caller_speech_fraction": Input(
            0.40, REASONED, "share of a call during which the caller speaks (ASR decoding)"
        ),
        "agent_speech_fraction": Input(0.45, REASONED, "share of a call during which the agent speaks (TTS producing)"),
        "turns_per_minute": Input(4.0, REASONED, "caller turns per minute"),
        "llm_share_of_turns": Input(0.30, REASONED, "turns that miss the fast path (ADR: 65-75% fast-path target)"),
        "llm_tokens_per_turn": Input(120, REASONED, "prompt + generated tokens billed to the LLM per LLM turn"),
        # ---- measured throughput (None = not measured: the GPU figure it feeds is UNDETERMINED)
        "asr_rtf": Input(None, MEASURED, "GPU-seconds per audio-second, at the tested concurrency"),
        "tts_x_realtime": Input(None, MEASURED, "audio-seconds synthesised per GPU-second"),
        "llm_tokens_per_s": Input(None, MEASURED, "aggregate tokens/s of ONE LLM device at the tested concurrency"),
        "llm_on_gpu": Input(0, STATED, "1 if the LLM shares the GPU pool (ADR 0001 default: CPU)"),
        "gpu_vram_gb": Input(24, STATED, "per-GPU memory of the target card"),
        # ---- topology
        "data_centres": Input(2, STATED, "Blueprint 3.5: two sites"),
        "active_active": Input(1, STATED, "1: each site carries half the peak; 0: standby site carries none"),
        "spare_gpus_per_dc": Input(1, REASONED, "N+1 within a site"),
    }


def apply_measured(inputs: dict[str, Input], measured: dict) -> dict[str, Input]:
    """Overlay a measurements file: {"asr_rtf": 0.08, ...}, optionally {"name": {"value":..,"source":..}}."""
    for k, v in measured.items():
        if k.startswith("_"):
            continue
        if k not in inputs:
            raise KeyError(f"unknown input {k!r}")
        if isinstance(v, dict):
            inputs[k] = Input(v.get("value"), v.get("source", MEASURED), v.get("note", ""))
        else:
            inputs[k] = Input(v, MEASURED if inputs[k].source == MEASURED else inputs[k].source, inputs[k].note)
    return inputs


# ------------------------------------------------------------------------------ queueing


def erlang_b(offered_erlangs: float, servers: int) -> float:
    """Blocking probability with `servers` lines and Poisson arrivals, blocked calls cleared."""
    if servers <= 0:
        return 1.0
    b = 1.0
    for n in range(1, servers + 1):
        b = offered_erlangs * b / (n + offered_erlangs * b)
    return b


def cap_for_blocking(offered_erlangs: float, max_blocking: float) -> int:
    n = 1
    while erlang_b(offered_erlangs, n) > max_blocking:
        n += 1
        if n > 100_000:
            raise ValueError("no cap reaches that blocking")
    return n


def appendix_h_row(concurrency: int) -> tuple[str, str]:
    for upper, stages, gpus in APPENDIX_H:
        if concurrency <= upper:
            return stages, gpus
    return APPENDIX_H[-1][1], APPENDIX_H[-1][2]


# ------------------------------------------------------------------------------ the model


def size(inp: dict[str, Input]) -> dict:
    v = {k: i.value for k, i in inp.items()}
    out: dict = {"inputs": {k: {"value": i.value, "source": i.source, "note": i.note} for k, i in inp.items()}}

    # ---- workload: how many calls are in progress at the busiest hour
    avg_calls_hr = v["calls_per_day"] / v["window_hours"]
    peak_calls_hr = avg_calls_hr * v["peak_hour_factor"]
    offered = peak_calls_hr * v["aht_min"] / 60.0  # Little's law: L = lambda * W
    cap = cap_for_blocking(offered, v["max_blocking"])
    blocked_peak = erlang_b(offered, cap)
    out["workload"] = {
        "average_calls_per_hour": round(avg_calls_hr, 1),
        "peak_calls_per_hour": round(peak_calls_hr, 1),
        "offered_load_erlangs_at_peak": round(offered, 1),
        "ai_cap_for_blocking_target": cap,
        "blocking_at_that_cap": round(blocked_peak, 4),
        "configured_cap": int(v["configured_cap"]),
        "blocking_at_configured_cap_peak": round(erlang_b(offered, int(v["configured_cap"])), 4),
        "blocking_at_configured_cap_average": round(
            erlang_b(avg_calls_hr * v["aht_min"] / 60.0, int(v["configured_cap"])), 4
        ),
        "calls_to_a_person_per_peak_hour": round(peak_calls_hr * blocked_peak, 1),
        "telephony_sessions_needed": math.ceil(offered * 1.5),
        "telephony_note": "AI lines plus overflow to people; 1.5x offered load is REASONED and is the SBC/trunk team's number to confirm",
    }
    target = cap
    out["target_concurrency"] = target

    # ---- per-call demand on each device class, per call-second
    fc, fa = v["caller_speech_fraction"], v["agent_speech_fraction"]
    llm_tok_s_per_call = (v["turns_per_minute"] / 60.0) * v["llm_share_of_turns"] * v["llm_tokens_per_turn"]
    demand: dict[str, float | None] = {
        "asr_gpu_s_per_call_s": None if v["asr_rtf"] is None else v["asr_rtf"] * fc,
        "tts_gpu_s_per_call_s": None if v["tts_x_realtime"] in (None, 0) else fa / v["tts_x_realtime"],
        "llm_tokens_per_s_per_call": llm_tok_s_per_call,
    }
    out["per_call_demand"] = {k: (None if x is None else round(x, 6)) for k, x in demand.items()}

    # ---- calls one GPU carries, ASR and TTS added (conservative), LLM added only if it shares the GPU
    missing = [n for n in ("asr_rtf", "tts_x_realtime") if v[n] in (None, 0)]
    llm_on_gpu = bool(v["llm_on_gpu"])
    if llm_on_gpu and v["llm_tokens_per_s"] in (None, 0):
        missing.append("llm_tokens_per_s")
    out["undetermined"] = missing

    calls_per_gpu = None
    if not missing:
        gpu_s = demand["asr_gpu_s_per_call_s"] + demand["tts_gpu_s_per_call_s"]
        if llm_on_gpu:
            gpu_s += llm_tok_s_per_call / v["llm_tokens_per_s"]
        calls_per_gpu = v["target_utilisation"] / gpu_s if gpu_s > 0 else None
    out["calls_per_gpu"] = None if calls_per_gpu is None else round(calls_per_gpu, 1)

    # ---- LLM devices when it is NOT on the GPU pool: the same arithmetic, its own pool
    llm_devices = None
    if v["llm_tokens_per_s"] not in (None, 0):
        llm_devices = math.ceil(target * llm_tok_s_per_call / (v["llm_tokens_per_s"] * v["target_utilisation"]))
    out["llm_devices_total"] = llm_devices
    if not llm_on_gpu and v["llm_tokens_per_s"] in (None, 0):
        out["undetermined"] = missing + ["llm_tokens_per_s (CPU/other device pool)"]

    # ---- GPUs per data centre
    share = 1.0 / v["data_centres"] if v["active_active"] else 1.0
    out["gpus_per_dc"] = None
    out["capacity_after_dc_loss_calls"] = None
    if calls_per_gpu:
        need = math.ceil(target * share / calls_per_gpu)
        per_dc = need + int(v["spare_gpus_per_dc"])
        out["gpus_per_dc"] = per_dc
        out["gpus_total"] = per_dc * int(v["data_centres"])
        carried = (per_dc - int(v["spare_gpus_per_dc"])) * calls_per_gpu
        # after losing one site, the survivors carry their own share PLUS what fits in their spare
        surviving = int(v["data_centres"]) - 1
        out["capacity_after_dc_loss_calls"] = (
            int(min(target, math.floor(per_dc * calls_per_gpu * surviving)))
            if v["active_active"]
            else int(min(target, math.floor(carried)))
        )
        out["headroom_calls"] = int(math.floor(out["gpus_total"] * calls_per_gpu)) - target
        out["headroom_pct"] = round(100.0 * out["headroom_calls"] / target, 1)
    else:
        out["headroom_calls"] = out["headroom_pct"] = None

    stages, gpus = appendix_h_row(target)
    out["appendix_h_indicative"] = {
        "tier_for_concurrency": target,
        "stages": stages,
        "gpus_per_dc": gpus,
        "status": "INDICATIVE, not measured; the measured result above replaces it",
    }
    out["at_capacity"] = at_capacity_statement(target, out)
    return out


def at_capacity_statement(target: int, out: dict) -> str:
    return (
        f"At {target} concurrent AI calls the admission controller (agent/admission.py) refuses the next call "
        "('capacity') and it goes to a person immediately -- it is never queued behind the GPU, because an unbounded "
        "queue turns overload into dead air for every caller. If recent p95 turn latency exceeds "
        "ADMISSION_SHED_P95_S, new calls are shed the same way BEFORE the cap is reached, and a failed model service "
        "closes the door entirely ('backend_down'). Calls already in progress continue. After the loss of one data "
        f"centre, capacity falls to {out['capacity_after_dc_loss_calls'] if out.get('capacity_after_dc_loss_calls') is not None else 'an UNDETERMINED number of'} calls "
        "and the cap must be lowered to match, or turn latency degrades for every caller."
    )


def report(out: dict) -> str:
    w = out["workload"]
    lines = ["# Capacity sizing (KCD-019)", "", "## Workload (arithmetic on the stated and reasoned inputs)", ""]
    lines += [
        f"- average {w['average_calls_per_hour']} calls/h; busiest hour {w['peak_calls_per_hour']} calls/h",
        f"- offered load at peak: {w['offered_load_erlangs_at_peak']} Erlangs (Little's law)",
        f"- AI cap that keeps refusals under target: {w['ai_cap_for_blocking_target']} "
        f"(blocking {w['blocking_at_that_cap']:.2%}) -> {w['calls_to_a_person_per_peak_hour']} calls/h go to a person at peak",
        f"- at the CONFIGURED cap of {w['configured_cap']}: {w['blocking_at_configured_cap_average']:.1%} of calls "
        f"go to a person in an average hour, {w['blocking_at_configured_cap_peak']:.1%} in the busiest",
        f"- telephony sessions: {w['telephony_sessions_needed']} ({w['telephony_note']})",
        "",
        "## GPUs",
        "",
    ]
    if out["undetermined"]:
        lines += ["**UNDETERMINED.** These measurements are missing, so no GPU count is stated:", ""]
        lines += [f"- `{m}`" for m in out["undetermined"]]
        lines += [
            "",
            "Run the bake-off (Appendix E) on the target GPU and fill "
            "`docs/capacity-measurements.template.json`; nothing else in this report needs to change.",
            "",
        ]
    else:
        lines += [
            f"- one GPU carries {out['calls_per_gpu']} calls at {out['inputs']['target_utilisation']['value']:.0%} utilisation",
            f"- **{out['gpus_per_dc']} GPU per data centre** ({out['gpus_total']} total), "
            f"headroom {out['headroom_calls']} calls ({out['headroom_pct']}%) over the {out['target_concurrency']}-call target",
            f"- capacity after losing one site: {out['capacity_after_dc_loss_calls']} calls",
        ]
        if out["llm_devices_total"] is not None:
            lines.append(f"- LLM devices needed in total: {out['llm_devices_total']}")
        lines.append("")
    ah = out["appendix_h_indicative"]
    lines += [
        f"Appendix H (INDICATIVE) for this tier: {ah['gpus_per_dc']} per DC -- {ah['stages']}.",
        "",
        "## At capacity",
        "",
        out["at_capacity"],
        "",
        "## Inputs",
        "",
        "| input | value | source |",
        "|---|---:|---|",
    ]
    lines += [f"| {k} | {i['value']} | {i['source']} |" for k, i in out["inputs"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", help="JSON of measured inputs (docs/capacity-measurements.template.json)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    inp = default_inputs()
    if args.measured:
        with open(args.measured, encoding="utf-8") as f:
            apply_measured(inp, json.load(f))
    out = size(inp)
    print(json.dumps(out, indent=2) if args.json else report(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
