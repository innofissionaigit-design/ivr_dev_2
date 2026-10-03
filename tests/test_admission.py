"""AdmissionController + HealthMonitor: pure logic, no GPU, no network."""

import asyncio

import pytest

from agent.admission import AdmissionController, HealthMonitor


def ctl(**kw):
    kw.setdefault("bypass_file", None)  # never let a stray file on the dev box close the door
    return AdmissionController(**kw)


def test_admits_up_to_cap_then_refuses_with_capacity():
    c = ctl(max_calls=2)
    a, b = c.try_admit(), c.try_admit()
    assert a.admitted and b.admitted
    third = c.try_admit()
    assert not third.admitted and third.reason == "capacity"
    assert c.active_calls == 2


def test_release_frees_a_slot_and_is_idempotent():
    c = ctl(max_calls=1)
    a = c.try_admit()
    c.release(a)
    c.release(a)  # second teardown of the same call must not free someone else's slot
    b = c.try_admit()
    assert b.admitted
    assert not c.try_admit().admitted
    c.release(a)  # stale ticket again
    assert c.active_calls == 1


def test_rejected_call_never_holds_a_slot():
    c = ctl(max_calls=0)
    r = c.try_admit()
    assert not r.admitted and r.ticket is None
    c.release(r)  # releasing a rejection is a no-op
    assert c.active_calls == 0


def test_bypass_flag_closes_the_door_and_clears():
    c = ctl(max_calls=5)
    c.set_bypass(True, actor="ops@test")
    r = c.try_admit()
    assert not r.admitted and r.reason == "bypass"
    assert c.snapshot()["bypass_by"] == "ops@test"
    c.set_bypass(False, actor="ops@test")
    assert c.try_admit().admitted


def test_bypass_file_works_without_any_api_call(tmp_path):
    flag = tmp_path / "ai_bypass"
    c = AdmissionController(max_calls=5, bypass_file=str(flag))
    assert c.try_admit().admitted
    flag.write_text("")
    assert c.try_admit().reason == "bypass"
    flag.unlink()
    assert c.try_admit().admitted


def test_bypass_does_not_drop_calls_already_in_progress():
    c = ctl(max_calls=5)
    a = c.try_admit()
    c.set_bypass(True)
    assert c.active_calls == 1  # existing call untouched; only NEW ones are refused
    c.release(a)


def test_backend_down_closes_the_door_and_names_the_backend():
    c = ctl(max_calls=5)
    c.set_backend_health("tts", False, "connection refused")
    r = c.try_admit()
    assert not r.admitted and r.reason == "backend_down:tts"
    c.set_backend_health("tts", True)
    assert c.try_admit().admitted


def test_latency_shed_needs_enough_samples_and_recovers():
    c = ctl(max_calls=10, latency_shed_p95_s=2.0, min_latency_samples=5, latency_window=10)
    for _ in range(4):
        c.record_turn_latency(9.0)
    assert c.try_admit().admitted, "4 samples is noise, not a trend"
    c.record_turn_latency(9.0)
    r = c.try_admit()
    assert not r.admitted and r.reason == "latency_shed"
    for _ in range(10):  # window fully replaced by good turns
        c.record_turn_latency(0.8)
    assert c.try_admit().admitted


def test_latency_shed_disabled_by_default():
    c = ctl(max_calls=10)
    for _ in range(50):
        c.record_turn_latency(30.0)
    assert c.try_admit().admitted


def test_snapshot_counts_rejections_by_reason():
    c = ctl(max_calls=1)
    c.try_admit()
    c.try_admit()
    c.set_bypass(True)
    c.try_admit()
    snap = c.snapshot()
    assert snap["counters"]["admitted"] == 1
    assert snap["counters"]["rejected:capacity"] == 1
    assert snap["counters"]["rejected:bypass"] == 1
    assert snap["peak_active"] == 1


def test_health_monitor_needs_consecutive_failures_and_recovers_on_first_success():
    c = ctl(max_calls=5)
    state = {"ok": True}

    async def probe():
        return state["ok"]

    m = HealthMonitor(c, {"tts": probe}, fail_threshold=3)

    async def run():
        state["ok"] = False
        await m.check_once()
        await m.check_once()
        assert c.try_admit().admitted, "two failures is below the threshold"
        await m.check_once()
        assert c.try_admit().reason == "backend_down:tts"
        state["ok"] = True
        await m.check_once()
        assert c.try_admit().admitted

    asyncio.run(run())


def test_health_monitor_treats_a_crashing_probe_as_a_failed_probe():
    c = ctl(max_calls=5)

    async def boom():
        raise RuntimeError("socket closed")

    m = HealthMonitor(c, {"ollama": boom}, fail_threshold=1)
    asyncio.run(m.check_once())
    assert c.snapshot()["backends_down"]["ollama"].startswith("RuntimeError")


def test_negative_cap_rejected():
    with pytest.raises(ValueError):
        AdmissionController(max_calls=-1)
