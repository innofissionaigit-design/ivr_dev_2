"""KCD-052 (agent/barge_in.py, agent/full_duplex.py): the caller talks over the
agent and is heard within 300 ms; the agent never triggers on its own audio;
false positives on noise are measured and bounded.

SYNTHETIC scenes only (tests/_synth_echo.py, tests/_synth_noise.py). The story
asks for 200 calls; tools/echo_eval.py runs that many on demand, and on real
recordings on a pod. In the suite a smaller set keeps the run short.

SAME-VOICE scenes are the deployed condition (the agent speaks in one TTS voice
per language); the scenes where the voice changes every few seconds are the worst
case and are tested separately (the hold-off after a voice change).

Measured on same-voice scenes: callers detected within budget; no false events on
echo alone or on street, clinic and fan noise.

    python -m pytest tests/test_barge_in.py -v
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth_echo import SR, caller, echo_scene, voice_segments
from _synth_noise import noise_profile

from agent.barge_in import BargeInConfig, BargeInDetector
from agent.full_duplex import FullDuplexProcessor

WARM_S = 6.0  # the canceller needs a few seconds of agent speech before events are trusted


def _run(mic, far, chunk=1600, dur_each=None, **kw):
    """Feed a scene through the full-duplex processor the way main.py will:
    the agent's speech is registered up front at t=0, one clip per voice, the
    microphone streams in."""
    fd = FullDuplexProcessor(**kw)
    if dur_each:
        for i, seg in enumerate(voice_segments(far, dur_each)):
            fd.place_reference(seg, voice=f"voice{i}")
    else:
        fd.place_reference(far)
    events, suppressed = [], 0
    for i in range(0, mic.size, chunk):
        r = fd.process(mic[i : i + chunk])
        if r.barge_in is not None:
            events.append((fd._mic_pos, r.barge_in))
        suppressed += int(r.suppressed_barge_in)
    return fd, events, suppressed


@pytest.fixture(scope="module")
def caller_runs():
    runs = []
    for s in range(8):
        far, mic, _ = echo_scene(s, dur_each=5.0, same_voice=True)
        c = caller(s, dur=1.5, f0=105 + 14 * s)
        on = int((8.0 + 0.31 * s) * SR)
        m = mic.copy()
        m[on : on + c.size] += c
        _, events, _ = _run(m, far)
        runs.append((on, c.size, events))
    return runs


def test_a_caller_talking_over_the_agent_is_detected_every_time(caller_runs):
    for on, n, events in caller_runs:
        hits = [ev for _, ev in events if on <= ev.detected_at_sample < on + n]
        assert hits, "caller not detected"


def test_detection_is_inside_300_ms_at_the_95th_percentile(caller_runs):
    lat = []
    for on, n, events in caller_runs:
        first = min(ev.detected_at_sample for _, ev in events if ev.detected_at_sample >= on)
        lat.append((first - on) / SR * 1000.0)
    assert np.percentile(lat, 95) <= 300.0, lat
    assert np.median(lat) <= 250.0, lat


def test_the_agent_never_triggers_on_its_own_audio():
    events = 0
    for s in range(12):
        far, mic, _ = echo_scene(s, dur_each=4.0, same_voice=True)
        _, ev, _ = _run(mic, far)
        events += len(ev)
    assert events == 0


@pytest.mark.parametrize("profile", ["clinic", "fan"])
def test_steady_noise_does_not_trigger(profile):
    events = 0
    for s in range(3):
        far, mic, _ = echo_scene(s, dur_each=5.0, same_voice=True)
        _, ev, _ = _run(mic + noise_profile(profile, mic.size, -42.0, seed=s), far)
        events += len(ev)
    assert events == 0, f"{profile}: {events} false events"


def test_loud_horn_bursts_are_the_known_limit_of_street_noise_and_are_bounded():
    # A horn is periodic and, at -36 dBFS over a -46 dBFS floor, looks like a voice to any
    # acoustic detector: periodicity cannot tell them apart (a pure single tone IS rejected,
    # a horn over traffic rumble is not -- the tonality of voices and of horn bursts overlaps).
    # MEASURED: 4.8 false events per minute of agent speech on the synthetic street, whose
    # horn bursts come every ~3 s -- far more often than a real street. The bound is stated,
    # not hidden; the manual interrupt and the two-second evidence rule are what remain.
    events, seconds = 0, 0.0
    for s in range(3):
        far, mic, _ = echo_scene(s, dur_each=5.0, same_voice=True)
        _, ev, _ = _run(mic + noise_profile("street", mic.size, -42.0, seed=s), far)
        events += len(ev)
        seconds += mic.size / SR
    assert events / seconds * 60.0 <= 9.0, events


def test_background_voices_are_the_known_limit_and_are_bounded():
    # A market's murmur IS speech: acoustics alone cannot tell a distant talker
    # from the caller. What is bounded is how often it fires; the level test
    # (relative to a tracked floor) keeps a steady murmur from doing it.
    per_call = []
    for s in range(3):
        far, mic, _ = echo_scene(s, dur_each=5.0, same_voice=True)
        _, ev, _ = _run(mic + noise_profile("market", mic.size, -42.0, seed=s), far)
        per_call.append(len(ev))
    assert max(per_call) <= 2, per_call


def test_events_before_the_canceller_has_converged_are_ignored_not_acted_on():
    far, mic, _ = echo_scene(0, dur_each=5.0, same_voice=True)
    c = caller(0, dur=1.0)
    m = mic.copy()
    m[int(0.5 * SR) : int(0.5 * SR) + c.size] += c  # a caller in the first second, before any convergence
    _, events, _ = _run(m, far)
    assert not any(ev.detected_at_sample < WARM_S * SR for _, ev in events)


def test_nothing_is_acted_on_when_the_agent_is_silent():
    # No reference at all: the detector may fire on a caller, but the processor
    # does not report a BARGE-IN -- there was nothing to barge in on.
    c = caller(0, dur=1.5)
    mic = np.zeros(6 * SR, np.float32)
    mic[2 * SR : 2 * SR + c.size] += c
    fd = FullDuplexProcessor()
    acted = 0
    for i in range(0, mic.size, 1600):
        acted += int(fd.process(mic[i : i + 1600]).barge_in is not None)
    assert acted == 0
    assert fd.suppressed >= 1  # it fired, and was recorded as suppressed


def test_an_isolated_click_or_pop_does_not_accumulate_into_a_trigger():
    d = BargeInDetector()
    x = np.zeros(SR, np.float32)
    for k in (2000, 6000, 10000, 14000):
        x[k : k + 40] = 0.6
    assert d.process(x, np.zeros_like(x)) is None


def test_reset_discards_evidence_from_the_previous_utterance():
    d = BargeInDetector(config=BargeInConfig(min_speech_s=5.0))
    c = caller(0, dur=0.3)
    d.process(c, np.zeros_like(c))
    assert d._score > 0
    d.reset()
    assert d._score == 0.0


def test_placing_reference_follows_the_clients_back_to_back_playback():
    fd = FullDuplexProcessor()
    a = fd.place_reference(np.zeros(SR, np.float32))
    b = fd.place_reference(np.zeros(SR // 2, np.float32))
    assert a == 0 and b == SR  # the second clip starts when the first ends
    assert fd.agent_speaking()


def test_cancel_pending_stops_expecting_audio_the_client_will_not_play():
    fd = FullDuplexProcessor()
    fd.place_reference(np.zeros(5 * SR, np.float32))
    fd.process(np.zeros(SR, np.float32))
    fd.cancel_pending()
    assert fd._ref_end == fd._mic_pos


def test_barge_in_is_held_off_for_two_seconds_after_the_agents_voice_changes():
    """A different TTS voice (the caller switched language) leaves the filter
    poorly adapted for a couple of seconds, and its residual looks like a caller.
    The server knows the voice changed, so it does not act on barge-in until the
    filter has re-learned; the caller keeps the manual interrupt."""
    from agent.full_duplex import VOICE_CHANGE_INHIBIT_S

    far, mic, _ = echo_scene(3, dur_each=6.0, voices=3)
    clips = voice_segments(far, 6.0)
    fd = FullDuplexProcessor()
    starts = [fd.place_reference(c, voice=f"v{i}") for i, c in enumerate(clips)]
    assert starts[1] == 6 * SR
    assert fd._inhibit_until == starts[2] + int(VOICE_CHANGE_INHIBIT_S * SR)  # set by the LAST change so far
    # the same voice again does not extend the hold
    before = fd._inhibit_until
    fd.place_reference(clips[0][:SR], voice="v2")
    assert fd._inhibit_until == before


def test_a_caller_is_still_heard_once_the_hold_has_passed():
    far, mic, _ = echo_scene(2, dur_each=5.0)  # the agent's voice changes at 5 s and 10 s
    c = caller(2, dur=1.5, f0=118)
    hold_ends = int((10.0 + 2.0) * SR)
    on = int(10.3 * SR)  # talking right after the change
    m1 = mic.copy()
    m1[on : on + c.size] += c
    _, held, _ = _run(m1, far, dur_each=5.0)
    assert not any(10 * SR <= ev.detected_at_sample < hold_ends for _, ev in held)  # nothing acted on inside the hold
    on2 = int(13.0 * SR)  # 3 s after the change: heard
    m2 = mic.copy()
    m2[on2 : on2 + c.size] += c
    _, heard, _ = _run(m2, far, dur_each=5.0)
    assert any(on2 <= ev.detected_at_sample < on2 + c.size for _, ev in heard)
