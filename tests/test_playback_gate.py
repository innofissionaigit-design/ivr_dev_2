"""KCD-050: agent/playback_gate.py -- the server half of the half-duplex gate.

The clock is injected, so every rule the story names is tested without waiting:
cumulative hold, resynchronisation past the muted region, and the backstop that
lifts the gate when the client never reports completion.

    python -m pytest tests/test_playback_gate.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.playback_gate import PLAYBACK_GUARD_S, RESYNC_REWIND_S, PlaybackGate


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_the_gate_starts_closed_so_the_greeting_cannot_be_heard_by_the_agent():
    g = PlaybackGate(clock=Clock())
    assert g.speaking and g.blocked()


def test_a_client_that_reports_completion_opens_the_gate_and_asks_for_a_resync():
    g = PlaybackGate(clock=Clock())
    g.release()
    assert not g.blocked() and g.resync_pending


def test_clips_queued_on_the_client_extend_the_deadline_cumulatively():
    c = Clock()
    g = PlaybackGate(clock=c)
    g.release()
    g.hold(2.0)
    first_deadline = g.deadline
    g.hold(3.0)  # queued behind the first: plays after it ends
    # extended from the FIRST deadline, not replaced by "now + 3": the second clip
    # cannot start until the first has finished. (The backstop guard is added per
    # clip -- it only ever lengthens the safety net, never the gate's real closing.)
    assert g.deadline == pytest.approx(first_deadline + 3.0 + PLAYBACK_GUARD_S)
    assert g.deadline > c.t + 2.0 + 3.0 + PLAYBACK_GUARD_S


def test_a_hold_after_the_gate_opened_starts_from_now_not_from_a_stale_deadline():
    c = Clock()
    g = PlaybackGate(clock=c)
    g.hold(1.0)
    g.release()
    c.t += 500.0
    g.hold(1.0)
    assert g.deadline == pytest.approx(c.t + 1.0 + PLAYBACK_GUARD_S)


def test_the_backstop_lifts_the_gate_if_playback_done_never_arrives():
    c = Clock()
    g = PlaybackGate(clock=c)
    g.release()
    g.hold(1.0)
    c.t += 1.0 + PLAYBACK_GUARD_S - 0.01
    assert g.blocked()
    c.t += 0.02
    assert not g.blocked()  # deadline passed: released
    assert g.backstop_releases == 1 and g.resync_pending


def test_the_backstop_is_counted_only_when_it_actually_fired():
    c = Clock()
    g = PlaybackGate(clock=c)
    g.release()
    g.hold(0.5)
    g.release()  # the client did report
    c.t += 100.0
    assert not g.blocked() and g.backstop_releases == 0


def test_resync_moves_the_marker_past_the_muted_region_with_a_small_rewind():
    g = PlaybackGate(clock=Clock())
    g.release()
    assert g.resync_target(buffer_end_s=12.0, processed_until_s=4.0) == pytest.approx(12.0 - RESYNC_REWIND_S)
    assert not g.resync_pending


def test_resync_never_moves_the_marker_backwards():
    g = PlaybackGate(clock=Clock())
    assert g.resync_target(buffer_end_s=5.0, processed_until_s=9.0) == 9.0
