"""KCD-050: the half-duplex gate, as a small testable object.

The agent must never hear itself. Two ends close the gate:

  * the CLIENT mutes capture while it plays audio (static/*.html), and
  * the SERVER refuses to run turn detection while `speaking`.

Neither end alone is enough: a muted client cannot tell the server when
playback finished (so the server needs the client's `playback_done`), and a
server-only gate cannot stop the caller's speaker leaking into the microphone.

The server half used to live as three loose attributes on CallSession in
main.py, where nothing about it could be tested off-pod. It is here now, with
the clock injected. main.py's CallSession delegates to one of these and keeps
its old attribute names as properties, so no call site changed.

Three behaviours the story names, each with a test:

  * hold() EXTENDS the deadline cumulatively -- clips queue on the client, so a
    second clip starts only after the first ends.
  * release() sets `resync_pending`, and the caller moves its processed marker
    past the muted region on the next tick (resync_target).
  * a BACKSTOP lifts the gate if the client never reports completion
    (expired()), so a lost `playback_done` cannot silence the agent for good.

PLAYBACK_GUARD_S is REASONED: enough for a network hop plus the client's own
output latency; it is the price of the backstop, paid only when
`playback_done` goes missing.
"""

from __future__ import annotations

import time
from collections.abc import Callable

PLAYBACK_GUARD_S = 3.0
RESYNC_REWIND_S = 0.25


class PlaybackGate:
    def __init__(
        self, guard_s: float = PLAYBACK_GUARD_S, clock: Callable[[], float] = time.time, start_closed: bool = True
    ):
        self.guard_s = guard_s
        self._clock = clock
        # Starts closed: the greeting goes out before the caller has said
        # anything, so the gate must already be shut when the first poll runs.
        self.speaking = start_closed
        self.deadline = clock() + guard_s if start_closed else 0.0
        self.resync_pending = False
        self.backstop_releases = 0

    def hold(self, audio_duration_s: float) -> None:
        """Called BEFORE each clip's bytes leave the server."""
        now = self._clock()
        base = max(self.deadline, now) if self.speaking else now
        self.speaking = True
        self.deadline = base + audio_duration_s + self.guard_s

    def release(self) -> None:
        """Playback is over (client said so, or the caller interrupted)."""
        self.speaking = False
        self.resync_pending = True

    def expired(self) -> bool:
        return self.speaking and self._clock() >= self.deadline

    def blocked(self) -> bool:
        """True while turn detection must not run. Applies the backstop as a
        side effect: a gate past its deadline is released, and counted."""
        if not self.speaking:
            return False
        if self.expired():
            self.backstop_releases += 1
            self.release()
            return False
        return True

    def resync_target(self, buffer_end_s: float, processed_until_s: float) -> float:
        """Where the processed marker moves to after playback: the end of the
        buffer, less a small rewind so the first syllable of a caller who
        started speaking as the gate lifted is not lost. Never moves backward."""
        self.resync_pending = False
        return max(processed_until_s, buffer_end_s - RESYNC_REWIND_S)
