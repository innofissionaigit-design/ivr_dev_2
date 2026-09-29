"""ADDED BY SOURAV -- "Caller goes silent" story (Epic: Conversation --
Difficult, Sensitive and Edge Cases).

story title: Caller goes silent
user story: As a caller who was distracted, I want a gentle prompt rather
    than a disconnection, so that I do not lose the call.
acceptance criteria: Two graduated prompts precede a graceful close, each
    different from the last. The close states what was and was not
    completed. Abandonment is logged with the turn index and the
    preceding prompt.

WHY THIS IS A SEPARATE, PURE MODULE, NOT WRITTEN INLINE IN
main.py/main_pcm.py's _turn_poll_loop a second time each: same drift
reason agent/correction_flow.py's own module docstring gives at length --
main.py and main_pcm.py each keep an independently-maintained copy of
_turn_poll_loop (a raw-PCM tail read vs a WAV re-decode is the only
INTENDED difference), so every decision that can be pure -- "what should
happen the next time the existing idle timeout trips" and "what can this
session honestly say about its own progress" -- is kept here once so it
cannot drift between the two copies.

ONE IMPORTANT DIFFERENCE FROM correction_flow.py's OWN CALL SITES, WORTH
FLAGGING FOR WHOEVER TOUCHES THIS NEXT: correction_flow's callers
(main.py's _continue_pending/_dispatch_turn_inner) sit inside
tools/make_pcm_variant.py's byte-identity-verified "reasoning half"
(_resolve_intent .. _resync_after_playback), so the generator itself
proves both transports stay in sync. _turn_poll_loop, where this module
is called from, sits OUTSIDE that verified span -- the generator copies
it through unchanged today only because none of its 11 fixed patch
targets happen to overlap the idle-timeout block. That currently holds
(confirmed by diffing both files' _turn_poll_loop bodies before this
story touched them), but it is not a mechanically-verified guarantee.
main_pcm.py must therefore still be regenerated with
tools/make_pcm_variant.py, never hand-edited, and the regenerated file
diffed against the previous version to confirm this module's call sites
landed identically in both -- see this story's own implementation report
for that diff.

THE STAGE MACHINE
------------------
Three per-call states, stored on CallSession (never globally -- a shared
or process-wide counter would leak one caller's silence into another
caller's call):

    0 (STAGE_NORMAL)       -> next silence action is Prompt 1
    1 (STAGE_PROMPT_1_SENT) -> next silence action is Prompt 2
    2 (STAGE_PROMPT_2_SENT) -> next silence action is the graceful close

_turn_poll_loop resets the stage back to 0 the moment the turn detector
(VAD) actually confirms an utterance end -- real caller speech, not
merely audio bytes arriving (CallSession.append() already bumps
last_activity on every raw chunk regardless of content, silence and
background noise included) and not the noise/silence the turn detector
already filters out before it ever reports an utterance end. That is the
one existing signal in this codebase that distinguishes "the caller
answered" from "audio is still arriving" without inventing a second VAD
pass this story was never asked to build.
"""
from __future__ import annotations

# Per-call silence-episode stage. Plain ints, matching every other bit of
# CallSession counter state (utt_seq, confirm_attempts) -- this codebase
# has no enum precedent for per-call counters, and CallSession is a plain
# container, not a state-machine object.
STAGE_NORMAL = 0
STAGE_PROMPT_1_SENT = 1
STAGE_PROMPT_2_SENT = 2

# next_silence_action()'s return values.
ACTION_PROMPT_1 = "prompt_1"
ACTION_PROMPT_2 = "prompt_2"
ACTION_ABANDON = "abandon"

# ADDED BY SOURAV -- bugfix for the "Caller goes silent" story's own close
# message (validation report Bug #2: a caller who never said a word was
# hearing "everything you asked has been taken care of", which is a false
# completion claim -- there was nothing to take care of). next_close_state()
# below is the function this bugfix adds; these are its three return
# values, same plain-string-constant style as ACTION_* above.
CLOSE_NO_ENGAGEMENT = "no_engagement"
CLOSE_COMPLETED = "completed"
CLOSE_UNFINISHED = "unfinished"


def next_silence_action(silence_stage: int) -> str:
    """Pure state transition: what _turn_poll_loop should do the next
    time the EXISTING IDLE_TIMEOUT_S condition trips (this story does not
    change that timeout's value or how it is measured -- only what
    happens when it fires), given how many graduated prompts this
    silence episode has already used.

        stage 0 (nothing sent yet)   -> ACTION_PROMPT_1
        stage 1 (prompt 1 sent)      -> ACTION_PROMPT_2
        stage 2 (prompt 2 sent) or
        anything else                -> ACTION_ABANDON

    Any value other than 0 or 1 abandons rather than raising: CallSession
    state should never reach anything else, and if it somehow did, ending
    the call is the safe direction for state this function does not
    recognise -- not looping a caller through prompts forever.
    """
    if silence_stage <= STAGE_NORMAL:
        return ACTION_PROMPT_1
    if silence_stage == STAGE_PROMPT_1_SENT:
        return ACTION_PROMPT_2
    return ACTION_ABANDON


def has_unfinished_business(pending: dict | None, deferred: dict | None) -> bool:
    """The one fact the completion-aware close (agent/reply_templates.py's
    silence_close_reply()) is allowed to state: whether this call has
    something genuinely left unfinished, straight from the two fields
    every flow in this codebase already uses for exactly that --
    session.pending ("a flow is mid-collection, waiting on one more
    field") and session.deferred ("part of an earlier multi-part question
    is still unanswered", see _run_parts()/_drain_deferred() in main.py).

    Deliberately does not try to name WHICH flow or WHICH question.
    session.pending's shape is not consistent across flows -- some
    "awaiting" states carry an "intent" key, report_status/report_send
    carry a "flow" key instead, and several (confirm_booking, date,
    doctor_choice, ...) carry neither (see agent/correction_flow.py's own
    module docstring for the same inconsistency noted from a different
    angle). Guessing a human-readable label from whichever key happens to
    be present would be exactly the invented completion tracking this
    story's own instructions rule out ("do not introduce fake completion
    tracking just to satisfy the wording"); a plain true/false is the one
    thing this state can say without ever being wrong.
    """
    return pending is not None or deferred is not None


# ADDED BY SOURAV -- bugfix for the "Caller goes silent" story's close
# message (validation report Bug #2). why this function exists: the close
# used to be picked from has_unfinished_business() alone (True/False), so
# a caller who never said a single word before going silent -- pending
# and deferred both None, exactly like a caller who DID talk and finished
# cleanly -- got the same "everything you asked has been taken care of"
# text as someone whose request was genuinely resolved. That is a false
# completion claim: nothing was ever asked, so nothing was "taken care
# of". has_unfinished_business() itself is left untouched (per the fix's
# own instructions) because it is still exactly the right answer to its
# own question ("is there a flow genuinely left mid-collection?") --
# this function only adds the ONE extra fact needed to tell "resolved"
# apart from "never happened": session.utt_seq, the same existing
# per-call counter _turn_poll_loop already increments on every VAD-
# confirmed utterance (see that function's own comment), never a new
# field or a new global.
#
# flow: has_unfinished_business() still wins outright when true (a
# caller who started something and then also went completely silent
# from turn zero cannot happen through this codebase's own state
# transitions -- pending/deferred are only ever set from inside a
# dispatched turn -- but even if it somehow did, "something is still
# mid-collection" is the more specific, more actionable fact and should
# not be shadowed by "no turns happened yet"). Otherwise: zero turns
# dispatched means the caller never engaged at all, which is a third,
# distinct thing from "engaged and finished cleanly".
def next_close_state(utt_seq: int, pending: dict | None, deferred: dict | None) -> str:
    """Which of the three silence-close texts (agent/reply_templates.py's
    silence_close_reply()) this abandonment should use:

        pending/deferred set          -> CLOSE_UNFINISHED
        utt_seq == 0, nothing pending -> CLOSE_NO_ENGAGEMENT (caller never
                                          engaged -- do not claim anything
                                          was completed or answered)
        utt_seq > 0, nothing pending  -> CLOSE_COMPLETED (caller engaged
                                          and nothing is left outstanding)

    `utt_seq` is session.utt_seq exactly as record_call_abandoned()'s own
    `turn_index` argument already uses it (see main.py's ACTION_ABANDON
    branch) -- the count of turns already dispatched, not a value invented
    for this fix.
    """
    if has_unfinished_business(pending, deferred):
        return CLOSE_UNFINISHED
    if utt_seq == 0:
        return CLOSE_NO_ENGAGEMENT
    return CLOSE_COMPLETED
