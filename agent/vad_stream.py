"""Turn-taking VAD for a *live* WebSocket call.

voice-to-rx-repo/voicerx/vad.py already wraps Silero VAD, but it is built
for an offline consultation recording: load the whole file once, return
every segment. That is the wrong shape here. A live caller needs the
opposite question answered continuously: "has the caller stopped talking
*yet*?" -- asked every ~500ms against a buffer that is still growing.

Rather than switch to Silero's separate low-level streaming API (new
integration surface, new failure modes, unproven on this stack), this
reuses the exact call this codebase already trusts --
`get_speech_timestamps()`, the same function voicerx/vad.py calls -- but
runs it repeatedly against only the *unprocessed tail* of the buffer, the
same fix server.py's `_process_slice()` already had to make:

    "The first version re-ran VAD over the whole growing file each chunk
    ... Silero's boundaries MOVE as more audio arrives, so a segment that
    straddled processed_until_s was excluded on every subsequent pass and
    never processed at all. A 3-minute live recording produced 2 segments."

The caller (main.py) is responsible for passing only the unprocessed tail
slice into `poll()` -- this module treats index 0 of whatever tensor it's
given as "now", and never looks at audio before that.
"""

from __future__ import annotations

import os

import torch

from agent.endpointing import EndpointConfig, TurnDecision, decide

_LOCAL_SILERO_REPO = os.environ.get("SILERO_VAD_REPO", "/workspace/silero-vad")

# Back-compat alias: main.py and older tests read `.utterance_end_s` and
# `.had_any_speech`, both still present on TurnDecision.
TurnResult = TurnDecision


class TurnDetector:
    """One instance per process, shared across calls (holds no per-call
    state itself -- main.py's CallSession tracks processed_until_s).

    This class only answers "where is the speech?" (Silero). Whether that
    means the caller has FINISHED is agent/endpointing.decide -- the guards
    (KCD-046), the semantic threshold (KCD-048) and the wake-up schedule
    (KCD-049) all live there, documented and tested without torch."""

    def __init__(
        self,
        silence_confirm_s: float = 1.0,
        tail_guard_s: float = 0.3,
        min_speech_s: float = 0.35,
        max_utterance_s: float = 20.0,
        config: EndpointConfig | None = None,
    ):
        # silence_confirm_s was raised from an initial 0.8s after real
        # testing showed that was cutting callers off mid-sentence. The
        # remaining values and their reasons are documented on EndpointConfig.
        self.config = config or EndpointConfig(
            silence_confirm_s=silence_confirm_s,
            tail_guard_s=tail_guard_s,
            min_speech_s=min_speech_s,
            max_utterance_s=max_utterance_s,
        )

        if os.path.isdir(_LOCAL_SILERO_REPO):
            self.model, utils = torch.hub.load(
                repo_or_dir=_LOCAL_SILERO_REPO,
                source="local",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
        else:
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
        self._get_speech_timestamps = utils[0]

    def spans(self, wav_tensor: torch.Tensor, sr: int) -> tuple[list[dict], float]:
        """Speech spans (seconds, relative to the tensor) and its duration."""
        if sr != 16000:
            wav_tensor = torch.nn.functional.interpolate(
                wav_tensor.view(1, 1, -1),
                scale_factor=16000 / sr,
                mode="linear",
                align_corners=False,
            ).view(-1)
            sr = 16000
        duration_s = wav_tensor.shape[-1] / sr
        if duration_s < 0.2:
            return [], duration_s
        return self._get_speech_timestamps(wav_tensor, self.model, sampling_rate=sr, return_seconds=True), duration_s

    def poll(
        self, wav_tensor: torch.Tensor, sr: int, completeness: str | None = None, semantic: bool = False
    ) -> TurnResult:
        """wav_tensor: the UNPROCESSED TAIL of the call's buffer only --
        i.e. audio already consumed by a prior completed turn must not be
        included. Index 0 is treated as "now".

        `completeness` is the verdict on a partial transcript of that tail
        (agent/endpointing.classify_completeness), or None if there is none;
        `semantic` turns the shortened/lengthened threshold on."""
        spans, duration_s = self.spans(wav_tensor, sr)
        return decide(spans, duration_s, self.config, completeness=completeness, semantic=semantic)
