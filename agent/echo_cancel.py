"""KCD-051: acoustic echo cancellation with the synthesised audio as the
far-end reference.

WHY THIS EXISTS. The half-duplex gate (agent/playback_gate.py) works by
deafness: the agent cannot hear anything while it speaks, so it cannot be
interrupted, and it costs a fixed guard after every reply. Browser echo
cancellation cannot help on this path -- the audio is synthesised here and
played through Web Audio, which the browser's canceller never sees as a
far-end signal. But THIS side knows exactly what it sent, so it can be the
canceller: subtract a filtered copy of the reference from the microphone and
what remains is the caller (plus noise and whatever the filter could not
model).

WHAT IS HERE (numpy only, no vendor SDK):
  * MDFCanceller -- a partitioned-block frequency-domain adaptive filter
    (overlap-save, NLMS-normalised per bin). Block 256 samples (16 ms at
    16 kHz), 8 partitions => 128 ms of echo path after alignment. About 3.5% of
    one core per call in numpy.
  * estimate_delay_long -- GCC-PHAT bulk delay between the reference as PLACED
    on the call timeline and the microphone, taking the EARLIEST strong peak
    (the direct path) rather than the tallest. The client's playback starts an
    unknown time after the server sent the clip (network + output buffer), so
    the filter is aligned by measurement, not assumed; a change of delay must
    be seen twice before it is adopted.
  * Double-talk handling. The filter must not learn from a caller talking over
    the agent. Two detectors -- a loosened Geigel test and a short-term ERLE
    COLLAPSE test against the best recent cancellation -- freeze adaptation
    (step 2% of normal). Both were needed: a first version compared against the
    running ERLE, which the caller itself dragged down, so it stopped noticing
    the caller, and the filter was wrecked (ERLE 26 dB -> 4 dB). Freezing the
    running figures while a caller talks fixed that.
  * Residual echo suppression -- attenuates only the spectral bins where the
    error is no larger than the residual the achieved ERLE predicts, so a
    caller (whose bins sit far above it) is not touched.
  * EchoPipeline -- the streaming wrapper the server uses: feed_reference()
    places far-end audio on the microphone timeline, process_ex() returns the
    cleaned microphone signal and the echo estimate, stats() reports ERLE and
    the delay.

MEASURED (tests/test_echo_cancel.py, synthetic scenes; see that file for the
scenes): steady-state ERLE 21-27 dB after about ten seconds of far-end speech;
17-24 dB with a distorting loudspeaker; delay tracked to within 25 ms for
120-400 ms; a caller talking over a converged filter comes through at 0.999
correlation and the filter is not damaged. KNOWN WEAK SPOT: when the far-end
VOICE changes the filter has never seen the newly excited frequencies and
cancellation dips to 0-7 dB for two to three seconds.

WHAT THIS DOES NOT CLAIM. Every number above is on SYNTHETIC echo paths. The
story asks for ERLE on REAL handsets and "never triggers a turn on its own
output over two hundred calls"; tools/echo_eval.py is the instrument for both
on a pod with recorded handset audio. A handset with strong speaker
non-linearity will cancel less than the linear model predicts; agent/barge_in.py
and agent/full_duplex.py are built not to trust the canceller before it has
measurably converged.
"""

from __future__ import annotations

import dataclasses

import numpy as np

BLOCK = 256


def erle_db(mic: np.ndarray, out: np.ndarray, skip: int = 0) -> float:
    """Echo return loss enhancement over echo-only audio: how much quieter
    the output is than the microphone, in dB. Call it on a window where only
    echo was present (no caller), after the filter has had time to converge."""
    m, o = np.asarray(mic[skip:], np.float64), np.asarray(out[skip:], np.float64)
    n = min(m.size, o.size)
    return float(10 * np.log10((np.sum(m[:n] ** 2) + 1e-12) / (np.sum(o[:n] ** 2) + 1e-12)))


# ------------------------------------------------------------------- delay


def estimate_delay(mic: np.ndarray, ref: np.ndarray, sr: int = 16000, max_delay_s: float = 0.6) -> tuple[int, float]:
    """(delay in samples, confidence). GCC-PHAT: the whitened cross-spectrum
    makes the correlation peak sharp on speech, whose energy is concentrated
    in a few harmonics. Confidence is peak height over the mean of the search
    window; ~1 means no echo of this reference was found."""
    n = min(mic.size, ref.size)
    if n < int(0.2 * sr):
        return 0, 0.0
    m, r = np.asarray(mic[-n:], np.float64), np.asarray(ref[-n:], np.float64)
    if np.sum(r**2) < 1e-8 or np.sum(m**2) < 1e-10:
        return 0, 0.0
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    cps = np.fft.rfft(m, nfft) * np.conj(np.fft.rfft(r, nfft))
    cps /= np.abs(cps) + 1e-9
    cc = np.abs(np.fft.irfft(cps, nfft))
    window = cc[: int(max_delay_s * sr) + 1]
    k = int(np.argmax(window))
    return k, float(window[k] / (np.mean(window) + 1e-12))


# ------------------------------------------------------------------ filter


class MDFCanceller:
    """One channel, fixed block size, fed aligned (mic block, ref block)."""

    def __init__(
        self,
        block: int = BLOCK,
        partitions: int = 8,
        mu: float = 0.5,
        geigel: float = 1.0,
        dt_hold_blocks: int = 8,
        dt_mu_fraction: float = 0.02,
    ):
        self.B, self.P, self.mu = block, partitions, mu
        self.geigel, self.dt_hold = geigel, dt_hold_blocks
        self.dt_mu_fraction = dt_mu_fraction
        n = block + 1
        self.W = np.zeros((partitions, n), np.complex128)
        self.X = np.zeros((partitions, n), np.complex128)
        self._xprev = np.zeros(block)
        self._power = np.full(n, 1e-6)
        self._err_power = np.full(n, 1e-6)
        self._eprev = np.zeros(block)
        self.step_decay = 3.0
        self.suppress = True
        # REASONED and swept on synthetic scenes (1e-3 .. 0.5): a larger floor on the
        # per-bin power keeps poorly excited bins from taking large, noisy steps,
        # which is what made the filter overshoot when the far-end voice changed.
        self.reg = 0.3
        self.suppress_beta = 1.5
        self.suppress_floor = 0.15
        self._dt_left = 0
        self._erle_best = 0.0
        self._pm_s = self._pe_s = 1e-12
        self._peaks: list[float] = []
        self.double_talk = False
        # running levels for ERLE, over single-talk blocks only
        self._pm = self._pe = 1e-12
        self.echo_estimate = np.zeros(block)

    def reset(self) -> None:
        self.W[:] = 0
        self.X[:] = 0
        self._power[:] = 1e-6

    @property
    def erle_db(self) -> float:
        return float(10 * np.log10(self._pm / self._pe))

    def process_block(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        B = self.B
        self._peaks = (self._peaks + [float(np.max(np.abs(ref)))])[-self.P :]
        ref_peak = max(self._peaks)
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = np.fft.rfft(np.concatenate([self._xprev, ref]))
        self._xprev = ref.copy()

        Yf = (self.W * self.X).sum(axis=0)
        y = np.fft.irfft(Yf)[B:]
        self.echo_estimate = y
        e = mic - y
        e_lin = e

        pm, pe = float(np.mean(mic**2)) + 1e-12, float(np.mean(e**2)) + 1e-12
        ref_active = ref_peak > 1e-4 and float(np.mean(ref**2)) > 1e-8

        # Double-talk detection, two independent tests.
        #  1. Geigel, loosened: the microphone is louder than the WHOLE
        #     reference window could produce. Loose because an echo path's
        #     summed reflections can exceed the reference peak, and a tight
        #     threshold reads ordinary echo as a caller.
        #  2. ERLE collapse: once the filter has converged, a block whose
        #     cancellation falls far below the long-run figure has something
        #     in it the filter never saw -- the caller. This is the test that
        #     works when the caller is quieter than the echo.
        geigel_hit = float(np.max(np.abs(mic))) > self.geigel * ref_peak and ref_peak > 1e-4
        # Short-term ERLE over ~5 blocks: a single 16 ms block is too noisy (a
        # pause in the far end makes echo and error both tiny and their ratio
        # meaningless), and would flag double-talk at random.
        self._pm_s = 0.75 * self._pm_s + 0.25 * pm
        self._pe_s = 0.75 * self._pe_s + 0.25 * pe
        block_erle = 10 * np.log10(self._pm_s / self._pe_s)
        # Compared against the BEST cancellation seen recently (decaying), not
        # the running figure: the running figure is dragged down by the caller
        # it is meant to detect, after which "collapse" could never fire again.
        margin = min(10.0, 0.7 * self._erle_best)  # a filter at 8 dB can only collapse so far
        collapse = self._erle_best > 3.0 and block_erle < self._erle_best - margin
        if ref_active and (geigel_hit or collapse):
            self._dt_left = self.dt_hold
        self.double_talk = self._dt_left > 0
        self._dt_left = max(0, self._dt_left - 1)

        if ref_active:
            # A single block in which cancellation collapses is the caller starting to
            # talk, and detection (smoothed, above) is a block or two behind it. Fed
            # into the running figures it would drag the reported ERLE from 20 dB to ~1
            # in ONE block (one caller-sized error against a tiny running residual),
            # after which nothing downstream would trust the canceller. So a block that
            # collapses against the best recent cancellation is not counted at all.
            instant_collapse = self._erle_best > 3.0 and 10 * np.log10(pm / pe) < self._erle_best - margin
            if not self.double_talk and not instant_collapse:
                self._pm = 0.98 * self._pm + 0.02 * pm
                self._pe = 0.98 * self._pe + 0.02 * pe
                self._erle_best = max(self.erle_db, self._erle_best - 0.005)
            else:
                # The running ERLE is frozen while a caller is talking -- it is
                # what sets the step size, and letting the caller drag it down
                # is exactly how the filter used to be wrecked. What keeps a
                # merely STALE filter (the far-end voice changed) from reading
                # as a caller forever is that the reference erodes here: after
                # a few seconds of persistent "collapse" it stops firing and
                # adaptation resumes.
                self._erle_best = max(self._erle_best - 0.05, 0.0)

        inst = (np.abs(self.X) ** 2).sum(axis=0)
        self._power = 0.9 * self._power + 0.1 * inst
        if ref_active:
            # Full-size normalised step in single-talk; nearly frozen while a
            # caller is talking over the agent (detected above, in THIS block,
            # before the step is taken). A per-bin "expected residual over
            # observed error" step was tried and dropped: the ERLE it relies on
            # is a whole-signal figure, so bins that were poorly excited looked
            # converged, took tiny steps and never learned.
            E = np.fft.rfft(np.concatenate([np.zeros(B), e]))
            self._err_power = 0.7 * self._err_power + 0.3 * (np.abs(E) ** 2)
            leak = float(np.clip(1.0 / (10 ** (max(self.erle_db, 0.0) / 10.0)), 0.01, 0.6))
            cap = self.dt_mu_fraction if self.double_talk else 1.0
            ratio = np.clip(leak * inst / (np.maximum(self._err_power, np.abs(E) ** 2) + 1e-12), 0.0, cap)
            G = (
                self.mu
                * ratio
                * np.conj(self.X)
                * E
                / (np.maximum(self._power, inst) + self.reg * float(np.mean(self._power)) + 1e-9)
            )
            self.W += G
            # gradient constraint: the true filter is causal and B taps long
            # per partition; zero the wrap-around half.
            w = np.fft.irfft(self.W, axis=1)
            w[:, B:] = 0.0
            self.W = np.fft.rfft(w, axis=1)
        return self._suppress_residual(e_lin, Yf, ref_active)

    def _suppress_residual(self, e: np.ndarray, Yf: np.ndarray, ref_active: bool) -> np.ndarray:
        """Residual echo suppression. The linear filter leaves an echo about
        `erle` dB below the original; this attenuates exactly those spectral
        bins where the error is no bigger than that expected residual, so it
        removes echo the filter could not model. Where the caller is talking
        the error is far above the expected residual, the gain stays at 1, and
        the caller is not touched -- which is what separates this from a
        blanket non-linear processor, and why barge-in can still see through
        it. Gains are smoothed across frequency so the equivalent filter is
        short and the overlap-save output is free of time-aliasing."""
        B = self.B
        window = np.concatenate([self._eprev, e])
        self._eprev = e.copy()
        if not ref_active or not self.suppress:
            return e
        Ew = np.fft.rfft(window)
        erle_lin = 10 ** (max(self.erle_db, 0.0) / 10.0)
        resid = (np.abs(Yf) ** 2) / max(erle_lin, 1.0)
        g = 1.0 - self.suppress_beta * resid / (np.abs(Ew) ** 2 + 1e-12)
        g = np.clip(g, self.suppress_floor, 1.0)
        g = np.convolve(g, np.ones(7) / 7.0, mode="same")
        return np.fft.irfft(Ew * g)[B:]


# ---------------------------------------------------------------- pipeline


@dataclasses.dataclass
class EchoStats:
    delay_ms: float
    delay_confidence: float
    erle_db: float
    double_talk_blocks: int
    blocks: int


class EchoPipeline:
    """Streaming echo cancellation on one call's microphone timeline.

    feed_reference(pcm, at_sample) places far-end audio at an ABSOLUTE
    position on the microphone timeline (sample index since the call began).
    The server computes that position as max(now, end of the previous clip)
    because the client plays clips back to back. The real echo arrives later
    by an unknown constant; DelayEstimator measures it.

    process(mic) takes the next microphone samples (timeline order) and
    returns cleaned samples. Output lags input by less than one block: the
    tail that does not fill a block is held for the next call.
    """

    def __init__(
        self,
        sr: int = 16000,
        block: int = BLOCK,
        partitions: int = 8,
        max_delay_s: float = 0.6,
        lead_s: float = 0.032,
        history_s: float = 4.0,
        delay_update_s: float = 0.5,
        min_confidence: float = 6.0,
        mu: float = 1.0,
        suppress: bool = True,
    ):
        self.sr, self.block = sr, block
        self.canceller = MDFCanceller(block, partitions, mu=mu)
        self.canceller.suppress = suppress
        self.max_delay_s, self.lead = max_delay_s, int(lead_s * sr)
        self.delay_update = int(delay_update_s * sr)
        self.min_confidence = min_confidence
        self._hist = int(history_s * sr)
        self._ref = np.zeros(0, np.float32)  # far-end timeline [_ref0, _ref0+len)
        self._ref0 = 0
        self._mic = np.zeros(0, np.float32)  # raw mic timeline, same indexing
        self._mic0 = 0
        self._pending = np.zeros(0, np.float32)
        self._pos = 0  # absolute index of the next unprocessed mic sample
        self._since_delay = 0
        self.delay = 0
        self._delay_candidate: int | None = None
        self.delay_confidence = 0.0
        self._blocks = self._dt_blocks = 0

    # -- reference placement -------------------------------------------
    def feed_reference(self, pcm: np.ndarray, at_sample: int) -> None:
        pcm = np.asarray(pcm, np.float32)
        end = at_sample + pcm.size
        if end > self._ref0 + self._ref.size:
            pad = end - (self._ref0 + self._ref.size)
            self._ref = np.concatenate([self._ref, np.zeros(pad, np.float32)])
        lo = at_sample - self._ref0
        if lo < 0:  # placed before what we still keep: drop the stale head
            pcm, lo = pcm[-lo:], 0
        self._ref[lo : lo + pcm.size] += pcm

    def reference_end(self) -> int:
        """Absolute sample index just past the last far-end audio placed."""
        return self._ref0 + self._ref.size

    def _ref_slice(self, a: int, b: int) -> np.ndarray:
        out = np.zeros(b - a, np.float32)
        lo, hi = max(a, self._ref0), min(b, self._ref0 + self._ref.size)
        if hi > lo:
            out[lo - a : hi - a] = self._ref[lo - self._ref0 : hi - self._ref0]
        return out

    # -- processing ------------------------------------------------------
    def process(self, mic: np.ndarray) -> np.ndarray:
        return self.process_ex(mic)[0]

    def process_ex(self, mic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(cleaned, echo_estimate) for the completed blocks. The estimate is
        what the filter subtracted; agent/barge_in.py compares the cleaned
        signal against it to tell residual echo from a caller."""
        mic = np.asarray(mic, np.float32)
        self._mic = np.concatenate([self._mic, mic])
        buf = np.concatenate([self._pending, mic])
        out, echo = [], []
        B = self.block
        while buf.size >= B:
            blk, buf = buf[:B], buf[B:]
            a = self._pos
            ref_blk = self._ref_slice(a - self.delay, a - self.delay + B)
            out.append(self.canceller.process_block(blk.astype(np.float64), ref_blk.astype(np.float64)))
            echo.append(self.canceller.echo_estimate)
            self._blocks += 1
            self._dt_blocks += int(self.canceller.double_talk)
            self._pos += B
            self._since_delay += B
            if self._since_delay >= self.delay_update:
                self._since_delay = 0
                self._update_delay()
        self._pending = buf
        self._trim()
        if not out:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        return np.concatenate(out).astype(np.float32), np.concatenate(echo).astype(np.float32)

    def _update_delay(self) -> None:
        win = int(1.5 * self.sr)
        end = self._pos
        start = max(self._mic0, end - win)
        mic = self._mic[start - self._mic0 : end - self._mic0]
        # The reference has to have been active for the estimate to mean anything.
        if float(np.mean(self._ref_slice(start, end) ** 2)) < 1e-7:
            return
        # The echo of `ref` lies up to max_delay AFTER it, so the reference is
        # taken from EARLIER than the mic window by the maximum delay.
        maxd = int(self.max_delay_s * self.sr)
        k, conf = estimate_delay_long(mic, self._ref_slice(start - maxd, end), maxd)
        self.delay_confidence = conf
        if conf >= self.min_confidence:
            new = max(0, k - self.lead)
            if abs(new - self.delay) <= 2 * self.block:
                self._delay_candidate = None  # agrees with what we use: nothing to change
            elif self._delay_candidate is not None and abs(new - self._delay_candidate) <= 2 * self.block:
                # seen twice in a row: a real change of path, not one odd window
                self.delay, self._delay_candidate = new, None
            else:
                self._delay_candidate = new

    def _trim(self) -> None:
        keep_from = self._pos - self._hist
        if keep_from > self._mic0:
            cut = keep_from - self._mic0
            self._mic, self._mic0 = self._mic[cut:], keep_from
        ref_from = keep_from - int(self.max_delay_s * self.sr) - self.block
        if ref_from > self._ref0:
            cut = min(ref_from - self._ref0, self._ref.size)
            self._ref, self._ref0 = self._ref[cut:], self._ref0 + cut

    def stats(self) -> EchoStats:
        return EchoStats(
            self.delay / self.sr * 1000.0, self.delay_confidence, self.canceller.erle_db, self._dt_blocks, self._blocks
        )


def estimate_delay_long(mic: np.ndarray, ref_long: np.ndarray, max_delay: int) -> tuple[int, float]:
    """GCC-PHAT of `mic` against a reference that starts `max_delay` samples
    EARLIER than the mic window. ref_long = ref[start-max_delay : end]; the
    returned lag k means mic[t] ~ ref[t - k] for k in [0, max_delay]."""
    n = mic.size
    if n < 3200 or ref_long.size < n:
        return 0, 0.0
    if np.sum(ref_long**2) < 1e-8 or np.sum(mic**2) < 1e-10:
        return 0, 0.0
    nfft = 1 << int(np.ceil(np.log2(ref_long.size + n)))
    M = np.fft.rfft(np.asarray(mic, np.float64), nfft)
    R = np.fft.rfft(np.asarray(ref_long, np.float64), nfft)
    cps = R * np.conj(M)  # correlation of ref against mic
    cps /= np.abs(cps) + 1e-9
    cc = np.abs(np.fft.irfft(cps, nfft))
    # cc[j]: ref_long delayed by -j relative to mic; mic sample t sits at
    # ref_long index t + max_delay, so lag k corresponds to j = max_delay - k
    # (mod nfft). Read j in [0, max_delay].
    window = cc[: max_delay + 1][::-1]  # window[k] for k = 0..max_delay
    peak = float(np.max(window))
    # The EARLIEST strong peak, not the tallest: an echo path's later
    # reflections can outrank the direct path, and the filter can only model
    # what comes AFTER the delay it is given.
    strong = np.where(window >= 0.6 * peak)[0]
    k = int(strong[0])
    return k, float(peak / (np.mean(window) + 1e-12))
