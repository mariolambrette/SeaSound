"""Utility for calculating STFT power from audio data."""

import numpy as np
from scipy import signal


def compute_stft_power(
    audio_pa: np.ndarray,
    sample_rate: int,
    nfft: int,
    win_length: int,
    hop_length: int,
    window: str = "hann",
    fmin_hz: float = 10.0,
    fmax_hz: float = 50000.0,
):
    """Compute STFT power for a single audio segment."""
    noverlap = win_length - hop_length
    freqs, times, Zxx = signal.stft( #pylint: disable-=invalid-name
        audio_pa,
        fs=sample_rate,
        window=window,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None, # pyright: ignore[reportArgumentType]
        padded=False,
    )
    power = np.abs(Zxx) ** 2
    mask = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    return freqs[mask], times, power[mask, :]


class StftAccumulator:
    """
    Streaming STFT over arbitrarily-sized contiguous sample blocks
    (refactor plan Stage 3, D8): frame-for-frame identical to a single
    full-signal ``compute_stft_power`` call, computed one block at a
    time with an overlap-save carry buffer.

    Frame *k* (0-based within the file) covers samples
    ``[k*hop, k*hop + win)``. After emitting frames up to ``m-1``, the
    carry holds the samples from ``m*hop`` onward — always fewer than
    ``win`` of them — and is prepended to the next block, so no frame
    is ever split, duplicated, or fabricated across a block seam.
    Identity is structural, not re-derived: each batch of frames is
    produced by the *same* ``compute_stft_power`` call the legacy
    full-file path uses, on a buffer that always starts exactly at a
    frame boundary.

    One accumulator serves exactly one file/channel — the per-file
    carry reset of D8 holds by construction. At ``finalise`` the carry
    (the trailing samples too few to complete a frame) is discarded,
    matching the legacy path's trailing-remainder drop. A whole file
    shorter than ``win_length`` therefore yields zero frames; the
    legacy ``scipy`` call would instead shrink ``nperseg`` with a
    warning on such degenerate input, which is silently wrong — the
    streaming path refuses to reproduce that.

    Memory: holds at most ``carry + one block`` of samples plus one
    block's frames — never the file.
    """

    def __init__(
        self,
        sample_rate: int,
        nfft: int,
        win_length: int,
        hop_length: int,
        window: str = "hann",
        fmin_hz: float = 10.0,
        fmax_hz: float = 50000.0,
    ):
        self._sample_rate = int(sample_rate)
        self._nfft = int(nfft)
        self._win = int(win_length)
        self._hop = int(hop_length)
        self._window = window
        self._fmin_hz = fmin_hz
        self._fmax_hz = fmax_hz

        self._carry = np.empty(0, dtype=np.float32)
        self._n_frames = 0
        self._finalised = False
        #: Masked frequency axis, available after the first emitted frames.
        self.freqs_hz: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        """Frames emitted so far (continuous within-file index)."""
        return self._n_frames

    def push(self, block: np.ndarray) -> np.ndarray | None:
        """
        Feed the next contiguous samples; return the newly completed
        power frames as ``(n_freq, k)`` in the compute dtype, or None
        if the buffered samples do not yet complete a frame.
        """
        if self._finalised:
            raise RuntimeError("push() after finalise() on StftAccumulator")

        if self._carry.size:
            buf = np.concatenate([self._carry, block])
        else:
            buf = np.asarray(block)

        if len(buf) < self._win:
            # Copy: the caller may reuse/mutate the block's storage.
            self._carry = np.array(buf, copy=True)
            return None

        n_avail = (len(buf) - self._win) // self._hop + 1
        consumed = (n_avail - 1) * self._hop + self._win

        freqs_hz, _times_s, power = compute_stft_power(
            audio_pa=buf[:consumed],
            sample_rate=self._sample_rate,
            nfft=self._nfft,
            win_length=self._win,
            hop_length=self._hop,
            window=self._window,
            fmin_hz=self._fmin_hz,
            fmax_hz=self._fmax_hz,
        )
        if self.freqs_hz is None:
            self.freqs_hz = freqs_hz

        # Overlap-save: the next frame starts at n_avail*hop; keep
        # everything from there (always < win samples — tiny copy).
        self._carry = np.array(buf[n_avail * self._hop:], copy=True)
        self._n_frames += n_avail
        return power

    def finalise(self) -> int:
        """
        End of file: discard the carry (the trailing samples too few to
        complete a frame — exactly what the legacy full-file call
        drops) and return the total frame count.
        """
        if self._finalised:
            raise RuntimeError("finalise() called twice on StftAccumulator")
        self._finalised = True
        self._carry = np.empty(0, dtype=np.float32)
        return self._n_frames
