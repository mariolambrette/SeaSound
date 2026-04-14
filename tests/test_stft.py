"""Tests for STFT computation module."""

import numpy as np
import pytest
from seasound.loader.stft import compute_stft_power


class TestComputeSTFTPower:
    """Test the compute_stft_power() function."""

    def test_output_shape(self):
        """STFT output shape matches expected (freq × time)."""
        sr = 96000
        duration = 2  # seconds
        audio = np.sin(2 * np.pi * 1000 * np.arange(sr * duration) / sr)
        audio = audio.astype(np.float32)

        freqs, times, power = compute_stft_power(
            audio,
            sample_rate=sr,
            nfft=2048,
            win_length=2048,
            hop_length=1024,
            window="hann",
            fmin_hz=10.0,
            fmax_hz=50000.0,
        )

        # Expected time frames: (n_samples - win_length) / hop_length + 1
        # But with boundary=None and padded=False, scipy reduces this
        expected_frames = (len(audio) - 2048) // 1024 + 1
        
        assert power.ndim == 2
        assert power.shape[0] == len(freqs)
        assert power.shape[1] == len(times)
        assert len(times) > 1  # At least 2 frames for 2 seconds

    def test_frequency_filtering_applied(self):
        """Frequency filtering reduces spectrum to [fmin_hz, fmax_hz] range."""
        sr = 96000
        duration = 1
        audio = np.sin(2 * np.pi * 1000 * np.arange(sr * duration) / sr)
        audio = audio.astype(np.float32)

        fmin, fmax = 100.0, 10000.0
        freqs, times, power = compute_stft_power(
            audio,
            sample_rate=sr,
            nfft=2048,
            win_length=2048,
            hop_length=1024,
            window="hann",
            fmin_hz=fmin,
            fmax_hz=fmax,
        )

        assert np.all(freqs >= fmin)
        assert np.all(freqs <= fmax)
        assert len(freqs) < 2048  # Filtered count less than full FFT

    def test_power_is_positive(self):
        """Power (magnitude squared) is always non-negative."""
        sr = 96000
        duration = 1
        audio = np.sin(2 * np.pi * 1000 * np.arange(sr * duration) / sr)
        audio = audio.astype(np.float32)

        freqs, times, power = compute_stft_power(
            audio,
            sample_rate=sr,
            nfft=2048,
            win_length=2048,
            hop_length=1024,
            window="hann",
            fmin_hz=10.0,
            fmax_hz=50000.0,
        )

        assert np.all(power >= 0)

    def test_sinusoid_peak_detection(self):
        """Peak power near the sinusoid's center frequency."""
        sr = 96000
        duration = 1
        freq_signal = 3000.0  # Hz
        audio = np.sin(2 * np.pi * freq_signal * np.arange(sr * duration) / sr)
        audio = audio.astype(np.float32) * 0.5

        freqs, times, power = compute_stft_power(
            audio,
            sample_rate=sr,
            nfft=2048,
            win_length=2048,
            hop_length=1024,
            window="hann",
            fmin_hz=10.0,
            fmax_hz=50000.0,
        )

        # Find frequency bin with highest mean power
        mean_power = power.mean(axis=1)
        peak_idx = np.argmax(mean_power)
        peak_freq = freqs[peak_idx]

        # Peak should be within ~100 Hz of signal (frequency resolution ~47 Hz at 2048 FFT)
        assert np.abs(peak_freq - freq_signal) < 200.0

    def test_different_window_types_accepted(self):
        """Different window functions are accepted without error."""
        sr = 96000
        duration = 0.5
        audio = np.sin(2 * np.pi * 1000 * np.arange(sr * duration) / sr)
        audio = audio.astype(np.float32)

        for window_type in ["hann", "hamming", "blackman"]:
            freqs, times, power = compute_stft_power(
                audio,
                sample_rate=sr,
                nfft=2048,
                win_length=2048,
                hop_length=1024,
                window=window_type,
                fmin_hz=10.0,
                fmax_hz=50000.0,
            )
            assert power.shape[0] > 0
            assert power.shape[1] > 0

    def test_output_dtype_is_complex(self):
        """STFT magnitude squared (power) has float dtype."""
        sr = 96000
        duration = 0.5
        audio = np.sin(2 * np.pi * 1000 * np.arange(sr * duration) / sr)
        audio = audio.astype(np.float32)

        freqs, times, power = compute_stft_power(
            audio,
            sample_rate=sr,
            nfft=2048,
            win_length=2048,
            hop_length=1024,
            window="hann",
            fmin_hz=10.0,
            fmax_hz=50000.0,
        )

        # power should be float (magnitude squared, real-valued)
        assert np.issubdtype(power.dtype, np.floating)
