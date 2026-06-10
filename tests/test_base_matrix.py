"""Tests for the base matrix computation."""

import numpy as np
import pytest
from tests.golden import base_matrix_from_array
from seasound.core.config import ProcessingConfig


class TestBaseMatrix:

    def test_sine_wave_energy_in_correct_band(self):
        """
        A 1000 Hz sine wave should produce high SPL in the 1000 Hz TOB
        band and much lower SPL in other bands.

        This is the fundamental correctness test for the entire DSP chain.
        """
        sr = 96000
        duration = 5  # seconds
        freq = 1000   # Hz
        amplitude_pa = 1.0  # 1 Pa RMS

        t = np.arange(sr * duration) / sr
        # amplitude_pa is RMS, peak is sqrt(2) * RMS for a sine
        audio_pa = amplitude_pa * np.sqrt(2) * np.sin(2 * np.pi * freq * t)

        config = ProcessingConfig(
            max_freq_hz=50000,
            min_freq_hz=10,
            base_resolution_s=1,
            reference_pressure_pa=1e-6,
            missing_band_strategy="nan",
        )

        matrix = base_matrix_from_array(audio_pa, sr, config)

        # Expected SPL for 1 Pa RMS: 20*log10(1 / 1e-6) = 120 dB
        # The actual value will be slightly less because the band only
        # captures energy within its edges, and windowing spreads some
        # energy to adjacent bins
        target_col = "1000.0Hz"
        assert target_col in matrix.columns

        target_spl = matrix[target_col].mean()
        # Should be close to 120 dB (within ~2 dB for windowing effects)
        assert 118 < target_spl < 122, f"Expected ~120 dB, got {target_spl:.1f}"

        # Other bands should be much lower
        other_cols = [c for c in matrix.columns if c != target_col and not np.isnan(matrix[c].iloc[0])]
        for col in other_cols:
            other_spl = matrix[col].mean()
            assert other_spl < target_spl - 20, (
                f"Band {col} ({other_spl:.1f} dB) is too close to "
                f"target band ({target_spl:.1f} dB)"
            )

    def test_output_shape(self):
        """5 seconds of audio at 1s resolution → 5 rows."""
        sr = 96000
        audio = np.random.randn(sr * 5) * 1e-3  # noise

        config = ProcessingConfig(
            max_freq_hz=50000, min_freq_hz=10,
            base_resolution_s=1, reference_pressure_pa=1e-6,
        )

        matrix = base_matrix_from_array(audio, sr, config)
        assert len(matrix) == 5
        assert "1000.0Hz" in matrix.columns

    def test_short_audio_returns_empty(self):
        """Audio shorter than base_resolution_s returns empty DataFrame."""
        sr = 96000
        audio = np.zeros(100)  # < 1 second

        config = ProcessingConfig(
            max_freq_hz=50000, min_freq_hz=10,
            base_resolution_s=1, reference_pressure_pa=1e-6,
        )

        matrix = base_matrix_from_array(audio, sr, config)
        assert len(matrix) == 0
