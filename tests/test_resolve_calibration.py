"""
tests/test_resolve_calibration.py

Tests for the PR 1 calibration split (resolve_calibration +
ResolvedCalibration) and the base-matrix config promotions.

The critical guarantee here is BIT-identity between the in-place block
path (used by streaming) and the legacy allocating path: each method's
calibrate_inplace must replay to_pascals' exact floating-point
operation order. A pre-combined single-scalar gain was measured to
diverge in the last bit for ~27% of float32 samples, so these tests use
np.array_equal, not allclose.
"""

import copy #pylint: disable=unused-import

import numpy as np
import pandas as pd
import pytest

from seasound.core.config import CalibrationConfig, ProcessingConfig
from seasound.core.exceptions import CalibrationError
from tests.golden import base_matrix_from_array
from seasound.loader.calibration import ( #pylint: disable=unused-import
    ResolvedCalibration,
    apply_calibration,
    load_calibration,
    resolve_calibration,
)
from seasound.loader.calibration_methods import (
    CALIBRATION_METHOD_REGISTRY,
    CalibrationMethod,
    get_calibration_method,
)
from seasound.loader.reader import AudioSegment


def _segment(data, serial="9999"):
    return AudioSegment(
        data=data,
        sample_rate=96000,
        serial=serial,
        datetime_start=None,
        channel=0,
        source_file="test.wav",
    )


def _config(sample_calibration_file, **overrides):
    defaults = dict(
        enabled=True,
        strict=False,
        file=sample_calibration_file,
        serial_column="Serial",
        sensitivity_column="High_Gain",
        method="soundtrap",
        vpp=2.0,
    )
    defaults.update(overrides)
    return CalibrationConfig(**defaults) #type: ignore


class TestInplaceBitIdentity:
    """calibrate_inplace must be bit-identical to to_pascals for every
    registered method — the foundation of the streaming path's
    bit-identity gate."""

    @pytest.mark.parametrize("method_name", sorted(CALIBRATION_METHOD_REGISTRY))
    @pytest.mark.parametrize("sens_db", [-176.0, -174.5, -188.2])
    def test_inplace_matches_to_pascals(self, method_name, sens_db):
        """For a range of methods and sensitivity values, calibrate_inplace must"""
        method = get_calibration_method(method_name)
        rng = np.random.default_rng(11)
        samples = rng.uniform(-1.0, 1.0, 200_000).astype(np.float32)
        samples[:3] = [0.0, -0.0, 1.0]  # exact edge values

        expected = method.to_pascals(samples, sens_db, vpp=2.0)
        block = samples.copy()
        result = method.calibrate_inplace(block, sens_db, vpp=2.0)

        assert result is block  # mutated in place, same object returned
        np.testing.assert_array_equal(
            result, expected,
            err_msg=(
                f"{method_name}.calibrate_inplace is not bit-identical to "
                f"to_pascals — operation order must be preserved exactly"
            ),
        )

    def test_inplace_works_on_noncontiguous_views(self):
        """Under channel_strategy='auto', segment.data is a column view;
        in-place calibration must be correct on views too."""
        method = get_calibration_method("soundtrap")
        rng = np.random.default_rng(12)
        stereo = rng.uniform(-1.0, 1.0, (50_000, 2)).astype(np.float32)
        column = stereo[:, 1]

        expected = method.to_pascals(column.copy(), -176.0, vpp=2.0)
        method.calibrate_inplace(column, -176.0, vpp=2.0)
        np.testing.assert_array_equal(stereo[:, 1], expected)

    def test_base_class_raises_for_nonscalar_methods(self):
        """Plan section 10 loud failure: a method without scalar
        in-place support must raise, never silently mis-calibrate."""

        class FrequencyDependentMethod(CalibrationMethod):
            """A dummy method that implements to_pascals but not calibrate_inplace"""
            name = "freq_dependent_dummy"

            def to_pascals(self, samples, sensitivity_db, vpp):
                return samples

        with pytest.raises(NotImplementedError):
            FrequencyDependentMethod().calibrate_inplace(
                np.zeros(4, dtype=np.float32), -176.0, 2.0
            )


class TestResolveCalibrationSemantics:
    """resolve_calibration must reproduce every apply_calibration
    branch: override, fallbacks, strict raises, and the flag."""

    def test_resolves_table_sensitivity(self, sample_calibration_file):
        """The sample calibration file contains a known serial with a known"""
        config = _config(sample_calibration_file)
        cal_df = load_calibration(config)
        resolved = resolve_calibration(_segment(np.zeros(1)), cal_df, config)

        assert resolved.calibrated is True
        assert resolved.sensitivity_db == -176.0
        assert resolved.method.name == "soundtrap" #type: ignore
        assert resolved.vpp == 2.0

    def test_override_bypasses_table(self, sample_calibration_file):
        """When sensitivity_db_override is set, the table is ignored and the"""
        config = _config(sample_calibration_file, sensitivity_db_override=-180.0)
        # load_calibration returns None when the override is set
        cal_df = load_calibration(config)
        assert cal_df is None

        resolved = resolve_calibration(_segment(np.zeros(1)), cal_df, config)
        assert resolved.calibrated is True
        assert resolved.sensitivity_db == -180.0

    def test_disabled_is_uncalibrated(self, sample_calibration_file):
        """When enabled=False, the config is disabled regardless of other settings."""
        config = _config(sample_calibration_file, enabled=False, strict=True)
        resolved = resolve_calibration(_segment(np.zeros(1)), None, config)
        assert resolved.calibrated is False

    def test_no_table_strict_raises(self, sample_calibration_file):
        """When enabled but no table, strict=True must raise; strict=False must"""
        config = _config(sample_calibration_file, strict=True)
        with pytest.raises(CalibrationError):
            resolve_calibration(_segment(np.zeros(1)), None, config)

    def test_missing_serial_non_strict(self, sample_calibration_file):
        """
        When the serial is not found in the table, strict=False must return an 
        uncalibrated segment.
        """
        config = _config(sample_calibration_file)
        cal_df = load_calibration(config)
        resolved = resolve_calibration(
            _segment(np.zeros(1), serial=None), cal_df, config #type: ignore
        )
        assert resolved.calibrated is False

    def test_missing_serial_strict_raises(self, sample_calibration_file):
        """When the serial is not found in the table, strict=True must raise."""
        config = _config(sample_calibration_file, strict=True)
        cal_df = load_calibration(config)
        with pytest.raises(CalibrationError):
            resolve_calibration(_segment(np.zeros(1), serial=None), cal_df, config) #type: ignore

    def test_unknown_serial_non_strict(self, sample_calibration_file):
        """
        When the serial is not found in the table, strict=False must return an 
        uncalibrated segment.
        """
        config = _config(sample_calibration_file)
        cal_df = load_calibration(config)
        resolved = resolve_calibration(
            _segment(np.zeros(1), serial="123456"), cal_df, config
        )
        assert resolved.calibrated is False

    def test_leading_zero_serial_falls_back(self, sample_calibration_file):
        """'09999' is not in the table; the integer-form fallback must
        resolve it to '9999'."""
        config = _config(sample_calibration_file)
        cal_df = load_calibration(config)
        resolved = resolve_calibration(
            _segment(np.zeros(1), serial="09999"), cal_df, config
        )
        assert resolved.calibrated is True
        assert resolved.sensitivity_db == -176.0

    def test_nan_sensitivity_non_strict(self, tmp_path):
        """
        When the sensitivity value is NaN, strict=False must return an 
        uncalibrated segment; strict=True must raise.
        """
        df = pd.DataFrame({"Serial": ["9999"], "High_Gain": [np.nan]})
        cal_path = str(tmp_path / "cal_nan.xlsx")
        df.to_excel(cal_path, index=False)
        config = _config(cal_path)
        cal_df = load_calibration(config)

        resolved = resolve_calibration(_segment(np.zeros(1)), cal_df, config)
        assert resolved.calibrated is False

    def test_metadata_only_object_accepted(self, sample_calibration_file):
        """
        The streaming path resolves before any samples exist; any
        object with .serial and .source_file must work.
        """

        class Meta:
            """Metadata-only object with the required attributes for resolution."""
            serial = "9471"
            source_file = "9471.260101120000.wav"

        config = _config(sample_calibration_file)
        cal_df = load_calibration(config)
        resolved = resolve_calibration(Meta(), cal_df, config) #type: ignore
        assert resolved.calibrated is True
        assert resolved.sensitivity_db == -174.5


class TestApplyCalibrationComposition:
    """apply_calibration must remain exactly resolve + apply."""

    def test_composition_matches_direct_call(self, sample_calibration_file):
        """
        For a known serial and config, the output of apply_calibration must
        be bit-identical to resolved.apply(). This is the legacy contract for 
        calibration; the streaming path relies on this bit-identity guarantee to 
        ensure that the new resolve + apply path produces the same calibrated 
        values as the old apply_calibration path, which is critical for 
        consistency across deployments and for trusting the new path.
        """
        config = _config(sample_calibration_file)
        cal_df = load_calibration(config)
        rng = np.random.default_rng(13)
        segment = _segment(rng.uniform(-1, 1, 10_000).astype(np.float32))

        audio_pa, calibrated = apply_calibration(segment, cal_df, config)
        resolved = resolve_calibration(segment, cal_df, config)

        assert calibrated is resolved.calibrated is True
        np.testing.assert_array_equal(audio_pa, resolved.apply(segment.data))

    def test_uncalibrated_returns_same_object(self, sample_calibration_file):
        """Legacy contract: the uncalibrated fallback returns
        segment.data itself, not a copy."""
        config = _config(sample_calibration_file)
        cal_df = load_calibration(config)
        segment = _segment(np.zeros(8, dtype=np.float32), serial="123456")

        audio_pa, calibrated = apply_calibration(segment, cal_df, config)
        assert calibrated is False
        assert audio_pa is segment.data

    def test_apply_does_not_mutate_input(self, sample_calibration_file):
        """apply() (the legacy path) must keep allocating: callers of
        apply_calibration may rely on segment.data being unchanged."""
        config = _config(sample_calibration_file)
        cal_df = load_calibration(config)
        raw = np.full(16, 0.5, dtype=np.float32)
        segment = _segment(raw.copy())

        apply_calibration(segment, cal_df, config)
        np.testing.assert_array_equal(segment.data, raw)


class TestBaseMatrixConfigPromotions:
    """nfft_padding_factor and sxx_dtype: baseline defaults must be
    bit-identical to the previous hardcoded behaviour."""

    @staticmethod
    def _audio(sr=8000, duration_s=10, seed=21):
        rng = np.random.default_rng(seed)
        return (0.1 * rng.standard_normal(duration_s * sr)).astype(np.float32), sr

    def test_defaults_match_explicit_baseline_values(self):
        """
        The default config values must produce a base matrix that is 
        bit-identical to the previous hardcoded defaults (nfft_padding_factor=4, 
        sxx_dtype='float32') to ensure that outputs remain comparable across 
        runs and deployments unless these knobs are intentionally changed
        (e.g. for a future refactor that alters the numerics of the STFT).
        """
        audio, sr = self._audio()
        default_cfg = ProcessingConfig()
        explicit_cfg = ProcessingConfig(nfft_padding_factor=4, sxx_dtype="float32")

        a = base_matrix_from_array(audio, sr, default_cfg)
        b = base_matrix_from_array(audio, sr, explicit_cfg)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_padding_factor_changes_numerics(self):
        """Sanity: the knob is live — a different factor must change
        values (this is why it is documented DO NOT CHANGE)."""
        audio, sr = self._audio()
        a = base_matrix_from_array(audio, sr, ProcessingConfig())
        b = base_matrix_from_array(
            audio, sr, ProcessingConfig(nfft_padding_factor=2)
        )
        mask = ~np.isnan(a.to_numpy())
        assert not np.array_equal(a.to_numpy()[mask], b.to_numpy()[mask])

    def test_float64_dtype_changes_numerics(self):
        """Sanity: the knob is live — a different dtype must change
        values (this is why it is documented DO NOT CHANGE)."""
        audio, sr = self._audio()
        a = base_matrix_from_array(audio, sr, ProcessingConfig())
        b = base_matrix_from_array(audio, sr, ProcessingConfig(sxx_dtype="float64"))
        assert a.shape == b.shape
        mask = ~np.isnan(a.to_numpy())
        assert not np.array_equal(a.to_numpy()[mask], b.to_numpy()[mask])
