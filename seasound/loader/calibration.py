"""
seasound/loader/calibration.py

Hydrophone calibration: load sensitivity data and convert normalised
audio samples to acoustic pressure in Pascals.

Calibration is the single most important step for producing physically
meaningful acoustic measurements. Without it, you have relative
numbers that cannot be compared between deployments, instruments,
or studies.

Calibration is split into two phases so the streaming pipeline can
resolve once per file and apply per block:

- resolve_calibration(): all per-file work — serial lookup (with
  leading-zero fallback), sensitivity_db_override, NaN handling,
  strict-vs-warn semantics, and method selection. Returns a
  ResolvedCalibration carrying the method and resolved sensitivity.
- ResolvedCalibration.apply() / .apply_inplace(): the per-sample work.
  apply() allocates (legacy behaviour); apply_inplace() mutates the
  block, applying the method's exact floating-point operation order so
  results are bit-identical to apply().

apply_calibration() remains the legacy single-call API and is now the
composition of the two phases.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from seasound.core.config import CalibrationConfig
from seasound.core.exceptions import CalibrationError
from seasound.loader.reader import AudioSegment
from seasound.loader.calibration_methods import (
    CalibrationMethod,
    get_calibration_method,
)

logger = logging.getLogger(__name__)


def load_calibration(config: CalibrationConfig) -> Optional[pd.DataFrame]:
    """
    Load calibration spreadsheet into a DataFrame indexed by serial number.

    The spreadsheet must have:
    - A column identifiable as serial number (named 'Serial' or the first column)
    - A sensitivity column matching config.sensitivity_column

    Parameters
    ----------
    config : CalibrationConfig

    Returns
    -------
    pd.DataFrame or None
        DataFrame indexed by serial (string), with the sensitivity column.
        Returns None only when calibration is disabled.

    Raises
    ------
    CalibrationError
        When strict=True and the file can't be read or is missing columns.
    """
    if not config.enabled:
        logger.info("Calibration disabled in config")
        return None

    if config.sensitivity_db_override is not None:
        logger.info(
            "Using sensitivity override: %.2f dB (calibration file will not be loaded)",
            config.sensitivity_db_override,
        )
        return None

    try:
        df = pd.read_excel(config.file, engine="openpyxl")
    except FileNotFoundError:
        msg = f"Calibration file not found: {config.file}"
        if config.strict:
            raise CalibrationError(msg) #pylint: disable=raise-missing-from
        logger.warning("%s — proceeding without calibration", msg)
        return None
    except Exception as exc: #pylint: disable=broad-exception-caught
        msg = f"Could not read calibration file {config.file}: {exc}"
        if config.strict:
            raise CalibrationError(msg) #pylint: disable=raise-missing-from
        logger.warning("%s — proceeding without calibration", msg)
        return None

    # --- Identify serial column ---
    serial_col = config.serial_column

    if serial_col not in df.columns:
        msg = (
            f"Serial column '{serial_col}' not found in {config.file}. "
            f"Available columns: {', '.join(df.columns.tolist())}. "
            f"Set calibration.serial_column in your config to match "
            f"the column containing hydrophone serial numbers."
        )
        if config.strict:
            raise CalibrationError(msg)
        logger.warning(msg)
        return None

    df[serial_col] = df[serial_col].astype(str).str.strip()
    df = df.set_index(serial_col)

    # --- Check sensitivity column exists ---
    sens_col = config.sensitivity_column
    if sens_col not in df.columns:
        msg = (
            f"Sensitivity column '{sens_col}' not found in {config.file}. "
            f"Available columns: {', '.join(df.columns.tolist())}"
        )
        if config.strict:
            raise CalibrationError(msg)
        logger.warning(msg)
        return None

    logger.info(
        "Loaded calibration for %d serial(s) from %s", len(df), config.file
    )
    return df


@dataclass(frozen=True)
class ResolvedCalibration:
    """
    The outcome of per-file calibration resolution.

    When ``calibrated`` is False, both application methods return the
    input unchanged (the legacy uncalibrated fallback). When True,
    ``method`` and ``sensitivity_db`` are set and application converts
    samples to Pascals.

    Attributes
    ----------
    method : CalibrationMethod or None
        Conversion method instance; None when uncalibrated.
    sensitivity_db : float or None
        Resolved sensitivity (from the table or the override);
        None when uncalibrated.
    vpp : float
        Peak-to-peak voltage from the config (used by some methods).
    calibrated : bool
        Whether calibration will actually be applied. Flows into
        SegmentArtifact / save_base_matrix unchanged.
    """
    method: Optional[CalibrationMethod]
    sensitivity_db: Optional[float]
    vpp: float
    calibrated: bool

    def apply(self, samples: np.ndarray) -> np.ndarray:
        """
        Convert samples to Pascals, allocating a new array (legacy
        apply_calibration behaviour). Uncalibrated: returns ``samples``
        itself, unchanged.
        """
        if not self.calibrated:
            return samples
        return self.method.to_pascals(samples, self.sensitivity_db, self.vpp) #type: ignore

    def apply_inplace(self, block: np.ndarray) -> np.ndarray:
        """
        Convert one block to Pascals in place and return it.

        Bit-identical to apply() — the method replays the same
        floating-point operations in the same order on the mutated
        array. Uncalibrated: returns ``block`` itself, untouched.

        Raises
        ------
        NotImplementedError
            If the resolved method does not support per-sample scalar
            in-place application (loud failure for future non-scalar
            methods, e.g. frequency-dependent calibration).
        """
        if not self.calibrated:
            return block
        return self.method.calibrate_inplace( #type: ignore
            block, self.sensitivity_db, self.vpp #type: ignore
        )


def resolve_calibration(
    segment: AudioSegment,
    cal_df: Optional[pd.DataFrame],
    config: CalibrationConfig,
) -> ResolvedCalibration:
    """
    Resolve per-file calibration state without touching sample data.

    All lookup, override, strict-vs-warn, and fallback semantics of the
    legacy apply_calibration() live here, with identical messages and
    identical CalibrationError behaviour under strict=True. The sample
    arithmetic is deferred to the returned ResolvedCalibration.

    Parameters
    ----------
    segment : AudioSegment
        Only ``segment.serial`` and ``segment.source_file`` are read, so
        any object carrying those attributes (e.g. file metadata in the
        streaming path, where no sample data exists yet) is accepted.
    cal_df : pd.DataFrame or None
        Calibration table from load_calibration().
    config : CalibrationConfig

    Returns
    -------
    ResolvedCalibration

    Raises
    ------
    CalibrationError
        When strict=True and calibration cannot be resolved.
    """
    uncalibrated = ResolvedCalibration(
        method=None, sensitivity_db=None, vpp=config.vpp, calibrated=False,
    )

    # --- sensitivity_db_override: bypass file lookup entirely ---
    if config.enabled and config.sensitivity_db_override is not None:
        try:
            method = get_calibration_method(config.method)
        except ValueError as exc:
            if config.strict:
                raise CalibrationError(str(exc)) from exc
            logger.warning(str(exc))
            return uncalibrated
        logger.debug(
            "Calibration applied via override: method=%s, sensitivity=%.1f dB",
            config.method,
            config.sensitivity_db_override,
        )
        return ResolvedCalibration(
            method=method,
            sensitivity_db=config.sensitivity_db_override,
            vpp=config.vpp,
            calibrated=True,
        )

    if cal_df is None:
        if config.strict and config.enabled:
            raise CalibrationError(
                f"No calibration data available for serial {segment.serial}"
            )
        return uncalibrated

    serial = segment.serial
    if serial is None:
        msg = (
            f"No serial number extracted from {segment.source_file}; "
            f"cannot look up calibration"
        )
        if config.strict:
            raise CalibrationError(msg)
        logger.warning("%s — returning uncalibrated data", msg)
        return uncalibrated

    # --- Look up serial in calibration table ---
    serial_str = str(serial).strip()

    # Try exact match, then integer form (handles leading zeros)
    if serial_str not in cal_df.index:
        # Try stripping leading zeros
        alt = str(int(serial_str)) if serial_str.isdigit() else serial_str
        if alt in cal_df.index:
            serial_str = alt
        else:
            msg = (
                f"Serial '{serial}' not found in calibration table. "
                f"Available serials: {', '.join(cal_df.index[:10].tolist())}"
                f"{'...' if len(cal_df) > 10 else ''}"
            )
            if config.strict:
                raise CalibrationError(msg)
            logger.warning("%s — returning uncalibrated data", msg)
            return uncalibrated

    # --- Get sensitivity value ---
    sens_col = config.sensitivity_column
    try:
        sens_db = float(cal_df.loc[serial_str, sens_col]) # pyright: ignore[reportArgumentType]
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Could not read {sens_col} for serial {serial_str}: {exc}"
        if config.strict:
            raise CalibrationError(msg) from exc
        logger.warning(msg)
        return uncalibrated

    if pd.isna(sens_db):
        msg = f"{sens_col} for serial {serial_str} is NaN"
        if config.strict:
            raise CalibrationError(msg)
        logger.warning(msg)
        return uncalibrated

    # --- Select the configured conversion method ---
    try:
        method = get_calibration_method(config.method)
    except ValueError as exc:
        msg = str(exc)
        if config.strict:
            raise CalibrationError(msg) from exc
        logger.warning(msg)
        return uncalibrated

    logger.debug(
        "Calibration applied: serial=%s, method=%s, sensitivity=%.1f dB",
        serial_str,
        config.method,
        sens_db,
    )

    return ResolvedCalibration(
        method=method,
        sensitivity_db=sens_db,
        vpp=config.vpp,
        calibrated=True,
    )


def apply_calibration(
    segment: AudioSegment,
    cal_df: Optional[pd.DataFrame],
    config: CalibrationConfig,
) -> tuple[np.ndarray, bool]:
    """
    Convert normalised audio samples to pressure in Pascals.

    Legacy single-call API: equivalent to resolve_calibration() followed
    by ResolvedCalibration.apply(). Allocates a calibrated copy; the
    streaming pipeline uses resolve_calibration() once per file and
    apply_inplace() per block instead.

    Parameters
    ----------
    segment : AudioSegment
        Audio data in normalised float range [-1, 1].
    cal_df : pd.DataFrame or None
        Calibration table from load_calibration().
    config : CalibrationConfig

    Returns
    -------
    tuple of (audio_pa, calibrated)
        audio_pa : np.ndarray — audio in Pascals (or unchanged if uncalibrated)
        calibrated : bool — whether calibration was actually applied

    Raises
    ------
    CalibrationError
        When strict=True and calibration cannot be applied.

    Notes
    -----
    **How hydrophone calibration works (for the bioacoustics newcomer):**

    A hydrophone converts acoustic pressure (in Pascals) into voltage.
    The sensitivity tells you the conversion factor:

        Sensitivity (dB re 1 V/µPa) = 20 x log10(volts_out / pressure_in_µPa)

    For a typical SoundTrap, the sensitivity might be -176 dB re 1 V/µPa.
    This means for 1 µPa of pressure, the output is 10^(-176/20) = 1.58e-9 volts.

    The recording chain is:
        Pressure (Pa) → Hydrophone → Voltage → ADC → Digital samples

    The WAV file contains digital samples normalised to [-1, 1].
    To reverse this chain:

    1. Samples → Volts: multiply by Vpp/2 (half the peak-to-peak voltage)
    2. Volts → µPa: divide by the linear sensitivity (10^(dB/20))
    3. µPa → Pa: multiply by 1e-6

    HOWEVER: SoundTrap calibration values in the "High_Gain" column are
    "end-to-end" sensitivities that already account for the ADC. The
    manufacturer specifies: multiply WAV samples by 10^(cal/20) to get µPa.
    This is a simpler conversion that skips the explicit voltage step.

    Always check your manufacturer's documentation for the exact meaning
    of their calibration values!
    """
    resolved = resolve_calibration(segment, cal_df, config)
    return resolved.apply(segment.data), resolved.calibrated
