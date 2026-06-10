"""
seasound/loader/calibration_methods.py

Calibration conversion methods for different hydrophone types.

Each method converts normalised WAV samples [-1, 1] to Pascals using
the hydrophone's sensitivity value and (optionally) the recording
system's peak-to-peak voltage.

To add a new method:
    1. Create a subclass of CalibrationMethod
    2. Implement to_pascals(), and — only if the method is a per-sample
       scalar operation — calibrate_inplace() with the SAME floating-point
       operation order. Methods that are not per-sample scalar (e.g.
       frequency-dependent calibration) must NOT implement
       calibrate_inplace(); the base-class implementation raises so the
       streaming pipeline fails loudly instead of silently mis-calibrating.
    3. Add it to CALIBRATION_METHOD_REGISTRY
    4. Add the method name to the valid_cal_methods set in config.py validate()
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class CalibrationMethod(ABC):
    """
    Base class for calibration conversion methods.

    Each subclass implements a specific relationship between
    WAV sample values, hydrophone sensitivity, and acoustic pressure.
    The exact relationship depends on the manufacturer's convention
    for what the sensitivity value represents.
    """
    name: str

    @abstractmethod
    def to_pascals(
        self,
        samples: np.ndarray,
        sensitivity_db: float,
        vpp: float,
    ) -> np.ndarray:
        """
        Convert normalised audio samples to pressure in Pascals.

        Parameters
        ----------
        samples : np.ndarray
            Audio samples in range [-1, 1] from soundfile.
        sensitivity_db : float
            Sensitivity value from the calibration spreadsheet.
            The interpretation depends on the method.
        vpp : float
            Peak-to-peak voltage of the recording system.
            Not used by all methods.

        Returns
        -------
        np.ndarray
            Audio data in Pascals.
        """

    def calibrate_inplace(
        self,
        block: np.ndarray,
        sensitivity_db: float,
        vpp: float,
    ) -> np.ndarray:
        """
        Calibrate one audio block in place and return it.

        Streaming counterpart of to_pascals(): mutates ``block`` instead
        of allocating a calibrated copy, applying the SAME floating-point
        operations in the SAME order so the result is bit-identical to
        ``to_pascals(block, ...)``. Do NOT collapse the per-step scalars
        into one pre-combined gain — float multiplication is not
        associative, and a single combined multiply diverges from the
        legacy result in the last bit for a substantial fraction of
        samples, breaking the refactor's bit-identity gates.

        The base implementation raises: any method that cannot be
        expressed as per-sample scalar arithmetic (e.g. a future
        frequency-dependent calibration, which must be applied in the
        spectral domain or via filtering) must fail loudly here rather
        than be silently mis-applied per block.
        """
        raise NotImplementedError(
            f"Calibration method '{self.name}' does not support in-place "
            f"per-block application. Streaming requires per-sample scalar "
            f"calibration; non-scalar methods must be applied in the "
            f"spectral domain and are not yet supported by the streaming "
            f"pipeline."
        )



class SoundTrapMethod(CalibrationMethod):
    """
    OceanInstruments SoundTrap end-to-end calibration.

    SoundTrap convention: the calibration value is an end-to-end
    sensitivity that accounts for the entire signal chain (hydrophone
    + preamp + ADC). The conversion is:

        µPa = samples x 10^(sensitivity_dB / 20)
        Pa  = µPa x 1e-6

    The Vpp parameter is not used because it's already incorporated
    into the end-to-end sensitivity.

    This method applies to SoundTrap ST300, ST400, ST500, and ST600
    when using the manufacturer-provided calibration values.
    """
    name = "soundtrap"

    def to_pascals(self, samples, sensitivity_db, vpp):
        cal_linear = 10.0 ** (sensitivity_db / 20.0)
        pressure_upa = samples * cal_linear
        return pressure_upa * 1e-6

    def calibrate_inplace(self, block, sensitivity_db, vpp):
        # Same operations, same order as to_pascals (bit-identity).
        cal_linear = 10.0 ** (sensitivity_db / 20.0)
        block *= cal_linear
        block *= 1e-6
        return block


class StandardMethod(CalibrationMethod):
    """
    Standard receive sensitivity calibration (dB re 1 V/µPa).

    Used by many hydrophone manufacturers (Reson/Teledyne, Brüel & Kjær,
    HTI, etc.) where the sensitivity specifies the voltage output per
    unit acoustic pressure:

        Sensitivity (dB re 1 V/µPa) = 20 × log10(V_out / P_in_µPa)

    Typical values: -170 to -220 dB re 1 V/µPa.

    Conversion chain:
        1. Samples → Volts:   V = samples × (Vpp / 2)
        2. Volts → µPa:       µPa = V / sensitivity_linear
           where sensitivity_linear = 10^(sensitivity_dB / 20)
        3. µPa → Pa:          Pa = µPa × 1e-6

    The Vpp parameter IS required for this method.
    """
    name = "standard"

    def to_pascals(self, samples, sensitivity_db, vpp):
        volts = samples * (vpp / 2.0)
        sensitivity_linear = 10.0 ** (sensitivity_db / 20.0)
        pressure_upa = volts / sensitivity_linear
        return pressure_upa * 1e-6

    def calibrate_inplace(self, block, sensitivity_db, vpp):
        # Same operations, same order as to_pascals (bit-identity).
        block *= (vpp / 2.0)
        sensitivity_linear = 10.0 ** (sensitivity_db / 20.0)
        block /= sensitivity_linear
        block *= 1e-6
        return block


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CALIBRATION_METHOD_REGISTRY: dict[str, type[CalibrationMethod]] = {
    "soundtrap": SoundTrapMethod,
    "standard": StandardMethod,
}


def get_calibration_method(method_name: str) -> CalibrationMethod:
    """
    Instantiate a calibration method by name.

    Parameters
    ----------
    method_name : str
        Key in CALIBRATION_METHOD_REGISTRY.

    Returns
    -------
    CalibrationMethod

    Raises
    ------
    ValueError
        If method_name is not recognised.
    """
    cls = CALIBRATION_METHOD_REGISTRY.get(method_name)
    if cls is None:
        available = ", ".join(sorted(CALIBRATION_METHOD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown calibration method '{method_name}'. "
            f"Available: {available}"
        )
    return cls()
