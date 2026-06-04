"""
seasound/utils/spectral.py

Third-octave band frequency calculations following ISO 266 and IEC 61260.

These are the shared building blocks used by both the loading engine
(to compute the base matrix) and analysis modules (to interpret it).
"""

import numpy as np

# ISO 266 standard third-octave band centre frequencies (Hz)
# This is the canonical list up to 50 kHz
ISO_266_TOB_CENTRES = np.array([
    10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000, 25000, 31500, 40000, 50000,
], dtype=float)

def tob_centre_frequencies(
    min_freq: float = 10.0,
    max_freq: float = 50000.0,
) -> np.ndarray:
    """
    Return ISO 266 third-octave band centre frequencies within [min, max] Hz.

    Parameters
    ----------
    min_freq : float
        Lower frequency limit (inclusive).
    max_freq : float
        Upper frequency limit (inclusive).

    Returns
    -------
    np.ndarray
        1-D array of centre frequencies in Hz.

    Examples
    --------
    >>> tob_centre_frequencies(100, 1000)
    array([ 100.,  125.,  160.,  200.,  250.,  315.,  400.,  500.,  630.,  800., 1000.])
    """
    mask = (ISO_266_TOB_CENTRES >= min_freq) & (ISO_266_TOB_CENTRES <= max_freq)
    return ISO_266_TOB_CENTRES[mask].copy()


def tob_band_edges(centres: np.ndarray) -> np.ndarray:
    """
    Compute IEC 61260 third-octave band edges.

    For a centre frequency fc, the band spans:
        f_lower = fc × 10^(-1/20) ≈ fc × 0.8913
        f_upper = fc × 10^(+1/20) ≈ fc × 1.1220

    These are the "exact" fractional-octave band edges per IEC 61260-1:2014.

    Parameters
    ----------
    centres : np.ndarray
        Array of centre frequencies in Hz.

    Returns
    -------
    np.ndarray
        Shape (n_bands, 2) with columns [f_lower, f_upper].

    Notes
    -------------
    A third-octave band spans a frequency ratio of 2^(1/3) ≈ 1.2599. The centre 
    of this ratio on a log scale gives:
        f_lower = fc / 2^(1/6) = fc × 10^(-1/20)
        f_upper = fc × 2^(1/6) = fc × 10^(1/20)
    
    The relationship 10^(1/20) = 2^(1/6) is approximate (they differ by ~0.04%)
    but the 10^(1/20) form is specified in IEC 61260 and used by JOMOPANS.
    """
    edges = np.zeros((len(centres), 2))
    edges[:, 0] = centres * 10 ** (-1 / 20)  # lower edge
    edges[:, 1] = centres * 10 ** (1 / 20)   # upper edge
    return edges


def nearest_band_index(target_hz: float, centres: np.ndarray) -> int:
    """
    Index of the TOB band whose centre is closest to target_hz.

    Parameters
    ----------
    target_hz : float
        Target frequency in Hz.
    centres : np.ndarray
        Array of TOB centre frequencies.

    Returns
    -------
    int
        Index into centres.
    """
    return int(np.argmin(np.abs(centres - target_hz)))


def extract_freq_hz(columns: list[str]) -> np.ndarray:
    """
    Parse numeric Hz values from column names like '63.0Hz'.

    Parameters
    ----------
    columns : list of str
        Column names from the base matrix DataFrame.

    Returns
    -------
    np.ndarray
        Numeric frequencies in Hz.
    """
    return np.array([float(c.replace("Hz", "")) for c in columns])


def freq_column_names(centres: np.ndarray) -> list[str]:
    """
    Convert centre frequencies to column name strings.

    Parameters
    ----------
    centres : np.ndarray
        TOB centre frequencies in Hz.

    Returns
    -------
    list of str
        Column names like ['10.0Hz', '12.5Hz', ...].
    """
    return [f"{f:.1f}Hz" for f in centres]