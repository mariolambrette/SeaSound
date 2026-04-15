"""
SeaSound analysis package.

This package is auto-initialised to register all built-in analysis modules on
import. User code should use:

    from seasound.analysis.registry import get_analysis
    module = get_analysis("ltsa")

Built-in modules:
- ltsa: Long-Term spectral average
- tob_levels: TOB-resolution summary statistics
- spectral_percentiles: Per-frequency percentile distributions
-spectrogram: Time frequency visualisation.
"""

# Import built-in analysis modules to trigger registration side effects.
# Order does not matter; each module calls register_analysis() on import.
from seasound.analysis import ltsa
from seasound.analysis import tob_levels
from seasound.analysis import spectral_percentiles
from seasound.analysis import spectrogram

__all__ = [
    "ltsa",
    "tob_levels",
    "spectral_percentiles",
    "spectrogram",
]
