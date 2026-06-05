"""
seasound/core/config.py

Defined Config dataclasses which are used by the pipeline. Handles the loading,
validation, and CLI merging of all configurable parameters. The default
configuration is stored in config/default_config.yaml.

Configuration validation occurs in three stages:
    1. load_yaml()    - Parse the YAML file into a dict
    2. merge_cli()    - Merge the CLI overrides
    3. validate()     - typecheck, range-check and return the final
                          PipelineConfig dataclass instance.

Validation errors raise ConfigError with a human-readable message
listing all problems (not just the first one found).
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import yaml

from seasound.core.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Dataclass definitions — one per config section
# ---------------------------------------------------------------------------


@dataclass
class InputConfig:
    """Configuration for audio file discovery and reading."""
    path: str = "./data/raw/"
    pattern: str = "*.wav"
    recursive: bool = True
    filename_format: str = "soundtrap"
    channel_strategy: str = "mono"
    selected_channel: int = 0
    custom_regex: Optional[str] = None
    custom_datetime_format: Optional[str] = None
    serial_override: Optional[str] = None
    start_datetime: Optional[str] = None
    per_file_trim_start_s: float = 3.0


@dataclass
class CalibrationConfig:
    """Configuration for hydrophone calibration."""
    enabled: bool = True
    strict: bool = True
    file: str = "./data/SoundTrapCalibration.xlsx"
    serial_column: str = "Serial"
    sensitivity_column: str = "High_Gain"
    method: str = "soundtrap"
    vpp: float = 2.0
    sensitivity_db_override: Optional[float] = None


@dataclass
class DeploymentBufferConfig:
    """Shared clipping buffer in hours, regardless of the clip source."""
    start: float = 0.0
    end: float = 0.0


@dataclass
class DeploymentConfig:
    """
    Configuration for trimming deployment start/end.

    Three clipping methods:
        1. "none" - default, do not explicity clip data, use the full recording.
        2. "manual" - explicit user-specified start and end datetimes (UTC)
            specified in the config file.
        3. "metadata" - Look up deployment start and end times based on a
            metadata table (see config/deployment_metadata_format.csv for an
            example).
    """
    enabled: bool = False
    clip_method: str = "none"

    # --- manual method ---
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None

    # --- auto method ---
    buffer_hours: DeploymentBufferConfig = field(
        default_factory=DeploymentBufferConfig
    )

    # --- metadata method ---
    metadata_file: str = ""
    metadata_format: str = "seasound"
    metadata_columns: dict = field(
        default_factory=lambda: {
            "location_id": "Location_ID",
            "hydrophone": "Hydrophone",
            "deploy": "DateTime_deploy_UTC",
            "retrieve": "DateTime_retrieve_UTC",
        }
    )
    location_id: str = ""
    hydrophone: str = ""


@dataclass
class OutputConfig:
    """Where and how outputs are written."""
    directory: str = "./output/"
    overwrite: bool = False
    naming: str = "{location_id}_{hydrophone}_{analysis}_{params}"


@dataclass
class ProcessingConfig:
    """Core pipeline processing parameters."""
    resume: bool = True
    workers: int = 0
    max_freq_hz: float = 50000.0
    min_freq_hz: float = 10.0
    base_resolution_s: int = 1
    reference_pressure_pa: float = 1e-6
    domain: str = "underwater"
    missing_band_strategy: str = "nan"
    # Base-matrix numeric levers. Defaults reproduce the float32 golden
    # baseline; DO NOT CHANGE either if outputs must stay comparable
    # across runs and deployments (both alter the numerics).
    nfft_padding_factor: int = 4   # JOMOPANS 4x zero-padding
    sxx_dtype: str = "float32"     # "float32" | "float64"
    cache_base_matrix: bool = True
    cache_directory: Optional[str] = None

    # Streaming Stage-1 path (refactor plan Stage 2). Transitional
    # dual-path flag: False keeps the legacy full-read path; True
    # streams whole-bin blocks (bit-identical output, bounded memory).
    # The flag and the legacy path are removed together at Stage 6.
    streaming_enabled: bool = True
    streaming_block_seconds: int = 60

    # Optional linear STFT support
    stft_cache_enabled: bool = False
    stft_nfft: int = 2048
    stft_win_length: int = 2048
    stft_hop_length: int = 1024
    stft_window: str = "hann"
    stft_fmin_hz: float = 10.0
    stft_fmax_hz: float = 50000.0
    stft_dtype: str = "float32"    # "float32" | "float16"


@dataclass
class PipelineConfig:
    """Top-level configuration container."""
    input: InputConfig = field(default_factory=InputConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pipeline: ProcessingConfig = field(default_factory=ProcessingConfig)
    analyses: dict = field(default_factory=dict)

    # Runtime flags (set by CLI, not YAML)
    load_only: bool = False
    analyse_only: bool = False
    dry_run: bool = False
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    """Parse a YAML config file into a raw dict."""
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    try:
        with open(path, "r") as f: #pylint: disable=unspecified-encoding
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if raw is None:
        raw = {}
    return raw


def merge_cli(base: dict, overrides: dict) -> dict:
    """
    Deep-merge CLI overrides in the YAML dict.

    CLI keys use dot notation mapped to the nested keys:
        "input.path" -> base["input"]["path"]

    Parameters
    ----------
    base : dict
        Parsed YAML config.
    overrides : dict
        CLI overrides as flat dot-notation keys.
        Example: {"input.path": "/data/wav", "pipeline.workers": 4}

    Returns
    -------
    dict
        Merged config dict.
    """
    if not overrides:
        return base

    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        d = base
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    return base


def _build_dataclass(cls, data: dict):
    """
    Recursively build a dataclass from a dict, ignoring unknown keys.
    Handles nested dataclasses by inspecting field types.
    """
    import dataclasses

    if not isinstance(data, dict):
        return data

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]

        # Check if the field type is itself a dataclass
        if dataclasses.is_dataclass(f.type):
            val = _build_dataclass(f.type, val if isinstance(val, dict) else {})

        kwargs[f.name] = val

    return cls(**kwargs)


def validate(raw: dict) -> PipelineConfig:
    """
    Convert raw dict to PipelineConfig, checking constraints.

    Collects all errors before raising so the user sees everything
    they need to fix in one pass.
    """
    errors = []

    # Build the typed config
    config = _build_dataclass(PipelineConfig, raw)

    # --- Input validation ---
    if not os.path.isdir(config.input.path) and not config.analyse_only:
        errors.append(
            f"input.path '{config.input.path}' is not a directory. "
            f"Create it or set the correct path."
        )

    valid_formats = {"soundtrap", "wildlife", "iclisten", "custom", "manual"}
    if config.input.filename_format not in valid_formats:
        errors.append(
            f"input.filename_format '{config.input.filename_format}' "
            f"must be one of: {', '.join(sorted(valid_formats))}"
        )

    if config.input.filename_format == "custom":
        if not config.input.custom_regex:
            errors.append(
                "input.custom_regex is required when "
                "filename_format is 'custom'"
            )
        if not config.input.custom_datetime_format:
            errors.append(
                "input.custom_datetime_format is required when "
                "filename_format is 'custom'"
            )

    valid_channels = {"mono", "auto", "select", "dual_gain"}
    if config.input.channel_strategy not in valid_channels:
        errors.append(
            f"input.channel_strategy '{config.input.channel_strategy}' "
            f"must be one of: {', '.join(sorted(valid_channels))}"
        )

    trim_s = config.input.per_file_trim_start_s
    if not isinstance(trim_s, (int, float)) or trim_s < 0:
        errors.append(
            f"input.per_file_trim_start_s must be a non-negative number; "
            f"got {trim_s}"
        )

    # --- STFT validation ---
    if config.pipeline.stft_win_length > config.pipeline.stft_nfft:
        errors.append("pipeline.stft_win_length must be <= pipeline.stft_nfft")

    if config.pipeline.stft_hop_length > config.pipeline.stft_win_length:
        errors.append("pipeline.stft_hop_length must be <= pipeline.stft_win_length")

    if config.pipeline.stft_fmin_hz >= config.pipeline.stft_fmax_hz:
        errors.append("pipeline.stft_fmin_hz must be < pipeline.stft_fmax_hz")

    # --- Calibration validation ---
    if (
        config.calibration.enabled
        and config.calibration.strict
        and config.calibration.sensitivity_db_override is None
    ):
        if not os.path.isfile(config.calibration.file) and not config.analyse_only:
            errors.append(
                f"calibration.file '{config.calibration.file}' not found. "
                f"Provide the file or set calibration.strict: false"
            )

    if config.calibration.vpp <= 0:
        errors.append("calibration.vpp must be positive")

    valid_cal_methods = {"soundtrap", "standard"}
    if config.calibration.method not in valid_cal_methods:
        errors.append(
            f"calibration.method '{config.calibration.method}' "
            f"must be one of: {', '.join(sorted(valid_cal_methods))}"
        )

    # --- Pipeline validation ---
    if config.pipeline.base_resolution_s < 1:
        errors.append("pipeline.base_resolution_s must be >= 1")

    if config.pipeline.reference_pressure_pa <= 0:
        errors.append("pipeline.reference_pressure_pa must be positive")

    valid_strategies = {"nan", "clip", "error"}
    if config.pipeline.missing_band_strategy not in valid_strategies:
        errors.append(
            f"pipeline.missing_band_strategy must be one of: "
            f"{', '.join(sorted(valid_strategies))}"
        )

    if config.pipeline.max_freq_hz <= config.pipeline.min_freq_hz:
        errors.append(
            "pipeline.max_freq_hz must be greater than pipeline.min_freq_hz"
        )

    if config.pipeline.nfft_padding_factor < 1:
        errors.append("pipeline.nfft_padding_factor must be >= 1")

    valid_sxx_dtypes = {"float32", "float64"}
    if config.pipeline.sxx_dtype not in valid_sxx_dtypes:
        errors.append(
            f"pipeline.sxx_dtype must be one of: "
            f"{', '.join(sorted(valid_sxx_dtypes))}"
        )

    bs = config.pipeline.streaming_block_seconds
    res = config.pipeline.base_resolution_s
    if not isinstance(bs, int) or bs < 1:
        errors.append(
            f"pipeline.streaming_block_seconds must be a positive integer; "
            f"got {bs}"
        )
    elif isinstance(res, int) and res >= 1 and bs % res != 0:
        errors.append(
            f"pipeline.streaming_block_seconds ({bs}) must be a whole "
            f"multiple of pipeline.base_resolution_s ({res})"
        )

    # --- Deployment validation ---
    if config.deployment.enabled:
        valid_clip_methods = {"none", "manual", "metadata"}
        method = config.deployment.clip_method
        if method not in valid_clip_methods:
            errors.append(
                f"deployment.clip_method '{method}' must be one of: "
                f"{', '.join(sorted(valid_clip_methods))}"
            )

        if method == "manual":
            if not config.deployment.start_utc:
                errors.append(
                    "deployment.start_utc is required when "
                    "clip_method is 'manual'"
                )
            if not config.deployment.end_utc:
                errors.append(
                    "deployment.end_utc is required when "
                    "clip_method is 'manual'"
                )

        if method == "metadata":
            if not config.deployment.metadata_file:
                errors.append(
                    "deployment.metadata_file is required when "
                    "clip_method is 'metadata'"
                )
            if not config.deployment.location_id:
                errors.append(
                    "deployment.location_id is required when "
                    "clip_method is 'metadata'"
                )

        if config.deployment.buffer_hours.start < 0:
            errors.append("deployment.buffer_hours.start must be >= 0")
        if config.deployment.buffer_hours.end < 0:
            errors.append("deployment.buffer_hours.end must be >= 0")

    # --- Raise all errors at once ---
    if errors:
        msg = "Configuration errors:\n" + "\n".join(f"  • {e}" for e in errors)
        raise ConfigError(msg)

    return config


def load_config(
    config_path: str,
    cli_overrides: Optional[dict] = None,
) -> PipelineConfig:
    """
    Public API: load YAML, merge CLI overrides, validate, return typed config.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    cli_overrides : dict, optional
        Flat dot-notation overrides from CLI.

    Returns
    -------
    PipelineConfig
    """
    raw = load_yaml(config_path)
    raw = merge_cli(raw, cli_overrides or {})
    return validate(raw)
