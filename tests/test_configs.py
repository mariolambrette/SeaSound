"""
tests/test_config_keys.py

Config-key hygiene tests for ``check_config_keys`` (seasound.core.config).

Unknown keys are silently dropped by ``_build_dataclass``; ``check_config_keys``
surfaces them as warnings, tailors the message for keys that have been removed,
and warns about keys that are inert once streaming is enabled. The function is
pure (returns messages, logs nothing), so these tests assert on the returned
list directly.
"""

from seasound.core.config import check_config_keys


# --- clean configs ---------------------------------------------------------

def test_clean_config_has_no_warnings():
    raw = {
        "input": {"path": "./data/raw/"},
        "pipeline": {"streaming_block_seconds": 60, "max_freq_hz": 48000.0},
        "analyses": {"spectrogram": {"enabled": True}},
    }
    assert check_config_keys(raw) == []


def test_empty_config_has_no_warnings():
    assert check_config_keys({}) == []


# --- unknown keys ----------------------------------------------------------

def test_unknown_key_is_flagged():
    raw = {"pipeline": {"frobnicate": True}}
    warnings = check_config_keys(raw)
    assert len(warnings) == 1
    assert "pipeline.frobnicate" in warnings[0]
    assert "Unknown" in warnings[0]


def test_nested_unknown_in_known_section_flagged():
    raw = {"calibration": {"vpp": 2.0, "bogus_cal_key": 1}}
    warnings = check_config_keys(raw)
    assert len(warnings) == 1
    assert "calibration.bogus_cal_key" in warnings[0]


def test_unknown_top_level_section_flagged_without_descending():
    # An unknown top-level section is reported once; its contents are not
    # walked (no schema to compare against), so no spurious child warnings.
    raw = {"notasection": {"a": 1, "b": 2}}
    warnings = check_config_keys(raw)
    assert len(warnings) == 1
    assert "notasection" in warnings[0]


# --- removed keys (tailored message) --------------------------------------

def test_removed_chunk_duration_gets_tailored_message():
    raw = {"pipeline": {"chunk_duration_s": 300}}
    warnings = check_config_keys(raw)
    assert len(warnings) == 1
    assert "pipeline.chunk_duration_s" in warnings[0]
    # Tailored guidance, not the generic "Unknown ... typo" message.
    assert "Unknown" not in warnings[0]
    assert "streaming_block_seconds" in warnings[0]


# --- inert-under-streaming keys -------------------------------------------

def test_inert_keys_warn_when_streaming_on():
    # streaming_enabled defaults to True, so these are inert and warned.
    raw = {"pipeline": {"stft_cache_enabled": True, "cache_base_matrix": True}}
    warnings = check_config_keys(raw)
    joined = " ".join(warnings)
    assert "pipeline.stft_cache_enabled" in joined
    assert "pipeline.cache_base_matrix" in joined
    assert len(warnings) == 2


def test_inert_keys_silent_when_streaming_off():
    # Under the legacy escape hatch these keys are live again — no warning.
    raw = {
        "pipeline": {
            "streaming_enabled": False,
            "stft_cache_enabled": True,
            "cache_base_matrix": True,
        }
    }
    assert check_config_keys(raw) == []


# --- dict-leaf sections must not be walked --------------------------------

def test_analyses_block_contents_not_flagged():
    # analyses is a free-form dict; its keys are analysis-defined and must
    # not be treated as unknown config keys.
    raw = {
        "analyses": {
            "spectrogram": {"enabled": True, "config": {"cmap": "viridis"}}
        }
    }
    assert check_config_keys(raw) == []


def test_metadata_columns_contents_not_flagged():
    # deployment.metadata_columns is a dict field, not a dataclass section;
    # its user-defined contents are leaves.
    raw = {"deployment": {"metadata_columns": {"location_id": "Loc", "x": "Y"}}}
    assert check_config_keys(raw) == []


# --- nested dataclass section still validated ------------------------------

def test_nested_dataclass_section_is_validated():
    # deployment.buffer_hours IS a dataclass, so its keys are checked.
    raw = {"deployment": {"buffer_hours": {"start": 1.0, "made_up": 9}}}
    warnings = check_config_keys(raw)
    assert len(warnings) == 1
    assert "deployment.buffer_hours.made_up" in warnings[0]
