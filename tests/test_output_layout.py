"""Tests for resolve_spectrogram_output_dir."""

import os

import pytest

from seasound.core.output_layout import resolve_spectrogram_output_dir


class _Entry:
    """Stand-in for one analysis's typed config entry."""

    def __init__(self, enabled: bool, config: dict | None = None):
        self.enabled = enabled
        self.config = config or {}


class FakeAnalyses:
    """Attribute-style analyses container (typed-object shape)."""

    def __init__(self, spec_enabled: bool, ann_enabled: bool):
        self.spectrogram = _Entry(spec_enabled)
        self.event_detection = _Entry(
            enabled=True if ann_enabled else False,
            config={"annotated_spectrogram": {"enabled": ann_enabled}},
        )


class FakePipelineConfig:
    def __init__(self, spec: bool, ann: bool):
        self.analyses = FakeAnalyses(spec, ann)


@pytest.mark.parametrize(
    "spec, ann, which, expected_suffix",
    [
        (True,  False, "raw",       "spectrograms"),
        (False, True,  "annotated", "spectrograms"),
        (True,  True,  "raw",       os.path.join("spectrograms", "raw")),
        (True,  True,  "annotated", os.path.join("spectrograms", "annotated")),
        (False, False, "raw",       "spectrograms"),
    ],
)
def test_resolve(spec, ann, which, expected_suffix, tmp_path):
    cfg = FakePipelineConfig(spec, ann)
    out = resolve_spectrogram_output_dir(str(tmp_path), cfg, which)
    assert out.endswith(expected_suffix)


def test_invalid_which_raises(tmp_path):
    cfg = FakePipelineConfig(True, True)
    with pytest.raises(ValueError, match="must be 'raw' or 'annotated'"):
        resolve_spectrogram_output_dir(str(tmp_path), cfg, "bogus")


def test_no_pipeline_config_returns_flat_layout(tmp_path):
    out = resolve_spectrogram_output_dir(str(tmp_path), None, "raw")
    assert out == os.path.join(str(tmp_path), "spectrograms")


def test_dict_shaped_analyses_works(tmp_path):
    """Dict-of-dicts analyses container works the same as the typed shape."""
    class CfgWithDict:
        analyses = {
            "spectrogram": {"enabled": True},
            "event_detection": {
                "enabled": True,
                "config": {"annotated_spectrogram": {"enabled": True}},
            },
        }
    out = resolve_spectrogram_output_dir(
        str(tmp_path), CfgWithDict(), "annotated",
    )
    assert out.endswith(os.path.join("spectrograms", "annotated"))
