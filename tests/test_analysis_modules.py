"""Unit tests for Phase 2 analysis modules."""

import os

import math
import numbers
import numpy as np
import pandas as pd
import pytest

from seasound.analysis.base import AnalysisModuleError
from seasound.analysis.ltsa import LTSAAnalysis
from seasound.analysis.registry import get_analysis, list_registered
from seasound.analysis.spectral_percentiles import SpectralPercentilesAnalysis
from seasound.analysis.spectrogram import SpectrogramAnalysis
from seasound.analysis.tob_levels import TOBLevelsAnalysis


def _seed_stft_store(wav_path, test_config, cache_dir):
    """Compute the real STFT for a WAV and write it into the shard store.

    Store-backed analog of the old ``input_files``/npz path: the
    spectrogram (refactor §8) now reads STFT from ``cache_dir/stft``
    shards rather than computing it on the fly, so tests that exercise a
    rendered spectrogram must seed the store first. This reuses the
    production STFT computation (``get_stft_for_file``) and the
    production shard writer, so the seeded data matches what the
    streaming loader would have written.
    """
    import soundfile as sf
    from seasound.analysis.calculate_stft import get_stft_for_file
    from seasound.loader.stft_store import (
        StftShardWriter, stft_dir_for, shard_name,
    )

    entries = get_stft_for_file(wav_path, test_config, cache_dir)
    sample_rate = sf.info(wav_path).samplerate
    hop = test_config.pipeline.stft_hop_length
    win = test_config.pipeline.stft_win_length
    basename = os.path.basename(wav_path)
    for entry in entries:
        shard_path = os.path.join(
            stft_dir_for(cache_dir), shard_name(basename, entry["channel"]),
        )
        writer = StftShardWriter(
            shard_path, entry["freqs_hz"], sample_rate, hop, win,
            entry["datetime_start"], channel=entry["channel"],
            serial=entry["serial"],
        )
        writer.append(entry["power"])
        writer.finalise()


class TestLTSAAnalysis:
    """Tests for LTSA module."""

    def test_ltsa_config_valid(self):
        module = LTSAAnalysis()
        cfg = {
            "time_resolution": "1h",
            "statistic": "median",
        }
        module.validate_config(cfg)  # Should not raise

    def test_ltsa_config_missing_time_resolution(self):
        module = LTSAAnalysis()
        cfg = {"statistic": "median"}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_ltsa_config_invalid_time_resolution_alias(self):
        module = LTSAAnalysis()
        cfg = {"time_resolution": "not_an_alias", "statistic": "median"}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_ltsa_config_invalid_statistic(self):
        module = LTSAAnalysis()
        cfg = {
            "time_resolution": "1h",
            "statistic": "invalid",
        }
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_ltsa_run_creates_output(self, synthetic_base_matrix, tmp_path):
        module = LTSAAnalysis()
        cfg = {
            "time_resolution": "6h",
            "statistic": "median",
        }
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        assert result.name == "ltsa"
        assert len(result.outputs) == 1
        assert os.path.isfile(result.outputs[0])
        assert "ltsa.csv" in result.outputs[0]

    def test_ltsa_run_respects_freq_range(self, synthetic_base_matrix, tmp_path):
        module = LTSAAnalysis()
        cfg = {
            "time_resolution": "1h",
            "statistic": "median",
            "freq_range": [500, 10000],
        }
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        output_df = pd.read_csv(result.outputs[0], index_col=0)
        for col in output_df.columns:
            if col.endswith("Hz"):
                freq = float(col[:-2])
                assert 500 <= freq <= 10000

    def test_ltsa_empty_matrix_raises(self, tmp_path):
        module = LTSAAnalysis()
        empty_matrix = pd.DataFrame()
        cfg = {"time_resolution": "1h", "statistic": "median"}

        with pytest.raises(AnalysisModuleError):
            module.run(empty_matrix, cfg, str(tmp_path))

    def test_ltsa_mean_uses_linear_power_not_db_average(
        self,
        synthetic_base_matrix,
        tmp_path,
    ):
        module = LTSAAnalysis()
        cfg = {"time_resolution": "1h", "statistic": "mean"}
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))
        out = pd.read_csv(result.outputs[0], index_col=0)

        raw_value = out.iat[0, 0]
        assert isinstance(raw_value, numbers.Real), f"Unexpected type: {type(raw_value)}"
        observed = float(raw_value)

        mean_linear = (10.0 ** (80.0 / 10.0) + 10.0 ** (90.0 / 10.0)) / 2.0
        expected = 10.0 * math.log10(mean_linear)

        assert observed == pytest.approx(expected, abs=1e-3)
        assert observed != pytest.approx(85.0, abs=1e-6)

    def test_ltsa_with_plot_enabled_produces_png(
        self, synthetic_base_matrix, tmp_path,
    ):
        module = LTSAAnalysis()
        cfg = {
            "time_resolution": "10min",
            "statistic": "median",
            "plot": {
                "enabled": True,
                "types": ["heatmap"],
                "heatmap": {"db_range": [60, 100]},
                "output_format": "png",
                "dpi": 100,
            },
        }
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))
        csvs = [p for p in result.outputs if p.endswith(".csv")]
        pngs = [p for p in result.outputs if p.endswith(".png")]
        assert len(csvs) == 1
        assert len(pngs) == 1
        assert result.summary["plots_generated"] == 1


class TestTOBLevelsAnalysis:
    """Tests for TOB Levels module."""

    def test_tob_levels_config_valid(self):
        module = TOBLevelsAnalysis()
        cfg = {"statistics": ["median", "p95"]}
        module.validate_config(cfg)  # Should not raise

    def test_tob_levels_config_missing_statistics(self):
        module = TOBLevelsAnalysis()
        cfg = {}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_tob_levels_config_invalid_stat_name(self):
        module = TOBLevelsAnalysis()
        cfg = {"statistics": ["bogus"]}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_tob_levels_config_invalid_window_s(self):
        module = TOBLevelsAnalysis()
        cfg = {"statistics": ["median"], "window_s": 0}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_tob_levels_run_creates_output(self, synthetic_base_matrix, tmp_path):
        module = TOBLevelsAnalysis()
        cfg = {"statistics": ["median", "mean", "p95"]}
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        assert result.name == "tob_levels"
        assert len(result.outputs) == 1
        assert os.path.isfile(result.outputs[0])

        output_df = pd.read_csv(result.outputs[0], index_col=0)
        assert len(output_df.columns) == 3  # median, mean, p95

    def test_tob_levels_windowed_output_schema(self, synthetic_base_matrix, tmp_path):
        module = TOBLevelsAnalysis()
        cfg = {"statistics": ["median"], "window_s": 600}  # 10 min
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        df = pd.read_csv(result.outputs[0])
        assert "window_start" in df.columns
        assert "window_end" in df.columns
        assert "frequency_band" in df.columns
        assert "median" in df.columns

        n_windows = 3600 // 600  # synthetic_base_matrix is 1 hour
        n_freq = len(synthetic_base_matrix.columns)
        assert len(df) == n_windows * n_freq


class TestSpectralPercentilesAnalysis:
    """Tests for Spectral Percentiles module."""

    def test_spectral_percentiles_config_valid(self):
        module = SpectralPercentilesAnalysis()
        cfg = {"percentiles": [5, 50, 95]}
        module.validate_config(cfg)  # Should not raise

    def test_spectral_percentiles_config_missing_percentiles(self):
        module = SpectralPercentilesAnalysis()
        cfg = {}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_spectral_percentiles_config_invalid_percentile_range(self):
        module = SpectralPercentilesAnalysis()
        cfg = {"percentiles": [150]}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_spectral_percentiles_invalid_window_alias(self):
        module = SpectralPercentilesAnalysis()
        cfg = {"percentiles": [5, 50, 95], "window": "bad_alias"}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_spectral_percentiles_run_full_window(self, synthetic_base_matrix, tmp_path):
        module = SpectralPercentilesAnalysis()
        cfg = {"percentiles": [5, 50, 95], "window": "full"}
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        assert result.name == "spectral_percentiles"
        assert len(result.outputs) == 1
        assert os.path.isfile(result.outputs[0])

        df = pd.read_csv(result.outputs[0])
        assert len(df) == 1
        assert "window_start" not in df.columns
        assert "window_end" not in df.columns

    def test_spectral_percentiles_run_windowed(self, synthetic_base_matrix, tmp_path):
        module = SpectralPercentilesAnalysis()
        cfg = {"percentiles": [5, 50, 95], "window": "10min"}
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        df = pd.read_csv(result.outputs[0])
        assert "window_start" in df.columns
        assert "window_end" in df.columns
        assert len(df) == 6  # 1 hour / 10 min

    def test_spectral_percentiles_with_plot_enabled_produces_png(
        self, synthetic_base_matrix, tmp_path,
    ):
        module = SpectralPercentilesAnalysis()
        cfg = {
            "percentiles": [5, 50, 95],
            "window": "full",
            "plot": {
                "enabled": True,
                "types": ["curves"],
                "curves": {"shaded_band": True},
                "output_format": "png",
                "dpi": 100,
            },
        }
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))
        pngs = [p for p in result.outputs if p.endswith(".png")]
        assert len(pngs) == 1


class TestSpectrogramAnalysis:
    """Tests for Spectrogram module."""

    def test_spectrogram_config_valid(self):
        module = SpectrogramAnalysis()
        cfg = {"output_format": "csv"}
        module.validate_config(cfg)  # Should not raise

    def test_spectrogram_config_invalid_format(self):
        module = SpectrogramAnalysis()
        cfg = {"output_format": "invalid"}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_spectrogram_config_invalid_db_range(self):
        module = SpectrogramAnalysis()
        cfg = {"output_format": "csv", "db_range": [120, 60]}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_spectrogram_config_invalid_time_chunk(self):
        module = SpectrogramAnalysis()
        cfg = {"output_format": "csv", "time_chunk": "not-a-frequency"}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_spectrogram_config_invalid_preserve_time_gaps_type(self):
        module = SpectrogramAnalysis()
        cfg = {"output_format": "csv", "preserve_time_gaps": "yes"}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_spectrogram_config_invalid_time_bins(self):
        module = SpectrogramAnalysis()
        cfg = {"output_format": "png", "time_bins": 0}
        with pytest.raises(ValueError):
            module.validate_config(cfg)

    def test_spectrogram_run_csv_format(
        self, synthetic_base_matrix, synthetic_wav, test_config, tmp_path
    ):
        module = SpectrogramAnalysis()
        _seed_stft_store(synthetic_wav, test_config, str(tmp_path))
        module.set_runtime_context(
            {
                "pipeline_config": test_config,
                "cache_dir": str(tmp_path),
            }
        )
        cfg = {"output_format": "csv"}
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        assert result.name == "spectrogram"
        assert len(result.outputs) == 1
        assert "spectrogram.csv" in result.outputs[0]
        assert os.path.isfile(result.outputs[0])
        assert result.summary["data_source"] == "STFT-derived matrix"

    def test_spectrogram_run_csv_chunked(
        self, synthetic_base_matrix, synthetic_wav, test_config, tmp_path
    ):
        module = SpectrogramAnalysis()
        _seed_stft_store(synthetic_wav, test_config, str(tmp_path))
        module.set_runtime_context(
            {
                "pipeline_config": test_config,
                "cache_dir": str(tmp_path),
            }
        )
        cfg = {"output_format": "csv", "time_chunk": "5s"}
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        assert len(result.outputs) >= 2
        assert all("spectrogram_" in p for p in result.outputs)
        assert all(os.path.isfile(p) for p in result.outputs)
        assert result.summary["data_source"] == "STFT-derived matrix"

    def test_spectrogram_run_png_format(
        self, synthetic_base_matrix, synthetic_wav, test_config, tmp_path
    ):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")

        module = SpectrogramAnalysis()
        _seed_stft_store(synthetic_wav, test_config, str(tmp_path))
        module.set_runtime_context(
            {
                "pipeline_config": test_config,
                "cache_dir": str(tmp_path),
            }
        )
        cfg = {"output_format": "png", "dpi": 150}
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        assert "spectrogram.png" in result.outputs[0]
        assert os.path.isfile(result.outputs[0])
        assert result.summary["data_source"] == "STFT-derived matrix"

    def test_spectrogram_run_png_preserve_time_gaps_summary(
        self, synthetic_base_matrix, synthetic_wav, test_config, tmp_path
    ):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")

        module = SpectrogramAnalysis()
        _seed_stft_store(synthetic_wav, test_config, str(tmp_path))
        module.set_runtime_context(
            {
                "pipeline_config": test_config,
                "cache_dir": str(tmp_path),
            }
        )
        cfg = {"output_format": "png", "preserve_time_gaps": True}
        result = module.run(synthetic_base_matrix, cfg, str(tmp_path))

        assert result.summary["preserve_time_gaps"] is True
        assert result.summary["missing_seconds_visualized"] >= 0

    def test_spectrogram_run_raises_without_stft_context(
        self, synthetic_base_matrix, tmp_path
    ):
        module = SpectrogramAnalysis()
        cfg = {"output_format": "csv"}
        with pytest.raises(AnalysisModuleError):
            module.run(synthetic_base_matrix, cfg, str(tmp_path))


class TestAnalysisRegistry:
    """Tests for analysis module registry."""

    def test_list_registered_contains_all_modules(self):
        modules = list_registered()
        expected = {
            "ltsa",
            "tob_levels",
            "spectral_percentiles",
            "spectrogram",
            "event_detection",
        }
        assert set(modules.keys()) == expected

    def test_get_analysis_ltsa(self):
        module = get_analysis("ltsa")
        assert isinstance(module, LTSAAnalysis)

    def test_get_analysis_unknown_raises(self):
        with pytest.raises(ValueError):
            get_analysis("nonexistent_module")
