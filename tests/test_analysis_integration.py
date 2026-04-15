"""Integration tests for Phase 2 analysis pipeline."""

import json
import os

import pandas as pd
import pytest

from seasound.core.pipeline import run_pipeline, run_analyses
from seasound.core.config import PipelineConfig


class TestAnalysisDispatch:
    """Tests for analysis dispatch and error handling."""
    
    def test_run_analyses_enabled_modules(self, test_config, synthetic_base_matrix):
        """Test that enabled analyses are executed."""
        test_config.analyses = {
            "ltsa": {
                "enabled": True,
                "required": True,
                "config": {
                    "time_resolution": "6h",
                    "statistic": "median",
                },
            },
            "tob_levels": {
                "enabled": True,
                "required": True,
                "config": {
                    "statistics": ["median"],
                },
            },
        }
        
        results = run_analyses(synthetic_base_matrix, test_config)
        
        assert "ltsa" in results
        assert "tob_levels" in results
        assert len(results["ltsa"]["outputs"]) >= 1
        assert len(results["tob_levels"]["outputs"]) >= 1
    
    def test_run_analyses_disabled_modules_skipped(self, test_config, synthetic_base_matrix):
        """Test that disabled analyses are skipped."""
        test_config.analyses = {
            "ltsa": {
                "enabled": False,
                "config": {"time_resolution": "1h", "statistic": "median"},
            },
        }
        
        results = run_analyses(synthetic_base_matrix, test_config)
        assert "ltsa" not in results
    
    def test_run_analyses_required_module_failure_propagates(
        self, test_config, synthetic_base_matrix
    ):
        """Test that required module failures abort pipeline."""
        test_config.analyses = {
            "ltsa": {
                "enabled": True,
                "required": True,
                "config": {
                    "time_resolution": "1h",
                    "statistic": "invalid_stat",  # Invalid config
                },
            },
        }
        
        with pytest.raises(ValueError):
            run_analyses(synthetic_base_matrix, test_config)
    
    def test_run_analyses_optional_module_failure_continues(
        self, test_config, synthetic_base_matrix
    ):
        """Test that optional module failures don't abort pipeline."""
        test_config.analyses = {
            "ltsa": {
                "enabled": True,
                "required": False,
                "config": {
                    "time_resolution": "1h",
                    "statistic": "invalid_stat",  # Invalid config
                },
            },
            "tob_levels": {
                "enabled": True,
                "required": True,
                "config": {"statistics": ["median"]},
            },
        }
        
        results = run_analyses(synthetic_base_matrix, test_config)
        assert "ltsa" not in results  # LTSA was optional and failed
        assert "tob_levels" in results  # TOB Levels still ran


class TestManifestIntegration:
    """Tests for manifest generation with analysis results."""
    
    def test_manifest_includes_analysis_results(self, test_config, synthetic_wav):
        """Test that run_manifest.json includes analysis outputs."""
        test_config.input.path = os.path.dirname(synthetic_wav)
        test_config.analyse_only = False
        test_config.load_only = False
        test_config.analyses = {
            "ltsa": {
                "enabled": True,
                "required": True,
                "config": {
                    "time_resolution": "6h",
                    "statistic": "median",
                },
            },
        }
        
        run_pipeline(test_config)
        
        manifest_path = os.path.join(test_config.output.directory, "run_manifest.json")
        assert os.path.isfile(manifest_path)
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        assert "analyses" in manifest
        assert "ltsa" in manifest["analyses"]
        assert "outputs" in manifest["analyses"]["ltsa"]
        assert "summary" in manifest["analyses"]["ltsa"]


class TestAnalyseOnlyMode:
    """Tests for --analyse-only pipeline mode."""
    
    def test_analyse_only_loads_from_cache(self, test_config, synthetic_wav):
        """Test that --analyse-only mode loads from cache."""
        test_config.input.path = os.path.dirname(synthetic_wav)
        
        # First run: load only
        test_config.load_only = True
        test_config.analyse_only = False
        run_pipeline(test_config)
        
        # Second run: analyse only
        test_config.load_only = False
        test_config.analyse_only = True
        test_config.analyses = {
            "ltsa": {
                "enabled": True,
                "config": {
                    "time_resolution": "6h",
                    "statistic": "median",
                },
            },
        }
        
        run_pipeline(test_config)
        
        # Verify LTSA output was created
        assert os.path.isfile(os.path.join(test_config.output.directory, "ltsa.csv"))