"""
Tests for growth rate data loading functionality.

This module tests the data loading functions for growth rates,
convergence diagnostics, and RT file discovery.
"""

import pytest
import pandas as pd
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, mock_open
import numpy as np

from antigentools.scoring.loaders import (
    RTFile,
    load_growth_rates,
    load_convergence_diagnostics,
    discover_rt_files,
    parse_rt_filename
)


class TestRTFile:
    """Test the RTFile dataclass."""
    
    def test_rtfile_creation(self):
        """Test creating an RTFile instance."""
        rt_file = RTFile(
            path=Path("results/build/estimates/FGA/rt_north_2025-01-01.tsv"),
            model="FGA",
            location="north",
            pivot_date="2025-01-01"
        )
        
        assert rt_file.model == "FGA"
        assert rt_file.location == "north"
        assert rt_file.pivot_date == "2025-01-01"
        assert isinstance(rt_file.path, Path)
    
    def test_rtfile_str_representation(self):
        """Test string representation of RTFile."""
        rt_file = RTFile(
            path=Path("results/build/estimates/FGA/rt_north_2025-01-01.tsv"),
            model="FGA",
            location="north",
            pivot_date="2025-01-01"
        )
        
        str_repr = str(rt_file)
        assert "FGA" in str_repr
        assert "north" in str_repr
        assert "2025-01-01" in str_repr


class TestParseRTFilename:
    """Test RT filename parsing."""
    
    def test_parse_standard_filename(self):
        """Test parsing standard RT filename."""
        path = Path("results/build/estimates/FGA/rt_north_2025-01-01.tsv")
        model, location, date = parse_rt_filename(path)
        
        assert model == "FGA"
        assert location == "north"
        assert date == "2025-01-01"
    
    def test_parse_different_locations(self):
        """Test parsing filenames with different locations."""
        test_cases = [
            ("results/build/estimates/GARW/rt_tropics_2025-04-01.tsv", "GARW", "tropics", "2025-04-01"),
            ("results/build/estimates/FGA/rt_south_2025-10-01.tsv", "FGA", "south", "2025-10-01"),
        ]
        
        for path_str, expected_model, expected_location, expected_date in test_cases:
            path = Path(path_str)
            model, location, date = parse_rt_filename(path)
            
            assert model == expected_model
            assert location == expected_location
            assert date == expected_date
    
    def test_parse_invalid_filename(self):
        """Test parsing invalid filenames raises appropriate errors."""
        # Wrong prefix
        with pytest.raises(ValueError, match="Invalid RT filename"):
            parse_rt_filename(Path("results/build/estimates/FGA/freq_north_2025-01-01.tsv"))
        
        # Missing parts
        with pytest.raises(ValueError, match="Invalid RT filename"):
            parse_rt_filename(Path("results/build/estimates/FGA/rt_north.tsv"))
        
        # No parent directory
        with pytest.raises(ValueError, match="Could not determine model"):
            parse_rt_filename(Path("rt_north_2025-01-01.tsv"))


class TestDiscoverRTFiles:
    """Test RT file discovery functionality."""
    
    def test_discover_rt_files_basic(self, tmp_path):
        """Test discovering RT files in a directory structure."""
        # Create test directory structure
        build_dir = tmp_path / "results" / "test-build"
        fga_dir = build_dir / "estimates" / "FGA"
        garw_dir = build_dir / "estimates" / "GARW"
        mlr_dir = build_dir / "estimates" / "MLR"
        
        fga_dir.mkdir(parents=True)
        garw_dir.mkdir(parents=True)
        mlr_dir.mkdir(parents=True)
        
        # Create test RT files
        rt_files = [
            fga_dir / "rt_north_2025-01-01.tsv",
            fga_dir / "rt_south_2025-01-01.tsv",
            garw_dir / "rt_north_2025-01-01.tsv",
            garw_dir / "rt_tropics_2025-04-01.tsv",
            mlr_dir / "rt_north_2025-01-01.tsv",  # Should be filtered out
        ]
        
        for file in rt_files:
            file.touch()
        
        # Also create non-RT files that should be ignored
        (fga_dir / "freq_north_2025-01-01.tsv").touch()
        (garw_dir / "data.csv").touch()
        
        # Discover RT files
        discovered = discover_rt_files("test-build", models=["FGA", "GARW"], base_dir=str(tmp_path))
        
        assert len(discovered) == 4
        assert all(isinstance(rt, RTFile) for rt in discovered)
        
        # Check models
        models = {rt.model for rt in discovered}
        assert models == {"FGA", "GARW"}
        
        # Check locations
        locations = {rt.location for rt in discovered}
        assert "north" in locations
        assert "south" in locations
        assert "tropics" in locations
    
    def test_discover_rt_files_empty_directory(self, tmp_path):
        """Test discovering RT files in empty directory."""
        # Create empty structure
        build_dir = tmp_path / "results" / "test-build" / "estimates"
        build_dir.mkdir(parents=True)
        
        discovered = discover_rt_files("test-build", models=["FGA"], base_dir=str(tmp_path))
        
        assert len(discovered) == 0
    
    def test_discover_rt_files_model_filtering(self, tmp_path):
        """Test that model filtering works correctly."""
        # Create test directory structure
        build_dir = tmp_path / "results" / "test-build"
        fga_dir = build_dir / "estimates" / "FGA"
        garw_dir = build_dir / "estimates" / "GARW"
        
        fga_dir.mkdir(parents=True)
        garw_dir.mkdir(parents=True)
        
        # Create files for both models
        (fga_dir / "rt_north_2025-01-01.tsv").touch()
        (garw_dir / "rt_north_2025-01-01.tsv").touch()
        
        # Test filtering for only FGA
        discovered = discover_rt_files("test-build", models=["FGA"], base_dir=str(tmp_path))
        assert len(discovered) == 1
        assert discovered[0].model == "FGA"
        
        # Test filtering for both
        discovered = discover_rt_files("test-build", models=["FGA", "GARW"], base_dir=str(tmp_path))
        assert len(discovered) == 2
    
    def test_discover_rt_files_sorted_order(self, tmp_path):
        """Test that RT files are returned in sorted order."""
        # Create test directory structure
        build_dir = tmp_path / "results" / "test-build"
        fga_dir = build_dir / "estimates" / "FGA"
        fga_dir.mkdir(parents=True)
        
        # Create files in non-alphabetical order
        files = [
            "rt_south_2025-10-01.tsv",
            "rt_north_2025-04-01.tsv",
            "rt_tropics_2025-01-01.tsv",
            "rt_north_2025-01-01.tsv",
        ]
        
        for file in files:
            (fga_dir / file).touch()
        
        discovered = discover_rt_files("test-build", models=["FGA"], base_dir=str(tmp_path))
        
        # Check sorted by (model, location, date)
        assert len(discovered) == 4
        # Since all same model, check location/date ordering
        assert discovered[0].location == "north" and discovered[0].pivot_date == "2025-01-01"
        assert discovered[1].location == "north" and discovered[1].pivot_date == "2025-04-01"
        assert discovered[2].location == "south"
        assert discovered[3].location == "tropics"


class TestLoadGrowthRates:
    """Test growth rate data loading."""
    
    def test_load_growth_rates_basic(self, tmp_path):
        """Test loading growth rates from file."""
        # Create test data
        test_data = pd.DataFrame({
            'variant': ['A', 'B', 'C'],
            'r_est': [0.1, 0.2, -0.1],
            'r_data': [0.12, 0.18, -0.08],
            'smoothed_frequency': [0.5, 0.3, 0.2],
            'smoothed_seq_counts': [100, 60, 40]
        })
        
        # Save to temp file
        rt_file = tmp_path / "results" / "test-build" / "estimates" / "FGA" / "rt_north_2025-01-01.tsv"
        rt_file.parent.mkdir(parents=True)
        test_data.to_csv(rt_file, sep='\t', index=False)
        
        # Load data
        loaded_df = load_growth_rates(
            build="test-build",
            model="FGA",
            location="north",
            pivot_date="2025-01-01",
            base_dir=str(tmp_path)
        )
        
        # Verify
        pd.testing.assert_frame_equal(loaded_df, test_data)
    
    def test_load_growth_rates_file_not_found(self, tmp_path):
        """Test loading growth rates with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_growth_rates(
                build="test-build",
                model="FGA",
                location="north",
                pivot_date="2025-01-01",
                base_dir=str(tmp_path)
            )
    
    def test_load_growth_rates_with_missing_columns(self, tmp_path):
        """Test loading growth rates handles missing columns gracefully."""
        # Create test data with only some columns
        test_data = pd.DataFrame({
            'variant': ['A', 'B'],
            'r_est': [0.1, 0.2]
        })
        
        # Save to temp file
        rt_file = tmp_path / "results" / "test-build" / "estimates" / "FGA" / "rt_north_2025-01-01.tsv"
        rt_file.parent.mkdir(parents=True)
        test_data.to_csv(rt_file, sep='\t', index=False)
        
        # Should load successfully
        loaded_df = load_growth_rates(
            build="test-build",
            model="FGA",
            location="north",
            pivot_date="2025-01-01",
            base_dir=str(tmp_path)
        )
        
        assert 'variant' in loaded_df.columns
        assert 'r_est' in loaded_df.columns
    
    def test_load_growth_rates_empty_file(self, tmp_path):
        """Test loading empty growth rates file."""
        # Create empty file with headers only
        rt_file = tmp_path / "results" / "test-build" / "estimates" / "FGA" / "rt_north_2025-01-01.tsv"
        rt_file.parent.mkdir(parents=True)
        rt_file.write_text("variant\tr_est\tr_data\n")
        
        loaded_df = load_growth_rates(
            build="test-build",
            model="FGA",
            location="north",
            pivot_date="2025-01-01",
            base_dir=str(tmp_path)
        )
        
        assert len(loaded_df) == 0
        assert 'variant' in loaded_df.columns


class TestLoadConvergenceDiagnostics:
    """Test convergence diagnostics loading."""
    
    def test_load_convergence_diagnostics_basic(self, tmp_path):
        """Test loading convergence diagnostics from JSON."""
        # Create test diagnostics data
        test_diagnostics = {
            "inference_method": "variational",
            "inference_settings": {
                "iterations": 10000,
                "learning_rate": 0.01,
                "num_samples": 100
            },
            "convergence_diagnostics": {
                "elbo_trajectory": {
                    "num_iterations": 10000,
                    "initial_loss": -1000.0,
                    "final_loss": -500.0,
                    "min_loss": -490.0,
                    "total_improvement": 500.0
                },
                "convergence": {
                    "converged": True,
                    "relative_change": 0.001,
                    "threshold": 0.01,
                    "window": 100,
                    "final_iteration": 9500
                }
            }
        }
        
        # Save to file
        diag_file = tmp_path / "diagnostics.json"
        with open(diag_file, 'w') as f:
            json.dump(test_diagnostics, f)
        
        # Load diagnostics
        loaded = load_convergence_diagnostics(diag_file)
        
        assert loaded == test_diagnostics
    
    def test_load_convergence_diagnostics_file_not_found(self):
        """Test loading non-existent diagnostics file."""
        result = load_convergence_diagnostics(Path("non_existent.json"))
        assert result is None
    
    def test_load_convergence_diagnostics_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        # Create invalid JSON
        diag_file = tmp_path / "invalid.json"
        diag_file.write_text("{ invalid json }")
        
        result = load_convergence_diagnostics(diag_file)
        assert result is None
    
    def test_load_convergence_diagnostics_empty_file(self, tmp_path):
        """Test loading empty diagnostics file."""
        # Create empty file
        diag_file = tmp_path / "empty.json"
        diag_file.write_text("{}")
        
        result = load_convergence_diagnostics(diag_file)
        assert result == {}
    
    def test_load_convergence_diagnostics_partial_data(self, tmp_path):
        """Test loading diagnostics with partial data."""
        # Create partial diagnostics
        test_diagnostics = {
            "inference_method": "variational",
            "convergence_diagnostics": {
                "convergence": {
                    "converged": False,
                    "relative_change": 0.1
                }
            }
        }
        
        # Save to file
        diag_file = tmp_path / "partial.json"
        with open(diag_file, 'w') as f:
            json.dump(test_diagnostics, f)
        
        # Load diagnostics
        loaded = load_convergence_diagnostics(diag_file)
        
        assert loaded["inference_method"] == "variational"
        assert loaded["convergence_diagnostics"]["convergence"]["converged"] is False