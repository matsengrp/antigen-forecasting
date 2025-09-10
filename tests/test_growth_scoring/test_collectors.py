"""
Tests for result collection dataclasses.

This module tests the dataclasses used to collect and store results
from growth rate analysis, replacing the dictionary-based approach.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from antigentools.scoring.growth_rate_results import (
    WindowResult,
    VariantResult,
    ConvergenceDiagnostic,
    GrowthRateResultsCollector
)


class TestWindowResult:
    """Test the WindowResult dataclass."""
    
    def test_window_result_creation(self):
        """Test creating a WindowResult instance."""
        result = WindowResult(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            correlation=0.85,
            mae=0.05,
            rmse=0.08,
            sign_disagreement_rate=0.15,
            overestimation_rate=0.25,
            n_seqs=1000,
            n_cases=500,
            seq_entropy=2.5,
            case_entropy=2.3,
            seq_entropy_norm=0.8,
            case_entropy_norm=0.75
        )
        
        assert result.pivot_date == "2025-01-01"
        assert result.model == "FGA"
        assert result.location == "north"
        assert result.correlation == 0.85
        assert result.mae == 0.05
        assert result.n_seqs == 1000
    
    def test_window_result_to_dict(self):
        """Test converting WindowResult to dictionary."""
        result = WindowResult(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            correlation=0.85,
            mae=0.05,
            rmse=0.08,
            sign_disagreement_rate=0.15,
            overestimation_rate=0.25,
            n_seqs=1000,
            n_cases=500,
            seq_entropy=2.5,
            case_entropy=2.3,
            seq_entropy_norm=0.8,
            case_entropy_norm=0.75
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['pivot_date'] == "2025-01-01"
        assert result_dict['model'] == "FGA"
        assert result_dict['correlation'] == 0.85
        assert len(result_dict) == 14  # All fields
    
    def test_window_result_validation(self):
        """Test validation of WindowResult fields."""
        # Valid ranges
        result = WindowResult(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            correlation=1.0,  # Max correlation
            mae=0.0,  # Min MAE
            rmse=0.0,  # Min RMSE
            sign_disagreement_rate=0.0,  # Min rate
            overestimation_rate=1.0,  # Max rate
            n_seqs=0,  # Min sequences
            n_cases=0,  # Min cases
            seq_entropy=0.0,  # Min entropy
            case_entropy=0.0,  # Min entropy
            seq_entropy_norm=0.0,  # Min normalized entropy
            case_entropy_norm=1.0   # Max normalized entropy
        )
        assert result.correlation == 1.0
        
        # Test invalid correlation
        with pytest.raises(ValueError, match="correlation must be between -1 and 1"):
            WindowResult(
                pivot_date="2025-01-01",
                model="FGA",
                location="north",
                correlation=1.5,  # Invalid
                mae=0.05,
                rmse=0.08,
                sign_disagreement_rate=0.15,
                overestimation_rate=0.25,
                n_seqs=1000,
                n_cases=500,
                seq_entropy=2.5,
                case_entropy=2.3,
                seq_entropy_norm=0.8,
                case_entropy_norm=0.75
            )
    
    def test_window_result_optional_fields(self):
        """Test WindowResult with None values for optional metrics."""
        result = WindowResult(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            correlation=None,  # Can be None if calculation failed
            mae=0.05,
            rmse=0.08,
            sign_disagreement_rate=0.15,
            overestimation_rate=0.25,
            n_seqs=1000,
            n_cases=500,
            seq_entropy=2.5,
            case_entropy=2.3,
            seq_entropy_norm=0.8,
            case_entropy_norm=0.75
        )
        
        assert result.correlation is None
        assert result.mae == 0.05


class TestVariantResult:
    """Test the VariantResult dataclass."""
    
    def test_variant_result_creation(self):
        """Test creating a VariantResult instance."""
        result = VariantResult(
            variant="A.1.2",
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            mae=0.03,
            normalized_mae=0.15,
            max_r_data=0.25,
            correlation=0.8,
            sign_disagreement_rate=0.1,
            overestimation_rate=0.2,
            n_points=50,
            total_sequences=5000,
            total_smoothed_sequences=4800,
            mean_variant_frequency=0.15,
            mean_smoothed_variant_frequency=0.16,
            max_variant_frequency=0.35,
            max_smoothed_variant_frequency=0.34
        )
        
        assert result.variant == "A.1.2"
        assert result.pivot_date == "2025-01-01"
        assert result.model == "FGA"
        assert result.mae == 0.03
        assert result.n_points == 50
    
    def test_variant_result_to_dict(self):
        """Test converting VariantResult to dictionary."""
        result = VariantResult(
            variant="A.1.2",
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            mae=0.03,
            normalized_mae=0.15,
            max_r_data=0.25,
            correlation=0.8,
            sign_disagreement_rate=0.1,
            overestimation_rate=0.2,
            n_points=50,
            total_sequences=5000,
            total_smoothed_sequences=4800,
            mean_variant_frequency=0.15,
            mean_smoothed_variant_frequency=0.16,
            max_variant_frequency=0.35,
            max_smoothed_variant_frequency=0.34
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['variant'] == "A.1.2"
        assert result_dict['mae'] == 0.03
        assert result_dict['n_points'] == 50
        assert len(result_dict) == 17  # All fields
    
    def test_variant_result_validation(self):
        """Test validation of VariantResult fields."""
        # Test invalid frequency
        with pytest.raises(ValueError, match="frequency values must be between 0 and 1"):
            VariantResult(
                variant="A.1.2",
                pivot_date="2025-01-01",
                model="FGA",
                location="north",
                mae=0.03,
                normalized_mae=0.15,
                max_r_data=0.25,
                correlation=0.8,
                sign_disagreement_rate=0.1,
                overestimation_rate=0.2,
                n_points=50,
                total_sequences=5000,
                total_smoothed_sequences=4800,
                mean_variant_frequency=1.5,  # Invalid
                mean_smoothed_variant_frequency=0.16,
                max_variant_frequency=0.35,
                max_smoothed_variant_frequency=0.34
            )


class TestConvergenceDiagnostic:
    """Test the ConvergenceDiagnostic dataclass."""
    
    def test_convergence_diagnostic_creation(self):
        """Test creating a ConvergenceDiagnostic instance."""
        diagnostic = ConvergenceDiagnostic(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            inference_method="variational",
            iterations=10000,
            learning_rate=0.01,
            num_samples=100,
            num_iterations=10000,
            initial_loss=-1000.0,
            final_loss=-500.0,
            min_loss=-490.0,
            total_improvement=500.0,
            converged=True,
            relative_change=0.001,
            threshold=0.01,
            window=100,
            final_iteration=9500
        )
        
        assert diagnostic.pivot_date == "2025-01-01"
        assert diagnostic.model == "FGA"
        assert diagnostic.inference_method == "variational"
        assert diagnostic.converged is True
    
    def test_convergence_diagnostic_optional_fields(self):
        """Test ConvergenceDiagnostic with None values."""
        diagnostic = ConvergenceDiagnostic(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            inference_method=None,
            iterations=None,
            learning_rate=None,
            num_samples=None,
            num_iterations=None,
            initial_loss=None,
            final_loss=None,
            min_loss=None,
            total_improvement=None,
            converged=None,
            relative_change=None,
            threshold=None,
            window=None,
            final_iteration=None
        )
        
        assert diagnostic.pivot_date == "2025-01-01"
        assert diagnostic.inference_method is None
        assert diagnostic.converged is None
    
    def test_convergence_diagnostic_to_dict(self):
        """Test converting ConvergenceDiagnostic to dictionary."""
        diagnostic = ConvergenceDiagnostic(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            inference_method="variational",
            iterations=10000,
            learning_rate=0.01,
            num_samples=100,
            num_iterations=10000,
            initial_loss=-1000.0,
            final_loss=-500.0,
            min_loss=-490.0,
            total_improvement=500.0,
            converged=True,
            relative_change=0.001,
            threshold=0.01,
            window=100,
            final_iteration=9500
        )
        
        result_dict = diagnostic.to_dict()
        
        assert result_dict['pivot_date'] == "2025-01-01"
        assert result_dict['inference_method'] == "variational"
        assert result_dict['converged'] is True
        assert len(result_dict) == 17  # All fields


class TestGrowthRateResultsCollector:
    """Test the GrowthRateResultsCollector class."""
    
    def test_results_collector_creation(self):
        """Test creating an empty GrowthRateResultsCollector."""
        collector = GrowthRateResultsCollector()
        
        assert len(collector.window_results) == 0
        assert len(collector.variant_results) == 0
        assert len(collector.convergence_diagnostics) == 0
    
    def test_add_window_result(self):
        """Test adding window results to collector."""
        collector = GrowthRateResultsCollector()
        
        result = WindowResult(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            correlation=0.85,
            mae=0.05,
            rmse=0.08,
            sign_disagreement_rate=0.15,
            overestimation_rate=0.25,
            n_seqs=1000,
            n_cases=500,
            seq_entropy=2.5,
            case_entropy=2.3,
            seq_entropy_norm=0.8,
            case_entropy_norm=0.75
        )
        
        collector.add_window_result(result)
        
        assert len(collector.window_results) == 1
        assert collector.window_results[0] == result
    
    def test_add_variant_result(self):
        """Test adding variant results to collector."""
        collector = GrowthRateResultsCollector()
        
        result = VariantResult(
            variant="A.1.2",
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            mae=0.03,
            normalized_mae=0.15,
            max_r_data=0.25,
            correlation=0.8,
            sign_disagreement_rate=0.1,
            overestimation_rate=0.2,
            n_points=50,
            total_sequences=5000,
            total_smoothed_sequences=4800,
            mean_variant_frequency=0.15,
            mean_smoothed_variant_frequency=0.16,
            max_variant_frequency=0.35,
            max_smoothed_variant_frequency=0.34
        )
        
        collector.add_variant_result(result)
        
        assert len(collector.variant_results) == 1
        assert collector.variant_results[0] == result
    
    def test_add_convergence_diagnostic(self):
        """Test adding convergence diagnostics to collector."""
        collector = GrowthRateResultsCollector()
        
        diagnostic = ConvergenceDiagnostic(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            inference_method="variational",
            iterations=10000,
            learning_rate=0.01,
            num_samples=100,
            num_iterations=10000,
            initial_loss=-1000.0,
            final_loss=-500.0,
            min_loss=-490.0,
            total_improvement=500.0,
            converged=True,
            relative_change=0.001,
            threshold=0.01,
            window=100,
            final_iteration=9500
        )
        
        collector.add_convergence_diagnostic(diagnostic)
        
        assert len(collector.convergence_diagnostics) == 1
        assert collector.convergence_diagnostics[0] == diagnostic
    
    def test_to_dataframes(self):
        """Test converting collector to DataFrames."""
        collector = GrowthRateResultsCollector()
        
        # Add sample data
        window_result = WindowResult(
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            correlation=0.85,
            mae=0.05,
            rmse=0.08,
            sign_disagreement_rate=0.15,
            overestimation_rate=0.25,
            n_seqs=1000,
            n_cases=500,
            seq_entropy=2.5,
            case_entropy=2.3,
            seq_entropy_norm=0.8,
            case_entropy_norm=0.75
        )
        collector.add_window_result(window_result)
        
        variant_result = VariantResult(
            variant="A.1.2",
            pivot_date="2025-01-01",
            model="FGA",
            location="north",
            mae=0.03,
            normalized_mae=0.15,
            max_r_data=0.25,
            correlation=0.8,
            sign_disagreement_rate=0.1,
            overestimation_rate=0.2,
            n_points=50,
            total_sequences=5000,
            total_smoothed_sequences=4800,
            mean_variant_frequency=0.15,
            mean_smoothed_variant_frequency=0.16,
            max_variant_frequency=0.35,
            max_smoothed_variant_frequency=0.34
        )
        collector.add_variant_result(variant_result)
        
        # Convert to DataFrames
        window_df, variant_df, convergence_df = collector.to_dataframes()
        
        # Check window DataFrame
        assert len(window_df) == 1
        assert window_df.iloc[0]['pivot_date'] == "2025-01-01"
        assert window_df.iloc[0]['model'] == "FGA"
        assert window_df.iloc[0]['correlation'] == 0.85
        
        # Check variant DataFrame
        assert len(variant_df) == 1
        assert variant_df.iloc[0]['variant'] == "A.1.2"
        assert variant_df.iloc[0]['mae'] == 0.03
        
        # Check convergence DataFrame (empty)
        assert len(convergence_df) == 0
    
    def test_empty_to_dataframes(self):
        """Test converting empty collector to DataFrames."""
        collector = GrowthRateResultsCollector()
        
        window_df, variant_df, convergence_df = collector.to_dataframes()
        
        # All should be empty but with correct columns
        assert len(window_df) == 0
        assert len(variant_df) == 0
        assert len(convergence_df) == 0
        
        # Check that columns are present
        assert 'pivot_date' in window_df.columns
        assert 'model' in window_df.columns
        assert 'variant' in variant_df.columns
        assert 'pivot_date' in convergence_df.columns
    
    def test_summary_stats(self):
        """Test getting summary statistics from collector."""
        collector = GrowthRateResultsCollector()
        
        # Add multiple results
        for i, model in enumerate(['FGA', 'GARW']):
            for j, location in enumerate(['north', 'south']):
                collector.add_window_result(WindowResult(
                    pivot_date=f"2025-0{i+1}-01",
                    model=model,
                    location=location,
                    correlation=0.8 + i*0.1,
                    mae=0.05 + j*0.01,
                    rmse=0.08,
                    sign_disagreement_rate=0.15,
                    overestimation_rate=0.25,
                    n_seqs=1000,
                    n_cases=500,
                    seq_entropy=2.5,
                    case_entropy=2.3,
                    seq_entropy_norm=0.8,
                    case_entropy_norm=0.75
                ))
        
        stats = collector.summary_stats()
        
        assert stats['total_windows'] == 4
        assert stats['unique_models'] == 2
        assert stats['unique_locations'] == 2
        assert stats['unique_dates'] == 2
        assert stats['total_variants'] == 0  # No variant results added
        assert stats['total_diagnostics'] == 0  # No diagnostics added