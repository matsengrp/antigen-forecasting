"""
Test suite for data processing functions.
"""

import pytest
import pandas as pd
import numpy as np

from antigentools.scoring.processors import (
    smooth_frequencies,
    merge_truth_pred,
    prep_frequency_data,
    sample_predictive_quantile,
    calculate_errors
)


class TestSmoothFrequencies:
    """Test frequency smoothing function."""
    
    def test_smooth_basic(self):
        """Test basic smoothing functionality."""
        df = pd.DataFrame({
            'truth_freq': [0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1],
            'date': pd.date_range('2027-01-01', periods=7)
        })
        
        result = smooth_frequencies(df, window_size=3)
        
        assert 'smoothed_freq' in result.columns
        assert len(result) == len(df)
        # Middle values should be averaged (0.3 + 0.4 + 0.3) / 3 = 0.333...
        assert abs(result.iloc[3]['smoothed_freq'] - 0.333333) < 0.001
    
    def test_smooth_edge_behavior(self):
        """Test smoothing behavior at edges."""
        df = pd.DataFrame({
            'truth_freq': [0.1, 0.9, 0.1]
        })
        
        result = smooth_frequencies(df, window_size=3)
        
        # With min_periods=1 and center=True, edges should handle gracefully
        assert np.all(np.isfinite(result['smoothed_freq']))
    
    def test_smooth_validation(self):
        """Test input validation."""
        df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Column 'truth_freq' not found"):
            smooth_frequencies(df)
        
        valid_df = pd.DataFrame({'truth_freq': [0.1, 0.2]})
        
        with pytest.raises(ValueError, match="window_size must be positive"):
            smooth_frequencies(valid_df, window_size=0)


class TestMergeTruthPred:
    """Test merging of truth and prediction data."""
    
    def test_merge_basic(self):
        """Test basic merge functionality."""
        truth_df = pd.DataFrame({
            'date': ['2027-01-01', '2027-01-02'],
            'country': ['north', 'north'],
            'variant': ['A', 'B'],
            'sequences': [10, 20],
            'total_seq': [30, 30],
            'truth_freq': [0.33, 0.67]
        })
        
        pred_df = pd.DataFrame({
            'date': ['2027-01-01', '2027-01-02'],
            'country': ['north', 'north'],
            'variant': ['A', 'B'],
            'pred_freq': [0.35, 0.65]
        })
        
        result = merge_truth_pred(pred_df, truth_df)
        
        assert len(result) == 2
        assert 'pred_freq' in result.columns
        assert 'truth_freq' in result.columns
        assert result['pred_freq'].notna().all()
    
    def test_merge_location_rename(self):
        """Test location column renaming."""
        truth_df = pd.DataFrame({
            'date': ['2027-01-01'],
            'country': ['north'],
            'variant': ['A'],
            'sequences': [10]
        })
        
        pred_df = pd.DataFrame({
            'date': ['2027-01-01'],
            'location': ['north'],  # Different column name
            'variant': ['A'],
            'pred_freq': [0.5]
        })
        
        result = merge_truth_pred(pred_df, truth_df)
        assert 'country' in result.columns
        assert 'location' not in result.columns
    
    def test_merge_missing_predictions(self):
        """Test handling of missing predictions."""
        truth_df = pd.DataFrame({
            'date': ['2027-01-01', '2027-01-02'],
            'country': ['north', 'north'],
            'variant': ['A', 'B'],
            'sequences': [10, 20]
        })
        
        pred_df = pd.DataFrame({
            'date': ['2027-01-01'],
            'country': ['north'],
            'variant': ['A'],
            'pred_freq': [0.5]
        })
        
        result = merge_truth_pred(pred_df, truth_df)
        
        # Should only include rows with predictions
        assert len(result) == 1
        assert result['variant'].iloc[0] == 'A'


class TestPrepFrequencyData:
    """Test frequency data preparation."""
    
    def test_prep_basic(self):
        """Test basic data preparation."""
        df = pd.DataFrame({
            'truth_freq': [0.1, 0.2, 0.3],
            'smoothed_freq': [0.12, 0.21, 0.29],
            'sequences': [10, 20, 30],
            'total_seq': [100, 100, 100],
            'pred_freq': [0.11, 0.19, 0.31],
            'ci_low': [0.05, 0.15, 0.25],
            'ci_high': [0.15, 0.25, 0.35]
        })
        
        result = prep_frequency_data(df)
        
        assert len(result) == 9
        assert result[0] is not None  # raw_freq
        assert result[1] is not None  # pred_freq
        assert np.array_equal(result[0], df['truth_freq'].values)
    
    def test_prep_missing_ci(self):
        """Test preparation without credible intervals."""
        df = pd.DataFrame({
            'truth_freq': [0.1, 0.2],
            'smoothed_freq': [0.1, 0.2],
            'sequences': [10, 20],
            'total_seq': [100, 100],
            'pred_freq': [0.1, 0.2]
        })
        
        result = prep_frequency_data(df)
        
        # CI values should be None
        assert result[5] is None  # ci_low
        assert result[6] is None  # ci_low_pred
        assert result[7] is None  # ci_high
        assert result[8] is None  # ci_high_pred
    
    def test_prep_empty_data(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()
        
        result = prep_frequency_data(df)
        
        # Should return tuple of Nones
        assert len(result) == 9
        assert all(x is None for x in result)


class TestSamplePredictiveQuantile:
    """Test predictive quantile sampling."""
    
    def test_sample_quantile_basic(self):
        """Test basic quantile sampling."""
        np.random.seed(42)
        
        total_seq = np.array([100, 100, 100])
        freq = np.array([0.1, 0.5, 0.9])
        
        result = sample_predictive_quantile(total_seq, freq, q=0.5, num_samples=1000)
        
        assert len(result) == len(total_seq)
        # Results should be close to input frequencies
        np.testing.assert_allclose(result, freq, rtol=0.1)
    
    def test_sample_quantile_edge_cases(self):
        """Test edge cases."""
        # All NaN total_seq
        total_seq = np.array([np.nan, np.nan])
        freq = np.array([0.5, 0.5])
        
        result = sample_predictive_quantile(total_seq, freq, q=0.5)
        assert np.all(np.isnan(result))
        
        # Zero total_seq
        total_seq = np.array([0, 100])
        freq = np.array([0.5, 0.5])
        
        result = sample_predictive_quantile(total_seq, freq, q=0.5)
        assert np.isnan(result[0])
        assert np.isfinite(result[1])
    
    def test_sample_quantile_validation(self):
        """Test input validation."""
        total_seq = np.array([100])
        freq = np.array([0.5])
        
        with pytest.raises(ValueError, match="Quantile q must be between 0 and 1"):
            sample_predictive_quantile(total_seq, freq, q=1.5)


class TestCalculateErrors:
    """Test error calculation function."""
    
    def test_calculate_errors_basic(self):
        """Test basic error calculation."""
        merged_df = pd.DataFrame({
            'date': ['2027-01-01', '2027-01-02'],
            'country': ['north', 'north'],
            'variant': ['A', 'A'],
            'sequences': [10, 15],
            'total_seq': [100, 100],
            'truth_freq': [0.1, 0.15],
            'smoothed_freq': [0.11, 0.14],
            'pred_freq': [0.12, 0.13]
        })
        
        result = calculate_errors(
            merged_df,
            pivot_date='2027-01-01',
            country='north',
            model='TestModel'
        )
        
        assert result is not None
        assert len(result) == 2
        assert 'MAE' in result.columns
        assert 'MSE' in result.columns
        assert 'loglik' in result.columns
        assert np.all(result['MAE'] >= 0)
    
    def test_calculate_errors_with_filtering(self):
        """Test error calculation with filtering."""
        merged_df = pd.DataFrame({
            'date': ['2027-01-01', '2027-01-02', '2027-01-03'],
            'country': ['north', 'north', 'north'],
            'variant': ['A', 'B', 'C'],
            'sequences': [10, 1, 50],
            'total_seq': [100, 100, 100],
            'truth_freq': [0.1, 0.01, 0.5],
            'smoothed_freq': [0.11, 0.008, np.nan],
            'pred_freq': [0.12, 0.015, 0.48]
        })
        
        # Test with frequency threshold
        result = calculate_errors(
            merged_df,
            pivot_date='2027-01-01',
            country='north',
            model='TestModel',
            min_freq_threshold=0.05,
            handle_missing_smoothed=True
        )
        
        # Should filter out B (below threshold) and C (missing smoothed)
        assert len(result) == 1
        assert result['variant'].iloc[0] == 'A'
    
    def test_calculate_errors_empty_after_filter(self):
        """Test when all data is filtered out."""
        merged_df = pd.DataFrame({
            'date': ['2027-01-01'],
            'country': ['north'],
            'variant': ['A'],
            'sequences': [1],
            'total_seq': [1000],
            'truth_freq': [0.001],
            'smoothed_freq': [0.001],
            'pred_freq': [0.001]
        })
        
        result = calculate_errors(
            merged_df,
            pivot_date='2027-01-01',
            country='north',
            model='TestModel',
            min_freq_threshold=0.1  # Filter out everything
        )
        
        assert result is None