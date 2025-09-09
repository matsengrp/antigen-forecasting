"""
Unit tests for score_models.py filtering functionality.

Tests the new variant filtering capabilities added to improve model scoring:
- Active variant filtering
- Frequency threshold filtering  
- Missing data handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))
from score_models import filter_active_variants, calculate_errors, merge_truth_pred, smooth_freqs


class TestFilterActiveVariants:
    """Test suite for the filter_active_variants function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample variant observation data."""
        dates = pd.date_range('2027-01-01', '2027-04-30', freq='7D')  # 4 months weekly
        data = []
        
        for date in dates:
            week_num = (date - dates[0]).days // 7
            
            # Variant 1: Always present with high counts
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'country': 'north',
                'variant': 'always_active',
                'sequences': 50 + np.random.randint(-10, 10)
            })
            
            # Variant 2: Disappears after week 8
            if week_num < 8:
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'country': 'north',
                    'variant': 'goes_extinct',
                    'sequences': 20
                })
            
            # Variant 3: Rarely observed
            if week_num % 5 == 0:  # Only every 5 weeks
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'country': 'north', 
                    'variant': 'rarely_seen',
                    'sequences': 3
                })
            
            # Variant 4: Low sequence count
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'country': 'north',
                'variant': 'low_count',
                'sequences': 1
            })
        
        return pd.DataFrame(data)
    
    def test_basic_filtering(self, sample_data):
        """Test basic active variant filtering."""
        pivot_date = '2027-04-15'
        
        result = filter_active_variants(
            sample_data,
            pivot_date=pivot_date,
            lookback_days=90,
            min_observations=3,
            min_sequences=10
        )
        
        # Only 'always_active' should pass all filters
        unique_variants = result['variant'].unique()
        assert 'always_active' in unique_variants
        assert 'goes_extinct' not in unique_variants  # Not seen in last 90 days
        assert 'rarely_seen' not in unique_variants   # Too few observations
        assert 'low_count' not in unique_variants     # Too few sequences
    
    def test_lookback_window(self, sample_data):
        """Test different lookback windows."""
        pivot_date = '2027-04-15'
        
        # Short lookback (30 days) - fewer variants
        result_short = filter_active_variants(
            sample_data,
            pivot_date=pivot_date,
            lookback_days=30,
            min_observations=1,
            min_sequences=1
        )
        
        # Long lookback (120 days) - more variants
        result_long = filter_active_variants(
            sample_data,
            pivot_date=pivot_date,
            lookback_days=120,
            min_observations=1,
            min_sequences=1
        )
        
        assert len(result_short) <= len(result_long)
    
    def test_edge_cases(self, sample_data):
        """Test edge cases for active variant filtering."""
        # Empty dataframe
        empty_df = pd.DataFrame(columns=['date', 'country', 'variant', 'sequences'])
        result = filter_active_variants(empty_df, pivot_date='2027-04-15')
        assert len(result) == 0
        
        # All variants filtered out (very strict criteria)
        result = filter_active_variants(
            sample_data,
            pivot_date='2027-04-15',
            lookback_days=10,  # Very short window
            min_observations=100,  # Impossible to meet
            min_sequences=10000
        )
        assert len(result) == 0


class TestCalculateErrors:
    """Test suite for the enhanced calculate_errors function."""
    
    @pytest.fixture
    def merged_data(self):
        """Create merged truth and prediction data."""
        dates = pd.date_range('2027-01-01', '2027-06-01', freq='7D')
        data = []
        
        for i, date in enumerate(dates):
            # High frequency variant
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'country': 'north',
                'variant': 'high_freq',
                'sequences': 50,
                'total_seq': 100,
                'truth_freq': 0.5,
                'pred_freq': 0.48,
                'smoothed_freq': 0.49
            })
            
            # Low frequency variant
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'country': 'north',
                'variant': 'low_freq',
                'sequences': 1,
                'total_seq': 100,
                'truth_freq': 0.01,
                'pred_freq': 0.015,
                'smoothed_freq': 0.008  # Below threshold
            })
            
            # Missing smoothed freq (simulates extinct variant)
            if i > 10:
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'country': 'north',
                    'variant': 'missing_smooth',
                    'sequences': 0,
                    'total_seq': 100,
                    'truth_freq': 0.0,
                    'pred_freq': 0.001,
                    'smoothed_freq': np.nan
                })
        
        return pd.DataFrame(data)
    
    def test_frequency_threshold_filtering(self, merged_data):
        """Test minimum frequency threshold filtering."""
        pivot_date = '2027-03-15'
        
        # No filtering
        result_no_filter = calculate_errors(
            merged_data.copy(),
            pivot_date=pivot_date,
            country='north',
            model='test',
            min_freq_threshold=None,
            handle_missing_smoothed=False
        )
        
        # With 1% threshold
        result_filtered = calculate_errors(
            merged_data.copy(),
            pivot_date=pivot_date,
            country='north',
            model='test',
            min_freq_threshold=0.01,
            handle_missing_smoothed=False
        )
        
        # Filtered result should have fewer rows
        assert len(result_filtered) < len(result_no_filter)
        
        # Check that low frequency variants are filtered out when below threshold
        low_freq_in_filtered = result_filtered[result_filtered['variant'] == 'low_freq']
        # Only keep rows where either smoothed_freq or pred_freq >= 0.01
        assert all((low_freq_in_filtered['smoothed_freq'] >= 0.01) | 
                  (low_freq_in_filtered['pred_freq'] >= 0.01))
    
    def test_missing_smoothed_handling(self, merged_data):
        """Test handling of missing smoothed frequency values."""
        pivot_date = '2027-05-15'
        
        # Don't handle missing
        result_with_nan = calculate_errors(
            merged_data.copy(),
            pivot_date=pivot_date,
            country='north',
            model='test',
            handle_missing_smoothed=False
        )
        
        # Handle missing (default behavior)
        result_no_nan = calculate_errors(
            merged_data.copy(),
            pivot_date=pivot_date,
            country='north',
            model='test',
            handle_missing_smoothed=True
        )
        
        # Result without NaN should have fewer rows
        assert len(result_no_nan) < len(result_with_nan)
        
        # No NaN values in smoothed_freq when handled
        assert not result_no_nan['smoothed_freq'].isna().any()
    
    def test_combined_filters(self, merged_data):
        """Test combining multiple filters."""
        pivot_date = '2027-05-15'
        
        # Apply all filters
        result = calculate_errors(
            merged_data.copy(),
            pivot_date=pivot_date,
            country='north',
            model='test',
            min_freq_threshold=0.02,  # 2% threshold
            handle_missing_smoothed=True
        )
        
        # Should only have high_freq variant
        assert len(result['variant'].unique()) == 1
        assert result['variant'].iloc[0] == 'high_freq'
    
    def test_error_calculations_preserved(self, merged_data):
        """Test that error calculations work correctly with filters."""
        pivot_date = '2027-03-15'
        
        result = calculate_errors(
            merged_data.copy(),
            pivot_date=pivot_date,
            country='north',
            model='test',
            min_freq_threshold=0.01,
            handle_missing_smoothed=True
        )
        
        # Check that error metrics are calculated
        assert 'MAE' in result.columns
        assert 'MSE' in result.columns
        assert 'loglik' in result.columns
        
        # Check MAE calculation is correct
        high_freq_row = result[result['variant'] == 'high_freq'].iloc[0]
        expected_mae = abs(high_freq_row['smoothed_freq'] - high_freq_row['pred_freq'])
        assert abs(high_freq_row['MAE'] - expected_mae) < 1e-10
        
        # Check MSE calculation
        expected_mse = (high_freq_row['smoothed_freq'] - high_freq_row['pred_freq']) ** 2
        assert abs(high_freq_row['MSE'] - expected_mse) < 1e-10


class TestSmoothFreqs:
    """Test suite for frequency smoothing function."""
    
    def test_smoothing_basic(self):
        """Test basic frequency smoothing."""
        # Create data with sharp changes
        data = pd.DataFrame({
            'truth_freq': [0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.1]
        })
        
        result = smooth_freqs(data.copy(), window_size=3)
        
        # Check smoothing reduces sharp transitions
        assert 'smoothed_freq' in result.columns
        
        # Middle values should be smoothed
        assert result['smoothed_freq'].iloc[2] < 0.5  # Smoothed down
        assert result['smoothed_freq'].iloc[4] < 0.5  # Smoothed down
    
    def test_smoothing_edge_cases(self):
        """Test smoothing with edge cases."""
        # Single data point
        data = pd.DataFrame({'truth_freq': [0.5]})
        result = smooth_freqs(data.copy(), window_size=7)
        assert result['smoothed_freq'].iloc[0] == 0.5
        
        # All same values
        data = pd.DataFrame({'truth_freq': [0.3] * 10})
        result = smooth_freqs(data.copy(), window_size=7)
        assert all(result['smoothed_freq'] == 0.3)


class TestIntegration:
    """Integration tests for the complete filtering pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete filtering pipeline as used in score_models.py."""
        # Create realistic data
        dates = pd.date_range('2026-10-01', '2027-06-01', freq='7D')
        truth_data = []
        pred_data = []
        
        for date in dates:
            week_num = (date - dates[0]).days // 7
            
            # Create truth data
            truth_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'country': 'north',
                'variant': 'variant_A',
                'sequences': 40 + np.random.randint(-5, 5)
            })
            
            # Create prediction data with slight error
            pred_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'country': 'north',
                'variant': 'variant_A',
                'pred_freq': 0.4 + np.random.uniform(-0.05, 0.05)
            })
        
        truth_df = pd.DataFrame(truth_data)
        pred_df = pd.DataFrame(pred_data)
        
        # Apply smoothing to truth data
        truth_df = truth_df.groupby(['country', 'variant'], group_keys=False).apply(
            smooth_freqs
        ).reset_index(drop=True)
        
        # Merge
        merged = merge_truth_pred(pred_df, truth_df)
        
        # Apply filters
        pivot_date = '2027-04-01'
        
        # Active variant filter
        filtered = filter_active_variants(
            merged,
            pivot_date=pivot_date,
            lookback_days=90,
            min_observations=3,
            min_sequences=10
        )
        
        # Calculate errors with additional filters
        errors = calculate_errors(
            filtered,
            pivot_date=pivot_date,
            country='north',
            model='test_model',
            min_freq_threshold=0.01,
            handle_missing_smoothed=True
        )
        
        # Verify we got results
        assert errors is not None
        assert len(errors) > 0
        assert 'MAE' in errors.columns