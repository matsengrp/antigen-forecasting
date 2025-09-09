"""
Test suite for NAIVE forecasting functions in antigentools.utils

These tests verify the naive_forecast and naive_forecast_full_window functions
that were implemented as part of issue #11 - Extend NAIVE Model Coverage to Full Training Window.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from datetime import datetime, timedelta

from antigentools.utils import (
    naive_forecast, 
    naive_forecast_full_window,
    _validate_naive_forecast_input,
    _calculate_variant_frequencies,
    _calculate_rolling_average_optimized
)


class TestNaiveForecast:
    """Test suite for the original naive_forecast function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample sequence count data for testing."""
        dates = pd.date_range('2027-03-15', '2027-04-15', freq='D')
        data = []
        
        countries = ['north', 'south']
        variants = ['A', 'B', 'C']
        
        for date in dates:
            for country in countries:
                for variant in variants:
                    # Create realistic sequence counts with some variation
                    if variant == 'A':
                        sequences = np.random.poisson(20)  # Dominant variant
                    elif variant == 'B':
                        sequences = np.random.poisson(5)   # Minor variant
                    else:
                        sequences = np.random.poisson(2)   # Rare variant
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'country': country,
                        'variant': variant,
                        'sequences': sequences
                    })
        
        return pd.DataFrame(data)

    def test_naive_forecast_basic_functionality(self, sample_data):
        """Test basic functionality of naive_forecast function."""
        pivot = '2027-04-01'
        result = naive_forecast(sample_data, pivot=pivot, n_days_to_average=7, period=30)
        
        # Check basic structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Check expected columns
        expected_cols = ['variant', 'country', 'freq', 'date', 'median_freq_nowcast', 'median_freq_forecast']
        assert all(col in result.columns for col in expected_cols)
        
        # Check data coverage (should have 60 days: 30 before + 30 after pivot)
        unique_dates = result['date'].unique()
        assert len(unique_dates) <= 60  # May be less if some dates have no data
        
        # Check that frequencies sum to 1 for each country-date combination
        for country in result['country'].unique():
            for date in result['date'].unique():
                country_date_data = result[(result['country'] == country) & (result['date'] == date)]
                if len(country_date_data) > 0:
                    freq_sum = country_date_data['freq'].sum()
                    assert abs(freq_sum - 1.0) < 0.001, f"Frequencies don't sum to 1 for {country} on {date}: {freq_sum}"

    def test_naive_forecast_nowcast_forecast_separation(self, sample_data):
        """Test that nowcast and forecast columns are properly separated."""
        pivot = '2027-04-01'
        result = naive_forecast(sample_data, pivot=pivot, period=10)
        
        pivot_dt = pd.to_datetime(pivot)
        
        # Check forecast dates have nowcast = NaN
        forecast_dates = result[result['date'] >= pivot]
        assert forecast_dates['median_freq_nowcast'].isna().all()
        assert forecast_dates['median_freq_forecast'].notna().all()
        
        # Check nowcast dates have forecast = NaN
        nowcast_dates = result[result['date'] < pivot]
        assert nowcast_dates['median_freq_forecast'].isna().all()
        assert nowcast_dates['median_freq_nowcast'].notna().all()

    def test_naive_forecast_edge_cases(self, sample_data):
        """Test edge cases for naive_forecast function."""
        pivot = '2027-04-01'
        
        # Test with minimal averaging window
        result = naive_forecast(sample_data, pivot=pivot, n_days_to_average=1, period=5)
        assert len(result) > 0
        
        # Test with large averaging window (should handle gracefully)
        result = naive_forecast(sample_data, pivot=pivot, n_days_to_average=100, period=5)
        assert len(result) > 0
        
        # Test with early pivot date (limited historical data)
        early_pivot = '2027-03-16'
        result = naive_forecast(sample_data, pivot=early_pivot, n_days_to_average=7, period=5)
        assert len(result) > 0

    def test_naive_forecast_empty_data(self):
        """Test naive_forecast with empty input data."""
        empty_data = pd.DataFrame(columns=['date', 'country', 'variant', 'sequences'])
        
        # Should raise ValueError due to input validation
        with pytest.raises(ValueError, match="cannot be empty"):
            naive_forecast(empty_data, pivot='2027-04-01', period=5)


class TestNaiveForecastFullWindow:
    """Test suite for the new naive_forecast_full_window function."""

    @pytest.fixture
    def extended_sample_data(self):
        """Create extended sample data covering several months."""
        dates = pd.date_range('2026-12-01', '2027-04-15', freq='D')  # ~4.5 months of data
        data = []
        
        countries = ['north', 'south']
        variants = ['A', 'B', 'C']
        
        for date in dates:
            for country in countries:
                for variant in variants:
                    # Create time-varying sequence counts
                    days_from_start = (date - dates[0]).days
                    if variant == 'A':
                        sequences = max(1, 25 - days_from_start // 10)  # Declining variant
                    elif variant == 'B':
                        sequences = max(1, 5 + days_from_start // 20)   # Growing variant
                    else:
                        sequences = np.random.poisson(3)               # Stable variant
                    
                    data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'country': country,
                        'variant': variant,
                        'sequences': sequences
                    })
        
        return pd.DataFrame(data)

    def test_full_window_basic_functionality(self, extended_sample_data):
        """Test basic functionality of naive_forecast_full_window."""
        pivot = '2027-04-01'
        result = naive_forecast_full_window(
            extended_sample_data, 
            pivot=pivot, 
            n_days_to_average=7, 
            forecast_period=180
        )
        
        # Check basic structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Check expected columns
        expected_cols = ['variant', 'country', 'freq', 'date', 'median_freq_nowcast', 'median_freq_forecast']
        assert all(col in result.columns for col in expected_cols)
        
        # Check that we have much more data than the original (should cover ~300+ days)
        unique_dates = result['date'].unique()
        assert len(unique_dates) > 200, f"Expected >200 days, got {len(unique_dates)}"

    def test_full_window_training_vs_forecast_separation(self, extended_sample_data):
        """Test that training and forecast dates are properly separated."""
        pivot = '2027-04-01'
        result = naive_forecast_full_window(
            extended_sample_data, 
            pivot=pivot, 
            forecast_period=30
        )
        
        pivot_dt = pd.to_datetime(pivot)
        
        # Check forecast dates (>= pivot) have nowcast = NaN
        forecast_mask = pd.to_datetime(result['date']) >= pivot_dt
        forecast_data = result[forecast_mask]
        assert forecast_data['median_freq_nowcast'].isna().all()
        assert forecast_data['median_freq_forecast'].notna().all()
        
        # Check training dates (< pivot) have forecast = NaN
        training_mask = pd.to_datetime(result['date']) < pivot_dt
        training_data = result[training_mask]
        assert training_data['median_freq_forecast'].isna().all()
        assert training_data['median_freq_nowcast'].notna().all()

    def test_full_window_with_training_window_limit(self, extended_sample_data):
        """Test full window function with limited training window."""
        pivot = '2027-04-01'
        training_window = 60  # 2 months
        
        result = naive_forecast_full_window(
            extended_sample_data, 
            pivot=pivot, 
            training_window=training_window,
            forecast_period=30
        )
        
        # Check that training dates don't go back more than training_window days
        min_date = pd.to_datetime(result['date']).min()
        pivot_dt = pd.to_datetime(pivot)
        days_back = (pivot_dt - min_date).days
        
        # Should be approximately the training window (may be slightly different due to data availability)
        assert days_back <= training_window + 7, f"Training window too large: {days_back} days"

    def test_full_window_forecast_consistency(self, extended_sample_data):
        """Test that forecast values are consistent across forecast dates."""
        pivot = '2027-04-01'
        result = naive_forecast_full_window(
            extended_sample_data, 
            pivot=pivot, 
            n_days_to_average=7,
            forecast_period=30
        )
        
        # Get forecast data
        pivot_dt = pd.to_datetime(pivot)
        forecast_data = result[pd.to_datetime(result['date']) >= pivot_dt]
        
        # For each country-variant combination, all forecast values should be the same
        for country in forecast_data['country'].unique():
            for variant in forecast_data['variant'].unique():
                country_variant_data = forecast_data[
                    (forecast_data['country'] == country) & 
                    (forecast_data['variant'] == variant)
                ]
                
                if len(country_variant_data) > 1:
                    forecast_values = country_variant_data['median_freq_forecast'].dropna()
                    if len(forecast_values) > 0:
                        # All forecast values should be identical (persistence model)
                        assert forecast_values.std() < 1e-10, \
                            f"Forecast values not consistent for {country}-{variant}: {forecast_values.tolist()}"

    def test_full_window_vs_original_consistency(self, extended_sample_data):
        """Test that full window and original functions give similar results for overlapping periods."""
        pivot = '2027-04-01'
        period = 30
        
        # Run original function
        original_result = naive_forecast(
            extended_sample_data, 
            pivot=pivot, 
            n_days_to_average=7, 
            period=period
        )
        
        # Run full window function with same period
        full_result = naive_forecast_full_window(
            extended_sample_data, 
            pivot=pivot, 
            n_days_to_average=7,
            forecast_period=period,
            training_window=period
        )
        
        # Compare forecast values for overlapping dates
        pivot_dt = pd.to_datetime(pivot)
        forecast_dates = [d for d in original_result['date'].unique() if pd.to_datetime(d) >= pivot_dt]
        
        for date in forecast_dates[:5]:  # Check first 5 forecast dates
            for country in ['north', 'south']:
                for variant in ['A', 'B', 'C']:
                    orig_val = original_result[
                        (original_result['date'] == date) & 
                        (original_result['country'] == country) & 
                        (original_result['variant'] == variant)
                    ]['median_freq_forecast']
                    
                    full_val = full_result[
                        (full_result['date'] == date) & 
                        (full_result['country'] == country) & 
                        (full_result['variant'] == variant)
                    ]['median_freq_forecast']
                    
                    if len(orig_val) > 0 and len(full_val) > 0:
                        # Values should be reasonably close (allowing for implementation differences)
                        assert abs(orig_val.iloc[0] - full_val.iloc[0]) < 0.05, \
                            f"Inconsistent forecast values for {date}, {country}, {variant}: orig={orig_val.iloc[0]}, full={full_val.iloc[0]}"

    def test_full_window_edge_cases(self, extended_sample_data):
        """Test edge cases for full window function."""
        pivot = '2027-04-01'
        
        # Test with no training window limit (None)
        result = naive_forecast_full_window(
            extended_sample_data, 
            pivot=pivot, 
            training_window=None,
            forecast_period=30
        )
        assert len(result) > 0
        
        # Test with very early pivot (limited data)
        early_pivot = '2026-12-05'
        result = naive_forecast_full_window(
            extended_sample_data, 
            pivot=early_pivot, 
            n_days_to_average=3,
            forecast_period=30
        )
        assert len(result) > 0
        
        # Test with large forecast period
        result = naive_forecast_full_window(
            extended_sample_data, 
            pivot=pivot, 
            forecast_period=365,
            training_window=90
        )
        assert len(result) > 0
        unique_dates = result['date'].unique()
        assert len(unique_dates) > 400  # Should have ~90 training + 365 forecast days

    def test_full_window_empty_data(self):
        """Test full window function with empty data."""
        empty_data = pd.DataFrame(columns=['date', 'country', 'variant', 'sequences'])
        
        # Should raise ValueError due to input validation
        with pytest.raises(ValueError, match="cannot be empty"):
            naive_forecast_full_window(
                empty_data, 
                pivot='2027-04-01', 
                forecast_period=30
            )


class TestNaiveForecastingIntegration:
    """Integration tests for NAIVE forecasting functions."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic time series data mimicking flu evolution."""
        dates = pd.date_range('2026-10-01', '2027-06-01', freq='D')  # ~8 months
        data = []
        
        countries = ['north', 'south', 'tropics']
        variants = ['variant_1', 'variant_2', 'variant_3', 'variant_4']
        
        for i, date in enumerate(dates):
            for country in countries:
                # Create realistic variant dynamics
                total_sequences = np.random.poisson(50)  # Variable sampling intensity
                
                # Variant frequencies change over time
                if i < 60:  # First 2 months
                    probs = [0.7, 0.2, 0.08, 0.02]
                elif i < 120:  # Next 2 months
                    probs = [0.5, 0.35, 0.12, 0.03]
                elif i < 180:  # Next 2 months  
                    probs = [0.3, 0.45, 0.2, 0.05]
                else:  # Last 2+ months
                    probs = [0.15, 0.35, 0.35, 0.15]
                
                # Add country-specific variation
                if country == 'tropics':
                    probs = [p * 0.9 for p in probs[:-1]] + [probs[-1] * 1.4]  # More variant_4
                elif country == 'north':
                    probs = [probs[0] * 1.1] + [p * 0.95 for p in probs[1:]]  # More variant_1
                
                # Normalize probabilities
                probs = [p / sum(probs) for p in probs]
                
                # Generate counts
                counts = np.random.multinomial(total_sequences, probs)
                
                for variant, count in zip(variants, counts):
                    if count > 0:  # Only include variants with sequences
                        data.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'country': country,
                            'variant': variant,
                            'sequences': count
                        })
        
        return pd.DataFrame(data)

    def test_realistic_forecasting_scenario(self, realistic_data):
        """Test both functions with realistic flu evolution data."""
        pivot = '2027-04-01'
        
        # Test original function
        original = naive_forecast(realistic_data, pivot=pivot, n_days_to_average=14, period=60)
        
        # Test full window function
        full_window = naive_forecast_full_window(
            realistic_data, 
            pivot=pivot, 
            n_days_to_average=14,
            forecast_period=180,
            training_window=None
        )
        
        # Both should work without errors
        assert len(original) > 0
        assert len(full_window) > 0
        
        # Full window should have much more data
        assert len(full_window) > len(original) * 2
        
        # Both should have consistent structure
        for result in [original, full_window]:
            # Check frequency constraints
            for country in result['country'].unique():
                for date in result['date'].unique():
                    country_date_data = result[(result['country'] == country) & (result['date'] == date)]
                    if len(country_date_data) > 0:
                        freq_sum = country_date_data['freq'].sum()
                        assert 0.99 <= freq_sum <= 1.01, f"Frequencies don't sum to 1: {freq_sum}"

    def test_performance_comparison(self, realistic_data):
        """Test that both functions produce reasonable and comparable results."""
        pivot = '2027-04-01'
        
        # Generate results
        full_result = naive_forecast_full_window(
            realistic_data, 
            pivot=pivot, 
            n_days_to_average=7,
            forecast_period=90
        )
        
        # Check that we have reasonable coverage
        unique_dates = full_result['date'].unique()
        pivot_dt = pd.to_datetime(pivot)
        
        # Count training vs forecast dates
        training_dates = [d for d in unique_dates if pd.to_datetime(d) < pivot_dt]
        forecast_dates = [d for d in unique_dates if pd.to_datetime(d) >= pivot_dt]
        
        assert len(training_dates) > 30, f"Too few training dates: {len(training_dates)}"
        assert len(forecast_dates) <= 90, f"Too many forecast dates: {len(forecast_dates)}"
        
        # Verify that forecast maintains realistic frequencies
        forecast_data = full_result[full_result['median_freq_forecast'].notna()]
        for country in forecast_data['country'].unique():
            country_data = forecast_data[forecast_data['country'] == country]
            # Each country should have reasonable variant distribution
            variant_means = country_data.groupby('variant')['median_freq_forecast'].mean()
            assert len(variant_means) > 0
            assert abs(variant_means.sum() - 1.0) < 0.02  # Should sum to ~1


class TestInputValidation:
    """Test suite for input validation functions."""
    
    def test_validate_naive_forecast_input_valid(self):
        """Test validation with valid inputs."""
        data = pd.DataFrame({
            'date': ['2027-01-01', '2027-01-02'],
            'country': ['north', 'north'],
            'variant': ['A', 'B'],
            'sequences': [10, 20]
        })
        
        result = _validate_naive_forecast_input(data, '2027-01-01', 7)
        assert isinstance(result, pd.Timestamp)
        assert result == pd.to_datetime('2027-01-01')
    
    def test_validate_missing_columns(self):
        """Test validation fails with missing columns."""
        data = pd.DataFrame({
            'date': ['2027-01-01'],
            'country': ['north'],
            'variant': ['A']
            # Missing 'sequences' column
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_naive_forecast_input(data, '2027-01-01', 7)
    
    def test_validate_empty_data(self):
        """Test validation fails with empty data."""
        data = pd.DataFrame(columns=['date', 'country', 'variant', 'sequences'])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_naive_forecast_input(data, '2027-01-01', 7)
    
    def test_validate_invalid_pivot_date(self):
        """Test validation fails with invalid pivot date."""
        data = pd.DataFrame({
            'date': ['2027-01-01'],
            'country': ['north'],
            'variant': ['A'],
            'sequences': [10]
        })
        
        with pytest.raises(ValueError, match="Invalid pivot date format"):
            _validate_naive_forecast_input(data, 'invalid-date', 7)
    
    def test_validate_invalid_n_days(self):
        """Test validation fails with invalid n_days_to_average."""
        data = pd.DataFrame({
            'date': ['2027-01-01'],
            'country': ['north'], 
            'variant': ['A'],
            'sequences': [10]
        })
        
        with pytest.raises(ValueError, match="must be positive"):
            _validate_naive_forecast_input(data, '2027-01-01', 0)
            
        with pytest.raises(ValueError, match="must be positive"):
            _validate_naive_forecast_input(data, '2027-01-01', -1)


class TestHelperFunctions:
    """Test suite for helper functions."""
    
    def test_calculate_variant_frequencies(self):
        """Test frequency calculation helper."""
        data = pd.DataFrame({
            'date': ['2027-01-01', '2027-01-01', '2027-01-02', '2027-01-02'],
            'country': ['north', 'north', 'north', 'north'],
            'variant': ['A', 'B', 'A', 'B'],
            'sequences': [30, 20, 40, 10]
        })
        
        result = _calculate_variant_frequencies(data)
        
        # Check that frequencies are calculated correctly
        jan1_data = result[result['date'] == '2027-01-01']
        assert abs(jan1_data[jan1_data['variant'] == 'A']['freq'].iloc[0] - 0.6) < 0.001
        assert abs(jan1_data[jan1_data['variant'] == 'B']['freq'].iloc[0] - 0.4) < 0.001
        
        jan2_data = result[result['date'] == '2027-01-02'] 
        assert abs(jan2_data[jan2_data['variant'] == 'A']['freq'].iloc[0] - 0.8) < 0.001
        assert abs(jan2_data[jan2_data['variant'] == 'B']['freq'].iloc[0] - 0.2) < 0.001
    
    def test_calculate_variant_frequencies_division_by_zero(self):
        """Test frequency calculation handles division by zero."""
        data = pd.DataFrame({
            'date': ['2027-01-01', '2027-01-01'],
            'country': ['north', 'north'],
            'variant': ['A', 'B'],
            'sequences': [0, 0]  # Zero sequences
        })
        
        result = _calculate_variant_frequencies(data)
        
        # Should handle zero division gracefully
        assert result['freq'].iloc[0] == 0.0
        assert result['freq'].iloc[1] == 0.0
    
    def test_rolling_average_optimized_basic(self):
        """Test rolling average calculation helper."""
        # Create test data
        data = pd.DataFrame({
            'date': pd.date_range('2027-01-01', periods=10, freq='D'),
            'country': ['north'] * 10,
            'variant': ['A'] * 10,
            'sequences': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            'freq': [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        })
        
        target_dates = pd.date_range('2027-01-05', periods=3, freq='D')
        pivot_dt = pd.to_datetime('2027-01-07')
        
        result = _calculate_rolling_average_optimized(
            data, target_dates, n_days=3, pivot_dt=pivot_dt, use_fixed_forecast=False
        )
        
        assert len(result) > 0
        assert 'freq' in result.columns
        assert all(result['freq'] >= 0)
        assert all(result['freq'] <= 1)


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling in refactored functions."""
    
    def test_naive_forecast_validation_errors(self):
        """Test that main functions properly handle validation errors."""
        # Test with missing columns
        bad_data = pd.DataFrame({'date': ['2027-01-01'], 'country': ['north']})
        
        with pytest.raises(ValueError):
            naive_forecast(bad_data, '2027-01-01')
            
        with pytest.raises(ValueError):
            naive_forecast_full_window(bad_data, '2027-01-01')
    
    def test_parameter_validation(self):
        """Test parameter validation in main functions."""
        data = pd.DataFrame({
            'date': ['2027-01-01'],
            'country': ['north'],
            'variant': ['A'],
            'sequences': [10]
        })
        
        # Test negative period
        with pytest.raises(ValueError, match="must be positive"):
            naive_forecast(data, '2027-01-01', period=-1)
            
        # Test negative forecast_period
        with pytest.raises(ValueError, match="must be positive"):
            naive_forecast_full_window(data, '2027-01-01', forecast_period=-1)
            
        # Test negative training_window
        with pytest.raises(ValueError, match="must be positive"):
            naive_forecast_full_window(data, '2027-01-01', training_window=-1)
    
    def test_single_variant_single_country(self):
        """Test functions work with minimal data."""
        data = pd.DataFrame({
            'date': ['2027-01-01', '2027-01-02', '2027-01-03'],
            'country': ['north', 'north', 'north'],
            'variant': ['A', 'A', 'A'],
            'sequences': [10, 20, 30]
        })
        
        result1 = naive_forecast(data, '2027-01-02', n_days_to_average=2, period=2)
        result2 = naive_forecast_full_window(data, '2027-01-02', n_days_to_average=2, forecast_period=2)
        
        assert len(result1) > 0
        assert len(result2) > 0
        
        # Frequencies should all be 1.0 (single variant)
        assert all(abs(result1['freq'] - 1.0) < 0.001)
        assert all(abs(result2['freq'] - 1.0) < 0.001)