"""
Test suite for filtering functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from antigentools.scoring.filters import (
    filter_active_variants,
    filter_by_frequency,
    filter_missing_data
)


class TestFilterActiveVariants:
    """Test active variant filtering."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2027-01-01', '2027-04-01', freq='D')
        data = []
        
        # Create data with different patterns
        for date in dates:
            # Variant A: Active throughout
            data.append({
                'date': date,
                'country': 'north',
                'variant': 'A',
                'sequences': np.random.poisson(20)
            })
            
            # Variant B: Dies out in February
            if date < pd.Timestamp('2027-02-15'):
                data.append({
                    'date': date,
                    'country': 'north',
                    'variant': 'B',
                    'sequences': np.random.poisson(10)
                })
            
            # Variant C: Emerges in March
            if date >= pd.Timestamp('2027-03-01'):
                data.append({
                    'date': date,
                    'country': 'north',
                    'variant': 'C',
                    'sequences': np.random.poisson(15)
                })
        
        return pd.DataFrame(data)
    
    def test_filter_active_basic(self, sample_data):
        """Test basic active variant filtering."""
        pivot_date = '2027-04-01'
        
        result = filter_active_variants(
            sample_data,
            pivot_date=pivot_date,
            lookback_days=30,
            min_observations=5,
            min_sequences=10
        )
        
        # Only variants A and C should be active in March
        active_variants = result['variant'].unique()
        assert 'A' in active_variants
        assert 'B' not in active_variants  # Died out
        assert 'C' in active_variants
    
    def test_filter_active_edge_cases(self, sample_data):
        """Test edge cases for active variant filtering."""
        # Test with very early pivot (no lookback data)
        early_result = filter_active_variants(
            sample_data,
            pivot_date='2027-01-02',
            lookback_days=30
        )
        assert len(early_result) == 0
        
        # Test with no active variants (very high thresholds)
        no_active = filter_active_variants(
            sample_data,
            pivot_date='2027-04-01',
            lookback_days=30,
            min_observations=1000,
            min_sequences=10000
        )
        assert len(no_active) == 0
    
    def test_filter_active_validation(self):
        """Test input validation."""
        df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            filter_active_variants(df, '2027-01-01')
        
        # Test invalid date format
        valid_df = pd.DataFrame({
            'date': ['2027-01-01'],
            'country': ['north'],
            'variant': ['A'],
            'sequences': [10]
        })
        
        with pytest.raises(ValueError, match="Invalid pivot date"):
            filter_active_variants(valid_df, 'invalid-date')
    
    def test_filter_active_performance(self, sample_data):
        """Test optimized filtering performance."""
        # Add more countries and variants for performance testing
        expanded_data = []
        for country in ['north', 'south', 'tropics']:
            country_data = sample_data.copy()
            country_data['country'] = country
            expanded_data.append(country_data)
        
        large_df = pd.concat(expanded_data)
        
        # Should handle large datasets efficiently
        result = filter_active_variants(
            large_df,
            pivot_date='2027-04-01',
            lookback_days=30
        )
        
        assert len(result) > 0
        # Check that filtering preserved data structure
        assert set(result.columns) == set(large_df.columns)


class TestFilterByFrequency:
    """Test frequency-based filtering."""
    
    def test_filter_frequency_basic(self):
        """Test basic frequency filtering."""
        df = pd.DataFrame({
            'variant': ['A', 'B', 'C', 'D'],
            'smoothed_freq': [0.5, 0.01, 0.005, 0.1],
            'pred_freq': [0.4, 0.015, 0.02, 0.09]
        })
        
        result = filter_by_frequency(df, min_freq_threshold=0.01)
        
        # Should keep A, B (smoothed >= 0.01), C (pred >= 0.01), and D
        assert len(result) == 4
        
        result_strict = filter_by_frequency(df, min_freq_threshold=0.1)
        # Should keep only A and D
        assert len(result_strict) == 2
        assert set(result_strict['variant']) == {'A', 'D'}
    
    def test_filter_frequency_validation(self):
        """Test frequency filter validation."""
        df = pd.DataFrame({'variant': ['A']})
        
        # Missing required columns
        with pytest.raises(ValueError, match="Missing required columns"):
            filter_by_frequency(df, 0.01)
        
        # Invalid threshold
        valid_df = pd.DataFrame({
            'smoothed_freq': [0.5],
            'pred_freq': [0.5]
        })
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            filter_by_frequency(valid_df, -0.1)
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            filter_by_frequency(valid_df, 1.5)


class TestFilterMissingData:
    """Test missing data filtering."""
    
    def test_filter_missing_any(self):
        """Test filtering with 'any' mode."""
        df = pd.DataFrame({
            'col1': [1.0, 2.0, np.nan, 4.0],
            'col2': [1.0, np.nan, 3.0, 4.0],
            'col3': [1.0, 2.0, 3.0, 4.0]
        })
        
        result = filter_missing_data(df, ['col1', 'col2'], how='any')
        
        # Should remove rows with ANY NaN in specified columns
        assert len(result) == 2  # Only first and last row remain
        assert result.index.tolist() == [0, 3]
    
    def test_filter_missing_all(self):
        """Test filtering with 'all' mode."""
        df = pd.DataFrame({
            'col1': [1.0, np.nan, np.nan, 4.0],
            'col2': [1.0, np.nan, 3.0, 4.0],
            'col3': [1.0, 2.0, 3.0, 4.0]
        })
        
        result = filter_missing_data(df, ['col1', 'col2'], how='all')
        
        # Should only remove rows where ALL specified columns are NaN
        assert len(result) == 3  # Only second row removed
        assert result.index.tolist() == [0, 2, 3]
    
    def test_filter_missing_validation(self):
        """Test missing data filter validation."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Non-existent columns
        with pytest.raises(ValueError, match="Columns not found"):
            filter_missing_data(df, ['col1', 'col2'])
        
        # Invalid 'how' parameter
        with pytest.raises(ValueError, match="how must be"):
            filter_missing_data(df, ['col1'], how='invalid')