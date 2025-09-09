"""Unit tests for antigentools.analysis module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from typing import List, Tuple, Optional
from unittest.mock import patch, MagicMock, mock_open

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from antigentools.analysis import (
    load_model_rt_values,
    get_variant_incidence,
    get_top_variants,
    get_growth_rates_df,
    get_filtered_growth_rates_df,
    diagnose_extreme_growth_rates,
    filter_growth_rates,
    evaluate_growth_rate_performance,
    calculate_variant_mae
)


class TestLoadModelRtValues:
    """Test load_model_rt_values function."""
    
    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_load_rt_values_success(self, mock_exists, mock_read_csv):
        """Test successful loading of RT values."""
        mock_exists.return_value = True
        mock_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'variant': [1, 2],
            'median_R': [1.1, 1.2]
        })
        mock_read_csv.return_value = mock_data
        
        result = load_model_rt_values('test_build', 'FGA', 'north', '2023-01-01')
        
        assert result is not None
        assert len(result) == 2
        pd.testing.assert_frame_equal(result, mock_data)
    
    @patch('os.path.exists')
    def test_load_rt_values_file_not_found(self, mock_exists):
        """Test loading RT values when file doesn't exist."""
        mock_exists.return_value = False
        
        result = load_model_rt_values('test_build', 'FGA', 'north', '2023-01-01')
        
        assert result is None
    
    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_load_rt_values_read_error(self, mock_exists, mock_read_csv):
        """Test loading RT values when read fails."""
        mock_exists.return_value = True
        mock_read_csv.side_effect = Exception("Read error")
        
        result = load_model_rt_values('test_build', 'FGA', 'north', '2023-01-01')
        
        assert result is None


class TestGetVariantIncidence:
    """Test get_variant_incidence function."""
    
    def test_get_variant_incidence_basic(self):
        """Test basic variant incidence extraction."""
        seqs_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=6),
            'variant': ['A', 'B', 'A', 'B', 'A', 'B'],
            'location': ['north'] * 6,
            'sequences': [10, 20, 15, 25, 12, 22]
        })
        
        result = get_variant_incidence(seqs_df, 'north', 'A')
        
        assert len(result) == 3  # 3 entries for variant A
        assert result.tolist() == [10, 15, 12]
        assert result.name == 'sequences'
    
    def test_get_variant_incidence_no_matches(self):
        """Test variant incidence when no matches found."""
        seqs_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=2),
            'variant': ['A', 'A'],
            'location': ['north', 'north'],
            'sequences': [10, 15]
        })
        
        result = get_variant_incidence(seqs_df, 'south', 'B')
        
        assert len(result) == 0
    
    def test_get_variant_incidence_empty_dataframe(self):
        """Test variant incidence with empty dataframe."""
        seqs_df = pd.DataFrame(columns=['date', 'variant', 'location', 'sequences'])
        
        result = get_variant_incidence(seqs_df, 'north', 'A')
        
        assert len(result) == 0


class TestGetTopVariants:
    """Test get_top_variants function."""
    
    def test_get_top_variants_basic(self):
        """Test basic top variants selection."""
        df = pd.DataFrame({
            'variant': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'location': ['tropics'] * 9,
            'growth_rate_r': [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 0.5, 0.6, 0.7],
            'growth_rate_r_data': [1.05, 1.15, 1.25, 2.05, 2.15, 2.25, 0.45, 0.55, 0.65]
        })
        
        result = get_top_variants(df, location='tropics', n=2)
        
        assert len(result) == 2
        # Results should be sorted by correlation (descending)
        assert result[0][1] >= result[1][1]  # First correlation >= second correlation
        
        # Check that variants are strings and correlations are floats
        for variant, corr, mae, n_points in result:
            assert isinstance(variant, str)
            assert isinstance(corr, float)
            assert isinstance(mae, float)
            assert isinstance(n_points, int)
    
    def test_get_top_variants_insufficient_data(self):
        """Test top variants when variants have insufficient data points."""
        df = pd.DataFrame({
            'variant': ['A', 'A', 'B'],  # Variant B has only 1 point
            'location': ['tropics'] * 3,
            'growth_rate_r': [1.0, 1.1, 2.0],
            'growth_rate_r_data': [1.05, 1.15, 2.05]
        })
        
        result = get_top_variants(df, location='tropics', n=2, min_points=3)
        
        assert len(result) == 0  # No variants have enough points
    
    def test_get_top_variants_wrong_location(self):
        """Test top variants with non-matching location."""
        df = pd.DataFrame({
            'variant': ['A', 'A', 'A'],
            'location': ['north'] * 3,
            'growth_rate_r': [1.0, 1.1, 1.2],
            'growth_rate_r_data': [1.05, 1.15, 1.25]
        })
        
        result = get_top_variants(df, location='tropics', n=2)
        
        assert len(result) == 0


class TestGetGrowthRatesDF:
    """Test get_growth_rates_df function."""
    
    @patch('antigentools.analysis.load_model_rt_values')
    @patch('pandas.read_csv')
    def test_get_growth_rates_df_basic(self, mock_read_csv, mock_load_rt):
        """Test basic growth rates dataframe creation."""
        # Mock sequence data
        seq_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'variant': ['A'] * 5,
            'location': ['north'] * 5,
            'sequences': [10, 12, 15, 18, 22]
        })
        
        # Mock case data
        case_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'location': ['north'] * 5,
            'cases': [100, 120, 150, 180, 220]
        })
        
        # Mock RT data
        rt_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'variant': ['A'] * 5,
            'median_R': [1.1, 1.2, 1.3, 1.4, 1.5]
        })
        
        mock_read_csv.side_effect = [seq_data, case_data]
        mock_load_rt.return_value = rt_data
        
        with patch('os.path.exists', return_value=True):
            result = get_growth_rates_df('test_build', 'FGA', 'north', '2023-01-01')
        
        assert result is not None
        assert 'growth_rate_r_data' in result.columns
        assert 'median_r' in result.columns
        assert len(result) > 0
    
    @patch('antigentools.analysis.load_model_rt_values')
    def test_get_growth_rates_df_no_rt_data(self, mock_load_rt):
        """Test growth rates df when RT data is missing."""
        mock_load_rt.return_value = None
        
        result = get_growth_rates_df('test_build', 'FGA', 'north', '2023-01-01')
        
        assert result is None


class TestGetFilteredGrowthRatesDF:
    """Test get_filtered_growth_rates_df function."""
    
    @patch('antigentools.analysis.get_growth_rates_df')
    def test_get_filtered_growth_rates_df_basic(self, mock_get_growth_rates):
        """Test basic filtered growth rates dataframe."""
        # Mock growth rates data
        growth_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'variant': ['A'] * 10,
            'location': ['north'] * 10,
            'smoothed_sequences': [10, 12, 15, 18, 22, 25, 28, 30, 32, 35],
            'frequency': [0.1] * 10,
            'growth_rate_r_data': np.random.normal(0.1, 0.02, 10),
            'median_r': np.random.normal(0.1, 0.02, 10)
        })
        
        mock_get_growth_rates.return_value = growth_data
        
        result = get_filtered_growth_rates_df(
            'test_build', 'FGA', 'north', '2023-01-01',
            min_sequence_count=5, min_variant_frequency=0.05
        )
        
        assert result is not None
        assert len(result) > 0
        # Check that filtering criteria are applied
        assert all(result['smoothed_sequences'] >= 5)
        assert all(result['frequency'] >= 0.05)
    
    @patch('antigentools.analysis.get_growth_rates_df')
    def test_get_filtered_growth_rates_df_no_data(self, mock_get_growth_rates):
        """Test filtered growth rates df when no base data."""
        mock_get_growth_rates.return_value = None
        
        result = get_filtered_growth_rates_df(
            'test_build', 'FGA', 'north', '2023-01-01'
        )
        
        assert result is None


class TestDiagnoseExtremeGrowthRates:
    """Test diagnose_extreme_growth_rates function."""
    
    def test_diagnose_extreme_growth_rates_basic(self):
        """Test basic extreme growth rates diagnosis."""
        df = pd.DataFrame({
            'variant': ['A', 'A', 'B', 'B'],
            'growth_rate_r_data': [0.05, 0.15, -0.05, -0.15],  # B has extreme values
            'date': pd.date_range('2023-01-01', periods=4)
        })
        
        result = diagnose_extreme_growth_rates(df, threshold=0.1)
        
        assert len(result) > 0
        # Should identify variants with extreme values
        extreme_variants = result['variant'].unique()
        assert 'B' in extreme_variants  # B has values beyond threshold
    
    def test_diagnose_extreme_growth_rates_no_extremes(self):
        """Test diagnosis when no extreme growth rates."""
        df = pd.DataFrame({
            'variant': ['A', 'A', 'B', 'B'],
            'growth_rate_r_data': [0.05, 0.08, 0.06, 0.07],  # All within threshold
            'date': pd.date_range('2023-01-01', periods=4)
        })
        
        result = diagnose_extreme_growth_rates(df, threshold=0.1)
        
        assert len(result) == 0


class TestFilterGrowthRates:
    """Test filter_growth_rates function."""
    
    def test_filter_growth_rates_basic(self):
        """Test basic growth rate filtering."""
        df = pd.DataFrame({
            'variant': ['A'] * 10,
            'date': pd.date_range('2023-01-01', periods=10),
            'growth_rate_r_data': [1, 2, np.nan, np.nan, 5, 6, 7, np.nan, 9, 10],
            'median_r': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1]
        })
        
        result = filter_growth_rates(df, min_segment_length=3)
        
        # Should keep segments of length >= 3
        assert len(result) < len(df)  # Some rows should be filtered out
        assert not result['growth_rate_r_data'].isna().any()  # No NaN values in result
    
    def test_filter_growth_rates_connect_gaps(self):
        """Test growth rate filtering with gap connection."""
        df = pd.DataFrame({
            'variant': ['A'] * 6,
            'date': pd.date_range('2023-01-01', periods=6),
            'growth_rate_r_data': [1, 2, np.nan, 4, 5, 6],
            'median_r': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
        })
        
        result = filter_growth_rates(df, connect_gaps=True, min_segment_length=2)
        
        # With gap connection, should treat as one segment
        assert len(result) > 0
        # NaN in the middle should be filled or segment should be kept


class TestEvaluateGrowthRatePerformance:
    """Test evaluate_growth_rate_performance function."""
    
    def test_evaluate_performance_basic(self):
        """Test basic growth rate performance evaluation."""
        df = pd.DataFrame({
            'growth_rate_r_data': [1.0, 2.0, 3.0, 4.0],
            'median_r': [1.1, 1.9, 3.2, 3.8]
        })
        
        result = evaluate_growth_rate_performance(df)
        
        assert 'correlation' in result
        assert 'mae' in result
        assert 'rmse' in result
        assert 'sign_disagreement_rate' in result
        assert 'overestimation_rate' in result
        
        # Check value ranges
        assert -1 <= result['correlation'] <= 1
        assert result['mae'] >= 0
        assert result['rmse'] >= 0
        assert 0 <= result['sign_disagreement_rate'] <= 1
        assert 0 <= result['overestimation_rate'] <= 1
    
    def test_evaluate_performance_perfect_correlation(self):
        """Test performance evaluation with perfect correlation."""
        df = pd.DataFrame({
            'growth_rate_r_data': [1.0, 2.0, 3.0, 4.0],
            'median_r': [1.0, 2.0, 3.0, 4.0]
        })
        
        result = evaluate_growth_rate_performance(df)
        
        assert abs(result['correlation'] - 1.0) < 1e-10
        assert result['mae'] == 0.0
        assert result['rmse'] == 0.0
        assert result['sign_disagreement_rate'] == 0.0
    
    def test_evaluate_performance_empty_data(self):
        """Test performance evaluation with empty data."""
        df = pd.DataFrame(columns=['growth_rate_r_data', 'median_r'])
        
        result = evaluate_growth_rate_performance(df)
        
        # Should handle empty data gracefully
        assert all(pd.isna(val) or val == 0 for val in result.values())


class TestCalculateVariantMAE:
    """Test calculate_variant_mae function."""
    
    def test_calculate_variant_mae_basic(self):
        """Test basic variant MAE calculation."""
        df = pd.DataFrame({
            'variant': ['A', 'A', 'A', 'B', 'B', 'B'],
            'country': ['north'] * 6,
            'model': ['FGA'] * 6,
            'analysis_date': ['2023-01-01'] * 6,
            'growth_rate_r_data': [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            'median_r': [1.1, 1.9, 3.2, 1.4, 2.6, 3.4]
        })
        
        result = calculate_variant_mae(df)
        
        assert len(result) == 2  # Two variants
        assert all(col in result.columns for col in ['variant', 'mae', 'normalized_mae'])
        assert all(result['mae'] >= 0)
        assert all(result['normalized_mae'] >= 0)
        
        # Check that we have results for both variants
        variants = set(result['variant'].tolist())
        assert variants == {'A', 'B'}
    
    def test_calculate_variant_mae_single_variant(self):
        """Test MAE calculation with single variant."""
        df = pd.DataFrame({
            'variant': ['A'] * 3,
            'country': ['north'] * 3,
            'model': ['FGA'] * 3,
            'analysis_date': ['2023-01-01'] * 3,
            'growth_rate_r_data': [1.0, 2.0, 3.0],
            'median_r': [1.1, 1.9, 3.2]
        })
        
        result = calculate_variant_mae(df)
        
        assert len(result) == 1
        assert result.iloc[0]['variant'] == 'A'
        assert result.iloc[0]['mae'] > 0
    
    def test_calculate_variant_mae_perfect_predictions(self):
        """Test MAE calculation with perfect predictions."""
        df = pd.DataFrame({
            'variant': ['A'] * 3,
            'country': ['north'] * 3,
            'model': ['FGA'] * 3,
            'analysis_date': ['2023-01-01'] * 3,
            'growth_rate_r_data': [1.0, 2.0, 3.0],
            'median_r': [1.0, 2.0, 3.0]
        })
        
        result = calculate_variant_mae(df)
        
        assert len(result) == 1
        assert result.iloc[0]['mae'] == 0.0
        assert result.iloc[0]['sign_disagreement_rate'] == 0.0


@pytest.fixture
def sample_growth_rates_df():
    """Fixture providing a sample growth rates dataframe."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'variant': ['A'] * 5 + ['B'] * 5,
        'location': ['north'] * 10,
        'growth_rate_r_data': np.random.normal(0.1, 0.02, 10),
        'median_r': np.random.normal(0.1, 0.02, 10),
        'smoothed_sequences': np.random.randint(10, 100, 10),
        'frequency': np.random.uniform(0.1, 0.9, 10)
    })


@pytest.fixture
def sample_rt_df():
    """Fixture providing a sample RT dataframe."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'variant': ['A'] * 5,
        'median_R': [1.1, 1.2, 1.3, 1.4, 1.5],
        'q025': [1.0, 1.1, 1.2, 1.3, 1.4],
        'q975': [1.2, 1.3, 1.4, 1.5, 1.6]
    })


if __name__ == "__main__":
    pytest.main([__file__])