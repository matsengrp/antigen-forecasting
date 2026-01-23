"""Unit tests for antigentools.utils module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from typing import Dict, Tuple
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from antigentools.utils import (
    read_estimates,
    map_dates,
    filter_to_lifespan,
    prune_fitness_dataframe,
    get_gamma_distribution_params,
    smooth_with_spline,
    convert_rt_to_growth_rate,
    calculate_distribution_entropy,
    get_deme_stats,
    get_outliers,
    calculate_sign_disagreement_rate,
    calculate_overestimation_rate,
)


class TestReadEstimates:
    """Test read_estimates function."""

    def test_read_estimates_single_file(self):
        """Test reading estimates from a single file."""
        # Create a temporary TSV file (read_estimates uses sep="\t")
        test_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'variant': [1, 2],
            'median_R': [1.1, 1.2]
        })

        # Use proper path format for pivot_date extraction: path/to/estimates_2023-01-01.tsv
        with tempfile.NamedTemporaryFile(mode='w', suffix='_2023-01-01.tsv', delete=False) as f:
            test_data.to_csv(f.name, index=False, sep='\t')

            result = read_estimates([f.name])

            assert len(result) == 2
            assert 'pivot_date' in result.columns
            assert result['variant'].tolist() == [1, 2]

        os.unlink(f.name)

    def test_read_estimates_multiple_files(self):
        """Test reading estimates from multiple files."""
        test_data1 = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'variant': [1, 2],
            'median_R': [1.1, 1.2]
        })

        test_data2 = pd.DataFrame({
            'date': ['2023-02-01', '2023-02-02'],
            'variant': [3, 4],
            'median_R': [1.3, 1.4]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='_2023-01-01.tsv', delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode='w', suffix='_2023-02-01.tsv', delete=False) as f2:

            test_data1.to_csv(f1.name, index=False, sep='\t')
            test_data2.to_csv(f2.name, index=False, sep='\t')

            result = read_estimates([f1.name, f2.name])

            assert len(result) == 4
            assert set(result['variant'].tolist()) == {1, 2, 3, 4}

        os.unlink(f1.name)
        os.unlink(f2.name)

    def test_read_estimates_empty_list(self):
        """Test reading estimates with empty file list raises ValueError."""
        # Implementation calls pd.concat([]) which raises ValueError
        with pytest.raises(ValueError, match="No objects to concatenate"):
            read_estimates([])


class TestMapDates:
    """Test map_dates function."""

    def test_map_dates_forward(self):
        """Test forward date mapping."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })

        mapping = map_dates(df)

        assert len(mapping) == 3
        # Keys are the unique dates from the dataframe
        dates = sorted(df['date'].unique())
        assert mapping[dates[0]] == 0
        assert mapping[dates[1]] == 1
        assert mapping[dates[2]] == 2

    def test_map_dates_reverse(self):
        """Test reverse date mapping."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })

        mapping = map_dates(df, reverse_mapping=True)

        assert len(mapping) == 3
        assert mapping[0] == pd.Timestamp('2023-01-01')
        assert mapping[1] == pd.Timestamp('2023-01-02')
        assert mapping[2] == pd.Timestamp('2023-01-03')


class TestFilterToLifespan:
    """Test filter_to_lifespan function."""

    def test_filter_to_lifespan_basic(self):
        """Test basic lifespan filtering."""
        # Implementation uses integer variant IDs and query with unquoted variant
        df = pd.DataFrame({
            'variant': [1, 1, 1, 1],
            'location': ['north', 'north', 'north', 'north'],
            't': [1, 2, 3, 4],
            'fitness': [0.1, 0.2, 0.3, 0.4],
            'seasonal_fitness': [0.1, 0.2, 0.3, 0.4]
        })

        # Lifespan: variant 1 in north from t=2 to t=3
        result = filter_to_lifespan(df, 1, 'north', (2, 3))

        assert pd.isna(result.iloc[0]['fitness'])  # t=1, outside lifespan
        assert result.iloc[1]['fitness'] == 0.2    # t=2, inside lifespan
        assert result.iloc[2]['fitness'] == 0.3    # t=3, inside lifespan
        assert pd.isna(result.iloc[3]['fitness'])  # t=4, outside lifespan

    def test_filter_to_lifespan_no_matching_data(self):
        """Test filtering when no data matches variant/deme."""
        df = pd.DataFrame({
            'variant': [2, 2],
            'location': ['south', 'south'],
            't': [1, 2],
            'fitness': [0.1, 0.2],
            'seasonal_fitness': [0.1, 0.2]
        })

        # Query for variant 1 in north - no matches, returns empty df
        result = filter_to_lifespan(df, 1, 'north', (1, 2))

        # Should return empty dataframe since no matching data
        assert len(result) == 0


class TestPruneFitnessDataframe:
    """Test prune_fitness_dataframe function."""

    def test_prune_fitness_dataframe_basic(self):
        """Test basic fitness dataframe pruning.

        Note: The implementation requires lifespan_df to have entries for ALL
        variant/location combinations that exist in fitness_df.
        """
        # Use single variant/location to avoid cross-product issue
        fitness_df = pd.DataFrame({
            'variant': [1, 1, 1, 1],
            'location': ['north', 'north', 'north', 'north'],
            't': [1, 2, 3, 4],
            'fitness': [0.1, 0.2, 0.3, 0.4],
            'seasonal_fitness': [0.1, 0.2, 0.3, 0.4]
        })

        # Implementation expects 'birth' and 'death' columns
        lifespan_df = pd.DataFrame({
            'variant': [1],
            'location': ['north'],
            'birth': [2],
            'death': [3]
        })

        result = prune_fitness_dataframe(fitness_df, lifespan_df)

        # Variant 1 in north: t=1 outside, t=2-3 inside, t=4 outside
        assert pd.isna(result[result['t'] == 1]['fitness'].iloc[0])
        assert result[result['t'] == 2]['fitness'].iloc[0] == 0.2
        assert result[result['t'] == 3]['fitness'].iloc[0] == 0.3
        assert pd.isna(result[result['t'] == 4]['fitness'].iloc[0])


class TestGammaDistributionParams:
    """Test get_gamma_distribution_params function."""

    def test_gamma_params_basic(self):
        """Test gamma distribution parameter calculation."""
        mean, std = 2.0, 1.0
        shape, scale = get_gamma_distribution_params(mean, std)

        # Check that parameters give correct mean and std
        calculated_mean = shape * scale
        calculated_var = shape * scale**2
        calculated_std = np.sqrt(calculated_var)

        assert abs(calculated_mean - mean) < 1e-10
        assert abs(calculated_std - std) < 1e-10

    def test_gamma_params_edge_cases(self):
        """Test gamma distribution parameters with edge cases."""
        # Test with very small std
        shape, scale = get_gamma_distribution_params(2.0, 0.1)
        assert shape > 0
        assert scale > 0

        # Test with large std
        shape, scale = get_gamma_distribution_params(1.0, 2.0)
        assert shape > 0
        assert scale > 0


class TestSmoothWithSpline:
    """Test smooth_with_spline function."""

    def test_smooth_with_spline_basic(self):
        """Test basic spline smoothing."""
        # Implementation requires 'country' column for grouping
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'country': ['north'] * 10,
            'variant': ['A'] * 10,
            'sequences': [1, 2, 4, 3, 5, 7, 6, 8, 9, 10]
        })

        result = smooth_with_spline(df)

        assert 'smoothed_sequences' in result.columns
        assert len(result) == 10
        # Smoothed values should be positive
        assert all(result['smoothed_sequences'] > 0)

    def test_smooth_with_spline_insufficient_data(self):
        """Test spline smoothing with insufficient data points."""
        # With only 2 points and k=3 (default), spline fitting is skipped
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=2),
            'country': ['north'] * 2,
            'variant': ['A'] * 2,
            'sequences': [1, 2]
        })

        # Should handle gracefully with few data points
        result = smooth_with_spline(df)
        assert 'smoothed_sequences' in result.columns


class TestConvertRtToGrowthRate:
    """Test convert_rt_to_growth_rate function."""

    def test_convert_rt_basic(self):
        """Test basic Rt to growth rate conversion."""
        rt_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'variant': ['A'] * 3,
            'median_R': [1.1, 1.2, 1.3]
        })

        result = convert_rt_to_growth_rate(rt_df)

        # Implementation returns 'growth_rate_r' column
        assert 'growth_rate_r' in result.columns
        assert len(result) == 3
        # Growth rates should be positive when R > 1
        assert all(result['growth_rate_r'] > 0)

    def test_convert_rt_below_one(self):
        """Test Rt to growth rate conversion when R < 1."""
        rt_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=2),
            'variant': ['A'] * 2,
            'median_R': [0.8, 0.9]
        })

        result = convert_rt_to_growth_rate(rt_df)

        # Growth rates should be negative when R < 1
        assert all(result['growth_rate_r'] < 0)


class TestCalculateDistributionEntropy:
    """Test calculate_distribution_entropy function."""

    def test_entropy_basic(self):
        """Test basic entropy calculation."""
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
            'variant': ['A', 'B', 'A', 'B'],
            'sequences': [50, 50, 25, 75],  # Equal vs unequal distribution
            'location': ['north'] * 4
        })

        entropy, norm_entropy = calculate_distribution_entropy(df, 'sequences', location='north')

        assert entropy >= 0
        assert 0 <= norm_entropy <= 1
        assert isinstance(entropy, float)
        assert isinstance(norm_entropy, float)


class TestGetOutliers:
    """Test get_outliers function."""

    def test_get_outliers_basic(self):
        """Test basic outlier detection."""
        # Create data with clear outliers
        df = pd.DataFrame({
            'model': ['A'] * 10,
            'location': ['north'] * 10,
            'metric': [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]  # 10 is clearly an outlier
        })

        outliers = get_outliers(df, 'metric')

        assert len(outliers) > 0
        assert 10 in outliers['metric'].values

    def test_get_outliers_no_outliers(self):
        """Test outlier detection with no outliers."""
        df = pd.DataFrame({
            'model': ['A'] * 5,
            'location': ['north'] * 5,
            'metric': [1, 1, 1, 1, 1]
        })

        outliers = get_outliers(df, 'metric')

        assert len(outliers) == 0


class TestSignDisagreementRate:
    """Test calculate_sign_disagreement_rate function."""

    def test_sign_disagreement_basic(self):
        """Test basic sign disagreement calculation."""
        # Row 0: data=1 (+), model=1.5 (+) -> agree
        # Row 1: data=-1 (-), model=1.5 (+) -> DISAGREE
        # Row 2: data=2 (+), model=2.5 (+) -> agree
        # Row 3: data=-2 (-), model=-1.5 (-) -> agree
        # Only 1 disagreement out of 4
        df = pd.DataFrame({
            'growth_rate_r_data': [1, -1, 2, -2],
            'median_r': [1.5, 1.5, 2.5, -1.5]
        })

        rate = calculate_sign_disagreement_rate(df)

        assert rate == 0.25  # 1 out of 4 disagree

    def test_sign_disagreement_all_agree(self):
        """Test sign disagreement when all signs agree."""
        df = pd.DataFrame({
            'growth_rate_r_data': [1, 2, 3],
            'median_r': [1.5, 2.5, 3.5]
        })

        rate = calculate_sign_disagreement_rate(df)

        assert rate == 0.0


class TestOverestimationRate:
    """Test calculate_overestimation_rate function."""

    def test_overestimation_basic(self):
        """Test basic overestimation rate calculation."""
        df = pd.DataFrame({
            'growth_rate_r_data': [1, 2, 3, 4],
            'median_r': [1.5, 1.5, 3.5, 3.5]  # 2 overestimations
        })

        rate = calculate_overestimation_rate(df)

        assert rate == 0.5  # 2 out of 4 overestimate

    def test_overestimation_with_tolerance(self):
        """Test overestimation rate with tolerance."""
        df = pd.DataFrame({
            'growth_rate_r_data': [1.0, 2.0],
            'median_r': [1.005, 2.1]  # One within tolerance, one overestimate
        })

        rate = calculate_overestimation_rate(df, tol=0.01)

        assert rate == 0.5  # Only second one counts as overestimation


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample dataframe for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'variant': ['A'] * 5 + ['B'] * 5,
        'location': ['north'] * 10,
        'sequences': range(1, 11),
        'cases': range(10, 20)
    })


if __name__ == "__main__":
    pytest.main([__file__])
