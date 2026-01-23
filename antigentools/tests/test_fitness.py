"""Unit tests for fitness calculation functions."""

import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from antigentools.analysis import (
    calculate_fitness_of_tips,
    calc_variance_over_time
)


class TestCalculateFitnessOfTips:
    """Test calculate_fitness_of_tips function."""

    def test_basic_fitness_calculation(self):
        """Test basic fitness calculation with known values."""
        tips_df = pd.DataFrame({
            'ag1': [10.0, 12.0],
            'ag2': [0.0, 0.0]
        })
        host_coords = (8.0, 0.0)

        result = calculate_fitness_of_tips(tips_df, host_coords)

        assert 'fitness' in result.columns
        # Distance = 2.0 and 4.0, risk = 0.14 and 0.28
        assert np.isclose(result['fitness'].iloc[0], 0.14, atol=0.01)
        assert np.isclose(result['fitness'].iloc[1], 0.28, atol=0.01)

    def test_fitness_bounded_lower(self):
        """Test that fitness is bounded at min_risk = 1 - homologous_immunity."""
        tips_df = pd.DataFrame({
            'ag1': [8.0],  # Same as host
            'ag2': [0.0]
        })
        host_coords = (8.0, 0.0)

        result = calculate_fitness_of_tips(tips_df, host_coords, homologous_immunity=0.95)

        # Distance = 0, so risk should be clamped to 0.05
        assert np.isclose(result['fitness'].iloc[0], 0.05)

    def test_fitness_bounded_upper(self):
        """Test that fitness is bounded at 1.0."""
        tips_df = pd.DataFrame({
            'ag1': [100.0],  # Very far from host
            'ag2': [100.0]
        })
        host_coords = (0.0, 0.0)

        result = calculate_fitness_of_tips(tips_df, host_coords)

        # Distance is very large, risk should be clamped to 1.0
        assert result['fitness'].iloc[0] == 1.0

    def test_custom_smith_conversion(self):
        """Test fitness with custom smith conversion factor."""
        tips_df = pd.DataFrame({
            'ag1': [10.0],
            'ag2': [0.0]
        })
        host_coords = (0.0, 0.0)

        # Distance = 10.0, with s=0.1, risk = 1.0 (clamped)
        result = calculate_fitness_of_tips(tips_df, host_coords, s=0.1)
        assert result['fitness'].iloc[0] == 1.0

        # Distance = 10.0, with s=0.01, risk = 0.1
        result = calculate_fitness_of_tips(tips_df, host_coords, s=0.01)
        assert np.isclose(result['fitness'].iloc[0], 0.1, atol=0.01)

    def test_preserves_original_columns(self):
        """Test that original DataFrame columns are preserved."""
        tips_df = pd.DataFrame({
            'ag1': [10.0],
            'ag2': [0.0],
            'name': ['tip1'],
            'year': [2023]
        })
        host_coords = (8.0, 0.0)

        result = calculate_fitness_of_tips(tips_df, host_coords)

        assert 'name' in result.columns
        assert 'year' in result.columns
        assert result['name'].iloc[0] == 'tip1'

    def test_does_not_modify_original(self):
        """Test that original DataFrame is not modified."""
        tips_df = pd.DataFrame({
            'ag1': [10.0],
            'ag2': [0.0]
        })
        host_coords = (8.0, 0.0)

        calculate_fitness_of_tips(tips_df, host_coords)

        assert 'fitness' not in tips_df.columns


class TestCalcVarianceOverTime:
    """Test calc_variance_over_time function."""

    def test_basic_variance_calculation(self):
        """Test variance calculation with simple data."""
        tips_df = pd.DataFrame({
            'ag1': [10.0, 12.0, 11.0, 13.0],
            'ag2': [0.0, 0.0, 0.0, 0.0],
            'year': [0.5, 0.5, 0.5, 0.5],
            'variant_test': [0, 0, 1, 1]
        })
        host_memory_df = pd.DataFrame({
            'year': [1.0],
            'ag1': [8.0],
            'ag2': [0.0]
        })

        result = calc_variance_over_time(tips_df, host_memory_df, ['variant_test'])

        assert len(result) == 1
        assert 'year' in result.columns
        assert 'method' in result.columns
        assert 'mean_variance' in result.columns
        assert 'n_variants' in result.columns
        assert result['method'].iloc[0] == 'test'

    def test_multiple_timepoints(self):
        """Test variance calculation across multiple timepoints."""
        tips_df = pd.DataFrame({
            'ag1': [10.0, 12.0, 14.0, 16.0],
            'ag2': [0.0, 0.0, 0.0, 0.0],
            'year': [0.5, 0.5, 1.5, 1.5],
            'variant_test': [0, 1, 0, 1]
        })
        host_memory_df = pd.DataFrame({
            'year': [1.0, 2.0],
            'ag1': [8.0, 10.0],
            'ag2': [0.0, 0.0]
        })

        result = calc_variance_over_time(tips_df, host_memory_df, ['variant_test'])

        assert len(result) == 2
        assert sorted(result['year'].tolist()) == [1.0, 2.0]

    def test_multiple_variant_methods(self):
        """Test variance calculation with multiple variant assignment methods."""
        tips_df = pd.DataFrame({
            'ag1': [10.0, 12.0, 11.0, 13.0],
            'ag2': [0.0, 0.0, 0.0, 0.0],
            'year': [0.5, 0.5, 0.5, 0.5],
            'variant_ag': [0, 0, 1, 1],
            'variant_phylo': [0, 1, 0, 1]
        })
        host_memory_df = pd.DataFrame({
            'year': [1.0],
            'ag1': [8.0],
            'ag2': [0.0]
        })

        result = calc_variance_over_time(tips_df, host_memory_df, ['variant_ag', 'variant_phylo'])

        assert len(result) == 2
        methods = set(result['method'].tolist())
        assert methods == {'ag', 'phylo'}

    def test_homogeneous_variants_low_variance(self):
        """Test that homogeneous variants have lower variance."""
        # Homogeneous: tips within variants are close together
        tips_homogeneous = pd.DataFrame({
            'ag1': [10.0, 10.1, 20.0, 20.1],
            'ag2': [0.0, 0.0, 0.0, 0.0],
            'year': [0.5, 0.5, 0.5, 0.5],
            'variant_test': [0, 0, 1, 1]
        })
        # Heterogeneous: tips within variants are far apart
        tips_heterogeneous = pd.DataFrame({
            'ag1': [10.0, 20.0, 10.0, 20.0],
            'ag2': [0.0, 0.0, 0.0, 0.0],
            'year': [0.5, 0.5, 0.5, 0.5],
            'variant_test': [0, 0, 1, 1]
        })
        host_memory_df = pd.DataFrame({
            'year': [1.0],
            'ag1': [0.0],
            'ag2': [0.0]
        })

        result_homo = calc_variance_over_time(tips_homogeneous, host_memory_df, ['variant_test'])
        result_hetero = calc_variance_over_time(tips_heterogeneous, host_memory_df, ['variant_test'])

        assert result_homo['mean_variance'].iloc[0] < result_hetero['mean_variance'].iloc[0]

    def test_n_variants_counts_window_only(self):
        """Test that n_variants counts only tips in [t-1, t] window."""
        # Tips in year 0.5 have variants 0,1; tips in year 1.5 have variant 2
        tips_df = pd.DataFrame({
            'ag1': [10.0, 12.0, 14.0],
            'ag2': [0.0, 0.0, 0.0],
            'year': [0.5, 0.5, 1.5],
            'variant_test': [0, 1, 2]
        })
        host_memory_df = pd.DataFrame({
            'year': [1.0, 2.0],
            'ag1': [8.0, 10.0],
            'ag2': [0.0, 0.0]
        })

        result = calc_variance_over_time(tips_df, host_memory_df, ['variant_test'])

        # At t=1.0, window is (0, 1], contains tips at year 0.5 with variants 0,1
        # At t=2.0, window is (1, 2], contains tip at year 1.5 with variant 2
        result_t1 = result[result['year'] == 1.0]
        result_t2 = result[result['year'] == 2.0]
        assert result_t1['n_variants'].iloc[0] == 2
        assert result_t2['n_variants'].iloc[0] == 1

    def test_variance_uses_all_tips(self):
        """Test that variance is calculated using ALL tips, not just window."""
        # All tips have same variant, spread across years
        tips_df = pd.DataFrame({
            'ag1': [10.0, 12.0, 14.0],
            'ag2': [0.0, 0.0, 0.0],
            'year': [0.5, 1.5, 2.5],
            'variant_test': [0, 0, 0]
        })
        host_memory_df = pd.DataFrame({
            'year': [1.0, 2.0, 3.0],
            'ag1': [8.0, 8.0, 8.0],
            'ag2': [0.0, 0.0, 0.0]
        })

        result = calc_variance_over_time(tips_df, host_memory_df, ['variant_test'])

        # All timepoints should have same variance since all tips are used
        # and host coordinates are constant
        variances = result['mean_variance'].tolist()
        assert all(np.isclose(v, variances[0]) for v in variances)

    def test_output_long_format(self):
        """Test that output is in long format suitable for aggregation."""
        tips_df = pd.DataFrame({
            'ag1': [10.0, 12.0],
            'ag2': [0.0, 0.0],
            'year': [0.5, 0.5],
            'variant_test': [0, 1]
        })
        host_memory_df = pd.DataFrame({
            'year': [1.0],
            'ag1': [8.0],
            'ag2': [0.0]
        })

        result = calc_variance_over_time(tips_df, host_memory_df, ['variant_test'])

        # Should have exactly these columns for aggregation
        expected_cols = {'year', 'method', 'mean_variance', 'n_variants'}
        assert expected_cols.issubset(set(result.columns))

    def test_single_tip_variant_returns_nan(self):
        """Test that single-tip variants have NaN variance."""
        tips_df = pd.DataFrame({
            'ag1': [10.0, 12.0, 14.0],
            'ag2': [0.0, 0.0, 0.0],
            'year': [0.5, 0.5, 0.5],
            'variant_test': [0, 1, 2]  # Each variant has only 1 tip
        })
        host_memory_df = pd.DataFrame({
            'year': [1.0],
            'ag1': [8.0],
            'ag2': [0.0]
        })

        result = calc_variance_over_time(tips_df, host_memory_df, ['variant_test'])

        # Mean of NaN variances is NaN
        assert pd.isna(result['mean_variance'].iloc[0])


if __name__ == "__main__":
    pytest.main([__file__])
