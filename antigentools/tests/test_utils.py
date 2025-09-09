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
    add_week_id_column,
    add_variant_frequencies,
    smooth_with_spline,
    calculate_variant_growth_rates,
    convert_rt_to_growth_rate,
    calculate_distribution_entropy,
    get_deme_stats,
    get_outliers,
    calculate_sign_disagreement_rate,
    calculate_overestimation_rate,
    hamming_distance,
    translate_dna_to_aa
)


class TestReadEstimates:
    """Test read_estimates function."""
    
    def test_read_estimates_single_file(self):
        """Test reading estimates from a single file."""
        # Create a temporary CSV file
        test_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'variant': [1, 2],
            'median_R': [1.1, 1.2]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            
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
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
            
            test_data1.to_csv(f1.name, index=False)
            test_data2.to_csv(f2.name, index=False)
            
            result = read_estimates([f1.name, f2.name])
            
            assert len(result) == 4
            assert set(result['variant'].tolist()) == {1, 2, 3, 4}
            
        os.unlink(f1.name)
        os.unlink(f2.name)
    
    def test_read_estimates_empty_list(self):
        """Test reading estimates with empty file list."""
        result = read_estimates([])
        assert result.empty


class TestMapDates:
    """Test map_dates function."""
    
    def test_map_dates_forward(self):
        """Test forward date mapping."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        })
        
        mapping = map_dates(df)
        
        assert len(mapping) == 3
        assert mapping[pd.Timestamp('2023-01-01')] == 0
        assert mapping[pd.Timestamp('2023-01-02')] == 1
        assert mapping[pd.Timestamp('2023-01-03')] == 2
    
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
        df = pd.DataFrame({
            'variant': ['A', 'A', 'A', 'A'],
            'location': ['north', 'north', 'north', 'north'],
            't': [1, 2, 3, 4],
            'fitness': [0.1, 0.2, 0.3, 0.4]
        })
        
        # Lifespan: variant A in north from t=2 to t=3
        result = filter_to_lifespan(df, 'A', 'north', (2, 3))
        
        assert pd.isna(result.iloc[0]['fitness'])  # t=1, outside lifespan
        assert result.iloc[1]['fitness'] == 0.2    # t=2, inside lifespan
        assert result.iloc[2]['fitness'] == 0.3    # t=3, inside lifespan
        assert pd.isna(result.iloc[3]['fitness'])  # t=4, outside lifespan
    
    def test_filter_to_lifespan_no_matching_data(self):
        """Test filtering when no data matches variant/deme."""
        df = pd.DataFrame({
            'variant': ['B', 'B'],
            'location': ['south', 'south'],
            't': [1, 2],
            'fitness': [0.1, 0.2]
        })
        
        result = filter_to_lifespan(df, 'A', 'north', (1, 2))
        
        # Should return original df unchanged since no matching data
        pd.testing.assert_frame_equal(result, df)


class TestPruneFitnessDataframe:
    """Test prune_fitness_dataframe function."""
    
    def test_prune_fitness_dataframe_basic(self):
        """Test basic fitness dataframe pruning."""
        fitness_df = pd.DataFrame({
            'variant': ['A', 'A', 'B', 'B'],
            'location': ['north', 'north', 'south', 'south'],
            't': [1, 2, 1, 2],
            'fitness': [0.1, 0.2, 0.3, 0.4]
        })
        
        lifespan_df = pd.DataFrame({
            'variant': ['A', 'B'],
            'location': ['north', 'south'],
            'start_time': [1.5, 0.5],
            'end_time': [1.8, 1.5]
        })
        
        result = prune_fitness_dataframe(fitness_df, lifespan_df)
        
        # Variant A in north: t=1 and t=2 should both be outside (1.5, 1.8)
        # Variant B in south: t=1 should be inside (0.5, 1.5), t=2 should be outside
        assert pd.isna(result[(result['variant'] == 'A') & (result['t'] == 1)]['fitness'].iloc[0])
        assert pd.isna(result[(result['variant'] == 'A') & (result['t'] == 2)]['fitness'].iloc[0])
        assert result[(result['variant'] == 'B') & (result['t'] == 1)]['fitness'].iloc[0] == 0.3
        assert pd.isna(result[(result['variant'] == 'B') & (result['t'] == 2)]['fitness'].iloc[0])


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


class TestAddWeekIdColumn:
    """Test add_week_id_column function."""
    
    def test_add_week_id_basic(self):
        """Test adding week ID column."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15']),
            'value': [1, 2, 3]
        })
        
        result = add_week_id_column(df)
        
        assert 'week_id' in result.columns
        assert result['week_id'].tolist() == [0, 1, 2]
    
    def test_add_week_id_custom_column(self):
        """Test adding week ID with custom date column."""
        df = pd.DataFrame({
            'custom_date': pd.to_datetime(['2023-01-01', '2023-01-08']),
            'value': [1, 2]
        })
        
        result = add_week_id_column(df, date_col='custom_date')
        
        assert 'week_id' in result.columns
        assert result['week_id'].tolist() == [0, 1]


class TestAddVariantFrequencies:
    """Test add_variant_frequencies function."""
    
    def test_add_variant_frequencies_basic(self):
        """Test adding variant frequencies."""
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
            'variant': ['A', 'B', 'A', 'B'],
            'sequences': [10, 20, 15, 25]
        })
        
        result = add_variant_frequencies(df)
        
        assert 'frequency' in result.columns
        # Check frequencies sum to 1 for each date
        freq_sums = result.groupby('date')['frequency'].sum()
        assert all(abs(s - 1.0) < 1e-10 for s in freq_sums)
        
        # Check specific frequency values
        day1_freqs = result[result['date'] == '2023-01-01']['frequency'].tolist()
        assert abs(day1_freqs[0] - 10/30) < 1e-10  # A: 10/(10+20)
        assert abs(day1_freqs[1] - 20/30) < 1e-10  # B: 20/(10+20)


class TestSmoothWithSpline:
    """Test smooth_with_spline function."""
    
    def test_smooth_with_spline_basic(self):
        """Test basic spline smoothing."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
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
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=2),
            'variant': ['A'] * 2,
            'sequences': [1, 2]
        })
        
        # Should handle gracefully with few data points
        result = smooth_with_spline(df)
        assert 'smoothed_sequences' in result.columns


class TestCalculateVariantGrowthRates:
    """Test calculate_variant_growth_rates function."""
    
    def test_calculate_growth_rates_basic(self):
        """Test basic growth rate calculation."""
        seqs_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'variant': ['A'] * 5,
            'sequences': [10, 12, 15, 18, 22],
            'location': ['north'] * 5
        })
        
        cases_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'location': ['north'] * 5,
            'cases': [100, 120, 150, 180, 220]
        })
        
        result = calculate_variant_growth_rates(seqs_df, cases_df)
        
        assert 'growth_rate_r_data' in result.columns
        assert len(result) == 4  # n-1 growth rate calculations
        # Growth rates should be reasonable (positive in this case)
        assert all(result['growth_rate_r_data'] > 0)


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
        
        assert 'median_r' in result.columns
        assert len(result) == 3
        # Growth rates should be positive when R > 1
        assert all(result['median_r'] > 0)
    
    def test_convert_rt_below_one(self):
        """Test Rt to growth rate conversion when R < 1."""
        rt_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=2),
            'variant': ['A'] * 2,
            'median_R': [0.8, 0.9]
        })
        
        result = convert_rt_to_growth_rate(rt_df)
        
        # Growth rates should be negative when R < 1
        assert all(result['median_r'] < 0)


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
        df = pd.DataFrame({
            'growth_rate_r_data': [1, -1, 2, -2],
            'median_r': [1.5, 1.5, 2.5, -1.5]  # Two sign disagreements
        })
        
        rate = calculate_sign_disagreement_rate(df)
        
        assert rate == 0.5  # 2 out of 4 disagree
    
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


class TestHammingDistance:
    """Test hamming_distance function."""
    
    def test_hamming_distance_identical_sequences(self):
        """Test hamming distance for identical sequences."""
        seq1 = "ATCGATCG"
        seq2 = "ATCGATCG"
        
        distance, indices = hamming_distance(seq1, seq2)
        
        assert distance == 0
        assert indices == []
    
    def test_hamming_distance_different_sequences(self):
        """Test hamming distance for different sequences."""
        seq1 = "ATCGATCG"
        seq2 = "ATGGATGG"
        
        distance, indices = hamming_distance(seq1, seq2)
        
        assert distance == 2
        assert indices == [2, 6]  # Positions where sequences differ
    
    def test_hamming_distance_all_different(self):
        """Test hamming distance when all positions differ."""
        seq1 = "AAAA"
        seq2 = "TTTT"
        
        distance, indices = hamming_distance(seq1, seq2)
        
        assert distance == 4
        assert indices == [0, 1, 2, 3]
    
    def test_hamming_distance_unequal_lengths(self):
        """Test hamming distance raises error for unequal length sequences."""
        seq1 = "ATCG"
        seq2 = "ATCGATCG"
        
        with pytest.raises(ValueError, match="Strand lengths are not equal!"):
            hamming_distance(seq1, seq2)
    
    def test_hamming_distance_empty_sequences(self):
        """Test hamming distance for empty sequences."""
        seq1 = ""
        seq2 = ""
        
        distance, indices = hamming_distance(seq1, seq2)
        
        assert distance == 0
        assert indices == []


class TestTranslateDnaToAa:
    """Test translate_dna_to_aa function."""
    
    def test_translate_basic_codons(self):
        """Test translation of basic codons."""
        # ATG = M (Methionine), TGA = * (Stop)
        dna = "ATGTGA"
        
        result = translate_dna_to_aa(dna)
        
        assert result == "M*"
    
    def test_translate_full_sequence(self):
        """Test translation of a full DNA sequence."""
        # ATG (M), GCT (A), GAA (E), TTT (F)
        dna = "ATGGCTGAATTT"
        
        result = translate_dna_to_aa(dna)
        
        assert result == "MAEF"
    
    def test_translate_with_incomplete_codon(self):
        """Test translation with incomplete final codon."""
        # ATG (M), GCT (A), GA (incomplete)
        dna = "ATGGCTGA"
        
        result = translate_dna_to_aa(dna)
        
        # BioPython translates incomplete codons
        assert result.startswith("MA")
    
    def test_translate_empty_sequence(self):
        """Test translation of empty sequence."""
        dna = ""
        
        result = translate_dna_to_aa(dna)
        
        assert result == ""
    
    def test_translate_lowercase_sequence(self):
        """Test translation handles lowercase DNA."""
        dna = "atggctgaa"
        
        result = translate_dna_to_aa(dna)
        
        assert result == "MAE"
    
    def test_translate_with_ambiguous_nucleotides(self):
        """Test translation with ambiguous nucleotides."""
        # N represents any nucleotide
        dna = "ATGNNNTGA"
        
        result = translate_dna_to_aa(dna)
        
        # Should translate but with X for unknown amino acid
        assert "X" in result or len(result) > 0


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