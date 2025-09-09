"""Unit tests for antigen_reader module."""
import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np
import tempfile
import os
import json

from antigentools.antigen_reader import (
    AntigenReader,
    process_file_path,
    split_string,
    extract_value,
    convert_to_snake_case,
    calculate_antigenic_movement_per_year,
    calculate_high_low_epitope_mutation_ratio,
    calculate_attack_rate,
    calculate_case_counts_over_time,
    count_branch_mutations,
    read_model_estimates
)


class TestAntigenReaderUtilityFunctions(unittest.TestCase):
    """Test utility functions used by AntigenReader."""
    
    def test_process_file_path(self):
        """Test path processing extracts values correctly."""
        # Test basic path
        path = "simulations/epitopeAcceptance_0.5_run_1"
        expected = {'epitope_acceptance': 0.5, 'run': 1.0}
        self.assertEqual(process_file_path(path), expected)
        
        # Test complex path with multiple underscores
        path = "simulations/epitopeAcceptance_0.5_nonEpitopeAcceptance_0.1_run_10"
        expected = {
            'epitope_acceptance': 0.5,
            'non_epitope_acceptance': 0.1,
            'run': 10.0
        }
        self.assertEqual(process_file_path(path), expected)
        
        # Test path with string values
        path = "simulations/model_test_run_5"
        expected = {'model': 'test', 'run': 5.0}
        result = process_file_path(path)
        self.assertEqual(result['run'], expected['run'])
        
    def test_split_string(self):
        """Test string splitting for complex parameter names."""
        # Test simple split
        self.assertEqual(split_string("param_1.0"), ["param_1.0"])
        
        # Test multiple parameters
        result = split_string("param1_0.5_param2_1.0")
        self.assertIn("param1_0.5", result)
        self.assertIn("param2_1.0", result)
        
    def test_extract_value(self):
        """Test value extraction from component strings."""
        # Test numeric value
        key, value = extract_value("param_1.5")
        self.assertEqual(key, "param")
        self.assertEqual(value, 1.5)
        
        # Test string value
        key, value = extract_value("model_test")
        self.assertEqual(key, "model")
        self.assertEqual(value, "test")
        
        # Test invalid component
        key, value = extract_value("invalid")
        self.assertIsNone(key)
        self.assertIsNone(value)
        
    def test_convert_to_snake_case(self):
        """Test camelCase to snake_case conversion."""
        self.assertEqual(convert_to_snake_case("epitopeAcceptance"), "epitope_acceptance")
        self.assertEqual(convert_to_snake_case("nonEpitopeAcceptance"), "non_epitope_acceptance")
        self.assertEqual(convert_to_snake_case("simple"), "simple")
        self.assertEqual(convert_to_snake_case("XMLHttpRequest"), "x_m_l_http_request")


class TestAntigenReaderCalculationFunctions(unittest.TestCase):
    """Test calculation functions used by AntigenReader."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample tips dataframe
        self.tips_df = pd.DataFrame({
            'year': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            'ag1': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            'ag2': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            'highEpitopeMutationCount': [10, 20, 30, 40, 50, 60],
            'lowEpitopeMutationCount': [5, 10, 15, 20, 25, 30]
        })
        
        # Create sample cases dataframe
        self.cases_df = pd.DataFrame({
            'totalS': [90000, 89000, 88000],
            'totalI': [1000, 2000, 3000],
            'totalCases': [1000, 2000, 3000],
            'northS': [30000, 29500, 29000],
            'northI': [500, 1000, 1500],
            'northCases': [500, 1000, 1500],
            'tropicsS': [30000, 29500, 29000],
            'tropicsI': [300, 600, 900],
            'tropicsCases': [300, 600, 900],
            'southS': [30000, 30000, 30000],
            'southI': [200, 400, 600],
            'southCases': [200, 400, 600]
        })
        
    def test_calculate_antigenic_movement_per_year(self):
        """Test antigenic movement calculation."""
        # Test with default parameters
        movement = calculate_antigenic_movement_per_year(self.tips_df)
        
        # Expected: average movement per year
        # Year 0-1: sqrt((2-0)^2 + (1-0)^2) = sqrt(5) ≈ 2.236
        # Year 1-2: sqrt((4-2)^2 + (2-1)^2) = sqrt(5) ≈ 2.236
        # Year 2-3: sqrt((5-4)^2 + (2.5-2)^2) = sqrt(1.25) ≈ 1.118
        expected_avg = (np.sqrt(5) + np.sqrt(5) + np.sqrt(1.25)) / 3
        self.assertAlmostEqual(movement, expected_avg, places=3)
        
        # Test with empty dataframe
        empty_df = pd.DataFrame(columns=['year', 'ag1', 'ag2'])
        movement = calculate_antigenic_movement_per_year(empty_df)
        self.assertTrue(np.isnan(movement))
        
    def test_calculate_high_low_epitope_mutation_ratio(self):
        """Test epitope mutation ratio calculation."""
        # Test with full data
        ratio = calculate_high_low_epitope_mutation_ratio(self.tips_df)
        # Using all data: sum(high)/sum(low) = 210/105 = 2.0
        self.assertEqual(ratio, 2.0)
        
        # Test with n_tips parameter
        ratio = calculate_high_low_epitope_mutation_ratio(self.tips_df, n_tips=2)
        # Last 2 tips: (50+60)/(25+30) = 110/55 = 2.0
        self.assertEqual(ratio, 2.0)
        
        # Test with missing columns
        df_no_cols = self.tips_df.drop(columns=['highEpitopeMutationCount'])
        ratio = calculate_high_low_epitope_mutation_ratio(df_no_cols)
        self.assertIsNone(ratio)
        
    def test_calculate_attack_rate(self):
        """Test attack rate calculation."""
        result = calculate_attack_rate(self.cases_df.copy())
        
        # Check total attack rate
        expected_total = self.cases_df['totalI'] / self.cases_df['totalS']
        pd.testing.assert_series_equal(result['total_attack_rate'], expected_total)
        
        # Check regional attack rates exist
        self.assertIn('north_attack_rate', result.columns)
        self.assertIn('tropics_attack_rate', result.columns)
        self.assertIn('south_attack_rate', result.columns)
        
    def test_calculate_case_counts_over_time(self):
        """Test case count normalization."""
        # Test with default population size
        result = calculate_case_counts_over_time(self.cases_df.copy())
        self.assertIn('cases_per_100k', result.columns)
        
        # Test with custom population size
        result = calculate_case_counts_over_time(self.cases_df.copy(), population_size=1000000)
        self.assertIn('cases_per_1000k', result.columns)
        
        # Verify calculation
        expected = self.cases_df['totalCases'] / 100000
        pd.testing.assert_series_equal(
            calculate_case_counts_over_time(self.cases_df.copy())['cases_per_100k'], 
            expected
        )


class TestCountBranchMutations(unittest.TestCase):
    """Test branch mutation counting function."""
    
    def test_count_branch_mutations_high_low_model(self):
        """Test mutation counting with high-low model."""
        # Create mock branch file content
        branch_content = '''{"child1",0.5,0.1,1,1,0,0,0.0,ATCG,1.0,2.0,10,5,3,7},{"parent1",0.4,0.1,1,0,0,0,0.0,ATCG,1.0,2.0,8,4,2,6}
{"child2",0.6,0.2,0,1,0,0,0.0,ATCG,1.0,2.0,15,8,5,10},{"parent2",0.5,0.1,1,0,0,0,0.0,ATCG,1.0,2.0,12,7,4,8}'''
        
        with patch('builtins.open', mock_open(read_data=branch_content)):
            result = count_branch_mutations('dummy.branches', high_low_model=True)
            
        # Check structure
        self.assertIn('trunk_epitope_mutations', result)
        self.assertIn('trunk_non-epitope_mutations', result)
        self.assertIn('side_branch_epitope_mutations', result)
        self.assertIn('side_branch_non-epitope_mutations', result)
        
        # Verify ratios are calculated
        self.assertIn('trunk_epitope_to_non-epitope_ratio', result)
        self.assertIn('side_branch_epitope_to_non-epitope_ratio', result)


class TestAntigenReaderClass(unittest.TestCase):
    """Test AntigenReader class methods."""
    
    def setUp(self):
        """Set up test environment."""
        self.reader = AntigenReader()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_antigen_reader_initialization(self):
        """Test AntigenReader initializes correctly."""
        self.assertEqual(self.reader.runs, [])
        self.assertEqual(self.reader.cases, {})
        self.assertEqual(self.reader.trees, {})
        self.assertEqual(self.reader.tips, {})
        self.assertEqual(self.reader.branches, {})
        self.assertEqual(self.reader.summary_files, {})
        self.assertEqual(self.reader.branches_stats, {})
        
    @patch('glob.glob')
    @patch('pandas.read_csv')
    def test_read_runs_basic(self, mock_read_csv, mock_glob):
        """Test basic read_runs functionality."""
        # Mock file paths
        mock_glob.return_value = ['simulations/run_1']
        
        # Mock summary file
        summary_data = pd.DataFrame({
            'parameter': ['diversity', 'tmrca'],
            'value': [5.0, 3.0]
        })
        mock_read_csv.return_value = summary_data
        
        # Run read_runs
        self.reader.read_runs()
        
        # Verify
        self.assertEqual(len(self.reader.runs), 1)
        self.assertIn('simulations/run_1', self.reader.runs)
        
    def test_write_tips_to_fasta(self):
        """Test FASTA file writing."""
        # Create test dataframe
        tips_df = pd.DataFrame({
            'name': ['seq1', 'seq2'],
            'nucleotideSequence': ['ATCG', 'GCTA'],
            'year': [2020, 2021],
            'fitness': [0.5, 0.6],
            'country': ['USA', 'UK'],
            'variant': ['A', 'B']
        })
        
        # Write to temp file
        output_path = os.path.join(self.temp_dir, 'test.fasta')
        self.reader.write_tips_to_fasta(tips_df, output_path)
        
        # Verify file exists and content
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn('>seq1', content)
            self.assertIn('ATCG', content)
            self.assertIn('>seq2', content)
            self.assertIn('GCTA', content)
            
    def test_write_tips_to_fasta_with_metadata(self):
        """Test FASTA file writing with metadata."""
        tips_df = pd.DataFrame({
            'name': ['seq1'],
            'nucleotideSequence': ['ATCG'],
            'year': [2020],
            'fitness': [0.5],
            'country': ['USA'],
            'variant': ['A']
        })
        
        output_path = os.path.join(self.temp_dir, 'test_meta.fasta')
        self.reader.write_tips_to_fasta(tips_df, output_path, write_metadata=True)
        
        # Check metadata file
        metadata_path = output_path.replace('.fasta', '_metadata.tsv')
        self.assertTrue(os.path.exists(metadata_path))
        
        # Verify metadata content
        metadata = pd.read_csv(metadata_path, sep='\t')
        self.assertIn('strain', metadata.columns)
        self.assertIn('clade_membership', metadata.columns)
        
    def test_parse_host(self):
        """Test host string parsing."""
        # Test empty host
        result = self.reader.parse_host('\n')
        self.assertEqual(result, [])
        
        # Test single infection
        result = self.reader.parse_host('(1.0,2.0)')
        self.assertEqual(result, [[1.0, 2.0]])
        
        # Test multiple infections
        result = self.reader.parse_host('(1.0,2.0)(3.0,4.0)')
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
        
    def test_aggregate_branch_stats(self):
        """Test branch stats aggregation."""
        # Create mock JSON files
        stats1 = {'trunk_epitope_mutations': 10, 'run': 1}
        stats2 = {'trunk_epitope_mutations': 20, 'run': 2}
        
        json_path1 = os.path.join(self.temp_dir, 'stats1.json')
        json_path2 = os.path.join(self.temp_dir, 'stats2.json')
        
        with open(json_path1, 'w') as f:
            json.dump(stats1, f)
        with open(json_path2, 'w') as f:
            json.dump(stats2, f)
            
        # Mock glob to return our files
        with patch('glob.glob', return_value=[json_path1, json_path2]):
            output_path = os.path.join(self.temp_dir, 'aggregated.csv')
            self.reader.aggregate_branch_stats(
                input_path="dummy/*.json",
                output_path=output_path
            )
            
        # Verify output
        self.assertTrue(os.path.exists(output_path))
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 2)
        self.assertIn('trunk_epitope_mutations', df.columns)
        
    def test_getters_and_setters(self):
        """Test getter and setter methods."""
        # Test cases
        cases_df = pd.DataFrame({'data': [1, 2, 3]})
        self.reader.set_cases('path1', cases_df)
        pd.testing.assert_frame_equal(self.reader.get_cases('path1'), cases_df)
        
        # Test tips
        tips_df = pd.DataFrame({'seq': ['A', 'T', 'C']})
        self.reader.set_tips('path1', tips_df)
        pd.testing.assert_frame_equal(self.reader.get_tips('path1'), tips_df)
        
        # Test branches stats
        stats = {'mutations': 10}
        self.reader.set_branches_stats('path1', stats)
        self.assertEqual(self.reader.get_branches_stats('path1'), stats)


class TestReadModelEstimates(unittest.TestCase):
    """Test read_model_estimates function."""
    
    @patch('pandas.read_csv')
    def test_read_model_estimates(self, mock_read_csv):
        """Test reading model estimates from paths."""
        # Mock data
        mock_df = pd.DataFrame({'value': [1, 2, 3]})
        mock_read_csv.return_value = mock_df
        
        # Test paths
        paths = [
            'results/FGA/north/estimates_2023-01-01.tsv',
            'results/MLR/south/estimates_2023-01-02.tsv'
        ]
        
        result = read_model_estimates(paths)
        
        # Verify calls
        self.assertEqual(mock_read_csv.call_count, 2)
        
        # Verify columns added
        self.assertIn('model', result.columns)
        self.assertIn('location', result.columns)
        self.assertIn('pivot_date', result.columns)
        
        # Verify parsing
        self.assertTrue((result['model'] == 'FGA').any())
        self.assertTrue((result['model'] == 'MLR').any())
        self.assertTrue((result['location'] == 'north').any())
        self.assertTrue((result['location'] == 'south').any())


if __name__ == '__main__':
    unittest.main()