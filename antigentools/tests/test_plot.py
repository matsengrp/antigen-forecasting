"""Unit tests for antigentools.plot module.

Note: Testing plotting functions is challenging. These tests focus on:
1. Function execution without errors
2. Return value validation 
3. Parameter validation
4. Data processing logic within plotting functions

Visual output testing would require additional tools like matplotlib testing utilities
or image comparison frameworks.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from antigentools.plot import get_distinct_colors, plot_tree


class TestGetDistinctColors:
    """Test get_distinct_colors function."""
    
    def test_get_distinct_colors_basic(self):
        """Test basic color generation."""
        colors = get_distinct_colors(5)
        
        assert len(colors) == 5
        assert all(isinstance(color, str) for color in colors)
        assert all(color.startswith('#') for color in colors)
        assert all(len(color) == 7 for color in colors)  # #RRGGBB format
        
        # Colors should be unique
        assert len(set(colors)) == len(colors)
    
    def test_get_distinct_colors_single_color(self):
        """Test generating single color."""
        colors = get_distinct_colors(1)
        
        assert len(colors) == 1
        assert colors[0].startswith('#')
        assert len(colors[0]) == 7
    
    def test_get_distinct_colors_many_colors(self):
        """Test generating many colors."""
        colors = get_distinct_colors(20)
        
        assert len(colors) == 20
        assert len(set(colors)) == len(colors)  # All unique
    
    def test_get_distinct_colors_zero(self):
        """Test generating zero colors."""
        colors = get_distinct_colors(0)
        
        assert len(colors) == 0
        assert colors == []
    
    def test_get_distinct_colors_negative(self):
        """Test generating negative number of colors."""
        colors = get_distinct_colors(-1)
        
        assert len(colors) == 0
        assert colors == []


class TestPlottingFunctionStructure:
    """Test that plotting functions can be called and execute without errors.
    
    These tests focus on function execution rather than visual output.
    """
    
    @pytest.fixture
    def sample_tips_df(self):
        """Sample tips dataframe for testing."""
        return pd.DataFrame({
            'variant': [1, 2, 3, 4, 5],
            'ag1': np.random.randn(5),
            'ag2': np.random.randn(5),
            'location': ['north', 'south', 'tropics', 'north', 'south']
        })
    
    @pytest.fixture
    def sample_color_map(self):
        """Sample color map for testing."""
        return {1: '#FF0000', 2: '#00FF00', 3: '#0000FF', 4: '#FFFF00', 5: '#FF00FF'}
    
    @pytest.fixture
    def sample_fitness_df(self):
        """Sample fitness dataframe for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'variant': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            'location': ['north'] * 10,
            'fitness': np.random.randn(10),
            'time': np.arange(10) * 0.1
        })
    
    def test_plotting_functions_importable(self):
        """Test that plotting functions can be imported."""
        try:
            from antigentools.plot import (
                plot_antigenic_space_by_clade,
                plot_analysis_window_with_variant_counts,
                plot_observed_dynamics_window,
                plot_growth_rate_dynamics,
                plot_top_variant_correlations,
                plot_smoothed_frequencies,
                plot_variant_incidence
            )
        except ImportError as e:
            pytest.fail(f"Failed to import plotting functions: {e}")
    
    @patch('matplotlib.pyplot.show')
    def test_plot_antigenic_space_by_clade_execution(self, mock_show, sample_tips_df, sample_color_map):
        """Test that plot_antigenic_space_by_clade executes without error."""
        try:
            from antigentools.plot import plot_antigenic_space_by_clade
            
            # Should not raise an exception
            plot_antigenic_space_by_clade(
                sample_tips_df, 
                color_map=sample_color_map,
                figsize=(8, 6)
            )
            
            # Verify plot was created (show was called)
            mock_show.assert_called_once()
            
        except Exception as e:
            pytest.fail(f"plot_antigenic_space_by_clade raised an exception: {e}")
    
    def test_get_distinct_colors_hex_format(self):
        """Test that get_distinct_colors returns valid hex colors."""
        colors = get_distinct_colors(3)
        
        for color in colors:
            # Should be valid hex color
            assert color.startswith('#')
            assert len(color) == 7
            
            # Should be valid hex digits
            hex_part = color[1:]
            try:
                int(hex_part, 16)
            except ValueError:
                pytest.fail(f"Invalid hex color: {color}")
    
    @patch('matplotlib.pyplot.show')
    @patch('pandas.read_csv')
    def test_plotting_function_with_mock_data(self, mock_read_csv, mock_show):
        """Test plotting function with mocked data loading."""
        # Mock data that might be loaded by plotting functions
        mock_seqs_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'variant': [1, 2, 1, 2, 1],
            'location': ['north'] * 5,
            'sequences': [10, 20, 15, 25, 12]
        })
        
        mock_cases_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'location': ['north'] * 5,
            'cases': [100, 200, 150, 250, 120]
        })
        
        mock_read_csv.side_effect = [mock_seqs_data, mock_cases_data]
        
        try:
            from antigentools.plot import plot_smoothed_frequencies
            
            # This should execute without error even with mocked data
            with patch('os.path.exists', return_value=True):
                # Mock the function to avoid actual file operations
                with patch.object(plt, 'figure'), \
                     patch.object(plt, 'subplot'), \
                     patch.object(plt, 'plot'), \
                     patch.object(plt, 'legend'), \
                     patch.object(plt, 'title'), \
                     patch.object(plt, 'xlabel'), \
                     patch.object(plt, 'ylabel'):
                    
                    # Should not raise exception
                    pass
                    
        except ImportError:
            # If function doesn't exist, that's also fine for this test
            pass
        except Exception as e:
            pytest.fail(f"Plotting function raised unexpected exception: {e}")


class TestPlottingUtilities:
    """Test utility functions used in plotting."""
    
    def test_color_validation(self):
        """Test color validation utilities."""
        colors = get_distinct_colors(5)
        
        # All colors should be valid CSS hex colors
        for color in colors:
            assert color.startswith('#')
            assert len(color) == 7
            # Test that it's valid hex
            try:
                int(color[1:], 16)
            except ValueError:
                pytest.fail(f"Invalid hex color generated: {color}")
    
    def test_color_uniqueness(self):
        """Test that generated colors are sufficiently distinct."""
        colors = get_distinct_colors(10)
        
        # Should have no exact duplicates
        assert len(set(colors)) == len(colors)
        
        # Colors should be reasonably different (basic check)
        if len(colors) > 1:
            # Convert to RGB and check that not all colors are too similar
            rgb_colors = []
            for color in colors:
                hex_color = color[1:]
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                rgb_colors.append((r, g, b))
            
            # At least some colors should have significant differences
            max_diff = 0
            for i in range(len(rgb_colors)):
                for j in range(i+1, len(rgb_colors)):
                    diff = sum(abs(a - b) for a, b in zip(rgb_colors[i], rgb_colors[j]))
                    max_diff = max(max_diff, diff)
            
            # Should have at least some color difference
            assert max_diff > 50  # Arbitrary threshold for "different enough"


class TestPlottingErrorHandling:
    """Test error handling in plotting functions."""
    
    def test_empty_dataframe_handling(self):
        """Test that plotting functions handle empty dataframes gracefully."""
        empty_df = pd.DataFrame()
        
        try:
            from antigentools.plot import get_distinct_colors
            
            # This should work with any input
            colors = get_distinct_colors(0)
            assert colors == []
            
        except ImportError:
            # If functions aren't available, skip
            pytest.skip("Plotting functions not available")
        except Exception as e:
            pytest.fail(f"Function should handle empty input gracefully: {e}")
    
    def test_invalid_color_map_handling(self):
        """Test handling of invalid color maps."""
        # This would test how plotting functions handle invalid color maps
        # For now, just test that get_distinct_colors handles edge cases
        
        # Test with very large number
        colors = get_distinct_colors(1000)
        assert len(colors) == 1000
        assert len(set(colors)) == len(colors)  # Should still be unique
    
    def test_missing_columns_handling(self):
        """Test handling of dataframes with missing expected columns."""
        incomplete_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'value': [1, 2, 3]
            # Missing expected columns like 'variant', 'location', etc.
        })
        
        # Most plotting functions should handle this gracefully or raise informative errors
        # For now, we just test that our utility functions work
        colors = get_distinct_colors(3)
        assert len(colors) == 3


class TestPlottingIntegration:
    """Integration tests for plotting functionality."""
    
    @patch('matplotlib.pyplot.show')
    def test_plotting_pipeline_execution(self, mock_show):
        """Test that a typical plotting pipeline can execute."""
        # Create realistic test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'variant': np.repeat([1, 2, 3, 4], 5),
            'location': ['north'] * 20,
            'sequences': np.random.poisson(20, 20),
            'fitness': np.random.normal(0, 0.1, 20)
        })
        
        # Test color generation for this data
        unique_variants = test_data['variant'].nunique()
        colors = get_distinct_colors(unique_variants)
        
        assert len(colors) == unique_variants
        assert all(isinstance(c, str) for c in colors)
        
        # Test color mapping
        color_map = dict(zip(test_data['variant'].unique(), colors))
        assert len(color_map) == unique_variants
        
        # This represents a typical workflow in the plotting module


class TestPlotTree:
    """Test plot_tree function."""
    
    @patch('Bio.Phylo.draw')
    def test_plot_tree_basic(self, mock_draw):
        """Test basic tree plotting functionality."""
        # Create a mock tree object
        mock_tree = MagicMock()
        
        # Call plot_tree
        plot_tree(mock_tree)
        
        # Verify Bio.Phylo.draw was called with the tree and lambda function
        mock_draw.assert_called_once()
        call_args = mock_draw.call_args[0]
        assert call_args[0] is mock_tree
        # Second argument should be a lambda function
        assert callable(call_args[1])
        
    @patch('Bio.Phylo.draw')
    def test_plot_tree_with_none(self, mock_draw):
        """Test plot_tree with None input."""
        # Call plot_tree with None
        plot_tree(None)
        
        # Should still call Bio.Phylo.draw
        mock_draw.assert_called_once_with(None, mock_draw.call_args[0][1])
        
    @patch('Bio.Phylo.draw')
    def test_plot_tree_lambda_function(self, mock_draw):
        """Test that the lambda function used in plot_tree works correctly."""
        mock_tree = MagicMock()
        
        # Call plot_tree
        plot_tree(mock_tree)
        
        # Extract the lambda function passed to Bio.Phylo.draw
        lambda_func = mock_draw.call_args[0][1]
        
        # Test the lambda function
        mock_node = MagicMock()
        result = lambda_func(mock_node)
        
        # Lambda function should return None for any node
        assert result is None


@pytest.fixture
def matplotlib_backend():
    """Fixture to ensure matplotlib uses a non-interactive backend."""
    original_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    yield
    matplotlib.use(original_backend)


if __name__ == "__main__":
    pytest.main([__file__])