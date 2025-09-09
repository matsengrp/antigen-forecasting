"""
Test suite for scoring metrics.
"""

import pytest
import numpy as np
from antigentools.scoring.metrics import MAE, MSE, Coverage, LogLoss


class TestMAE:
    """Test Mean Absolute Error metric."""
    
    def test_mae_basic(self):
        """Test MAE with basic inputs."""
        mae = MAE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        
        result = mae.evaluate(y_true, y_pred)
        expected = np.array([0.5, 0.5, 0.5])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_mae_perfect_predictions(self):
        """Test MAE with perfect predictions."""
        mae = MAE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()
        
        result = mae.evaluate(y_true, y_pred)
        expected = np.zeros(3)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_mae_shape_mismatch(self):
        """Test MAE with mismatched shapes."""
        mae = MAE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            mae.evaluate(y_true, y_pred)


class TestMSE:
    """Test Mean Squared Error metric."""
    
    def test_mse_basic(self):
        """Test MSE with basic inputs."""
        mse = MSE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        
        result = mse.evaluate(y_true, y_pred)
        expected = np.array([0.25, 0.25, 0.25])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_mse_shape_mismatch(self):
        """Test MSE with mismatched shapes."""
        mse = MSE()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            mse.evaluate(y_true, y_pred)


class TestCoverage:
    """Test Coverage metric."""
    
    def test_coverage_basic(self):
        """Test coverage calculation."""
        coverage = Coverage()
        y_true = np.array([0.5, 0.3, 0.7])
        y_pred = np.array([0.5, 0.3, 0.7])  # Not used
        ci_low = np.array([0.4, 0.2, 0.8])
        ci_high = np.array([0.6, 0.4, 0.9])
        
        result = coverage.evaluate(y_true, y_pred, ci_low=ci_low, ci_high=ci_high)
        expected = np.array([1, 1, 0])  # First two covered, third not
        
        np.testing.assert_array_equal(result, expected)
    
    def test_coverage_missing_intervals(self):
        """Test coverage with missing interval bounds."""
        coverage = Coverage()
        y_true = np.array([0.5, 0.3, 0.7])
        y_pred = np.array([0.5, 0.3, 0.7])
        
        with pytest.raises(ValueError, match="ci_low and ci_high must be provided"):
            coverage.evaluate(y_true, y_pred)
    
    def test_coverage_shape_mismatch(self):
        """Test coverage with mismatched shapes."""
        coverage = Coverage()
        y_true = np.array([0.5, 0.3, 0.7])
        y_pred = np.array([0.5, 0.3, 0.7])
        ci_low = np.array([0.4, 0.2])
        ci_high = np.array([0.6, 0.4, 0.9])
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            coverage.evaluate(y_true, y_pred, ci_low=ci_low, ci_high=ci_high)


class TestLogLoss:
    """Test LogLoss metric."""
    
    def test_logloss_basic(self):
        """Test log-likelihood calculation."""
        logloss = LogLoss()
        seq_count = np.array([10, 20, 30])
        total_seq = np.array([100, 100, 100])
        pred_freq = np.array([0.1, 0.2, 0.3])
        
        result = logloss.evaluate(
            seq_count, pred_freq,
            seq_count=seq_count, total_seq=total_seq
        )
        
        # Result should be log probabilities (negative values)
        assert np.all(result <= 0)
        assert np.all(np.isfinite(result))
    
    def test_logloss_edge_cases(self):
        """Test log-likelihood with edge cases."""
        logloss = LogLoss()
        
        # Test with zero total sequences
        seq_count = np.array([10, 0, 30])
        total_seq = np.array([100, 0, 100])
        pred_freq = np.array([0.1, 0.5, 0.3])
        
        result = logloss.evaluate(
            seq_count, pred_freq,
            seq_count=seq_count, total_seq=total_seq
        )
        
        # Should handle zero total_seq gracefully
        assert np.isnan(result[1])
        assert np.isfinite(result[0]) and np.isfinite(result[2])
    
    def test_logloss_invalid_probabilities(self):
        """Test log-likelihood with invalid probabilities."""
        logloss = LogLoss()
        
        seq_count = np.array([10, 20, 30])
        total_seq = np.array([100, 100, 100])
        pred_freq = np.array([0.1, 1.5, -0.1])  # Invalid probabilities
        
        result = logloss.evaluate(
            seq_count, pred_freq,
            seq_count=seq_count, total_seq=total_seq
        )
        
        # Should return NaN for invalid probabilities
        assert np.isfinite(result[0])
        assert np.isnan(result[1])  # p > 1
        assert np.isnan(result[2])  # p < 0
    
    def test_logloss_missing_total_seq(self):
        """Test log-likelihood without total_seq."""
        logloss = LogLoss()
        seq_count = np.array([10, 20, 30])
        pred_freq = np.array([0.1, 0.2, 0.3])
        
        with pytest.raises(ValueError, match="total_seq must be provided"):
            logloss.evaluate(seq_count, pred_freq)