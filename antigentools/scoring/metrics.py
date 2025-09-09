"""
Model evaluation metrics for antigen forecasting.

This module provides various metrics for evaluating model performance:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- Coverage (Credible interval coverage)
- LogLoss (Log-likelihood based on binomial distribution)
"""

import numpy as np
from scipy.stats import binom
from typing import Protocol, Union, Optional
import logging

logger = logging.getLogger(__name__)


class Metric(Protocol):
    """Protocol for model evaluation metrics."""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the metric."""
        ...


class MAE:
    """Mean Absolute Error metric."""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the absolute error between true and predicted values.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns
        -------
        np.ndarray
            Absolute errors for each prediction
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        return np.abs(y_true - y_pred)


class MSE:
    """Mean Squared Error metric."""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the squared error between true and predicted values.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns
        -------
        np.ndarray
            Squared errors for each prediction
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
            
        return np.square(y_true - y_pred)


class Coverage:
    """Credible interval coverage metric."""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> np.ndarray:
        """
        Determine if the true value is covered by the credible intervals.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Not used, included for interface consistency
        ci_low : np.ndarray
            Lower bounds of credible intervals (passed via kwargs)
        ci_high : np.ndarray
            Upper bounds of credible intervals (passed via kwargs)
            
        Returns
        -------
        np.ndarray
            Binary array indicating coverage (1) or not (0)
        """
        ci_low = kwargs.get('ci_low')
        ci_high = kwargs.get('ci_high')
        
        if ci_low is None or ci_high is None:
            raise ValueError("ci_low and ci_high must be provided in kwargs")
            
        if y_true.shape != ci_low.shape or y_true.shape != ci_high.shape:
            raise ValueError("Shape mismatch between y_true, ci_low, and ci_high")
            
        return ((y_true >= ci_low) & (y_true <= ci_high)).astype(int)


class LogLoss:
    """Log-likelihood metric based on binomial distribution."""
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute log-likelihood of observed counts given predicted frequencies.
        
        Parameters
        ----------
        y_true : np.ndarray
            Observed sequence counts (passed as seq_count in kwargs)
        y_pred : np.ndarray
            Predicted frequencies
        total_seq : np.ndarray
            Total sequence counts (passed via kwargs)
            
        Returns
        -------
        np.ndarray
            Log-likelihood values
        """
        seq_count = kwargs.get('seq_count', y_true)
        total_seq = kwargs.get('total_seq')
        
        if total_seq is None:
            raise ValueError("total_seq must be provided in kwargs")
            
        if seq_count.shape != y_pred.shape or seq_count.shape != total_seq.shape:
            raise ValueError("Shape mismatch between seq_count, y_pred, and total_seq")
        
        # Handle edge cases
        valid_mask = (total_seq > 0) & np.isfinite(y_pred) & (y_pred >= 0) & (y_pred <= 1)
        
        loglik = np.full_like(seq_count, fill_value=np.nan, dtype=float)
        
        if np.any(valid_mask):
            loglik[valid_mask] = binom.logpmf(
                k=seq_count[valid_mask].astype(int),
                n=total_seq[valid_mask].astype(int),
                p=y_pred[valid_mask]
            )
        
        return loglik