"""
Scoring module for antigen forecasting model evaluation.

This module provides functionality for:
- Computing various error metrics (MAE, MSE, LogLoss, Coverage)
- Loading and preprocessing data
- Filtering variants based on activity and frequency
- Calculating model performance scores
"""

from .metrics import MAE, MSE, Coverage, LogLoss
from .loaders import load_data, load_truthset
from .filters import filter_active_variants
from .processors import (
    smooth_frequencies, 
    prep_frequency_data, 
    merge_truth_pred,
    calculate_errors
)

__all__ = [
    'MAE', 'MSE', 'Coverage', 'LogLoss',
    'load_data', 'load_truthset',
    'filter_active_variants',
    'smooth_frequencies', 'prep_frequency_data', 
    'merge_truth_pred', 'calculate_errors'
]