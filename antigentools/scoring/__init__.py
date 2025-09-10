"""
Scoring module for antigen forecasting model evaluation.

This module provides functionality for:
- Computing various error metrics (MAE, MSE, LogLoss, Coverage)
- Loading and preprocessing data
- Filtering variants based on activity and frequency
- Calculating model performance scores
"""

from .metrics import MAE, MSE, Coverage, LogLoss
from .config import (
    load_config, parse_config, ScoringConfig, MainConfig,
    GrowthRateConfig, ConvergenceConfig, parse_growth_rate_config
)
from .loaders import (
    load_data, load_truthset,
    RTFile, discover_rt_files, parse_rt_filename,
    load_growth_rates, load_convergence_diagnostics
)
from .filters import filter_active_variants
from .processors import (
    smooth_frequencies, 
    prep_frequency_data, 
    merge_truth_pred,
    calculate_errors
)
from .growth_rate_results import (
    WindowResult, VariantResult, ConvergenceDiagnostic,
    GrowthRateResultsCollector
)

__all__ = [
    'MAE', 'MSE', 'Coverage', 'LogLoss',
    'load_config', 'parse_config', 'ScoringConfig', 'MainConfig',
    'GrowthRateConfig', 'ConvergenceConfig', 'parse_growth_rate_config',
    'load_data', 'load_truthset',
    'RTFile', 'discover_rt_files', 'parse_rt_filename',
    'load_growth_rates', 'load_convergence_diagnostics',
    'filter_active_variants',
    'smooth_frequencies', 'prep_frequency_data', 
    'merge_truth_pred', 'calculate_errors',
    'WindowResult', 'VariantResult', 'ConvergenceDiagnostic',
    'GrowthRateResultsCollector'
]