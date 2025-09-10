"""
Result collection dataclasses for growth rate scoring.

This module provides type-safe dataclasses for collecting and organizing
results from growth rate analysis.
"""

import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class WindowResult:
    """Results for a single growth rate analysis window.
    
    Represents window-level metrics for growth rate analysis,
    combining model predictions with evaluation metrics.
    
    Attributes
    ----------
    pivot_date : str
        Analysis date in YYYY-MM-DD format
    model : str
        Model name (e.g., 'FGA', 'GARW')
    location : str
        Location name (e.g., 'north', 'south', 'tropics')
    correlation : Optional[float]
        Correlation between predictions and observations
    mae : float
        Mean absolute error
    rmse : float
        Root mean squared error
    sign_disagreement_rate : float
        Rate of sign disagreements between prediction and observation
    overestimation_rate : float
        Rate of overestimation by the model
    n_seqs : int
        Number of sequences in the window
    n_cases : int
        Number of cases in the window
    seq_entropy : float
        Sequence count entropy
    case_entropy : float
        Case count entropy
    seq_entropy_norm : float
        Normalized sequence entropy
    case_entropy_norm : float
        Normalized case entropy
    """
    
    pivot_date: str
    model: str
    location: str
    correlation: Optional[float]
    mae: float
    rmse: float
    sign_disagreement_rate: float
    overestimation_rate: float
    n_seqs: int
    n_cases: int
    seq_entropy: float
    case_entropy: float
    seq_entropy_norm: float
    case_entropy_norm: float
    
    def __post_init__(self):
        """Validate field values after initialization."""
        # Validate correlation if not None and not NaN
        import math
        if (self.correlation is not None and 
            not math.isnan(self.correlation) and 
            not -1 <= self.correlation <= 1):
            raise ValueError("correlation must be between -1 and 1")
        
        # Validate non-negative metrics
        if self.mae < 0:
            raise ValueError("mae must be >= 0")
        if self.rmse < 0:
            raise ValueError("rmse must be >= 0")
        
        # Validate rates (0-1, allowing small floating point errors)
        if not (0 <= self.sign_disagreement_rate <= 1.001):
            raise ValueError("sign_disagreement_rate must be between 0 and 1")
        if not (0 <= self.overestimation_rate <= 1.001):
            raise ValueError("overestimation_rate must be between 0 and 1")
        
        # Validate counts
        if self.n_seqs < 0:
            raise ValueError("n_seqs must be >= 0")
        if self.n_cases < 0:
            raise ValueError("n_cases must be >= 0")
        
        # Validate entropy values
        if self.seq_entropy < 0:
            raise ValueError("seq_entropy must be >= 0")
        if self.case_entropy < 0:
            raise ValueError("case_entropy must be >= 0")
        
        # Validate normalized entropy (0-1)
        if not 0 <= self.seq_entropy_norm <= 1:
            raise ValueError("seq_entropy_norm must be between 0 and 1")
        if not 0 <= self.case_entropy_norm <= 1:
            raise ValueError("case_entropy_norm must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class VariantResult:
    """Results for a single variant growth rate analysis.
    
    Represents variant-level metrics for growth rate analysis,
    including variant-specific performance measures.
    
    Attributes
    ----------
    variant : str
        Variant identifier
    pivot_date : str
        Analysis date in YYYY-MM-DD format
    model : str
        Model name
    location : str
        Location name
    mae : float
        Mean absolute error for this variant
    normalized_mae : float
        Normalized mean absolute error
    max_r_data : float
        Maximum observed growth rate for this variant
    correlation : Optional[float]
        Correlation between predictions and observations for this variant
    sign_disagreement_rate : float
        Rate of sign disagreements for this variant
    overestimation_rate : float
        Rate of overestimation for this variant
    n_points : int
        Number of data points for this variant
    total_sequences : int
        Total sequence count for this variant
    total_smoothed_sequences : int
        Total smoothed sequence count for this variant
    mean_variant_frequency : float
        Mean frequency of this variant
    mean_smoothed_variant_frequency : float
        Mean smoothed frequency of this variant
    max_variant_frequency : float
        Maximum frequency of this variant
    max_smoothed_variant_frequency : float
        Maximum smoothed frequency of this variant
    """
    
    variant: str
    pivot_date: str
    model: str
    location: str
    mae: float
    normalized_mae: float
    max_r_data: float
    correlation: Optional[float]
    sign_disagreement_rate: float
    overestimation_rate: float
    n_points: int
    total_sequences: int
    total_smoothed_sequences: int
    mean_variant_frequency: float
    mean_smoothed_variant_frequency: float
    max_variant_frequency: float
    max_smoothed_variant_frequency: float
    
    def __post_init__(self):
        """Validate field values after initialization."""
        # Validate correlation if not None and not NaN
        import math
        if (self.correlation is not None and 
            not math.isnan(self.correlation) and 
            not -1 <= self.correlation <= 1):
            raise ValueError("correlation must be between -1 and 1")
        
        # Validate non-negative metrics
        if self.mae < 0:
            raise ValueError("mae must be >= 0")
        if self.normalized_mae < 0:
            raise ValueError("normalized_mae must be >= 0")
        
        # Validate rates (0-1, allowing small floating point errors)
        if not (0 <= self.sign_disagreement_rate <= 1.001):
            raise ValueError("sign_disagreement_rate must be between 0 and 1")
        if not (0 <= self.overestimation_rate <= 1.001):
            raise ValueError("overestimation_rate must be between 0 and 1")
        
        # Validate counts
        if self.n_points < 0:
            raise ValueError("n_points must be >= 0")
        if self.total_sequences < 0:
            raise ValueError("total_sequences must be >= 0")
        if self.total_smoothed_sequences < 0:
            raise ValueError("total_smoothed_sequences must be >= 0")
        
        # Validate frequency values (0-1)
        frequency_fields = [
            ('mean_variant_frequency', self.mean_variant_frequency),
            ('mean_smoothed_variant_frequency', self.mean_smoothed_variant_frequency),
            ('max_variant_frequency', self.max_variant_frequency),
            ('max_smoothed_variant_frequency', self.max_smoothed_variant_frequency)
        ]
        
        for field_name, value in frequency_fields:
            if not 0 <= value <= 1:
                raise ValueError("frequency values must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class ConvergenceDiagnostic:
    """Convergence diagnostic information for a model run.
    
    Represents convergence diagnostics from variational inference,
    including ELBO trajectory and convergence status.
    
    Attributes
    ----------
    pivot_date : str
        Analysis date
    model : str
        Model name
    location : str
        Location name
    inference_method : Optional[str]
        Inference method used
    iterations : Optional[int]
        Number of iterations requested
    learning_rate : Optional[float]
        Learning rate used
    num_samples : Optional[int]
        Number of samples used
    num_iterations : Optional[int]
        Actual number of iterations completed
    initial_loss : Optional[float]
        Initial ELBO loss
    final_loss : Optional[float]
        Final ELBO loss
    min_loss : Optional[float]
        Minimum ELBO loss achieved
    total_improvement : Optional[float]
        Total improvement in ELBO
    converged : Optional[bool]
        Whether convergence was achieved
    relative_change : Optional[float]
        Final relative change in ELBO
    threshold : Optional[float]
        Convergence threshold used
    window : Optional[int]
        Window size for convergence check
    final_iteration : Optional[int]
        Iteration at which convergence was achieved
    """
    
    pivot_date: str
    model: str
    location: str
    inference_method: Optional[str] = None
    iterations: Optional[int] = None
    learning_rate: Optional[float] = None
    num_samples: Optional[int] = None
    num_iterations: Optional[int] = None
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    min_loss: Optional[float] = None
    total_improvement: Optional[float] = None
    converged: Optional[bool] = None
    relative_change: Optional[float] = None
    threshold: Optional[float] = None
    window: Optional[int] = None
    final_iteration: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class GrowthRateResultsCollector:
    """Collects and manages results from growth rate analysis.
    
    Provides type-safe collection and management of growth rate results.
    
    Attributes
    ----------
    window_results : List[WindowResult]
        Collection of window-level results
    variant_results : List[VariantResult]
        Collection of variant-level results
    convergence_diagnostics : List[ConvergenceDiagnostic]
        Collection of convergence diagnostics
    """
    
    def __init__(self):
        """Initialize empty collector."""
        self.window_results: List[WindowResult] = []
        self.variant_results: List[VariantResult] = []
        self.convergence_diagnostics: List[ConvergenceDiagnostic] = []
    
    def add_window_result(self, result: WindowResult) -> None:
        """Add a window result to the collection."""
        self.window_results.append(result)
    
    def add_variant_result(self, result: VariantResult) -> None:
        """Add a variant result to the collection."""
        self.variant_results.append(result)
    
    def add_convergence_diagnostic(self, diagnostic: ConvergenceDiagnostic) -> None:
        """Add convergence diagnostic to the collection."""
        self.convergence_diagnostics.append(diagnostic)
    
    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert collections to DataFrames.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Window results, variant results, and convergence diagnostics DataFrames
        """
        # Convert window results
        if self.window_results:
            window_dicts = [result.to_dict() for result in self.window_results]
            window_df = pd.DataFrame(window_dicts)
        else:
            # Create empty DataFrame with correct columns
            window_df = pd.DataFrame(columns=[
                'pivot_date', 'model', 'location', 'correlation', 'mae', 'rmse',
                'sign_disagreement_rate', 'overestimation_rate', 'n_seqs', 'n_cases',
                'seq_entropy', 'case_entropy', 'seq_entropy_norm', 'case_entropy_norm'
            ])
        
        # Convert variant results
        if self.variant_results:
            variant_dicts = [result.to_dict() for result in self.variant_results]
            variant_df = pd.DataFrame(variant_dicts)
        else:
            # Create empty DataFrame with correct columns
            variant_df = pd.DataFrame(columns=[
                'variant', 'pivot_date', 'model', 'location', 'mae', 'normalized_mae',
                'max_r_data', 'correlation', 'sign_disagreement_rate', 'overestimation_rate',
                'n_points', 'total_sequences', 'total_smoothed_sequences',
                'mean_variant_frequency', 'mean_smoothed_variant_frequency',
                'max_variant_frequency', 'max_smoothed_variant_frequency'
            ])
        
        # Convert convergence diagnostics
        if self.convergence_diagnostics:
            convergence_dicts = [diag.to_dict() for diag in self.convergence_diagnostics]
            convergence_df = pd.DataFrame(convergence_dicts)
        else:
            # Create empty DataFrame with correct columns
            convergence_df = pd.DataFrame(columns=[
                'pivot_date', 'model', 'location', 'inference_method', 'iterations',
                'learning_rate', 'num_samples', 'num_iterations', 'initial_loss',
                'final_loss', 'min_loss', 'total_improvement', 'converged',
                'relative_change', 'threshold', 'window', 'final_iteration'
            ])
        
        return window_df, variant_df, convergence_df
    
    def summary_stats(self) -> Dict[str, int]:
        """Get summary statistics for the collection."""
        stats = {}
        
        # Window results stats
        stats['total_windows'] = len(self.window_results)
        if self.window_results:
            stats['unique_models'] = len(set(r.model for r in self.window_results))
            stats['unique_locations'] = len(set(r.location for r in self.window_results))
            stats['unique_dates'] = len(set(r.pivot_date for r in self.window_results))
        else:
            stats['unique_models'] = 0
            stats['unique_locations'] = 0
            stats['unique_dates'] = 0
        
        # Variant results stats
        stats['total_variants'] = len(self.variant_results)
        
        # Convergence diagnostics stats
        stats['total_diagnostics'] = len(self.convergence_diagnostics)
        
        return stats