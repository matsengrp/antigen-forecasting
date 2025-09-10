"""
Configuration handling for model scoring.

This module provides configuration validation and data classes
for scoring parameters and growth rate analysis.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for model scoring parameters."""
    
    min_frequency_threshold: Optional[float] = None
    active_window_days: Optional[int] = None
    min_sequences: int = 10
    min_observations: int = 3
    handle_missing_smoothed: bool = True
    smoothing_window: int = 7
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.min_frequency_threshold is not None:
            if not 0 <= self.min_frequency_threshold <= 1:
                raise ValueError(
                    f"min_frequency_threshold must be between 0 and 1, "
                    f"got {self.min_frequency_threshold}"
                )
        
        if self.active_window_days is not None:
            if self.active_window_days <= 0:
                raise ValueError(
                    f"active_window_days must be positive, "
                    f"got {self.active_window_days}"
                )
        
        if self.min_sequences < 0:
            raise ValueError(
                f"min_sequences must be non-negative, "
                f"got {self.min_sequences}"
            )
        
        if self.min_observations < 1:
            raise ValueError(
                f"min_observations must be positive, "
                f"got {self.min_observations}"
            )
        
        if self.smoothing_window < 1:
            raise ValueError(
                f"smoothing_window must be positive, "
                f"got {self.smoothing_window}"
            )


@dataclass
class MainConfig:
    """Configuration for main analysis parameters."""
    
    estimation_dates: List[str]
    locations: List[str]
    models: List[str]
    
    def __post_init__(self):
        """Validate configuration values."""
        if not self.estimation_dates:
            raise ValueError("estimation_dates cannot be empty")
        
        if not self.locations:
            raise ValueError("locations cannot be empty")
        
        if not self.models:
            raise ValueError("models cannot be empty")


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Parameters
    ----------
    config_path : Path
        Path to configuration file
        
    Returns
    -------
    Dict[str, Any]
        Loaded configuration dictionary
        
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If configuration is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")
    
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    return config


def parse_config(config_dict: Dict[str, Any]) -> Tuple[MainConfig, ScoringConfig]:
    """
    Parse configuration dictionary into structured config objects.
    
    Parameters
    ----------
    config_dict : Dict[str, Any]
        Raw configuration dictionary
        
    Returns
    -------
    Tuple[MainConfig, ScoringConfig]
        Parsed configuration objects
        
    Raises
    ------
    ValueError
        If required sections are missing
    """
    # Validate required sections
    if "main" not in config_dict:
        raise ValueError("Configuration missing required 'main' section")
    
    # Parse main configuration
    main_dict = config_dict["main"]
    main_config = MainConfig(
        estimation_dates=main_dict.get("estimation_dates", []),
        locations=main_dict.get("locations", []),
        models=main_dict.get("models", [])
    )
    
    # Parse scoring configuration (optional section)
    scoring_dict = config_dict.get("scoring", {})
    scoring_config = ScoringConfig(
        min_frequency_threshold=scoring_dict.get("min_frequency_threshold"),
        active_window_days=scoring_dict.get("active_window_days"),
        min_sequences=scoring_dict.get("min_sequences", 10),
        min_observations=scoring_dict.get("min_observations", 3),
        handle_missing_smoothed=scoring_dict.get("handle_missing_smoothed", True),
        smoothing_window=scoring_dict.get("smoothing_window", 7)
    )
    
    return main_config, scoring_config


# Growth rate configuration classes

@dataclass
class GrowthRateConfig:
    """Configuration for growth rate analysis parameters.
    
    Attributes
    ----------
    connect_gaps : bool
        Whether to connect gaps in the growth rate data when plotting
    min_segment_length : int
        Minimum segment length to trust growth rate calculations
    min_sequence_count : int
        Minimum smoothed sequence count per variant
    min_variant_frequency : float
        Minimum variant frequency to consider
    epsilon : float
        Tolerance threshold for overestimation rate calculations
    min_total_sequences : Optional[int]
        Minimum total sequences per window (None to disable)
    spline_smoothing_factor : float
        Smoothing factor for spline interpolation
    spline_order : int
        Order of spline interpolation (1-5)
    """
    
    connect_gaps: bool = True
    min_segment_length: int = 3
    min_sequence_count: int = 10
    min_variant_frequency: float = 0.01
    epsilon: float = 1e-3
    min_total_sequences: Optional[int] = 300
    spline_smoothing_factor: float = 1.0
    spline_order: int = 3
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        if self.min_segment_length < 1:
            raise ValueError("min_segment_length must be >= 1")
        
        if self.min_sequence_count < 0:
            raise ValueError("min_sequence_count must be >= 0")
        
        if not 0 <= self.min_variant_frequency <= 1:
            raise ValueError("min_variant_frequency must be between 0 and 1")
        
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        
        if self.min_total_sequences is not None and self.min_total_sequences < 0:
            raise ValueError("min_total_sequences must be >= 0 or None")
        
        if not 1 <= self.spline_order <= 5:
            raise ValueError("spline_order must be between 1 and 5")
        
        if self.spline_smoothing_factor <= 0:
            raise ValueError("spline_smoothing_factor must be > 0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.
        
        Returns
        -------
        Dict[str, Any]
            Configuration as dictionary
        """
        return asdict(self)


@dataclass
class ConvergenceConfig:
    """Configuration for convergence diagnostics.
    
    Attributes
    ----------
    threshold : float
        Threshold for convergence diagnostics (0-1)
    check_diagnostics : bool
        Whether to check and load convergence diagnostics
    required_models : List[str]
        List of models that require convergence diagnostics
    """
    
    threshold: float = 0.5
    check_diagnostics: bool = True
    required_models: List[str] = None
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        if self.required_models is None:
            self.required_models = ["FGA", "GARW"]
        
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        
        if not isinstance(self.required_models, list):
            raise TypeError("required_models must be a list")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.
        
        Returns
        -------
        Dict[str, Any]
            Configuration as dictionary
        """
        return asdict(self)


def parse_growth_rate_config(config_dict: Dict[str, Any]) -> Tuple[GrowthRateConfig, ConvergenceConfig]:
    """Parse configuration dictionary into growth rate config objects.
    
    Parameters
    ----------
    config_dict : Dict[str, Any]
        Raw configuration dictionary
    
    Returns
    -------
    Tuple[GrowthRateConfig, ConvergenceConfig]
        Parsed and validated configuration objects
    
    Raises
    ------
    ValueError
        If configuration values fail validation
    """
    # Extract growth rate config
    growth_rate_dict = config_dict.get('growth_rate', {})
    
    # Filter to only include valid fields
    growth_rate_fields = {
        k: v for k, v in growth_rate_dict.items()
        if k in GrowthRateConfig.__annotations__
    }
    
    growth_config = GrowthRateConfig(**growth_rate_fields)
    
    # Extract convergence config
    convergence_dict = config_dict.get('convergence', {})
    
    # Filter to only include valid fields
    convergence_fields = {
        k: v for k, v in convergence_dict.items()
        if k in ConvergenceConfig.__annotations__
    }
    
    conv_config = ConvergenceConfig(**convergence_fields)
    
    return growth_config, conv_config