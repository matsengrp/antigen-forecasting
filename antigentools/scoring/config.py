"""
Configuration handling for model scoring.

This module provides configuration validation and data classes
for scoring parameters.
"""

from dataclasses import dataclass
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