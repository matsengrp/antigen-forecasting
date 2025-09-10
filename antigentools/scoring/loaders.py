"""
Data loading functionality for model scoring.

This module provides functions to load:
- Model predictions/estimates
- Truth set data
- Configuration files
- Growth rate (RT) files
- Convergence diagnostics
"""

import pandas as pd
import numpy as np
import itertools
import json
import glob
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def load_data(filepath: Union[str, Path], sep: str = '\t') -> pd.DataFrame:
    """
    Load summary data of forecast frequencies.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the file
    sep : str, default='\t'
        Separator for the file
        
    Returns
    -------
    pd.DataFrame
        DataFrame of summary statistics from predictions
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If required columns are missing
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading predictions from {filepath}")
    
    try:
        predictions_df = pd.read_csv(filepath, sep=sep)
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise
    
    # Validate required columns
    required_cols = ['median_freq_forecast']
    missing_cols = set(required_cols) - set(predictions_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Use median frequency as the predictions
    predictions_df['pred_freq'] = predictions_df['median_freq_forecast']
    
    # Fill any missing prediction values with median nowcast frequencies
    if 'median_freq_nowcast' in predictions_df.columns:
        null_count = predictions_df['pred_freq'].isnull().sum()
        if null_count > 0:
            logger.warning(f"Filling {null_count} missing predictions with nowcast values")
            predictions_df['pred_freq'].fillna(predictions_df['median_freq_nowcast'], inplace=True)
    
    # Process credible intervals if available
    _process_credible_intervals(predictions_df)
    
    return predictions_df


def _process_credible_intervals(df: pd.DataFrame) -> None:
    """
    Process credible interval columns in the predictions dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process (modified in place)
    """
    # Check for forecast credible intervals
    if ("freq_forecast_upper_95" in df.columns) and ("freq_forecast_lower_95" in df.columns):
        logger.debug("Processing forecast credible intervals")
        df['ci_low'] = df['freq_forecast_lower_95']
        df['ci_high'] = df['freq_forecast_upper_95']
        
        # Merge with nowcast intervals if available
        if "freq_lower_95" in df.columns:
            df['ci_low'] = df[['ci_low', 'freq_lower_95']].min(axis=1)
        if "freq_upper_95" in df.columns:
            df['ci_high'] = df[['ci_high', 'freq_upper_95']].max(axis=1)
            
        # Validate intervals
        invalid_intervals = (df['ci_low'] > df['ci_high']).sum()
        if invalid_intervals > 0:
            logger.warning(f"Found {invalid_intervals} invalid credible intervals (low > high)")


def load_truthset(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load retrospective frequencies (truth set).
    
    Parameters
    ----------
    path : Union[str, Path]
        Path to the truth set file
        
    Returns
    -------
    pd.DataFrame
        DataFrame of the truth set observations with smoothed frequencies
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If required columns are missing
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Truth set file not found: {path}")
    
    logger.info(f"Loading truth set from {path}")
    
    try:
        truth_set = pd.read_csv(path, sep="\t")
    except Exception as e:
        logger.error(f"Error reading truth set file {path}: {e}")
        raise
    
    # Validate required columns
    required_cols = ['date', 'country', 'variant', 'sequences']
    missing_cols = set(required_cols) - set(truth_set.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in truth set: {missing_cols}")
    
    # Convert date column to datetime for consistent handling
    truth_set['date'] = pd.to_datetime(truth_set['date'])
    
    # Get unique values for each dimension
    all_dates = truth_set["date"].unique()
    all_locations = truth_set["country"].unique()
    all_variants = truth_set["variant"].unique()
    
    logger.info(f"Truth set contains: {len(all_dates)} dates, "
                f"{len(all_locations)} locations, {len(all_variants)} variants")
    
    # Create complete grid of all combinations
    combined = [all_dates, all_locations, all_variants]
    complete_grid = pd.DataFrame(
        columns=["date", "country", "variant"],
        data=list(itertools.product(*combined))
    )
    
    # Merge with actual data, filling missing with 0
    new_truth = complete_grid.merge(truth_set, how="left", on=["date", "country", "variant"])
    new_truth['sequences'].fillna(0, inplace=True)
    
    # Calculate total sequences per date/location
    new_truth["total_seq"] = new_truth.groupby(["date", "country"])["sequences"].transform("sum")
    
    # Calculate frequencies with safe division
    new_truth["truth_freq"] = np.divide(
        new_truth["sequences"],
        new_truth["total_seq"],
        out=np.zeros_like(new_truth["sequences"], dtype=float),
        where=new_truth["total_seq"] != 0
    )
    
    # Sort for consistent processing
    new_truth = new_truth.sort_values(by=["country", "variant", "date"])
    
    # Apply smoothing to each country/variant combination
    logger.info("Applying frequency smoothing")
    from .processors import smooth_frequencies
    
    new_truth = (
        new_truth.groupby(["country", "variant"], group_keys=False)
        .apply(smooth_frequencies)
        .reset_index(drop=True)
    )
    
    # Convert date back to string for compatibility
    new_truth['date'] = new_truth['date'].dt.strftime('%Y-%m-%d')
    
    return new_truth


# Growth rate (RT) file loading functionality

@dataclass
class RTFile:
    """Represents a discovered RT file with metadata.
    
    Attributes
    ----------
    path : Path
        Full path to the RT file
    model : str
        Model name (e.g., 'FGA', 'GARW')
    location : str
        Location name (e.g., 'north', 'south', 'tropics')
    pivot_date : str
        Pivot date in YYYY-MM-DD format
    """
    path: Path
    model: str
    location: str
    pivot_date: str
    
    def __str__(self) -> str:
        """String representation of RTFile."""
        return f"RTFile({self.model}_{self.location}_{self.pivot_date})"


def parse_rt_filename(path: Path) -> Tuple[str, str, str]:
    """Parse RT filename to extract model, location, and date.
    
    Expected format: rt_{location}_{date}.tsv
    Model is extracted from the parent directory name.
    
    Parameters
    ----------
    path : Path
        Path to RT file
    
    Returns
    -------
    Tuple[str, str, str]
        Model name, location, pivot date
    
    Raises
    ------
    ValueError
        If filename doesn't match expected format
    """
    filename = path.name
    
    # Check filename starts with 'rt_'
    if not filename.startswith('rt_'):
        raise ValueError(f"Invalid RT filename format: {filename}")
    
    # Split filename parts
    parts = filename.replace('.tsv', '').split('_')
    
    if len(parts) < 3:
        raise ValueError(f"Invalid RT filename format: {filename}")
    
    # Extract location and date (everything after 'rt_')
    location = parts[1]
    date_parts = parts[2:]  # Handle dates with multiple parts
    pivot_date = '_'.join(date_parts)
    
    # Extract model from parent directory
    if path.parent.name == '':
        raise ValueError(f"Could not determine model from path: {path}")
    
    model = path.parent.name
    
    return model, location, pivot_date


def discover_rt_files(
    build: str,
    models: List[str],
    base_dir: str = "."
) -> List[RTFile]:
    """Discover RT files in the expected directory structure.
    
    Searches for files matching the pattern:
    {base_dir}/results/{build}/estimates/{model}/rt_*.tsv
    
    Parameters
    ----------
    build : str
        Build name (e.g., 'flu-simulated-150k-samples')
    models : List[str]
        List of model names to include
    base_dir : str
        Base directory for search (default: current directory)
    
    Returns
    -------
    List[RTFile]
        List of discovered RT files, sorted by (model, location, date)
    """
    rt_files = []
    
    # Search pattern for RT files
    pattern = f"{base_dir}/results/{build}/estimates/*/rt_*.tsv"
    rt_paths = glob.glob(pattern)
    
    logger.info(f"Found {len(rt_paths)} RT files matching pattern: {pattern}")
    
    for path_str in rt_paths:
        path = Path(path_str)
        
        try:
            model, location, pivot_date = parse_rt_filename(path)
            
            # Filter by requested models
            if model not in models:
                continue
            
            rt_files.append(RTFile(
                path=path,
                model=model,
                location=location,
                pivot_date=pivot_date
            ))
        
        except ValueError as e:
            # Skip files that don't match expected format
            logger.debug(f"Skipping file {path}: {e}")
            continue
    
    # Sort by model, location, then date
    rt_files.sort(key=lambda x: (x.model, x.location, x.pivot_date))
    
    logger.info(f"Discovered {len(rt_files)} valid RT files for models: {models}")
    
    return rt_files


def load_growth_rates(
    build: str,
    model: str,
    location: str,
    pivot_date: str,
    base_dir: str = ".",
    data_path: str = "data/"
) -> pd.DataFrame:
    """Load growth rate data from TSV file.
    
    Parameters
    ----------
    build : str
        Build name
    model : str
        Model name
    location : str
        Location name
    pivot_date : str
        Pivot date in YYYY-MM-DD format
    base_dir : str
        Base directory (default: current directory)
    data_path : str
        Data directory path (for compatibility, not currently used)
    
    Returns
    -------
    pd.DataFrame
        Growth rate data
    
    Raises
    ------
    FileNotFoundError
        If the RT file does not exist
    """
    # Construct file path
    rt_file = Path(base_dir) / "results" / build / "estimates" / model / f"rt_{location}_{pivot_date}.tsv"
    
    if not rt_file.exists():
        raise FileNotFoundError(f"RT file not found: {rt_file}")
    
    logger.info(f"Loading growth rates from {rt_file}")
    
    # Load data
    try:
        df = pd.read_csv(rt_file, sep='\t')
        logger.debug(f"Loaded {len(df)} growth rate records")
        return df
    except Exception as e:
        logger.error(f"Error loading RT file {rt_file}: {e}")
        raise ValueError(f"Error loading RT file {rt_file}: {e}")


def load_convergence_diagnostics(diagnostics_path: Path) -> Optional[Dict[str, Any]]:
    """Load convergence diagnostics from JSON file.
    
    Parameters
    ----------
    diagnostics_path : Path
        Path to diagnostics JSON file
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Diagnostics data, or None if file not found or invalid
    """
    try:
        if not diagnostics_path.exists():
            logger.debug(f"Convergence diagnostics file not found: {diagnostics_path}")
            return None
        
        logger.debug(f"Loading convergence diagnostics from {diagnostics_path}")
        
        with open(diagnostics_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error loading convergence diagnostics from {diagnostics_path}: {e}")
        # Return None for any loading errors
        return None