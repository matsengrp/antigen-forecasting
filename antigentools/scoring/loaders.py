"""
Data loading functionality for model scoring.

This module provides functions to load:
- Model predictions/estimates
- Truth set data
- Configuration files
"""

import pandas as pd
import numpy as np
import itertools
from pathlib import Path
from typing import Optional, Union
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