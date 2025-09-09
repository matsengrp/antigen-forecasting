"""
Filtering functions for variant selection in model scoring.

This module provides filters to focus evaluation on:
- Active variants (recently observed)
- Common variants (above frequency threshold)
- Variants with sufficient data
"""

import pandas as pd
import numpy as np
from typing import Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


def filter_active_variants(
    df: pd.DataFrame, 
    pivot_date: str, 
    lookback_days: int = 90,
    min_observations: int = 3, 
    min_sequences: int = 10
) -> pd.DataFrame:
    """
    Filter to only include variants that have been recently observed.
    
    This function identifies "active" variants based on their presence in a
    lookback window before the pivot date. A variant is considered active if it
    meets minimum thresholds for observations and total sequences.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with variant observations. Must contain columns:
        'date', 'country', 'variant', 'sequences'
    pivot_date : str
        The analysis/pivot date (YYYY-MM-DD format)
    lookback_days : int, default=90
        Number of days to look back for active variants
    min_observations : int, default=3
        Minimum number of times variant must be observed in lookback window
    min_sequences : int, default=10
        Minimum total sequences for a variant in lookback window
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only active variants
        
    Raises
    ------
    ValueError
        If required columns are missing or date formats are invalid
    """
    # Validate input
    required_cols = {'date', 'country', 'variant', 'sequences'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        logger.warning("Empty dataframe provided to filter_active_variants")
        return df
    
    # Ensure date columns are datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
    
    try:
        pivot_dt = pd.to_datetime(pivot_date)
    except Exception as e:
        raise ValueError(f"Invalid pivot date format: {pivot_date}") from e
    
    lookback_start = pivot_dt - pd.Timedelta(days=lookback_days)
    
    logger.info(f"Filtering active variants from {lookback_start} to {pivot_dt}")
    
    # Get variants observed in the lookback window
    recent_mask = (df['date'] >= lookback_start) & (df['date'] < pivot_dt)
    recent_data = df[recent_mask]
    
    if recent_data.empty:
        logger.warning(f"No data found in lookback window for {pivot_date}")
        return df[df['date'] < lookback_start]  # Return empty with same structure
    
    # Calculate observation counts and total sequences per variant (optimized)
    active_stats = recent_data.groupby(['country', 'variant'])['sequences'].agg(
        observation_count='count',
        total_sequences='sum'
    ).reset_index()
    
    # Filter based on criteria
    active_mask = (
        (active_stats['observation_count'] >= min_observations) &
        (active_stats['total_sequences'] >= min_sequences)
    )
    active_variants = active_stats[active_mask]
    
    logger.info(f"Found {len(active_variants)} active variant-country combinations "
                f"out of {len(active_stats)} total")
    
    # Create set of active variant-country combinations for fast lookup
    active_keys = set(
        zip(active_variants['country'], active_variants['variant'])
    )
    
    # Optimized filtering using boolean indexing instead of apply
    if not active_keys:
        return df.iloc[0:0]  # Return empty dataframe with same structure
    
    # Create multi-index for efficient filtering
    original_index = df.index
    df_indexed = df.set_index(['country', 'variant'])
    mask = df_indexed.index.isin(active_keys)
    filtered_df = df_indexed[mask].reset_index()
    
    # Restore original order
    filtered_df = filtered_df.reindex(columns=df.columns)
    
    logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows")
    
    return filtered_df


def filter_by_frequency(
    df: pd.DataFrame,
    min_freq_threshold: float,
    freq_column: str = 'smoothed_freq',
    pred_column: str = 'pred_freq'
) -> pd.DataFrame:
    """
    Filter variants by minimum frequency threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with frequency data
    min_freq_threshold : float
        Minimum frequency threshold (e.g., 0.01 for 1%)
    freq_column : str, default='smoothed_freq'
        Column name for observed frequencies
    pred_column : str, default='pred_freq'
        Column name for predicted frequencies
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
        
    Raises
    ------
    ValueError
        If frequency columns are missing or threshold is invalid
    """
    if min_freq_threshold < 0 or min_freq_threshold > 1:
        raise ValueError(f"min_freq_threshold must be between 0 and 1, got {min_freq_threshold}")
    
    required_cols = {freq_column, pred_column}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Keep if either observed or predicted frequency exceeds threshold
    freq_mask = (df[freq_column] >= min_freq_threshold) | (df[pred_column] >= min_freq_threshold)
    filtered_df = df[freq_mask].copy()
    
    logger.info(f"Frequency filter: {len(df)} → {len(filtered_df)} rows "
                f"(threshold: {min_freq_threshold:.1%})")
    
    return filtered_df


def filter_missing_data(
    df: pd.DataFrame,
    columns: list,
    how: str = 'any'
) -> pd.DataFrame:
    """
    Filter rows with missing data in specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter
    columns : list
        List of column names to check for missing data
    how : str, default='any'
        'any': Remove row if any specified column is NaN
        'all': Remove row only if all specified columns are NaN
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    missing_cols = set(columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    if how not in ['any', 'all']:
        raise ValueError(f"how must be 'any' or 'all', got {how}")
    
    initial_len = len(df)
    
    if how == 'any':
        filtered_df = df.dropna(subset=columns).copy()
    else:
        filtered_df = df.dropna(subset=columns, how='all').copy()
    
    removed = initial_len - len(filtered_df)
    if removed > 0:
        logger.info(f"Removed {removed} rows with missing data in {columns}")
    
    return filtered_df