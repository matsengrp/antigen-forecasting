"""
Data processing functions for model scoring.

This module provides functions for:
- Frequency smoothing
- Data preparation and transformation
- Merging predictions with truth data
- Error calculation
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging

from .metrics import MAE, MSE, LogLoss, Coverage
from .filters import filter_by_frequency, filter_missing_data

logger = logging.getLogger(__name__)


def smooth_frequencies(
    freq_df: pd.DataFrame, 
    window_size: int = 7,
    freq_column: str = 'truth_freq'
) -> pd.DataFrame:
    """
    Apply rolling mean smoothing to frequency data.
    
    Parameters
    ----------
    freq_df : pd.DataFrame
        DataFrame containing frequency data
    window_size : int, default=7
        Size of the rolling window for smoothing
    freq_column : str, default='truth_freq'
        Column name containing frequencies to smooth
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'smoothed_freq' column
        
    Raises
    ------
    ValueError
        If freq_column is missing or window_size is invalid
    """
    if freq_column not in freq_df.columns:
        raise ValueError(f"Column '{freq_column}' not found in dataframe")
    
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}")
    
    freq_df = freq_df.copy()
    
    # Apply rolling mean with centered window
    raw_freqs = freq_df[freq_column]
    smoothed_freqs = raw_freqs.rolling(
        window=window_size, 
        min_periods=1, 
        center=True
    ).mean()
    
    freq_df['smoothed_freq'] = smoothed_freqs
    
    # Log statistics about smoothing
    nan_count = smoothed_freqs.isna().sum()
    if nan_count > 0:
        logger.warning(f"Smoothing resulted in {nan_count} NaN values")
    
    return freq_df


def merge_truth_pred(
    predictions_df: pd.DataFrame,
    truth_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge prediction and truth dataframes.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame of model predictions
    truth_df : pd.DataFrame
        DataFrame of truth set data
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with predictions and truth data
    """
    # Standardize column names
    if "location" in predictions_df.columns:
        predictions_df = predictions_df.rename(columns={"location": "country"})
    
    # Validate merge keys
    merge_keys = ["date", "country", "variant"]
    for df, name in [(predictions_df, "predictions"), (truth_df, "truth")]:
        missing_keys = set(merge_keys) - set(df.columns)
        if missing_keys:
            raise ValueError(f"Missing merge keys in {name} dataframe: {missing_keys}")
    
    # Perform merge
    merged_set = pd.merge(
        truth_df, 
        predictions_df, 
        how="left", 
        on=merge_keys
    )
    
    # Recalculate totals to ensure consistency
    if "total_seq" not in merged_set.columns or merged_set["total_seq"].isna().any():
        merged_set["total_seq"] = merged_set.groupby(["date", "country"])["sequences"].transform("sum")
    
    # Recalculate truth frequencies with safe division
    merged_set["truth_freq"] = np.divide(
        merged_set["sequences"],
        merged_set["total_seq"],
        out=np.zeros_like(merged_set["sequences"], dtype=float),
        where=merged_set["total_seq"] != 0
    )
    
    # Filter to rows with predictions
    merged_with_preds = merged_set[merged_set["pred_freq"].notnull()]
    
    logger.info(f"Merged data: {len(merged_set)} total rows, "
                f"{len(merged_with_preds)} with predictions")
    
    return merged_with_preds


def prep_frequency_data(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], ...]:
    """
    Extract and prepare frequency arrays for analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing frequency data
        
    Returns
    -------
    tuple
        Tuple containing:
        - raw_freq: Raw frequency data
        - pred_freq: Predicted frequencies
        - seq_count: Sequence counts for a variant
        - total_seq: Total sequence counts
        - smoothed_freq: Smoothed frequency data
        - ci_low: Lower bound of credible interval
        - ci_low_pred: Lower bound of predictive interval
        - ci_high: Upper bound of credible interval
        - ci_high_pred: Upper bound of predictive interval
    """
    if df.empty:
        logger.warning("Empty dataframe provided to prep_frequency_data")
        return (None,) * 9
    
    # Extract arrays, ensuring 1D shape
    def safe_extract(column: str, required: bool = True) -> Optional[np.ndarray]:
        if column in df.columns:
            return np.asarray(df[column]).ravel()
        elif required:
            raise ValueError(f"Required column '{column}' not found")
        return None
    
    try:
        raw_freq = safe_extract("truth_freq")
        smoothed_freq = safe_extract("smoothed_freq")
        seq_count = safe_extract("sequences")
        total_seq = safe_extract("total_seq")
        pred_freq = safe_extract("pred_freq")
        
        # Extract credible intervals if available
        ci_low = safe_extract("ci_low", required=False)
        ci_high = safe_extract("ci_high", required=False)
        
        # Calculate predictive intervals if credible intervals exist
        ci_low_pred = None
        ci_high_pred = None
        
        if ci_low is not None and total_seq is not None:
            ci_low_pred = sample_predictive_quantile(total_seq, ci_low, q=0.025)
        
        if ci_high is not None and total_seq is not None:
            ci_high_pred = sample_predictive_quantile(total_seq, ci_high, q=0.975)
        
        return (
            raw_freq,
            pred_freq,
            seq_count,
            total_seq,
            smoothed_freq,
            ci_low,
            ci_low_pred,
            ci_high,
            ci_high_pred,
        )
        
    except Exception as e:
        logger.error(f"Error preparing frequency data: {e}")
        raise


def sample_predictive_quantile(
    total_seq: np.ndarray, 
    freq: np.ndarray, 
    q: float,
    num_samples: int = 100
) -> np.ndarray:
    """
    Sample the predictive quantile of the data.
    
    Parameters
    ----------
    total_seq : np.ndarray
        Total sequence counts
    freq : np.ndarray
        Frequencies
    q : float
        Quantile to sample (between 0 and 1)
    num_samples : int, default=100
        Number of samples to take
        
    Returns
    -------
    np.ndarray
        Predicted frequencies at the specified quantile
    """
    if not 0 <= q <= 1:
        raise ValueError(f"Quantile q must be between 0 and 1, got {q}")
    
    freq_pred = np.full_like(total_seq, fill_value=np.nan, dtype=float)
    
    if np.all(np.isnan(total_seq)):
        return freq_pred
    
    # Create valid mask
    valid_mask = (
        np.isfinite(total_seq) & 
        np.isfinite(freq) & 
        (total_seq > 0) & 
        (freq >= 0) & 
        (freq <= 1)
    )
    
    if not np.any(valid_mask):
        return freq_pred
    
    # Sample from binomial distribution
    valid_total = total_seq[valid_mask].astype(int)
    valid_freq = freq[valid_mask]
    
    sample_counts = np.random.binomial(
        n=valid_total,
        p=valid_freq,
        size=(num_samples, len(valid_total))
    )
    
    # Calculate quantiles
    sample_quants = np.nanquantile(sample_counts, q, axis=0)
    
    # Convert back to frequencies
    freq_pred[valid_mask] = sample_quants / valid_total
    
    return freq_pred


def calculate_errors(
    merged: pd.DataFrame,
    pivot_date: str,
    country: str,
    model: str,
    min_freq_threshold: Optional[float] = None,
    handle_missing_smoothed: bool = True
) -> Optional[pd.DataFrame]:
    """
    Calculate errors between truth and predictions.
    
    Parameters
    ----------
    merged : pd.DataFrame
        DataFrame of merged truth and predictions
    pivot_date : str
        Date of the analysis (YYYY-MM-DD format)
    country : str
        Name of the country
    model : str
        Name of the model
    min_freq_threshold : Optional[float]
        Minimum frequency threshold to include variants
    handle_missing_smoothed : bool, default=True
        Whether to filter out rows with missing smoothed frequencies
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame of errors, or None if no valid data
    """
    if merged.empty:
        logger.warning(f"Empty dataframe for {model} {country} {pivot_date}")
        return None
    
    # Apply filters
    filtered = merged.copy()
    
    # Filter missing smoothed frequencies
    if handle_missing_smoothed:
        filtered = filter_missing_data(filtered, columns=['smoothed_freq'])
        if filtered.empty:
            logger.warning(f"No data after filtering missing smoothed_freq for {model} {country} {pivot_date}")
            return None
    
    # Apply frequency threshold
    if min_freq_threshold is not None:
        filtered = filter_by_frequency(filtered, min_freq_threshold)
        if filtered.empty:
            logger.warning(f"No data after frequency filtering for {model} {country} {pivot_date}")
            return None
    
    # Ensure dates are datetime
    if not pd.api.types.is_datetime64_any_dtype(filtered['date']):
        filtered['date'] = pd.to_datetime(filtered['date'])
    
    # Calculate lead times
    pivot_dt = pd.to_datetime(pivot_date)
    lead = (filtered['date'] - pivot_dt).dt.days
    
    # Create base error dataframe
    error_df = pd.DataFrame({
        "country": country,
        "model": model,
        "pivot_date": pivot_dt,
        "lead": lead,
        "variant": filtered["variant"].values,
        "date": filtered["date"].values,
    })
    
    # Prepare frequency data
    frequency_data = prep_frequency_data(filtered)
    
    if frequency_data[0] is None:
        logger.error(f"Failed to prepare frequency data for {model} {country} {pivot_date}")
        return None
    
    (
        raw_freq, pred_freq, seq_count, total_seq, smoothed_freq,
        ci_low, ci_low_pred, ci_high, ci_high_pred
    ) = frequency_data
    
    # Calculate metrics
    mae = MAE()
    error_df["MAE"] = mae.evaluate(smoothed_freq, pred_freq)
    
    mse = MSE()
    error_df["MSE"] = mse.evaluate(smoothed_freq, pred_freq)
    
    logloss = LogLoss()
    error_df["loglik"] = logloss.evaluate(
        seq_count, pred_freq,
        seq_count=seq_count, total_seq=total_seq
    )
    
    # Calculate coverage if intervals available
    coverage = Coverage()
    if ci_low is not None and ci_high is not None:
        error_df["coverage_posterior"] = coverage.evaluate(
            smoothed_freq, pred_freq,
            ci_low=ci_low, ci_high=ci_high
        )
    
    if ci_low_pred is not None and ci_high_pred is not None:
        error_df["coverage_predictive"] = coverage.evaluate(
            smoothed_freq, pred_freq,
            ci_low=ci_low_pred, ci_high=ci_high_pred
        )
    
    # Add frequency columns for diagnostics
    error_df["total_seq"] = total_seq
    error_df["raw_freq"] = raw_freq
    error_df["smoothed_freq"] = smoothed_freq
    error_df["pred_freq"] = pred_freq
    
    logger.info(f"Calculated errors for {len(error_df)} predictions")
    
    return error_df