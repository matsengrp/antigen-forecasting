"""
Compute model scores with focused evaluation on meaningful variants.

This script computes the scores of the models based on predictions and truth set data,
with enhanced filtering to focus on epidemiologically relevant variants. The scoring
excludes rare, extinct, and non-circulating variants to provide clearer model
performance differentiation.

Key filtering features:
- Active variant filtering: Only scores variants observed recently (configurable window)
- Frequency threshold: Excludes very rare variants below a threshold
- Missing data handling: Properly handles NaN values in smoothed frequencies

Usage:
    python score_models.py --config config.yaml --truth-set truth_set.tsv --estimates-path estimates --output-path scores.tsv

Required Arguments:
    --config            Path to the configuration file with scoring parameters
    --truth-set         Path to the truth set of sequences
    --estimates-path    Path to the model estimates
    --output-path       Path to save the output scores

Configuration (in YAML):
    scoring:
      min_frequency_threshold: 0.01  # Minimum frequency to include (1%)
      active_window_days: 90         # Look back window for active variants
      min_sequences: 10              # Minimum sequences for a variant
      min_observations: 3            # Minimum observations in window
      handle_missing_smoothed: true  # Filter missing smoothed frequencies

Output:
    - scores.tsv        Model scores with filtering applied

Dependencies:
    - Requires Python 3.x
    - pandas
    - numpy
    - pyyaml
    - scipy

Author: Zorian Thornton (@zorian15)
Date: 2025-02-06
"""
import pandas as pd
import numpy as np
import os
import yaml
import itertools
import argparse
from scipy.stats import binom
from abc import ABC, abstractmethod
from datetime import datetime

class ModelScores(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass


class MAE(ModelScores):
    def __init__(self) -> None:
        pass

    def evaluate(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Compute the absollute error between true and predicted values.
        """
        return np.abs(y_true - y_pred)

class MSE(ModelScores):
    def __init__(self) -> None:
        pass
    
    def evaluate(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Compute the mean squared error between true and predicted values.
        """
        return np.square(y_true - y_pred)
    
class Coverage(ModelScores):
    def __init__(self) -> None:
        pass
    
    def compute_coverage(self, y_true: np.array, ci_low: np.array, ci_high: np.array) -> float:
        """
        Determine if the true value is covered under the credible intervals.
        """
        return ((y_true >= ci_low) & (y_true <= ci_high)).astype(int)

class LogLoss(ModelScores):
    def __init__(self):
        pass

    def evaluate(self, seq_value: int, tot_seq: int, pred_values: np.array) -> float:
        loglik = binom.logpmf(k=seq_value, n=tot_seq, p=pred_values)
        return loglik

def smooth_freqs(freq_df: pd.DataFrame, window_size=7) -> pd.DataFrame:
    """
    Smooth the frequencies of the data using a 1D filter.
    """
    raw_freqs = pd.Series(freq_df['truth_freq'])
    smoothed_freqs = (raw_freqs.rolling(window=window_size, min_periods=1, center=True).mean().values)
    freq_df['smoothed_freq'] = smoothed_freqs
    return freq_df

def load_data(filepath:str, sep:str='\t') -> pd.DataFrame:
    """
    Load in summary data of forecast frequencies.

    Parameters
    ----------
    filepath (str): 
        Path to the file.
    sep (str): 
        Separator for the file.

    Returns
    -------
    pd.DataFrame: 
        DataFrame of summary statistics from predictions.
    """
    # Check for file existence
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load in the dataframe
    predictions_df = pd.read_csv(filepath, sep=sep)

    # Use median frequency as the predictions
    predictions_df['pred_freq'] = predictions_df['median_freq_forecast']
    
    # Fill any missing prediction values with median nowcast frequencies
    if 'median_freq_nowcast' in predictions_df.columns:
        predictions_df['pred_freq'].fillna(predictions_df['median_freq_nowcast'], inplace=True)

    # Check for credible intervals and add to dataframe
    if ("freq_forecast_upper_95" in predictions_df.columns) and ("freq_forecast_lower_95" in predictions_df.columns):
        # Define the credible intervals for forecasted frequencies
        predictions_df['ci_low'] = predictions_df['freq_forecast_lower_95']
        predictions_df['ci_high'] = predictions_df['freq_forecast_upper_95']
        # Now do the same for the nowcasted frequencies
        if "freq_lower_95" in predictions_df.columns:
            predictions_df['ci_low'] = predictions_df[['ci_low', 'freq_lower_95']].min(axis=1)
        if "freq_upper_95" in predictions_df.columns:
            predictions_df['ci_high'] = predictions_df[['ci_high', 'freq_upper_95']].max(axis=1)
    # Return
    return predictions_df

def load_truthset(path: str) -> pd.DataFrame:
    """
    Load retrospective frequencies.

    Parameters
    ----------
    path (str):
        Path to the truth set.

    Returns
    -------
    pd.DataFrame:
        DataFrame of the truth set observations.
    """
    truth_set = pd.read_csv(path, sep="\t")

    all_dates = pd.unique(truth_set["date"])
    all_loc = pd.unique(truth_set["country"])
    all_var = pd.unique(truth_set["variant"])

    combined = [all_dates, all_loc, all_var]
    df = pd.DataFrame(
        columns=["date", "country", "variant"],
        data=list(itertools.product(*combined)),
    )

    new_truth = df.merge(truth_set, how="left").fillna(0)
    new_truth["total_seq"] = new_truth.groupby(["date", "country"])[
        "sequences"
    ].transform("sum")

    new_truth["truth_freq"] = new_truth["sequences"] / new_truth["total_seq"]
    new_truth = new_truth.sort_values(by=["country", "variant", "date"])
    new_truth = (
        new_truth.groupby(["country", "variant"], group_keys=False)
        .apply(smooth_freqs)
        .reset_index(drop=True)
    )
    return new_truth

def sample_predictive_quantile(total_seq: np.array, freq:np.array, q:float, num_samples:int=100):
    """
    Sample the predictive quantile of the data.
    
    Parameters
    ----------
    total_seq (np.array): 
        Total sequence counts.
    freq (np.array):
        Frequencies.
    q (float):
        Quantile to sample for.
    num_samples (int):
        Number of samples to take.
        
    Returns
    -------
    np.array:
        Predicted frequencies.
    """
    freq_pred = np.full_like(total_seq, fill_value=np.nan)
    if np.isnan(total_seq).all():
        return freq_pred
    sample_counts = np.random.binomial(
        total_seq.astype(int), freq, size=(num_samples, len(total_seq))
    )
    sample_quants = np.nanquantile(sample_counts, q, axis=0)
    np.divide(sample_quants, total_seq, out=freq_pred, where=total_seq != 0)
    return freq_pred

def merge_truth_pred(df, location_truth):
    """
    Merge the truth and prediction dataframes.

    Parameters
    ----------
    df (pd.DataFrame):
        DataFrame of the predictions.
    location_truth (pd.DataFrame):
        DataFrame of the truth set.

    Returns
    -------
    pd.DataFrame:
        Merged DataFrame of the truth and predictions.
    """
    if "location" in df.columns:
        df.rename(columns={"location": "country"}, inplace=True)
    merged_set = pd.merge(
        location_truth, df, how="left", on=["date", "country", "variant"]
    )

    # Compute total sequences for each location and date
    merged_set["total_seq"] = merged_set.groupby(["date", "country"])[
        "sequences"
    ].transform("sum")

    # Compute retrospective frequencies for each variant
    merged_set["truth_freq"] = (
        merged_set["sequences"] / merged_set["total_seq"]
    )
    return merged_set[merged_set["pred_freq"].notnull()]


def filter_active_variants(df: pd.DataFrame, pivot_date: str, lookback_days: int = 90, 
                          min_observations: int = 3, min_sequences: int = 10) -> pd.DataFrame:
    """
    Filter to only include variants that have been recently observed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with variant observations.
    pivot_date : str
        The analysis/pivot date.
    lookback_days : int
        Number of days to look back for active variants.
    min_observations : int
        Minimum number of times variant must be observed in lookback window.
    min_sequences : int
        Minimum total sequences for a variant in lookback window.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only active variants.
    """
    pivot_dt = pd.to_datetime(pivot_date)
    lookback_start = pivot_dt - pd.Timedelta(days=lookback_days)
    
    # Get variants observed in the lookback window
    recent_data = df[(pd.to_datetime(df['date']) >= lookback_start) & 
                     (pd.to_datetime(df['date']) < pivot_dt)]
    
    # Calculate observation counts and total sequences per variant
    active_variants = recent_data.groupby(['country', 'variant']).agg({
        'sequences': ['count', 'sum']
    }).reset_index()
    
    # Flatten column names
    active_variants.columns = ['country', 'variant', 'observation_count', 'total_sequences']
    
    # Filter based on criteria
    active_variants = active_variants[
        (active_variants['observation_count'] >= min_observations) &
        (active_variants['total_sequences'] >= min_sequences)
    ]
    
    # Get list of active variant-country combinations
    active_keys = set(zip(active_variants['country'], active_variants['variant']))
    
    # Filter original dataframe to only include active variants
    mask = df.apply(lambda row: (row['country'], row['variant']) in active_keys, axis=1)
    
    return df[mask]

def calculate_errors(merged: pd.DataFrame, pivot_date: str, country: str, model: str, 
                    min_freq_threshold: float = None, handle_missing_smoothed: bool = True):
    """
    Calculate the errors between the truth and predictions.

    Parameters
    ----------
    merged : pd.DataFrame
        DataFrame of the merged truth and predictions.
    pivot_date : str
        Date of the analysis.
    country : str
        Name of the country.
    model : str
        Name of the model.
    min_freq_threshold : float, optional
        Minimum frequency threshold to include variants (applied to smoothed_freq).
        If None, no filtering.
    handle_missing_smoothed : bool
        Whether to filter out rows with missing smoothed frequencies.

    Returns
    -------
    pd.DataFrame
        DataFrame of the errors.
    """
    # Filter out rows with missing smoothed_freq if requested
    if handle_missing_smoothed:
        merged = merged[merged['smoothed_freq'].notna()].copy()
        if len(merged) == 0:
            return None
    
    # Apply minimum frequency threshold if specified
    # Use smoothed_freq for consistency since that's what errors are calculated against
    if min_freq_threshold is not None:
        freq_mask = (merged['smoothed_freq'] >= min_freq_threshold) | (merged['pred_freq'] >= min_freq_threshold)
        merged = merged[freq_mask].copy()
        if len(merged) == 0:
            return None
    # Compute model dates and leads from pivot_date
    model_dates = pd.to_datetime(merged["date"])
    lead = (model_dates - pd.to_datetime(pivot_date)).dt.days

    error_df = pd.DataFrame(
        {
            "country": country,
            "model": model,
            "pivot_date": pd.to_datetime(pivot_date),
            "lead": lead,
            "variant": merged["variant"],
        }
    )

    # unpacking prepped_data values
    (
        raw_freq,
        pred_freq,
        seq_count,
        total_seq,
        smoothed_freq,
        ci_low,
        ci_low_pred,
        ci_high,
        ci_high_pred,
    ) = prep_frequency_data(merged)

    if raw_freq is None:
        return None

    # Computing metrics
    # MAE
    mae = MAE()
    error_df["MAE"] = mae.evaluate(smoothed_freq, pred_freq)

    # MSE
    mse = MSE()
    error_df["MSE"] = mse.evaluate(smoothed_freq, pred_freq)

    # Logloss error
    logloss = LogLoss()
    error_df["loglik"] = logloss.evaluate(seq_count, total_seq, pred_freq)

    # Computing Coverage
    coverage = Coverage()
    if ci_low is not None and ci_high is not None:
        error_df["coverage_posterior"] = coverage.compute_coverage(
            smoothed_freq, ci_low, ci_high
        )
    if ci_low_pred is not None and ci_high_pred is not None:
        error_df["coverage_predictive"] = coverage.compute_coverage(
            smoothed_freq, ci_low_pred, ci_high_pred
        )
    # Adding frequencies columns for comparison and diagnostics
    error_df["total_seq"] = total_seq
    error_df["raw_freq"] = raw_freq
    error_df["smoothed_freq"] = smoothed_freq
    error_df["pred_freq"] = pred_freq
    error_df["date"] = model_dates
    return error_df

def prep_frequency_data(df: pd.DataFrame) -> tuple:
    """
    Prepare the frequency data for analysis.

    Resulting dictionary should have the following keys:
    - 'raw_freq': Raw frequency data.
    - 'pred_freq': Predicted frequencues.
    - 'seq_count': Sequence counts for a variant.
    - 'total_seq': Total sequence counts.
    - 'smoothed_freq': Smoothed frequency data.
    - 'ci_low': Lower bound of the credible interval.
    - 'ci_low_pred': Lower bound of the predicted credible interval.
    - 'ci_high': Upper bound of the credible interval.
    - 'ci_high_pred': Upper bound of the predicted credible interval.

    Parameters
    ----------
    df (pd.DataFrame): 
        DataFrame of the frequency data.

    Returns
    -------
    tuple:
        Tuple of the prepared data.
    """
    
    # Convert frequencies to arrays
    raw_freq = np.squeeze(df[["truth_freq"]].to_numpy(), axis=-1)

    if len(raw_freq) == 0:
        return (None,) * 9

    # Convert smoothed frequencies to arrays
    smoothed_freq = np.squeeze(df[["smoothed_freq"]].to_numpy())

    # Convert sequences and total sequnces to arrays
    seq_count = np.squeeze(df[["sequences"]].to_numpy())
    total_seq = np.squeeze(df[["total_seq"]].to_numpy())

    # Convert predicted frequencies to arrays
    pred_freq = np.squeeze(df[["pred_freq"]].to_numpy())

    # Convert credible intervals to arrays
    if "ci_low" in df.columns:
        ci_low = np.squeeze(df[["ci_low"]].to_numpy())
        ci_low_pred = sample_predictive_quantile(total_seq, ci_low, q=0.025)
    else:
        ci_low = None
        ci_low_pred = None

    if "ci_high" in df.columns:
        ci_high = np.squeeze(df[["ci_high"]].to_numpy())
        ci_high_pred = sample_predictive_quantile(total_seq, ci_high, q=0.975)
    else:
        ci_high = None
        ci_high_pred = None

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute model scores.')

    parser.add_argument('--config', type=str, default="../configs/benchmark_config.yaml", help='Path to the configuration file.')
    parser.add_argument('--truth-set', type=str, help='Path to the truth set of sequences.')
    parser.add_argument('--estimates-path', type=str, help='Path to the estimates.')
    parser.add_argument('--output-path', type=str, help='Path to save the output.')
    args = parser.parse_args()
    
    with open(args.config, "r") as config:
        config = yaml.safe_load(config)

    dates = config["main"]["estimation_dates"]
    locations = config["main"]["locations"]
    models = config["main"]["models"]
    
    # Load scoring configuration parameters
    scoring_config = config.get("scoring", {})
    min_freq_threshold = scoring_config.get("min_frequency_threshold", None)
    active_window_days = scoring_config.get("active_window_days", None)
    min_sequences = scoring_config.get("min_sequences", 10)
    min_observations = scoring_config.get("min_observations", 3)
    handle_missing_smoothed = scoring_config.get("handle_missing_smoothed", True)
    
    print(f"Scoring configuration:")
    print(f"  Min frequency threshold: {min_freq_threshold}")
    print(f"  Active window days: {active_window_days}")
    print(f"  Min sequences: {min_sequences}")
    print(f"  Min observations: {min_observations}")
    print(f"  Handle missing smoothed: {handle_missing_smoothed}")

    # Retrospective sequence counts
    truth_set = load_truthset(
        args.truth_set
    )

    # Loading model predictions and computing errors
    score_df_list = []
    for model in models:
        for location in locations:
            pred_dic = {}
            # Filtering to location of increast
            location_truth = truth_set[truth_set["country"] == location]
            if len(location_truth) == 0:
                print(f"{location} is not in the retrospective frequencies.")
                continue

            for pivot_date in dates:
                filepath = args.estimates_path + f"/{model}/freq_{location}_{pivot_date}.tsv"
                # if no .tsv, check for .csv
                if not os.path.exists(filepath):
                    filepath = args.estimates_path + f"/{model}/freq_{location}_{pivot_date}.csv"

                # Load data
                sep = "\t" if filepath.endswith(".tsv") else ","
                try:
                    raw_pred = load_data(filepath, sep=sep)
                except FileNotFoundError:
                    raw_pred = None
                if raw_pred is None:
                    print(
                        f"No {model} predictions found for country {location} on analysis date {pivot_date}"
                    )
                    continue

                # Merge predictions and truth set
                merged = merge_truth_pred(raw_pred, location_truth)
                
                # Track original size for diagnostics
                original_size = len(merged)
                
                # Apply active variant filtering if configured
                if active_window_days is not None:
                    merged = filter_active_variants(
                        merged, 
                        pivot_date=pivot_date,
                        lookback_days=active_window_days,
                        min_observations=min_observations,
                        min_sequences=min_sequences
                    )
                    active_filtered_size = len(merged)
                    print(f"  Active variant filter: {original_size} → {active_filtered_size} rows")
                
                # Make dataframe containing the errors with additional filtering
                error_df = calculate_errors(
                    merged, 
                    pivot_date, 
                    country=location, 
                    model=model,
                    min_freq_threshold=min_freq_threshold,
                    handle_missing_smoothed=handle_missing_smoothed
                )
                
                if error_df is None:
                    print(f"  No valid data after filtering for {model} {location} {pivot_date}")
                    continue
                
                final_size = len(error_df)
                print(f"  Final size after all filters: {final_size} rows")

                score_df_list.append(error_df)
    score_df = pd.concat(score_df_list)

    # Save score output to a tsv file
    error_filepath = args.output_path
    date = datetime.now().strftime("%Y-%m-%d")
    score_df.to_csv(error_filepath, sep="\t", index=False)