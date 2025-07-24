import glob
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress, gamma, entropy
from typing import Optional, Tuple, Dict, Any
from scipy.interpolate import UnivariateSpline
import json


def read_estimates(paths: list) -> pd.DataFrame:
    """ Read evofr forecasting estimates from a list of paths.

    Parses the pivot date from the path to add to the dataframe.

    Parameters:
    ---------------
        paths (list): List of paths to read

    Returns:
    ---------------
        pd.DataFrame: Dataframe of estimates
    """
    dfs = []
    for path in paths:
        pivot_date = path.split("/")[-1].split("_")[-1].split(".")[0]
        df = pd.read_csv(path, sep="\t")
        df['pivot_date'] = pivot_date
        dfs.append(df)
    return pd.concat(dfs)


def map_dates(df: pd.DataFrame, reverse_mapping: bool = False) -> Dict[Any, Any]:
    """Map dates in dataframe to time index.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with column 'date'
    reverse_mapping : bool, default=False
        If True, return a mapping from index to date

    Returns:
    --------
    Dict[Any, Any]
        Dictionary mapping dates to time indices (or vice versa)
    """
    date_mapping = {date: t for t, date in enumerate(sorted(df['date'].unique()))}
    if reverse_mapping:
        date_mapping = {value: key for key, value in date_mapping.items()}
    return date_mapping


def filter_to_lifespan(df: pd.DataFrame, variant: str, deme: str, lifespan: tuple, time_col_name: str='t') -> pd.DataFrame:
    """ Pre-process fitness dataframe to set all fitness values outside of the lifespan of a variant to NaN.

    Parameters:
    ---------------
        df (pd.DataFrame): Dataframe with columns 'time', 'fitness', 'variant'
        variant (str): Variant name
        deme (str): Deme/location name
        lifespan (tuple): Tuple of start and end dates
        time_col_name (str): Name of time column

    Returns:
    ---------------
        pd.DataFrame: Filtered dataframe
    """
    start, end = lifespan
    # Set fitness values of variant outside of lifespan to NaN
    var_df = df.query(f"variant == {variant} and location == '{deme}'")
    # Set fitness values outside of lifespan to NaN
    var_df.loc[(var_df[time_col_name] < start) | (var_df[time_col_name] > end), 'fitness'] = np.nan
    var_df.loc[(var_df[time_col_name] < start) | (var_df[time_col_name] > end), 'seasonal_fitness'] = np.nan

    return var_df


def prune_fitness_dataframe(fitness_df: pd.DataFrame, lifespan_df: pd.DataFrame, var_col_name: str='variant', 
                           loc_col_name: str='location', time_col_name: str='t') -> pd.DataFrame:
    """ Prune fitness dataframe to only include data within lifespan.

    Parameters:
    ---------------
        fitness_df (pd.DataFrame): Dataframe with columns 'time', 'fitness', 'variant'
        lifespan_df (pd.DataFrame): Dataframe with columns 'variant', 'birth', 'death'
        var_col_name (str): Name of variant column
        loc_col_name (str): Name of location column
        time_col_name (str): Name of time column

    Returns:
    ---------------
        pruned_df (pd.DataFrame): Dataframe
    """
    variant_dfs = []
    for variant in fitness_df[var_col_name].unique():
        for deme in fitness_df[loc_col_name].unique():
            # Get lifespan of variant
            lifespan = lifespan_df.query(f"variant == {variant} and location == '{deme}'")[['birth', 'death']].values[0]
            # Prune dataframe
            pruned_df = filter_to_lifespan(fitness_df, variant, deme, lifespan, time_col_name)
            variant_dfs.append(pruned_df)
    pruned_df = pd.concat(variant_dfs)
    return pruned_df

def get_gamma_distribution_params(mean, std):
    """
    Calculate the shape and scale parameters of a gamma distribution given its mean and standard deviation.
    
    Parameters:
    -----------
    mean : float
        The mean of the gamma distribution.
    std : float
        The standard deviation of the gamma distribution.
    
    Returns:
    --------
    tuple
        A tuple containing the shape (alpha) and scale (θ) parameters of the gamma distribution.
    """
    alpha = (mean / std)**2
    theta = std**2 / mean
    return alpha, theta

def add_week_id_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add a week_id column to a dataframe that represents the chronological order of dates.
    
    This function creates a new column 'week_id' that assigns a unique integer ID
    to each unique date in the dataframe, with earlier dates receiving smaller IDs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing a date column
    date_col : str, default='date'
        Name of the column containing date information
        
    Returns:
    --------
    pd.DataFrame
        A copy of the input DataFrame with an additional 'week_id' column
        
    Notes:
    ------
    - The function ensures that dates are properly sorted chronologically
    - All rows with the same date will receive the same week_id
    - The earliest date will have week_id=0
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
        result_df[date_col] = pd.to_datetime(result_df[date_col])
    
    # Get unique dates and sort them chronologically
    unique_dates = sorted(result_df[date_col].unique())
    
    # Create a mapping from date to week_id
    date_to_id = {date: i for i, date in enumerate(unique_dates)}
    
    # Add the week_id column based on the mapping
    result_df['week_id'] = result_df[date_col].map(date_to_id)
    
    return result_df

def add_variant_frequencies(seqs_df: pd.DataFrame, count_column: str = 'sequences', ensure_complete: bool = True) -> pd.DataFrame:
    """
    Calculate and add variant frequencies to the dataframe.
    
    This function takes a dataframe containing variant sequence counts and adds a new column
    with the relative frequency of each variant at each location and time point.
    
    Parameters:
    -----------
    seqs_df : pd.DataFrame
        DataFrame containing sequence data with 'country', 'variant', 'date',
        and sequence count columns.
    count_column : str, default='smoothed_sequences'
        Name of the column containing sequence counts to use for frequency calculation.
        Could be raw 'sequences' or 'smoothed_sequences'.
    ensure_complete : bool, default=True
        If True, ensures all variants have entries for all dates (missing combinations 
        get 0.0 frequency). If False, uses original behavior where missing combinations
        are not included.
    
    Returns:
    --------
    pd.DataFrame
        A copy of the input DataFrame with an additional 'variant_frequency' column
        containing the relative frequency of each variant.
    
    Notes:
    ------
    - If a location and date combination has no sequences at all, frequencies will be NaN
    - Frequencies for each location and date should sum to 1.0
    - NaN values in the count column will be excluded from frequency calculations
    - When ensure_complete=True, all variant-date combinations will be present with 0.0
      frequency for unobserved variants
    """
    # Create a copy of the input DataFrame
    result_df = seqs_df.copy()
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(result_df['date']):
        result_df['date'] = pd.to_datetime(result_df['date'])
    
    if ensure_complete:
        # Get all unique values
        all_countries = result_df['country'].unique()
        all_variants = result_df['variant'].unique()
        all_dates = result_df['date'].unique()
        
        # Create a complete index of all combinations
        complete_index = pd.MultiIndex.from_product(
            [all_countries, all_variants, all_dates],
            names=['country', 'variant', 'date']
        )
        
        # Reindex to include all combinations, filling missing values with 0
        result_df = result_df.set_index(['country', 'variant', 'date']).reindex(complete_index)
        
        # Fill missing sequence counts with 0
        if count_column in result_df.columns:
            result_df[count_column] = result_df[count_column].fillna(0)
        
        # Also fill 'sequences' column if it exists and is different from count_column
        if 'sequences' in result_df.columns and 'sequences' != count_column:
            result_df['sequences'] = result_df['sequences'].fillna(0)
            
        # Reset index to get back to regular DataFrame
        result_df = result_df.reset_index()
    
    # Add a column for the variant frequencies, initialize with NaN
    result_df['variant_frequency'] = np.nan
    
    # Calculate the total counts for each location and date
    total_counts = result_df.groupby(['country', 'date'])[count_column].transform('sum')
    
    # Calculate the variant frequency as the proportion of each variant
    # Avoid division by zero
    mask = total_counts > 0
    result_df.loc[mask, 'variant_frequency'] = result_df.loc[mask, count_column] / total_counts[mask]
    result_df.loc[~mask, 'variant_frequency'] = 0.0
    
    # Validate frequencies (should sum to 1.0 for each location and date)
    # Group by location and date, and check the sum of frequencies
    freq_sums = result_df.groupby(['country', 'date'])['variant_frequency'].sum().reset_index()
    freq_sums = freq_sums.rename(columns={'variant_frequency': 'freq_sum'})
    
    # Check if any sum deviates significantly from 1.0
    validation_threshold = 0.01  # Allow 1% deviation due to floating point errors
    invalid_sums = freq_sums[(abs(freq_sums['freq_sum'] - 1.0) > validation_threshold) & 
                            (~freq_sums['freq_sum'].isna()) &
                            (freq_sums['freq_sum'] > 0)]  # Only check non-zero sums
    
    if len(invalid_sums) > 0:
        print(f"Warning: {len(invalid_sums)} location-date combinations have frequency sums that deviate from 1.0")
        print(invalid_sums.head())
    
    return result_df

def smooth_with_spline(
    df: pd.DataFrame, 
    col_to_smooth: str = 'sequences',
    output_col: str = 'smoothed_sequences',
    s: float = 0.5, 
    k: int = 3,
    log_transform: bool = False
) -> pd.DataFrame:
    """
    Smooth variant sequence counts using 1-D spline fitting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing sequence counts data with columns 'country', 'variant', 
        'date', and `col_to_smooth`.
    col_to_smooth : str, default='sequences'
        The name of the column containing the sequence counts to be smoothed.
    output_col : str, default='smoothed_sequences'
        The name of the column to store the smoothed sequence counts.
    log_transform : bool, default=False
        If True, applies log1p transformation to the sequence counts before smoothing.
        This can help stabilize variance for highly skewed data, and counts.
    s : float, default=0.5
        Positive smoothing factor. Larger values produce smoother curves.
        s=0 forces the spline to interpolate through all data points exactly.
        Recommended range: 0.1-10 depending on noise level.
    k : int, default=3
        Degree of the smoothing spline. Must be 1 <= k <= 5.
        k=3 gives cubic splines which are generally smooth and flexible.
    
    Returns:
    --------
    pd.DataFrame
        A copy of the input DataFrame with an additional `output_col` column
        containing the spline-smoothed sequence counts.
    
    Notes:
    ------
    - Higher s values will produce smoother curves but may miss important trends
    - Lower s values will follow the data more closely but may overfit noise
    - For noisy data, s values between 1-5 often work well
    - For cleaner data, s values between 0.1-1 may be more appropriate
    """
    # Validate parameters
    if k < 1 or k > 5:
        raise ValueError("k must be between 1 and 5")
    if s < 0:
        raise ValueError("s must be non-negative")
    
    # Create a copy of the input DataFrame
    result_df = df.copy()
    result_df.dropna(subset=[col_to_smooth], inplace=True)
    
    # Add a column for the smoothed counts, initialize with NaN
    result_df[output_col] = np.nan
    
    # Convert date to datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df['date']):
        result_df['date'] = pd.to_datetime(result_df['date'])
    
    # For each location and variant combination, perform the smoothing
    for location in result_df['country'].unique():
        for variant in result_df[result_df['country'] == location]['variant'].unique():
            # Get data for this specific location and variant
            mask = (result_df['country'] == location) & (result_df['variant'] == variant)
            variant_data = result_df[mask].sort_values('date')
            
            # Skip if we don't have enough data points
            # Need at least k+1 data points for a degree k spline
            if len(variant_data) < k + 1:
                print(f"Skipping smoothing for {location}, variant {variant}: "
                      f"not enough data points ({len(variant_data)})")
                continue
            
            # Create a numerical index for dates (days since first observation)
            variant_data = variant_data.copy()
            variant_data['day_index'] = (variant_data['date'] - variant_data['date'].min()).dt.days
            
            # Apply spline fitting to the sequence counts
            # Convert to 1-indexed arrays for scipy
            x = variant_data['day_index'].values
            y = variant_data[col_to_smooth].values
            
            try:
                # If log_transform is enabled, apply log1p transformation
                if log_transform:
                    y = np.log1p(y)
                
                # Fit spline to log-transformed data
                spline = UnivariateSpline(x, y, k=k, s=s)
                
                # Predict smoothed values for all points
                smoothed_y = spline(x)
                
                if log_transform:
                    smoothed_y = np.expm1(smoothed_y)  # Transform back from log space
                
                # Update the result DataFrame with smoothed values
                result_df.loc[variant_data.index, output_col] = smoothed_y
            
            except Exception as e:
                # If spline fitting fails, fall back to original counts
                print(f"Warning: Spline fitting failed for {location}, variant {variant}: {e}")
                result_df.loc[variant_data.index, output_col] = variant_data[col_to_smooth]
    
    return result_df

def calculate_variant_growth_rates(seqs_df: pd.DataFrame, cases_df: pd.DataFrame, use_freqs: bool = False) -> pd.DataFrame:
    """
    Calculate empirical growth rates (r_data) for each variant in each location based on smoothed sequence counts and total case counts.
    
    Parameters:
    -----------
    seqs_df : pd.DataFrame
        DataFrame containing sequence data with 'country', 'variant', 'date', 'sequences',
        'smoothed_sequences', and 'week_id' columns.
    cases_df : pd.DataFrame
        DataFrame containing case count data with 'country', 'date', 'cases', and 'week_id' columns.
        Used to adjust the growth rates by incorporating total case counts.
    use_freqs : bool, default=False
        If True, use variant frequencies instead of smoothed counts to calculate growth rates.
    
    Returns:
    --------
    pd.DataFrame
        A copy of the input DataFrame with an additional 'growth_rate_r_data' column.
    """
    # Create a copy of the input DataFrame
    result_df = seqs_df.copy()
    
    # Add a column for the growth rates, initialize with NaN
    result_df['growth_rate_r_data'] = np.nan
    
    # Ensure date columns are datetime
    result_df['date'] = pd.to_datetime(result_df['date'])
    cases_df['date'] = pd.to_datetime(cases_df['date'])
    
    # For each location and variant combination, calculate growth rates
    for location in result_df['country'].unique():
        for variant in result_df[result_df['country'] == location]['variant'].unique():
            # Get data for this specific location and variant
            mask = (result_df['country'] == location) & (result_df['variant'] == variant)
            variant_data = result_df[mask].sort_values('week_id')  # Sort by week_id instead of date
            
            # Skip if we don't have enough data points
            if len(variant_data) < 2:
                continue
            
            # Calculate day differences between consecutive observations
            variant_data = variant_data.copy()  # Avoid SettingWithCopyWarning
            variant_data['days_diff'] = variant_data['date'].diff().dt.days
            
            # Calculate growth rate as (ln(C_t * total_cases_t) - ln(C_{t-1} * total_cases_{t-1})) / (t - t_{t-1})
            for i in range(1, len(variant_data)):
                # Get current and previous values
                current_week_id = variant_data.iloc[i]['week_id']
                prev_week_id = variant_data.iloc[i-1]['week_id']
                current_count = variant_data.iloc[i]['smoothed_sequences']
                prev_count = variant_data.iloc[i-1]['smoothed_sequences']
                days_diff = variant_data.iloc[i]['days_diff']

                # Get total cases for current and previous week_ids for this location
                current_cases = cases_df[(cases_df['country'] == location) & (cases_df['week_id'] == current_week_id)]['cases'].values
                prev_cases = cases_df[(cases_df['country'] == location) & (cases_df['week_id'] == prev_week_id)]['cases'].values
                
                # Skip if we have zero counts or missing data
                if (current_count <= 0 or np.isnan(current_count) or prev_count <= 0 or np.isnan(prev_count) or days_diff == 0
                    or len(current_cases) == 0 or len(prev_cases) == 0 or current_cases[0] <= 0 or prev_cases[0] <= 0):
                    continue
                
                # Calculate growth rate with case count adjustment
                if use_freqs:
                    current_freq = variant_data.iloc[i]['variant_frequency_smoothed']
                    prev_freq = variant_data.iloc[i-1]['variant_frequency_smoothed']
                    # Use frequencies to adjust counts
                    growth_rate = (np.log(current_freq * current_cases[0]) - 
                                   np.log(prev_freq * prev_cases[0])) / days_diff
                else:
                    growth_rate = (np.log(current_count) - np.log(prev_count)) / days_diff
                
                # Store in the result DataFrame
                result_df.loc[variant_data.index[i], 'growth_rate_r_data'] = growth_rate
    
    return result_df

def convert_rt_to_growth_rate(rt_df: pd.DataFrame, gamma_shape: float = 2.5, gamma_scale: float = 1.5, rt_column: str = 'median_R') -> pd.DataFrame:
    """
    Convert reproduction numbers (Rt) to growth rates (r) using the Euler-Lotka equation.
    
    Parameters:
    -----------
    rt_df : pd.DataFrame
        DataFrame containing Rt values loaded from model result files.
    gamma_shape : float, default=2.5
        Shape parameter k for the gamma distribution of generation time.
    gamma_scale : float, default=1.5
        Scale parameter θ for the gamma distribution of generation time.
    rt_column : str, default='median_R'
        Name of the column in rt_df containing the Rt values.
    
    Returns:
    --------
    pd.DataFrame
        A copy of the input DataFrame with an additional 'growth_rate_r' column.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = rt_df.copy()
    
    # Apply the Euler-Lotka equation: r = (Rt^(1/k) - 1) / θ
    result_df['growth_rate_r'] = (result_df[rt_column]**(1.0/gamma_shape) - 1.0) / gamma_scale
    
    return result_df

def calculate_distribution_entropy(
    df: pd.DataFrame, 
    count_column: str, 
    location_column: str = 'location', 
    location: Optional[str] = None
) -> Tuple[float, float]:
    """
    Calculate Shannon entropy of a distribution over time
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date, count, and location data
    count_column : str
        Name of the column containing count values
    location_column : str, default='location'
        Name of the column containing location
    location : str, optional
        If provided, filter for a specific location
    
    Returns
    -------
    Tuple[float, float]
        Tuple of (entropy_value, normalized_entropy)
    """
    # Filter for location if specified
    if location and location_column in df.columns:
        df_filtered = df[df[location_column] == location].copy()
    else:
        df_filtered = df.copy()
        
    # Get total counts by date
    counts = df_filtered.groupby('date')[count_column].sum().values
    
    # Avoid division by zero
    total = counts.sum()
    if total == 0:
        return (0, 0)
    
    # Calculate probabilities
    probabilities = counts / total
    
    # Remove zeros to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    
    # Calculate entropy
    entropy_value = entropy(probabilities, base=2)
    
    # Calculate max possible entropy (if distribution were uniform)
    max_entropy = np.log2(len(counts)) if len(counts) > 0 else 0
    
    # Normalized entropy (0-1 scale)
    normalized_entropy = entropy_value / max_entropy if max_entropy > 0 else 0
    
    return entropy_value, normalized_entropy

def get_deme_stats(
    analysis_date: str, 
    location: str, 
    build: str,
    data_dir: str = "../data"
) -> Dict[str, Any]:
    """
    Calculate statistics (counts and entropy) for cases and sequences in a deme.
    
    Parameters
    ----------
    analysis_date : str
        The analysis date to load the data from
    location : str
        The location to filter the dataframe by
    build : str
        The build name to load sequence and case counts from
    data_dir : str, default="../data"
        Base directory for data files
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing various statistics:
        - 'seq_counts': Total sequence counts
        - 'case_counts': Total case counts
        - 'seq_entropy': Raw entropy value for sequences
        - 'seq_norm_entropy': Normalized entropy (0-1) for sequences
        - 'case_entropy': Raw entropy value for cases
        - 'case_norm_entropy': Normalized entropy (0-1) for cases
    """
    results = {
        'seq_counts': None,
        'case_counts': None,
        'seq_entropy': None,
        'seq_norm_entropy': None,
        'case_entropy': None,
        'case_norm_entropy': None
    }
    
    # Load in the data
    data_path = f"{data_dir}/{build}/time-stamped/{analysis_date}/"
    seq_counts_df = pd.read_csv(f"{data_path}/seq_counts.tsv", sep="\t")
    case_counts_df = pd.read_csv(f"{data_path}/case_counts.tsv", sep="\t")
    
    # Filter the dataframes to only include the specified location
    location_seqs_df = seq_counts_df.query("country == @location")
    location_cases_df = case_counts_df.query("country == @location")
    
    # Get the counts for the specified location
    results['seq_counts'] = location_seqs_df["sequences"].sum()
    results['case_counts'] = location_cases_df["cases"].sum()
    
    # Calculate entropy statistics for sequences and cases
    seq_entropy, seq_norm_entropy = calculate_distribution_entropy(
        location_seqs_df, 
        count_column="sequences", 
        location_column="country"
    )
    results['seq_entropy'] = seq_entropy
    results['seq_norm_entropy'] = seq_norm_entropy
    
    case_entropy, case_norm_entropy = calculate_distribution_entropy(
        location_cases_df, 
        count_column="cases", 
        location_column="country"
    )
    results['case_entropy'] = case_entropy
    results['case_norm_entropy'] = case_norm_entropy
    
    return results

def get_outliers(df, metric_column, by_columns=['model', 'location']):
    """
    Identify outliers in a dataframe for a specific metric.
    Uses the IQR method (values outside 1.5*IQR from Q1 and Q3).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    metric_column : str
        Column name for the metric to find outliers in (e.g., 'correlation', 'mae', 'rmse')
    by_columns : list
        List of columns to group by (e.g., ['model', 'location'])
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing only the outlier rows
    """
    outliers_list = []
    
    # Iterate through combinations of grouping variables
    for group_name, group_df in df.groupby(by_columns):
        # Calculate Q1, Q3, and IQR for this group
        q1 = group_df[metric_column].quantile(0.25)
        q3 = group_df[metric_column].quantile(0.75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers in this group
        group_outliers = group_df[
            (group_df[metric_column] < lower_bound) | 
            (group_df[metric_column] > upper_bound)
        ]
        
        if not group_outliers.empty:
            outliers_list.append(group_outliers)
    
    if outliers_list:
        return pd.concat(outliers_list).sort_values(metric_column)
    else:
        return pd.DataFrame()  # Empty DataFrame if no outliers found
    
def calculate_sign_disagreement_rate(growth_rates_df: pd.DataFrame, r_data_col: str = 'growth_rate_r_data', r_model_col: str = 'median_r') -> float:
    """Calculate the rate of sign disagreement between r_model and r_data.

    This function computes the proportion of instances where the signs of the growth rates
    from the model and the data disagree. It returns NaN if there are no valid data points
    to compare.
    
    Parameters
    ----------
    growth_rates_df : pd.DataFrame
        DataFrame containing growth rates with columns for model and data growth rates.
    r_data_col : str, optional
        The column name for the growth rates from the data, by default 'growth_rate_r_data'
    r_model_col : str, optional
        The column name for the growth rates from the model, by default 'median_r'
    
    Returns
    -------
    float
        The proportion of instances where the signs of the growth rates disagree.
        Returns NaN if there are no valid data points to compare.
    """
    valid_data = growth_rates_df.dropna(subset=[r_data_col, r_model_col])
    
    if len(valid_data) == 0:
        return np.nan
    
    # Count where signs disagree
    sign_disagreements = (
        (valid_data[r_data_col] > 0) & (valid_data[r_model_col] < 0) |
        (valid_data[r_data_col] < 0) & (valid_data[r_model_col] > 0)
    ).sum()
    
    return sign_disagreements / len(valid_data)

def calculate_overestimation_rate(growth_rates_df: pd.DataFrame, 
                                r_data_col: str = 'growth_rate_r_data',
                                r_model_col: str = 'median_r',
                                tol: float = 0.01) -> float:
    """
    Calculate the proportion of points where r_model > r_data (above y=x line).
    
    Parameters:
    -----------
    growth_rates_df : pd.DataFrame
        DataFrame with growth rate data
    r_data_col : str
        Column name for empirical growth rates
    r_model_col : str  
        Column name for model growth rates
    tol : float, default=0.01
        Tolerance for considering overestimation (to allow some wiggle room)
        
    Returns:
    --------
    float
        Proportion of points where model overestimates growth rate
    """
    # Remove NaN values
    valid_data = growth_rates_df.dropna(subset=[r_data_col, r_model_col])
    
    if len(valid_data) == 0:
        return np.nan
    
    # Count points above y=x line (r_model > r_data)
    overestimation_count = (valid_data[r_model_col] >= valid_data[r_data_col] + tol).sum()
    
    return overestimation_count / len(valid_data)


def compute_vi_convergence_diagnostics(posterior, window=100, threshold=1e-2):
    """
    Compute convergence diagnostics for variational inference.
    
    Parameters
    ----------
    posterior : PosteriorHandler
        Posterior object from evofr containing samples and losses
    window : int, default=100
        Number of iterations to use for computing relative change
    threshold : float, default=1e-2
        Threshold for relative change to consider converged
        
    Returns
    -------
    dict
        Dictionary containing convergence diagnostics with keys:
        - 'elbo_trajectory': Dict with basic ELBO statistics
        - 'convergence': Dict with convergence status and metrics
        - 'error': Error message if losses not found
    """
    diagnostics_results = {}
    
    # Extract ELBO losses (negative ELBO values)
    if 'losses' in posterior.samples:
        losses = posterior.samples['losses']
        
        # Convert to numpy array if needed
        if hasattr(losses, 'numpy'):
            losses = losses.numpy()
        
        # Basic statistics
        diagnostics_results['elbo_trajectory'] = {
            'num_iterations': len(losses),
            'initial_loss': float(losses[0]),
            'final_loss': float(losses[-1]),
            'min_loss': float(np.min(losses)),
            'total_improvement': float(losses[0] - losses[-1])
        }
        
        # Convergence check using relative change
        if len(losses) >= window:
            relative_change = abs(losses[-1] - losses[-window]) / abs(losses[-window])
            converged = relative_change < threshold
            
            diagnostics_results['convergence'] = {
                'converged': bool(converged),
                'relative_change': float(relative_change),
                'threshold': threshold,
                'window': window,
                'final_iteration': len(losses)
            }
        else:
            diagnostics_results['convergence'] = {
                'converged': False,
                'message': f'Not enough iterations ({len(losses)}) for window size ({window})'
            }
    else:
        diagnostics_results['error'] = 'No losses found in posterior.samples'
    
    return diagnostics_results


def save_vi_convergence_diagnostics(posterior, model_name, location, analysis_date, 
                                   inference_method, inference_settings, 
                                   output_dir="results/convergence_diagnostics",
                                   window=100, threshold=1e-2):
    """
    Compute and save VI convergence diagnostics to JSON file.
    
    Parameters
    ----------
    posterior : PosteriorHandler
        Posterior object from evofr containing samples and losses
    model_name : str
        Name of the model (e.g., 'FGA', 'MLR', 'GARW')
    location : str
        Location/deme name (e.g., 'tropics', 'north', 'south')
    analysis_date : str
        Analysis date in format YYYY-MM-DD
    inference_method : str
        Name of inference method (e.g., 'InferFullRank')
    inference_settings : dict
        Dictionary containing inference settings (iterations, lr, num_samples)
    output_dir : str, default="results/convergence_diagnostics"
        Directory to save diagnostics files
    window : int, default=100
        Number of iterations to use for computing relative change
    threshold : float, default=1e-2
        Threshold for relative change to consider converged
        
    Returns
    -------
    str
        Path to the saved diagnostics file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute convergence diagnostics
    convergence_stats = compute_vi_convergence_diagnostics(posterior, window, threshold)
    
    # Prepare output structure
    diagnostics_output = {
        'model': model_name,
        'location': location,
        'analysis_date': analysis_date,
        'inference_method': inference_method,
        'inference_settings': inference_settings,
        'convergence_diagnostics': convergence_stats
    }
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(v) for v in obj)
        return obj
    
    # Convert to serializable format
    diagnostics_output = convert_to_serializable(diagnostics_output)
    
    # Create filename
    filename = f"{model_name}_{location}_{analysis_date}_vi_diagnostics.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(diagnostics_output, f, indent=2)
    
    return filepath