import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from scipy.stats import pearsonr
from antigentools.utils import (
    add_week_id_column,
    smooth_with_spline,
    add_variant_frequencies,
    calculate_variant_growth_rates,
    calculate_sign_disagreement_rate,
    calculate_overestimation_rate,
)

def load_model_rt_values(build: str, model: str, location: str, pivot_date: str) -> Optional[pd.DataFrame]:
    """
    Load Rt values directly from model result files.
    
    Parameters:
    -----------
    build : str
        The build name (e.g., 'flu-simulated-150k-samples').
    model : str
        The model name (e.g., 'FGA', 'GARW').
    location : str
        The location name (e.g., 'north', 'tropics').
    pivot_date : str
        The pivot date in YYYY-MM-DD format.
    
    Returns:
    --------
    Optional[pd.DataFrame]
        A DataFrame containing the Rt values, or None if the file is not found.
    """
    # Construct path to the Rt file for this model, location, and date
    rt_path = f"../results/{build}/estimates/{model}/rt_{location}_{pivot_date}.tsv"
    
    try:
        # Read the file
        rt_df = pd.read_csv(rt_path, sep='\t')
        return rt_df
    except FileNotFoundError:
        print(f"Could not find Rt file: {rt_path}")
        return None
    
# Prepare incidence data for a specific location and variant
def get_variant_incidence(seqs_df: pd.DataFrame, location: str, variant_id: str) -> pd.Series:
    """Extract time series of sequence counts for a specific variant and location.
    
    Parameters:
    -----------
    seqs_df : pd.DataFrame
        DataFrame containing sequence data with 'country', 'variant', 'date', and 'sequences' columns.
    location : str
        The location name (e.g., 'north', 'tropics').
    variant_id : str
        The variant identifier (e.g., 'H3N2', 'H1N1').
    
    Returns:
    --------
    pd.Series
        A time-indexed series of sequence counts for the specified variant and location.
    """
    variant_data = seqs_df[(seqs_df['country'] == location) & 
                          (seqs_df['variant'] == variant_id)].sort_values('date')
    
    # Create a time-indexed series of sequence counts
    incidence_series = pd.Series(variant_data['sequences'].values, index=pd.to_datetime(variant_data['date']))
    
    return incidence_series

def get_top_variants(
    df: pd.DataFrame, 
    location: str = 'tropics', 
    n: int = 4, 
    min_points: int = 3, 
    r_model_col: str = 'growth_rate_r', 
    r_data_col: str = 'growth_rate_r_data'
) -> List[Tuple[str, float, float, int]]:
    """Get the top n variants with the highest correlation between model growth rate and empirical growth rate.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing growth rate data with columns 'country', 'variant', 
        'growth_rate_r' (model), and 'growth_rate_r_data' (empirical).
    location : str, default='tropics'
        The location to filter the data by (e.g., 'tropics').
    n : int, default=4
        The number of top variants to return based on correlation.
    min_points : int, default=3
        Minimum number of data points required for a variant to be considered.
    r_model_col : str, default='growth_rate_r'
        The column name for the model growth rate.
    r_data_col : str, default='growth_rate_r_data'
        The column name for the empirical growth rate.
        
    Returns:
    --------
    List[Tuple[str, float, float, int]]
        A list of tuples containing the variant, correlation coefficient, p-value, and number of data points.
        Each tuple is in the form (variant, correlation, p_value, num_points).
    """
    # Filter by location
    location_data = df[df['country'] == location].copy()
    
    # Convert variant to string to ensure consistency with color map keys
    location_data['variant'] = location_data['variant'].astype(str)
    
    # Clean data (remove NaNs)
    location_data = location_data.dropna(subset=[r_data_col, r_model_col])
    
    # Calculate correlation for each variant
    correlations = []
    for variant in location_data['variant'].unique():
        variant_data = location_data[location_data['variant'] == variant]
        
        # Only include variants with enough data points
        if len(variant_data) >= min_points:
            try:
                corr, p_value = pearsonr(
                    variant_data[r_model_col], 
                    variant_data[r_data_col]
                )
                correlations.append((variant, corr, p_value, len(variant_data)))
            except Exception:
                pass
    
    # Sort by absolute correlation (ignoring sign)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    return correlations[:n]

def get_growth_rates_df(
    build: str, 
    model: str, 
    location: str, 
    pivot_date: str, 
    spline_smoothing_factor: float = 0.5,
    spline_order: int = 3,
    use_freqs: bool = True,
    renormalize_frequencies: bool = True
) -> pd.DataFrame:
    """
    Load growth rates from model results and convert Rt to growth rates.
    
    Parameters:
    -----------
    build : str
        The build name (e.g., 'flu-simulated-150k-samples').
    model : str
        The model name (e.g., 'FGA', 'GARW').
    location : str
        The location name (e.g., 'north', 'tropics').
    pivot_date : str
        The pivot date in YYYY-MM-DD format.
    spline_smoothing_factor : float, default=0.5
        Smoothing factor for splines (larger = smoother).
    spline_order : int, default=3
        Degree of the spline (1-5).
    use_freqs : bool, default=True
        Whether to use frequencies instead of raw counts.
    renormalize_frequencies : bool, default=True
        Whether to renormalize smoothed frequencies to ensure they sum to 1.
        If False, accepts small deviations from sum=1 to avoid jaggedness artifacts.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing growth rates with columns 'country', 'variant', 
        'date', 'growth_rate_r', 'growth_rate_r_data', 'variant_incidence', 
        and 'variant_incidence_smoothed' (if smoothed frequencies are available).
    """
    seqs_path = f"../data/{build}/time-stamped/{pivot_date}/seq_counts.tsv"
    cases_path = f"../data/{build}/time-stamped/{pivot_date}/case_counts.tsv"
    seqs_df = pd.read_csv(seqs_path, sep='\t')
    cases_df = pd.read_csv(cases_path, sep='\t')

    # Subset seqs and cases DataFrames for the specified location
    seqs_df = seqs_df[seqs_df['country'] == location]
    cases_df = cases_df[cases_df['country'] == location]

    # Add week_id column to both DataFrames for weekly calculations
    seqs_df = add_week_id_column(seqs_df, date_col='date')
    cases_df = add_week_id_column(cases_df, date_col='date')
    rt_df = load_model_rt_values(build, model, location, pivot_date)

    # Add variant frequencies and then smooth.
    smoothed_seqs_df = seqs_df.copy()
    smoothed_seqs_df = add_variant_frequencies(smoothed_seqs_df, count_column='sequences')
    smoothed_seqs_df = smooth_with_spline(smoothed_seqs_df, col_to_smooth='sequences', output_col='smoothed_sequences', s=spline_smoothing_factor, k=spline_order, log_transform=False)
    # Smooth sequences
    if use_freqs:
        output_col = 'variant_frequency_smoothed'
        smoothed_seqs_df = smooth_with_spline(smoothed_seqs_df, col_to_smooth='variant_frequency', output_col=output_col, s=spline_smoothing_factor, k=spline_order)
        
        # Optionally normalize the smoothed frequencies to ensure they sum to 1 for each date
        if renormalize_frequencies:
            smoothed_seqs_df[output_col] = smoothed_seqs_df.groupby('date')[output_col].transform(lambda x: x / x.sum())
    
    # Calculate growth rates based on the smoothed counts
    growth_rates_df = calculate_variant_growth_rates(seqs_df=smoothed_seqs_df, cases_df=cases_df, use_freqs=use_freqs)

    # Add variant incidence calculation (case counts * variant frequency)
    # First merge with cases data to get case counts for each date
    cases_df_for_merge = cases_df[['date', 'cases']].drop_duplicates()
    growth_rates_df = pd.merge(
        growth_rates_df,
        cases_df_for_merge,
        on='date',
        how='left'
    )
    
    # Calculate variant incidence = cases * variant_frequency
    growth_rates_df['variant_incidence'] = growth_rates_df['cases'] * growth_rates_df['variant_frequency']
    
    # Also calculate smoothed variant incidence if smoothed frequencies are available
    if 'variant_frequency_smoothed' in growth_rates_df.columns:
        growth_rates_df['variant_incidence_smoothed'] = growth_rates_df['cases'] * growth_rates_df['variant_frequency_smoothed']

    try:
        # Change date column to string format for merging
        growth_rates_df['date'] = growth_rates_df['date'].dt.strftime('%Y-%m-%d')
        rt_df['date'] = rt_df['date'].dt.strftime('%Y-%m-%d')
    except AttributeError:
        # If date is already a string, do nothing
        pass
    # Merge with the model data for comparison
    growth_rates_df = pd.merge(
        growth_rates_df, 
        rt_df[['date', 'variant', 'median_r', 'model', 'analysis_date']], 
        on=['date', 'variant'], 
        how='left', 
        suffixes=('', '_model')
    ).drop_duplicates()
    # Calculate the absolute error between empirical and model growth rates
    growth_rates_df['abs_error'] = (
        growth_rates_df['growth_rate_r_data'] - growth_rates_df['median_r']
    ).abs() 
    
    # Set median_r to NaN for variants with zero frequency
    # This prevents them from affecting the y-axis range in plots
    zero_freq_mask = growth_rates_df['variant_frequency'] == 0.0
    growth_rates_df.loc[zero_freq_mask, 'median_r'] = np.nan

    # Return the final DataFrame with growth rates
    return growth_rates_df

# Helper function to get growth rates with automatic filtering of unreliable early points
def get_filtered_growth_rates_df(
    build, 
    model, 
    location, 
    pivot_date, 
    spline_smoothing_factor=5.0, 
    spline_order=3,
    min_sequence_count=5,
    min_variant_frequency=0.05,
    skip_first_n_points=2,
    use_freqs=True
):
    """
    Get growth rates dataframe with automatic filtering of unreliable data points.
    
    Parameters:
    -----------
    min_sequence_count : float
        Minimum smoothed sequence count to trust growth rate calculations
    min_variant_frequency : float  
        Minimum variant frequency to trust growth rate calculations
    skip_first_n_points : int
        Number of initial points to skip for each variant (often noisy)
    """
    # Get the base growth rates
    growth_rates_df = get_growth_rates_df(
        build=build, 
        model=model, 
        location=location, 
        pivot_date=pivot_date,
        spline_smoothing_factor=spline_smoothing_factor,
        spline_order=spline_order,
        use_freqs=use_freqs
    )
    
    # Filter out low count/frequency data points
    low_count_mask = (
        (growth_rates_df['smoothed_sequences'] < min_sequence_count) | 
        (growth_rates_df['variant_frequency_smoothed'] < min_variant_frequency)
    )
    growth_rates_df.loc[low_count_mask, 'growth_rate_r_data'] = np.nan
    
    # Skip the first few points for each variant
    if skip_first_n_points > 0:
        for variant in growth_rates_df['variant'].unique():
            variant_mask = growth_rates_df['variant'] == variant
            variant_indices = growth_rates_df[variant_mask].index
            if len(variant_indices) > skip_first_n_points:
                growth_rates_df.loc[variant_indices[:skip_first_n_points], 'growth_rate_r_data'] = np.nan
    
    return growth_rates_df

def diagnose_extreme_growth_rates(
    growth_rates_df: pd.DataFrame,
    r_data_col: str = 'growth_rate_r_data',
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    Identify and diagnose variants with extreme growth rate values.
    
    Parameters:
    -----------
    growth_rates_df : pd.DataFrame
        DataFrame containing growth rates and related data
    r_data_col : str
        Column name for empirical growth rates
    threshold : float
        Threshold for considering a growth rate "extreme" (absolute value)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with extreme growth rate cases and diagnostic information
    """
    # Find extreme values
    extreme_mask = growth_rates_df[r_data_col].abs() > threshold
    extreme_cases = growth_rates_df[extreme_mask].copy()
    
    if len(extreme_cases) == 0:
        return pd.DataFrame()
    
    # Add diagnostic columns
    extreme_cases['abs_r_data'] = extreme_cases[r_data_col].abs()
    
    # Sort by absolute growth rate
    extreme_cases = extreme_cases.sort_values('abs_r_data', ascending=False)
    
    # Select relevant columns for diagnosis
    diagnostic_cols = [
        'date', 'variant', 'country', r_data_col, 'variant_frequency', 
        'variant_frequency_smoothed', 'sequences', 'smoothed_sequences',
        'cases', 'variant_incidence_smoothed'
    ]
    
    # Only include columns that exist
    available_cols = [col for col in diagnostic_cols if col in extreme_cases.columns]
    
    return extreme_cases[available_cols]

def filter_growth_rates(
    df: pd.DataFrame,
    r_data_col: str = 'growth_rate_r_data',
    connect_gaps: bool = False,
    min_segment_length: Optional[int] = None
) -> pd.DataFrame:
    """
    Filter growth rate data to only include valid segments based on segment length criteria.
    
    This function applies the same filtering logic used in plotting to ensure consistency
    between plotting and performance evaluation functions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing growth rate data with columns including the growth rate column
        and 'variant' for grouping.
    r_data_col : str, default='growth_rate_r_data'
        The column name for the growth rate data to filter on.
    connect_gaps : bool, default=False
        If True, connects all non-NaN points ignoring gaps. If False, finds continuous 
        segments of non-NaN values.
    min_segment_length : Optional[int], default=None
        Minimum number of consecutive non-NaN points required for a segment to be valid.
        If None, no minimum length is enforced.
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame containing only data from valid segments.
    """
    if min_segment_length is None:
        # No filtering, just remove NaN values
        return df.dropna(subset=[r_data_col])
    
    filtered_rows = []
    
    # Process each variant separately
    for variant in df['variant'].unique():
        variant_data = df[df['variant'] == variant].sort_values('date' if 'date' in df.columns else df.index)
        
        if variant_data[r_data_col].isna().all():
            continue
            
        if connect_gaps:
            # For connected gaps, treat all non-NaN points as one segment
            valid_indices = variant_data[~variant_data[r_data_col].isna()].index.tolist()
            if len(valid_indices) >= min_segment_length:
                filtered_rows.extend(valid_indices)
        else:
            # Find continuous segments of non-NaN values
            is_valid = ~variant_data[r_data_col].isna()
            segments = []
            start = None
            
            for i, (idx, valid) in enumerate(zip(variant_data.index, is_valid)):
                if valid:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        if i - start >= min_segment_length:
                            segment_indices = variant_data.index[start:i].tolist()
                            segments.extend(segment_indices)
                        start = None
            
            # Don't forget the last segment if it ends at the last point
            if start is not None and len(variant_data) - start >= min_segment_length:
                segment_indices = variant_data.index[start:].tolist()
                segments.extend(segment_indices)
            
            filtered_rows.extend(segments)
    
    # Return filtered DataFrame
    if filtered_rows:
        return df.loc[filtered_rows]
    else:
        # Return empty DataFrame with same columns
        return df.iloc[0:0].copy()


def evaluate_growth_rate_performance(growth_rates_df: pd.DataFrame, r_data_col: str = 'growth_rate_r_data', r_model_col: str = 'median_r', overestimation_tol: float = 1e-3, connect_gaps: bool = False, min_segment_length: Optional[int] = None, min_sequence_count: Optional[float] = None, min_variant_frequency: Optional[float] = None) -> dict:
    """
    Evaluate the performance of growth rate predictions by comparing model and empirical values.
    
    Parameters:
    -----------
    growth_rates_df : pd.DataFrame
        DataFrame containing growth rate data with columns 'growth_rate_r_data' (empirical)
        and 'median_r' (model).
    r_data_col : str, default='growth_rate_r_data'
        The column name for the empirical growth rate.
    r_model_col : str, default='median_r'
        The column name for the model growth rate.
    overestimation_tol : float, default=1e-3
        Tolerance for overestimation rate calculation.
    connect_gaps : bool, default=False
        If True, connects all non-NaN points ignoring gaps. If False, finds continuous 
        segments of non-NaN values.
    min_segment_length : Optional[int], default=None
        Minimum number of consecutive non-NaN points required for a segment to be valid.
        If None, no minimum length is enforced.
    min_sequence_count : Optional[float], default=None
        Minimum smoothed sequence count to trust growth rate calculations.
        If provided, filters out points below this threshold.
    min_variant_frequency : Optional[float], default=None
        Minimum variant frequency to trust growth rate calculations.
        If provided, filters out points below this threshold.

    Returns:
    --------
    dict
        A dictionary containing evaluation metrics:
        - 'correlation': Pearson correlation coefficient between model and empirical growth rates
        - 'mae': Mean Absolute Error between model and empirical growth rates
        - 'rmse': Root Mean Square Error between model and empirical growth rates
        - 'sign_disagreement_rate': Rate at which model and empirical growth rates disagree on sign
        - 'overestimation_rate': Rate at which model overestimates growth rates
        - 'n_points': Number of valid data points used in the evaluation
    """
    # First apply sequence count and frequency filters if provided
    clean_data = growth_rates_df.copy()
    
    if min_sequence_count is not None or min_variant_frequency is not None:
        # Build filter mask
        filter_mask = pd.Series(True, index=clean_data.index)
        
        if min_sequence_count is not None and 'smoothed_sequences' in clean_data.columns:
            filter_mask &= clean_data['smoothed_sequences'] >= min_sequence_count
            
        if min_variant_frequency is not None and 'variant_frequency_smoothed' in clean_data.columns:
            filter_mask &= clean_data['variant_frequency_smoothed'] >= min_variant_frequency
            
        # Apply the filter
        clean_data = clean_data[filter_mask]
    
    # Apply the same filtering logic as used in plotting
    clean_data = filter_growth_rates(
        clean_data, 
        r_data_col=r_data_col, 
        connect_gaps=connect_gaps, 
        min_segment_length=min_segment_length
    )
    
    # Additional filtering to ensure both columns have valid data
    clean_data = clean_data.dropna(subset=[r_data_col, r_model_col])
    
    # Correlation (captures linear relationship)
    correlation = clean_data[r_data_col].corr(clean_data[r_model_col])
    
    # Mean Absolute Error (captures magnitude of differences)
    mae = (clean_data[r_data_col] - clean_data[r_model_col]).abs().mean()
    
    # Root Mean Square Error (penalizes larger errors more)
    rmse = np.sqrt(((clean_data[r_data_col] - clean_data[r_model_col])**2).mean())

    # Calculate sign disagreement rate
    sign_disagreement_rate = calculate_sign_disagreement_rate(clean_data, r_data_col, r_model_col)

    # Calculate overestimation rate
    overestimation_rate = calculate_overestimation_rate(clean_data, r_data_col, r_model_col, tol=overestimation_tol)
    
    return {
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse,
        'sign_disagreement_rate': sign_disagreement_rate,
        'overestimation_rate': overestimation_rate,
        'n_points': len(clean_data)
    }


def calculate_variant_mae(
    growth_rates_df: pd.DataFrame,
    r_data_col: str = 'growth_rate_r_data',
    r_model_col: str = 'median_r',
    overestimation_tol: float = 1e-3,
    min_sequence_count: Optional[float] = None,
    min_variant_frequency: Optional[float] = None
) -> pd.DataFrame:
    """
    Calculate MAE and max r_data value for each variant in the growth rates dataframe.
    
    For each unique variant in the dataframe, this function calculates:
    - Mean Absolute Error (MAE) between model and data growth rates
    - Maximum absolute r_data value for that variant
    - Pearson correlation between model and data growth rates
    - Sign disagreement rate (fraction of times model and data disagree on sign)
    - Overestimation rate (fraction of times model overestimates growth)
    
    Parameters:
    -----------
    growth_rates_df : pd.DataFrame
        DataFrame containing growth rate data with columns for variant identification,
        model growth rates, and data growth rates. Expected columns:
        - 'variant': variant identifier
        - 'country': location/country
        - 'model': model name
        - 'analysis_date': analysis date
        - r_data_col: empirical growth rate column
        - r_model_col: model growth rate column
    r_data_col : str, default='growth_rate_r_data'
        Column name for empirical growth rates
    r_model_col : str, default='median_r'
        Column name for model growth rates
    overestimation_tol : float, default=1e-3
        Tolerance for overestimation rate calculation
    min_sequence_count : Optional[float], default=None
        Minimum smoothed sequence count to trust growth rate calculations.
        If provided, filters out points below this threshold.
    min_variant_frequency : Optional[float], default=None
        Minimum variant frequency to trust growth rate calculations.
        If provided, filters out points below this threshold.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per variant containing:
        - variant: variant identifier
        - country: location
        - model: model name
        - analysis_date: analysis date
        - mae: Mean Absolute Error for that variant
        - normalized_mae: Normalized Mean Absolute Error (MAE / max_r_data)
        - max_r_data: Maximum absolute r_data value for that variant
        - correlation: Pearson correlation coefficient
        - sign_disagreement_rate: Fraction of times signs disagree
        - overestimation_rate: Fraction of times model overestimates
        - n_points: Number of data points used in calculations
    """
    # First apply sequence count and frequency filters if provided
    clean_df = growth_rates_df.copy()
    
    if min_sequence_count is not None or min_variant_frequency is not None:
        # Build filter mask
        filter_mask = pd.Series(True, index=clean_df.index)
        
        if min_sequence_count is not None and 'smoothed_sequences' in clean_df.columns:
            filter_mask &= clean_df['smoothed_sequences'] >= min_sequence_count
            
        if min_variant_frequency is not None and 'variant_frequency_smoothed' in clean_df.columns:
            filter_mask &= clean_df['variant_frequency_smoothed'] >= min_variant_frequency
            
        # Apply the filter
        clean_df = clean_df[filter_mask]
    
    # Filter out rows with NaN values in either growth rate column
    clean_df = clean_df.dropna(subset=[r_data_col, r_model_col])
    
    # Group by variant and calculate metrics
    variant_metrics = []
    
    for variant in clean_df['variant'].unique():
        variant_data = clean_df[clean_df['variant'] == variant]
        
        # Skip if no data for this variant
        if len(variant_data) == 0:
            continue
            
        # Calculate MAE for this variant
        mae = (variant_data[r_data_col] - variant_data[r_model_col]).abs().mean()
        
        # Calculate max absolute r_data value
        max_r_data = variant_data[r_data_col].abs().max()

        # Normalize MAE by max r_data value (to avoid division by zero)
        if max_r_data > 0:
            normalized_mae = mae / max_r_data
        else:
            normalized_mae = np.nan
        
        # Calculate correlation
        if len(variant_data) >= 2:  # Need at least 2 points for correlation
            try:
                correlation, _ = pearsonr(variant_data[r_data_col], variant_data[r_model_col])
            except:
                correlation = np.nan
        else:
            correlation = np.nan
        
        # Calculate sign disagreement rate
        sign_disagreement_rate = calculate_sign_disagreement_rate(
            variant_data, r_data_col, r_model_col
        )
        
        # Calculate overestimation rate
        overestimation_rate = calculate_overestimation_rate(
            variant_data, r_data_col, r_model_col, tol=overestimation_tol
        )
        
        # Get metadata (these should be consistent within a variant)
        country = variant_data['country'].iloc[0]
        model = variant_data['model'].iloc[0] if 'model' in variant_data.columns else None
        analysis_date = variant_data['analysis_date'].iloc[0] if 'analysis_date' in variant_data.columns else None
        
        variant_metrics.append({
            'variant': variant,
            'country': country,
            'model': model,
            'analysis_date': analysis_date,
            'mae': mae,
            'normalized_mae': normalized_mae,
            'max_r_data': max_r_data,
            'correlation': correlation,
            'sign_disagreement_rate': sign_disagreement_rate,
            'overestimation_rate': overestimation_rate,
            'n_points': len(variant_data)
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(variant_metrics)
    
    # Sort by variant for consistency
    result_df = result_df.sort_values('variant').reset_index(drop=True)
    
    return result_df