import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from scipy.stats import pearsonr
from antigentools.utils import (
    smooth_with_spline,
    calculate_sign_disagreement_rate,
    calculate_overestimation_rate,
)


def calculate_fitness_of_tips(
    tips_df: pd.DataFrame,
    host_coordinates: Tuple[float, float],
    s: float = 0.07,
    homologous_immunity: float = 0.95
) -> pd.DataFrame:
    """
    Calculate fitness of tips based on infection risk against host coordinates.

    Fitness = infection risk: variants antigenically distant from host immune
    memory have higher fitness (higher chance of infection).

    Parameters
    ----------
    tips_df : pd.DataFrame
        DataFrame with 'ag1', 'ag2' columns.
    host_coordinates : Tuple[float, float]
        Tuple of (ag1, ag2) for host immune memory centroid.
    s : float, default=0.07
        Smith conversion factor scaling antigenic distance to infection risk.
    homologous_immunity : float, default=0.95
        Immunity against identical antigens (bounds minimum fitness).

    Returns
    -------
    pd.DataFrame
        Copy of tips_df with added 'fitness' column.
    """
    result_df = tips_df.copy()

    # Euclidean distance from each tip to host coordinates
    distances = np.sqrt(
        (tips_df['ag1'] - host_coordinates[0])**2 +
        (tips_df['ag2'] - host_coordinates[1])**2
    )

    # Calculate risk of infection, bounded to [min_risk, 1.0]
    risk = distances * s
    min_risk = 1.0 - homologous_immunity
    risk = np.clip(risk, min_risk, 1.0)

    result_df['fitness'] = risk
    return result_df


def calc_variance_over_time(
    tips_df: pd.DataFrame,
    host_memory_df: pd.DataFrame,
    variant_cols: List[str],
    s: float = 0.07,
    homologous_immunity: float = 0.95
) -> pd.DataFrame:
    """
    Calculate mean within-variant fitness variance over time.

    For each timepoint t, calculates fitness for ALL tips using host coordinates
    at t, then computes mean variance of fitness within each variant group.
    n_variants counts unique variants in [t-1, t] window only.

    Parameters
    ----------
    tips_df : pd.DataFrame
        DataFrame with 'ag1', 'ag2', 'year', and variant_* columns.
    host_memory_df : pd.DataFrame
        DataFrame with 'year', 'ag1', 'ag2' columns (host immune memory).
    variant_cols : List[str]
        List of variant column names (e.g., ['variant_ag', 'variant_phylo']).
    s : float, default=0.07
        Smith conversion factor.
    homologous_immunity : float, default=0.95
        Immunity against identical antigens.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: year, method, mean_variance, n_variants.
    """
    results = []
    time_points = sorted(host_memory_df['year'].unique())

    for t in time_points:
        # Get host coordinates at time t
        host_row = host_memory_df[host_memory_df['year'] == t].iloc[0]
        host_coords = (host_row['ag1'], host_row['ag2'])

        # Calculate fitness for ALL tips
        tips_with_fitness = calculate_fitness_of_tips(
            tips_df,
            host_coordinates=host_coords,
            s=s,
            homologous_immunity=homologous_immunity
        )

        # Calculate variance for each variant assignment method
        for var_col in variant_cols:
            # Variance across ALL tips
            variances = tips_with_fitness.groupby(var_col)['fitness'].var()
            mean_variance = variances.mean()

            # Count variants in [t-1, t] window only
            year_mask = (tips_df['year'] > t - 1) & (tips_df['year'] <= t)
            n_variants = tips_df.loc[year_mask, var_col].nunique()

            # Strip 'variant_' prefix for method name
            method = var_col.replace('variant_', '')

            results.append({
                'year': t,
                'method': method,
                'mean_variance': mean_variance,
                'n_variants': n_variants
            })

    return pd.DataFrame(results)

def load_model_rt_values(build: str, model: str, location: str, pivot_date: str, results_path: str = "../results/") -> Optional[pd.DataFrame]:
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
    results_path : str, default="../results/"
        Base directory for results files.
    
    Returns:
    --------
    Optional[pd.DataFrame]
        A DataFrame containing the Rt values, or None if the file is not found.
    """
    # Construct path to the Rt file for this model, location, and date
    rt_path = f"{results_path}{build}/estimates/{model}/rt_{location}_{pivot_date}.tsv"
    
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
    min_variant_incidence=50.0,
    skip_first_n_points=2,
    use_freqs=True,
    use_smoothed_incidence=True,
    data_path="../data/",
):
    """
    Get growth rates dataframe with automatic filtering of unreliable data points.
    
    Uses improved variant incidence calculation pipeline with proper date alignment
    and complete variant×time frequency matrix. Includes handling for missing case 
    data and frequency re-normalization after spline smoothing.
    
    Key Processing Steps:
    -------------------
    1. Aligns all dates to weekly boundaries (Monday week starts)
    2. Creates complete variant×time frequency matrix (fills missing combinations with 0)
    3. Applies spline smoothing to both sequence counts and variant frequencies
    4. Re-normalizes smoothed frequencies to ensure they sum to 1.0 at each time point
    5. Handles missing case data by taking first chronological observation per week
    6. Fills remaining missing case data using nearest neighbor approach (±2 days)
    7. Calculates variant incidence and empirical growth rates
    8. Merges with model results and applies filtering
    
    Data Alignment & Missing Data Handling:
    ---------------------------------------
    - When multiple case observations exist for the same week, takes the first chronologically
    - For missing case data, searches for observations within ±2 days and uses closest match
    - Spline-smoothed frequencies are clipped to [0,1] and re-normalized to sum to 1.0
    
    Parameters:
    -----------
    build : str
        The build name (e.g., 'flu-simulated-150k-samples')
    model : str
        The model name (e.g., 'FGA', 'GARW')
    location : str
        The location name (e.g., 'north', 'tropics')
    pivot_date : str
        The pivot date in YYYY-MM-DD format
    spline_smoothing_factor : float, default=5.0
        Smoothing factor for splines (larger = smoother)
    spline_order : int, default=3
        Degree of the spline (1-5)
    min_sequence_count : float, default=5
        Minimum smoothed sequence count to trust growth rate calculations
    min_variant_frequency : float, default=0.05
        Minimum variant frequency to trust growth rate calculations
    min_variant_incidence : float, default=50.0
        Minimum smoothed variant incidence to trust growth rate calculations
    skip_first_n_points : int, default=2
        Number of initial points to skip for each variant (often noisy)
    use_freqs : bool, default=True
        Whether to use frequencies instead of raw counts
    use_smoothed_incidence : bool, default=True
        Whether to use smoothed incidence for r_data calculations.
        If False, uses raw incidence (variant_frequency × cases).
        If True, uses smoothed incidence (variant_frequency_smoothed × cases).
    data_path : str, default='../data/'
        Directory where data lives.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing growth rates with proper variant incidence calculations
    """
    # Load raw data
    seqs_path = f"{data_path}{build}/time-stamped/{pivot_date}/seq_counts.tsv"
    cases_path = f"{data_path}{build}/time-stamped/{pivot_date}/case_counts.tsv"
    seqs_df = pd.read_csv(seqs_path, sep='\t')
    cases_df = pd.read_csv(cases_path, sep='\t')
    
    # Filter for location
    seqs_df = seqs_df[seqs_df['country'] == location]
    cases_df = cases_df[cases_df['country'] == location]
    
    # Step 1: Align weekly dates
    def align_weekly_dates(df, date_col='date'):
        """Align dates to week start (Monday) for consistent merging."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        # Map to week start (Monday)
        df['week_start'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='d')
        # Create year_week based on week_start
        df['year_week'] = df['week_start'].dt.strftime('%Y-%U')
        return df
    
    seqs_aligned = align_weekly_dates(seqs_df)
    cases_aligned = align_weekly_dates(cases_df)
    
    # Step 2: Calculate complete variant frequencies (with 0.0 for missing combinations)
    def calculate_variant_frequencies_complete(seqs_df):
        """Calculate variant frequencies with complete variant×time matrix."""
        # Get all unique combinations
        all_variants = seqs_df['variant'].unique()
        all_week_starts = seqs_df['week_start'].unique()
        
        # Create complete combinations
        from itertools import product
        complete_combinations = list(product([location], all_variants, all_week_starts))
        complete_df = pd.DataFrame(complete_combinations, columns=['country', 'variant', 'week_start'])
        
        # Add week_start derived columns
        complete_df['week_start'] = pd.to_datetime(complete_df['week_start'])
        complete_df['year_week'] = complete_df['week_start'].dt.strftime('%Y-%U')
        
        # Merge with original data
        merged_df = pd.merge(
            complete_df,
            seqs_df[['country', 'variant', 'week_start', 'sequences', 'date']],
            on=['country', 'variant', 'week_start'],
            how='left'
        )
        
        # Fill missing sequences with 0
        merged_df['sequences'] = merged_df['sequences'].fillna(0.0)
        
        # Calculate frequencies
        total_by_week = merged_df.groupby(['country', 'week_start'])['sequences'].transform('sum')
        merged_df['variant_frequency'] = np.where(
            total_by_week > 0,
            merged_df['sequences'] / total_by_week,
            0.0
        )
        
        return merged_df
    
    seqs_with_freq = calculate_variant_frequencies_complete(seqs_aligned)
    
    # Step 3: Apply spline smoothing
    seqs_smoothed = smooth_with_spline(
        seqs_with_freq, 
        col_to_smooth='sequences', 
        output_col='smoothed_sequences',
        s=spline_smoothing_factor, 
        k=spline_order,
        date_col='week_start'
    )
    
    seqs_smoothed = smooth_with_spline(
        seqs_smoothed,
        col_to_smooth='variant_frequency',
        output_col='variant_frequency_smoothed', 
        s=spline_smoothing_factor,
        k=spline_order,
        date_col='week_start'
    )
    
    # Re-normalize smoothed frequencies to ensure they sum to 1.0 and are in [0, 1]
    # First, clip negative values to 0
    seqs_smoothed['variant_frequency_smoothed'] = seqs_smoothed['variant_frequency_smoothed'].clip(lower=0.0)
    
    # Re-normalize so frequencies sum to 1.0 at each time point
    # The only edge case is if sum is 0 (all variants have 0 frequency), then keep as 0
    seqs_smoothed['variant_frequency_smoothed'] = seqs_smoothed.groupby(['country', 'week_start'])['variant_frequency_smoothed'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )
    
    # Step 4: Calculate variant incidence
    # Merge with case counts - take first observation chronologically when multiple exist
    cases_weekly = cases_aligned.sort_values('date').groupby(['country', 'week_start']).first().reset_index()[['country', 'week_start', 'cases']]
    
    growth_rates_df = pd.merge(
        seqs_smoothed,
        cases_weekly,
        on=['country', 'week_start'],
        how='left'
    )
    
    # Fill any missing case counts using nearest neighbor approach (±2 days)
    missing_cases = growth_rates_df['cases'].isna()
    if missing_cases.any():
        missing_weeks = growth_rates_df[missing_cases]['week_start'].unique()
        location_cases_raw = cases_aligned[cases_aligned['country'] == location].copy()
        
        for missing_week in missing_weeks:
            # Find case data within ±2 days of this week_start (using original raw dates)
            time_diff = (location_cases_raw['date'] - missing_week).abs()
            nearby_mask = time_diff <= pd.Timedelta(days=2)
            nearby_cases = location_cases_raw[nearby_mask]
            
            if len(nearby_cases) > 0:
                # Use the closest case count (by original date)
                closest_idx = time_diff[nearby_mask].idxmin()
                closest_cases = location_cases_raw.loc[closest_idx, 'cases']
                
                # Fill all rows for this location and week_start
                fill_mask = (growth_rates_df['week_start'] == missing_week) & (growth_rates_df['country'] == location)
                growth_rates_df.loc[fill_mask, 'cases'] = closest_cases
    
    # Calculate incidence
    growth_rates_df['variant_incidence'] = growth_rates_df['variant_frequency'] * growth_rates_df['cases']
    growth_rates_df['variant_incidence_smoothed'] = growth_rates_df['variant_frequency_smoothed'] * growth_rates_df['cases']
    
    # Apply variant incidence filtering if specified
    if min_variant_incidence is not None and min_variant_incidence > 0:
        incidence_mask = growth_rates_df['variant_incidence_smoothed'] >= min_variant_incidence
        growth_rates_df = growth_rates_df[incidence_mask].copy()
    
    # Ensure we have a proper date column for plotting compatibility
    # Use the original date if available, otherwise use week_start
    if 'date' not in growth_rates_df.columns or growth_rates_df['date'].isna().any():
        growth_rates_df['date'] = growth_rates_df['week_start']
    
    # Ensure date column is datetime type
    growth_rates_df['date'] = pd.to_datetime(growth_rates_df['date'])
    
    # Step 5: Calculate empirical growth rates
    def calculate_growth_rates(df, use_smoothed=False):
        """Calculate empirical growth rates from variant incidence."""
        df = df.copy()
        df['growth_rate_r_data'] = np.nan
        
        # Choose which incidence column to use
        incidence_col = 'variant_incidence_smoothed' if use_smoothed else 'variant_incidence'
        
        # Sort by week_start within each variant
        df = df.sort_values(['variant', 'week_start'])
        
        for variant in df['variant'].unique():
            variant_mask = df['variant'] == variant
            variant_data = df[variant_mask].copy()
            
            if len(variant_data) < 2:
                continue
                
            # Calculate time differences in days
            variant_data['days_diff'] = variant_data['week_start'].diff().dt.days
            
            # Calculate growth rates
            for i in range(1, len(variant_data)):
                curr_incidence = variant_data.iloc[i][incidence_col]
                prev_incidence = variant_data.iloc[i-1][incidence_col] 
                days_diff = variant_data.iloc[i]['days_diff']
                
                if (curr_incidence > 0 and prev_incidence > 0 and 
                    not pd.isna(curr_incidence) and not pd.isna(prev_incidence) and
                    days_diff > 0):
                    
                    growth_rate = (np.log(curr_incidence) - np.log(prev_incidence)) / days_diff
                    df.loc[variant_data.index[i], 'growth_rate_r_data'] = growth_rate
        
        return df
    
    growth_rates_df = calculate_growth_rates(growth_rates_df, use_smoothed=use_smoothed_incidence)
    
    # Load model results and merge
    rt_df = load_model_rt_values(build, model, location, pivot_date, results_path=data_path.replace('data/', 'results/'))
    if rt_df is not None:
        # Ensure both date columns are datetime type
        growth_rates_df['date'] = pd.to_datetime(growth_rates_df['date'])
        rt_df['date'] = pd.to_datetime(rt_df['date'])
        
        # Try direct merge first (exact date matching)
        # Include all confidence interval columns if they exist
        merge_cols = ['date', 'variant', 'median_r', 'model', 'analysis_date']
        
        # Add confidence interval columns if they exist in rt_df
        confidence_cols = ['r_lower_95', 'r_upper_95', 'r_lower_80', 'r_upper_80', 'r_lower_50', 'r_upper_50']
        for col in confidence_cols:
            if col in rt_df.columns:
                merge_cols.append(col)
        
        growth_rates_df = pd.merge(
            growth_rates_df,
            rt_df[merge_cols],
            on=['date', 'variant'],
            how='left'
        )
        
        # Calculate absolute error where we have both values
        mask = ~(growth_rates_df['growth_rate_r_data'].isna() | growth_rates_df['median_r'].isna())
        growth_rates_df.loc[mask, 'abs_error'] = (
            growth_rates_df.loc[mask, 'growth_rate_r_data'] - growth_rates_df.loc[mask, 'median_r']
        ).abs()
        
        # Set median_r to NaN for variants with zero frequency (like in original function)
        zero_freq_mask = growth_rates_df['variant_frequency'] == 0.0
        growth_rates_df.loc[zero_freq_mask, 'median_r'] = np.nan
    
    # Apply filtering
    # Filter out low count/frequency data points
    if 'smoothed_sequences' in growth_rates_df.columns and 'variant_frequency_smoothed' in growth_rates_df.columns:
        low_count_mask = (
            (growth_rates_df['smoothed_sequences'] < min_sequence_count) | 
            (growth_rates_df['variant_frequency_smoothed'] < min_variant_frequency)
        )
        growth_rates_df.loc[low_count_mask, 'growth_rate_r_data'] = np.nan
    
    # Skip the first few points for each variant
    if skip_first_n_points > 0:
        for variant in growth_rates_df['variant'].unique():
            variant_mask = growth_rates_df['variant'] == variant
            variant_indices = growth_rates_df[variant_mask].sort_values('week_start').index
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
        - 'r2': R² (coefficient of determination) between model and empirical growth rates
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
    
    # Mean Square Error
    mse = ((clean_data[r_data_col] - clean_data[r_model_col])**2).mean()
    
    # Root Mean Square Error (penalizes larger errors more)
    rmse = np.sqrt(mse)

    # R² (coefficient of determination)
    ss_res = ((clean_data[r_data_col] - clean_data[r_model_col])**2).sum()
    ss_tot = ((clean_data[r_data_col] - clean_data[r_data_col].mean())**2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Calculate sign disagreement rate
    sign_disagreement_rate = calculate_sign_disagreement_rate(clean_data, r_data_col, r_model_col)

    # Calculate overestimation rate
    overestimation_rate = calculate_overestimation_rate(clean_data, r_data_col, r_model_col, tol=overestimation_tol)

    return {
        'correlation': correlation,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
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
    min_variant_frequency: Optional[float] = None,
    min_variant_incidence: Optional[float] = None
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
    min_variant_incidence : Optional[float], default=None
        Minimum smoothed variant incidence to trust growth rate calculations.
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
        - mse: Mean Squared Error for that variant
        - normalized_mae: Normalized Mean Absolute Error (MAE / max_r_data)
        - max_r_data: Maximum absolute r_data value for that variant
        - correlation: Pearson correlation coefficient
        - r2: R² (coefficient of determination)
        - sign_disagreement_rate: Fraction of times signs disagree
        - overestimation_rate: Fraction of times model overestimates
        - n_points: Number of data points used in calculations
        - total_sequences: Sum of raw sequence counts across all timepoints
        - total_smoothed_sequences: Sum of smoothed sequence counts across all timepoints
        - mean_variant_frequency: Mean of variant frequencies across timepoints
        - mean_smoothed_variant_frequency: Mean of smoothed variant frequencies across timepoints
        - max_variant_frequency: Maximum variant frequency achieved
        - max_smoothed_variant_frequency: Maximum smoothed variant frequency achieved
    """
    # First apply sequence count and frequency filters if provided
    clean_df = growth_rates_df.copy()
    
    if min_sequence_count is not None or min_variant_frequency is not None or min_variant_incidence is not None:
        # Build filter mask
        filter_mask = pd.Series(True, index=clean_df.index)
        
        if min_sequence_count is not None and 'smoothed_sequences' in clean_df.columns:
            filter_mask &= clean_df['smoothed_sequences'] >= min_sequence_count
            
        if min_variant_frequency is not None and 'variant_frequency_smoothed' in clean_df.columns:
            filter_mask &= clean_df['variant_frequency_smoothed'] >= min_variant_frequency
            
        if min_variant_incidence is not None and 'variant_incidence_smoothed' in clean_df.columns:
            filter_mask &= clean_df['variant_incidence_smoothed'] >= min_variant_incidence
            
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
        
        # Calculate MSE for this variant
        mse = ((variant_data[r_data_col] - variant_data[r_model_col])**2).mean()
        
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

        # Calculate R²
        if len(variant_data) >= 2:
            ss_res = ((variant_data[r_data_col] - variant_data[r_model_col])**2).sum()
            ss_tot = ((variant_data[r_data_col] - variant_data[r_data_col].mean())**2).sum()
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        else:
            r2 = np.nan
        
        # Calculate sign disagreement rate
        sign_disagreement_rate = calculate_sign_disagreement_rate(
            variant_data, r_data_col, r_model_col
        )
        
        # Calculate overestimation rate
        overestimation_rate = calculate_overestimation_rate(
            variant_data, r_data_col, r_model_col, tol=overestimation_tol
        )
        
        # Calculate sequence and frequency summaries
        total_sequences = variant_data['sequences'].sum() if 'sequences' in variant_data.columns else np.nan
        total_smoothed_sequences = variant_data['smoothed_sequences'].sum() if 'smoothed_sequences' in variant_data.columns else np.nan
        mean_variant_frequency = variant_data['variant_frequency'].mean() if 'variant_frequency' in variant_data.columns else np.nan
        mean_smoothed_variant_frequency = variant_data['variant_frequency_smoothed'].mean() if 'variant_frequency_smoothed' in variant_data.columns else np.nan
        max_variant_frequency = variant_data['variant_frequency'].max() if 'variant_frequency' in variant_data.columns else np.nan
        max_smoothed_variant_frequency = variant_data['variant_frequency_smoothed'].max() if 'variant_frequency_smoothed' in variant_data.columns else np.nan
        
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
            'mse': mse,
            'normalized_mae': normalized_mae,
            'max_r_data': max_r_data,
            'correlation': correlation,
            'r2': r2,
            'sign_disagreement_rate': sign_disagreement_rate,
            'overestimation_rate': overestimation_rate,
            'n_points': len(variant_data),
            'total_sequences': total_sequences,
            'total_smoothed_sequences': total_smoothed_sequences,
            'mean_variant_frequency': mean_variant_frequency,
            'mean_smoothed_variant_frequency': mean_smoothed_variant_frequency,
            'max_variant_frequency': max_variant_frequency,
            'max_smoothed_variant_frequency': max_smoothed_variant_frequency
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(variant_metrics)
    
    # Sort by variant for consistency
    result_df = result_df.sort_values('variant').reset_index(drop=True)
    
    return result_df