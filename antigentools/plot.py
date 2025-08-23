#!/usr/bin/env python3
"""
Visualization utilities for antigen forecasting.

This module contains functions for visualizing observed and predicted frequencies, 
growth advantages (Rt), fitness, and case counts for variants across different locations.
"""
import glob
import os
import random
import pandas as pd
import numpy as np
import evofr as ef

# Plotting imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Antigentools imports
from antigentools.utils import (
    read_estimates
)
from antigentools.analysis import (
    get_top_variants,
    filter_growth_rates
)

# Mute warnings
import warnings
warnings.filterwarnings('ignore')

from typing import Any, Dict, List, Optional, Tuple, Union

def get_distinct_colors(n: int) -> List[str]:
    """Generates n distinct colors in hex format.

    Parameters:
    -----------
    n : int
        Number of colors to generate

    Returns:
    --------
    List[str]
        List of hex colors
    """
    cmap = plt.get_cmap('hsv')  # Use a colormap like 'hsv' or 'rainbow'
    colors = [cmap(i / n) for i in range(n)]
    hex_colors = [plt.cm.colors.to_hex(c) for c in colors]
    # Shuffle colors to avoid adjacent colors being too similar
    random.shuffle(hex_colors)
    return hex_colors


def make_color_map(v_names: List[str], palette: List[str]) -> Dict[str, str]:
    """Create a color map for variants.

    Parameters:
    -----------
    v_names : List[str]
        List of variant names
    palette : List[str]
        List of colors to use for variants

    Returns:
    --------
    Dict[str, str]
        A dictionary mapping variant names to colors
    """
    v_colors = palette[:len(v_names)]
    color_map = dict(zip(v_names, v_colors))
    return color_map

def plot_antigenic_space_by_clade(
    tips_df: pd.DataFrame, 
    color_map: Dict[str, str], 
    figsize: Tuple[int, int] = (10, 10), 
    variant_col: str = 'variant'
) -> None:
    """Plot antigenic space colored by clade.

    Parameters:
    -----------
    tips_df : pd.DataFrame
        Tips dataframe containing antigenic coordinates and variant information
    color_map : Dict[str, str]
        Dictionary mapping clades to colors
    figsize : Tuple[int, int], default=(10, 10)
        Figure size as (width, height)
    variant_col : str, default='variant'
        Column name for variant
    
    Returns:
    --------
    None
    """
    fig, ax = plt.subplots(figsize=figsize)
    n_variants = tips_df[variant_col].nunique()

    # Now plot each variant in the antigenic space
    for var_id in range(n_variants):
        variant_df = tips_df.query(f"{variant_col} == {var_id}")
        ax.scatter(variant_df['ag1'], variant_df['ag2'], color=color_map[var_id], s=40, edgecolors='black', alpha=0.8, label=var_id)
    ax.set_xlabel("ag1")
    ax.set_ylabel("ag2")
    ax.set_title("Antigenic space")

    # Create a legend
    patches = [Patch(color=c, label=l) for l, c in color_map.items()]
    legend = fig.legend(patches, list(color_map.keys()), ncol=10, bbox_to_anchor=(0.5, -0.15), 
                       loc="lower center", title="Variant")  
    legend.get_frame().set_linewidth(2.)
    legend.get_frame().set_edgecolor("k")
    fig.tight_layout()
    plt.show()

def plot_observed_cases(
    ax: plt.Axes, 
    flu_data: Dict[str, Any], 
    deme: str, 
    pivot_idx: Optional[List[int]] = None,
    label_fontsize: int = 14
) -> None:
    """Plot observed cases.

    Parameters:
    -----------
    ax : plt.Axes
        Axis to plot on
    flu_data : Dict[str, Any]
        Dictionary of `evofr` data
    deme : str
        Location/deme name (e.g., 'north', 'tropics', 'south')
    pivot_idx : Optional[List[int]], default=None
        List of time indices to mark with vertical lines
    label_fontsize : int, default=14
        Font size for axis labels
        
    Returns:
    --------
    None
    """
    t = np.arange(0, flu_data[deme].cases.shape[0])
    ax.bar(t, flu_data[deme].cases, width=5.0, color='black', alpha=0.5)
    if pivot_idx:
        for idx in pivot_idx:
            ax.axvline(x=idx, color='black', linestyle='--', lw=2.0)
    ax.set_ylabel("Cases", fontsize=label_fontsize)


def plot_observed_freqs(ax: plt.Axes, ef_data: Dict[str, Any], deme: str, color_map: Dict[str, str], label_fontsize: int = 14) -> None:
    """ Plot observed frequencies of variants.
    
    Parameters:
    --------------
        ax (matplotlib.pyplot.Axes): Axes object
        ef_data (Dict): Dictionary of `evofr` data
        deme (str): Demographic unit
        color_map (dict): Dictionary mapping variant names to colors
        label_fontsize (int, optional): Font size for axis labels. Defaults to 14.
    
    Returns:
    --------------
        None
    """
    # Get variant names
    v_names = sorted(ef_data[deme].var_names)
    n_variant = len(v_names)
    
    # Calculate observed frequencies
    obs_freq = np.divide(ef_data[deme].seq_counts, ef_data[deme].seq_counts.sum(axis=1)[:, None])
    t = np.arange(0, obs_freq.shape[0])

    for variant in range(n_variant):
        v_name = v_names[variant]
        # Plot observed frequencies
        ax.scatter(t, obs_freq[:, variant], color=color_map[v_name], alpha=0.8, s=150, edgecolors='black')
    
    # Axes
    ax.set_ylabel("Frequency", fontsize=label_fontsize)


def plot_frequencies(
    ax: plt.Axes, 
    ef_data: Dict[str, Any], 
    freqs: pd.DataFrame, 
    deme: str, 
    model: str, 
    color_map: Dict[str, str], 
    p: int = 50, 
    pivot_idx: Optional[List[int]] = None,
    label_fontsize: int = 14
) -> None:
    """ Plot observed and predicted frequencies of variants.

    Parameters:
    ---------------
        ax (matplotlib.axes.Axes): Axis to plot on
        ef_data (dict): Dictionary of evofr data
        freqs (pd.DataFrame): Observed and predicted frequencies dataframe
        deme (str): Deme name
        model (str): Model name
        color_map (dict): Dictionary of colors to use for variants
        p (int): Percentile to plot
        pivot_idx (list): Index to pivot plot
        label_fontsize (int, optional): Font size for axis labels. Defaults to 14.

    Returns:
    ---------------
        None
    """
    # Get variant names
    v_names = sorted(ef_data[deme].var_names)
    n_variant = len(v_names)
    
    # Calculate observed frequencies
    obs_freq = np.divide(ef_data[deme].seq_counts, ef_data[deme].seq_counts.sum(axis=1)[:, None])
    t = np.arange(0, obs_freq.shape[0])

    # Grab predicted frequencies of interest
    deme_freq = freqs.query(f'location == "{deme}" and model == "{model}"')
    for variant in range(n_variant):
        v_name = v_names[variant]
        variant_df = deme_freq.query(f'variant == {v_name}')
        # Plot observed frequencies
        ax.scatter(t, obs_freq[:, variant], color=color_map[v_name], alpha=0.8, s=150, edgecolors='black')
        # Plot nowcasts
        ax.plot(variant_df['t'], variant_df['median_freq_nowcast'], lw=4.5, label=v_name, color=color_map[v_name])
        ax.fill_between(variant_df['t'], variant_df[f'freq_lower_{p}'], variant_df[f'freq_upper_{p}'], color=color_map[v_names[variant]], alpha=0.5)
        # Plot forecasts
        ax.plot(variant_df['t'], variant_df['median_freq_forecast'], lw=5.5, label=v_name, color=color_map[v_name], linestyle='--')
        ax.fill_between(variant_df['t'], variant_df[f'freq_forecast_lower_{p}'], variant_df[f'freq_forecast_upper_{p}'], color=color_map[v_names[variant]], alpha=0.5)
    if pivot_idx:
        for idx in pivot_idx:
            ax.axvline(x=idx, color='black', linestyle='--', lw=2.0)
    ax.set_xticks([])
    ax.set_ylabel("Frequency", fontsize=label_fontsize)


def plot_rt(
    ax: plt.Axes, 
    inferred_rt: pd.DataFrame, 
    deme: str, 
    model: str, 
    color_map: Dict[str, str], 
    p: int = 95, 
    pivot_idx: Optional[List[int]] = None,
    label_fontsize: int = 14
) -> None:
    """ Plot posteriors of inferred variant growth advantages.

    Parameters:
    ---------------
        ax (matplotlib.axes.Axes): Axis to plot on
        inferred_rt (pd.DataFrame): Inferred Rt dataframe
        deme (str): Deme/location name
        model (str): Model name 
        color_map (dict): Dictionary of colors to use for variants
        p (int): Percentile to plot
        pivot_idx (list): List of pivot indices
        label_fontsize (int, optional): Font size for axis labels. Defaults to 14.

    Returns:
    ---------------
        None
    """
    v_names = sorted(inferred_rt['variant'].unique())
    n_variant = len(v_names)
    
    # Create an integer-keyed version of the color map for integer variant IDs
    # This handles both cases where variant IDs are integers but color_map has string keys
    int_color_map = {}
    for k, v in color_map.items():
        try:
            int_key = int(k)
            int_color_map[int_key] = v
        except (ValueError, TypeError):
            # Skip keys that aren't convertible to integers
            pass
    
    # Create a consolidated color map that works with both string and integer keys
    combined_color_map = {**color_map, **int_color_map}
    
    # Grab predicted frequencies of interest
    deme_ga = inferred_rt.query(f'location == "{deme}" and model == "{model}"')
    for variant in range(n_variant):
        v_name = v_names[variant]
        variant_df = deme_ga.query(f'variant == {v_name}')
        
        # Get the appropriate color for this variant
        variant_str_key = str(v_name)
        variant_color = None
        if variant_str_key in combined_color_map:
            variant_color = combined_color_map[variant_str_key]
        elif v_name in combined_color_map:
            variant_color = combined_color_map[v_name]
            
        if variant_color is None:
            # Skip variants with no color
            continue
            
        ax.plot(variant_df['t'], variant_df['median_R'], lw=5.5, label=variant_str_key, color=variant_color)
        ax.fill_between(variant_df['t'], variant_df[f'R_lower_{p}'], variant_df[f'R_upper_{p}'], color=variant_color, alpha=0.5)
    
    if pivot_idx:
        for idx in pivot_idx:
            ax.axvline(x=idx, color='black', linestyle='--', lw=2.0)
    ax.axhline(y=1.0, color='red', linestyle='--', lw=1.5)
    ax.set_xlabel('')
    ax.set_ylabel(r"$R_t$", fontsize=label_fontsize)
    
    # Calculate and set appropriate y-axis limits based on data range if there's data to plot
    if not deme_ga.empty and 'median_R' in deme_ga.columns:
        min_y = deme_ga['median_R'].min()
        max_y = deme_ga['median_R'].max()
        
        # Add some padding (5% of the range)
        if min_y != max_y:  # Avoid division by zero if all values are the same
            padding = 0.05 * (max_y - min_y)
            min_y -= padding
            max_y += padding
        else:
            # If all values are the same, add a small absolute padding
            min_y -= 0.1
            max_y += 0.1
            
        # Ensure 1.0 is included in the y-range since it's a significant reference value for Rt
        min_y = min(min_y, 0.95)
        max_y = max(max_y, 1.05)
        
        ax.set_ylim(min_y, max_y)


def plot_fitness(
    ax: plt.Axes, 
    fitness_df: pd.DataFrame, 
    deme: str, 
    color_map: Dict[str, str], 
    pivot_idx: Optional[List[int]] = None,
    label_fontsize: int = 14
) -> None:
    """ Plot simulated fitness of variants.

    Parameters:
    ---------------
        ax (matplotlib.axes.Axes): Axis to plot on
        fitness_df (pd.DataFrame): Fitness dataframe
        deme (str): Deme name
        color_map (dict): Dictionary of colors to use for variants
        pivot_idx (list): List of pivot indices
        label_fontsize (int, optional): Font size for axis labels. Defaults to 14.

    Returns:
    ---------------
        None
    """
    v_names = sorted(fitness_df.dropna()['variant'].unique())
    n_variant = len(v_names)
    max_fitness = fitness_df['fitness'].max()
    min_fitness = fitness_df['fitness'].min()
        
    # Grab predicted frequencies of interest
    deme_fitness = fitness_df.query(f'location == "{deme}"')
    for variant in range(n_variant):
        v_name = v_names[variant]
        if v_name not in color_map:
            continue
        variant_df = deme_fitness.query(f'variant == {v_name}')
        ax.plot(variant_df['t'], variant_df['seasonal_fitness'], lw=6.0, label=v_name, color=color_map[v_name], linestyle='-')
    
    # Plot dashed vertical lines at pivot points
    if pivot_idx:
        for idx in pivot_idx:
            ax.axvline(x=idx, color='black', linestyle='--', lw=2.0)
    ax.set_xlabel('')
    ax.set_ylim(min_fitness - 0.1, max_fitness + 0.1)
    ax.set_ylabel("Fitness", fontsize=label_fontsize)


def plot_r_model(
    ax: plt.Axes, 
    rt_df: pd.DataFrame, 
    color_map: Dict[str, str], 
    analysis_date: Optional[str] = None, # pivot_idx removed
    plot_legend: bool = False
) -> None:
    """Plot model-inferred growth rates (r) for different variants.

    This function plots the 'growth_rate_r' column from the provided DataFrame
    against the 't' column. Each variant is plotted as a separate line, colored
    according to the 'color_map'. If the 't' column is not present, it will be
    derived from the 'date' column. If 'analysis_date' is provided, a vertical
    dashed line will be drawn at the 't' value corresponding to this date.
    A legend can be added by setting 'plot_legend' to True.

    Parameters:
    -----------
        ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
        rt_df (pd.DataFrame): A DataFrame that must contain 'variant', 
                              'growth_rate_r', and either 't' or ('date').
                              The 'variant' column should contain identifiers
                              (preferably strings) that are keys in the 'color_map'.
                              If 'analysis_date' is used, a 'date' column (convertible
                              to datetime) is typically required to map the date to a 't' value.
        color_map (Dict[str, str]): A dictionary mapping variant names to 
                                    hex color codes.
        analysis_date (Optional[str]): A date string (e.g., YYYY-MM-DD). If provided,
                                       a vertical dashed line will be drawn at the 't'
                                       value corresponding to this date. This requires
                                       the DataFrame to have a 'date' column if 't' is
                                       not derived, or if 't' is derived from 'date'.
                                       Defaults to None.
        plot_legend (bool): If True, a legend is added below the plot. 
                            Defaults to False.

    Raises:
    -------
        ValueError: If the 'rt_df' DataFrame does not contain the required columns
                    ('variant', 'growth_rate_r', and 't' or 'date').
    """
    rt_df_processed = rt_df.copy()

    required_base_cols = ['variant', 'growth_rate_r']
    for col in required_base_cols:
        if col not in rt_df_processed.columns:
            raise ValueError(f"Input DataFrame `rt_df` must contain a '{col}' column.")

    # Ensure 'date' column is datetime if it exists, for 't' derivation or analysis_date processing
    date_col_exists_and_is_datetime = False
    if 'date' in rt_df_processed.columns:
        if not pd.api.types.is_datetime64_any_dtype(rt_df_processed['date']):
            try:
                rt_df_processed['date'] = pd.to_datetime(rt_df_processed['date'])
                date_col_exists_and_is_datetime = True
            except Exception as e:
                print(f"Warning: Could not convert 'date' column to datetime: {e}")
        else:
            date_col_exists_and_is_datetime = True

    if 't' not in rt_df_processed.columns:
        if date_col_exists_and_is_datetime:
            # Normalize dates before creating mapping to ensure consistency
            rt_df_processed['date_normalized_for_t_mapping'] = rt_df_processed['date'].dt.normalize()
            unique_dates = sorted(rt_df_processed['date_normalized_for_t_mapping'].unique())
            date_to_t_mapping = {date_val: i for i, date_val in enumerate(unique_dates)}
            rt_df_processed['t'] = rt_df_processed['date_normalized_for_t_mapping'].map(date_to_t_mapping)
            rt_df_processed.drop(columns=['date_normalized_for_t_mapping'], inplace=True)
        else:
            raise ValueError("Input DataFrame `rt_df` must contain either a 't' column or a 'date' column (convertible to datetime) to derive 't'.")

    v_names = sorted(rt_df_processed['variant'].unique())
    
    # Create an integer-keyed version of the color map for integer variant IDs
    # This handles both cases where variant IDs in rt_df are integers but color_map has string keys
    int_color_map = {}
    for k, v in color_map.items():
        try:
            int_key = int(k)
            int_color_map[int_key] = v
        except (ValueError, TypeError):
            # Skip keys that aren't convertible to integers
            pass
    
    # Create a consolidated color map that works with both string and integer keys
    combined_color_map = {**color_map, **int_color_map}
    
    plotted_labels_handles = {} # To store unique labels and their corresponding handles for the legend

    for v_name in v_names:
        # Get the original variant ID for dataframe filtering
        original_variant_id = v_name
        # Convert to string for color map lookup and label
        variant_str_key = str(v_name)
        
        # Try to find a color for this variant - check both string and original format
        variant_color = None
        if variant_str_key in combined_color_map:
            variant_color = combined_color_map[variant_str_key]
        elif original_variant_id in combined_color_map:
            variant_color = combined_color_map[original_variant_id]
        
        if variant_color is None:
            # No color found for this variant, skip it
            # print(f"Warning: Variant '{variant_str_key}' not found in color_map. Skipping.")
            continue

        variant_df = rt_df_processed[rt_df_processed['variant'] == original_variant_id].sort_values(by='t')
        
        if not variant_df.empty:
            line, = ax.plot( # Capture the line artist
                variant_df['t'], 
                variant_df['growth_rate_r'], 
                lw=4.5,
                label=variant_str_key, 
                color=variant_color
            )
            if variant_str_key not in plotted_labels_handles:
                 plotted_labels_handles[variant_str_key] = line

    # Plot vertical line for analysis_date
    if analysis_date is not None:
        if date_col_exists_and_is_datetime:
            try:
                analysis_datetime_norm = pd.to_datetime(analysis_date).normalize()
                # Ensure the DataFrame's date column is also normalized for comparison
                # Use the original 'date' column for matching, not the potentially dropped normalized one
                matching_rows = rt_df_processed[rt_df_processed['date'].dt.normalize() == analysis_datetime_norm]
                if not matching_rows.empty:
                    t_values_for_analysis_date = matching_rows['t'].unique()
                    if len(t_values_for_analysis_date) > 0:
                        ax.axvline(x=t_values_for_analysis_date[0], color='black', linestyle='--', lw=2.0)
                        if len(t_values_for_analysis_date) > 1:
                            print(f"Warning: Multiple 't' values found for analysis_date '{analysis_date}'. Using the first: {t_values_for_analysis_date[0]}.")
                    # else: # This case should ideally not happen if matching_rows is not empty and 't' was derived
                        # print(f"Warning: analysis_date '{analysis_date}' found but no corresponding 't' value. No vertical line drawn.")
                else:
                    print(f"Warning: analysis_date '{analysis_date}' not found in rt_df. No vertical line drawn.")
            except Exception as e:
                print(f"Warning: Could not process analysis_date '{analysis_date}' for vertical line: {e}. No vertical line drawn.")
        else:
            print("Warning: 'analysis_date' provided for vertical line, but 'date' column is missing or not datetime in rt_df.")
    
    ax.axhline(y=0.0, color='red', linestyle='--', lw=1.5)
    ax.set_xlabel("") # Keep x-label empty as per previous context
    ax.set_ylabel("$r_{\\text{model}}$")
    
    # Calculate and set appropriate y-axis limits based on data range if there's data to plot
    if not rt_df_processed.empty and 'growth_rate_r' in rt_df_processed.columns:
        min_y = rt_df_processed['growth_rate_r'].min()
        max_y = rt_df_processed['growth_rate_r'].max()
        
        # Add some padding (5% of the range)
        if min_y != max_y:  # Avoid division by zero if all values are the same
            padding = 0.05 * (max_y - min_y)
            min_y -= padding
            max_y += padding
        else:
            # If all values are the same, add a small absolute padding
            min_y -= 0.1
            max_y += 0.1
            
        ax.set_ylim(min_y, max_y)

    if plot_legend:
        fig = ax.get_figure()
        # Use the handles and labels from plotted_labels_handles to ensure correct legend items
        # Sorting by label (variant name) for consistent legend order
        sorted_labels = sorted(plotted_labels_handles.keys())
        handles = [plotted_labels_handles[label] for label in sorted_labels]
        
        if handles: # Only add legend if there are items to show
            legend_obj = fig.legend(handles, sorted_labels, 
                                   ncol=10, 
                                   bbox_to_anchor=(0.5, -0.15), # Adjusted position slightly
                                   loc="lower center", 
                                   title="Variant")
            legend_obj.get_frame().set_linewidth(2.)
            legend_obj.get_frame().set_edgecolor("k")


def plot_variant_counts(
    ax: plt.Axes, 
    ef_data: Dict[str, Any], 
    deme: str, 
    color_map: Dict[str, str], 
    pivot_idx: Optional[List[int]] = None,
    label_fontsize: int = 14
) -> None:
    """ Plot histogram of variant sequence counts with optional pivot lines.

    Parameters:
    ---------------
        ax (matplotlib.axes.Axes): Axis to plot on
        ef_data (dict): Dictionary of `evofr` data
        deme (str): Deme to plot
        color_map (dict): Dictionary mapping variant names to colors
        pivot_idx (list, optional): List of pivot indices to draw vertical lines at. Defaults to None.
        label_fontsize (int, optional): Font size for axis labels. Defaults to 14.

    Returns:
    ---------------
        None
    """
    # Get variant names
    v_names = sorted(ef_data[deme].var_names)
    n_variant = len(v_names)
    
    # Get sequence counts matrix
    seq_counts = ef_data[deme].seq_counts
    t = np.arange(0, seq_counts.shape[0])
    
    # Create a stacked bar plot
    bottom = np.zeros(len(t))
    
    for variant_idx in range(n_variant):
        v_name = v_names[variant_idx]
        # Plot sequence counts for this variant on top of previous variants
        ax.bar(t, seq_counts[:, variant_idx], bottom=bottom, width=5.0, 
               color=color_map.get(v_name, 'gray'), # Use gray if variant not in color_map
               label=v_name, alpha=0.8, edgecolor='black', linewidth=0.5)
        # Update the bottom for the next variant
        bottom += seq_counts[:, variant_idx]
    
    if pivot_idx:
        for idx in pivot_idx:
            ax.axvline(x=idx, color='black', linestyle='--', lw=2.0)
            
    ax.set_ylabel("Sampled Sequences", fontsize=label_fontsize)
    ax.set_xlabel("")


def plot_observed_dynamics(
    seqs_df: pd.DataFrame,
    cases_df: pd.DataFrame, 
    deme: str, 
    color_map: Dict[str, str], 
    figsize: Tuple[int, int] = (30, 15),
    title_fontsize: int = 30,
    label_fontsize: int = 14,
    tick_labelsize: int = 12,
    legend_fontsize: int = 12,
    legend_title_fontsize: int = 14,
    title: Optional[str] = None
) -> None:
    """ Plot observed dynamics for entire simulation using time-stamped truth data.
    
    This creates a three-panel plot including:
    1. Case counts over time
    2. Variant sequence counts (stacked histogram) 
    3. Variant frequencies over time

    Parameters:
    ---------------
        seqs_df (pd.DataFrame): Sequence counts dataframe with columns: date, country, variant, sequences
        cases_df (pd.DataFrame): Case counts dataframe with columns: date, country, cases  
        deme (str): Location/deme name to plot
        color_map (dict): Dictionary mapping variant names to colors
        figsize (tuple): Size of the figure. Defaults to (30, 15).
        title_fontsize (int): Font size for the figure title. Defaults to 30.
        label_fontsize (int): Font size for axis labels. Defaults to 14.
        tick_labelsize (int): Font size for tick labels. Defaults to 12.
        legend_fontsize (int): Font size for legend text. Defaults to 12.
        legend_title_fontsize (int): Font size for legend title. Defaults to 14.
        title (str, optional): Custom title for the plot. If None, uses default format.

    Returns:
    ---------------
        None
    """
    # Setup figure and grid with shared x-axis
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)
    
    # Add a title
    if title:
        fig.suptitle(title, fontsize=title_fontsize)
    else:
        fig.suptitle(f"Observed dynamics for {deme} deme", fontsize=title_fontsize)
    
    # Filter data for the specified deme
    deme_seqs = seqs_df[seqs_df['country'] == deme].copy()
    deme_cases = cases_df[cases_df['country'] == deme].copy()
    
    # Create evofr data object for this deme
    ef_data_deme = ef.CaseFrequencyData(raw_cases=deme_cases, raw_seq=deme_seqs)
    
    # Subset color map to only include variants present in the deme
    available_variants = ef_data_deme.var_names
    filtered_color_map = {k: v for k, v in color_map.items() if k in available_variants}
    
    # Sort color map by variant name
    filtered_color_map = {k: filtered_color_map[k] for k in sorted(filtered_color_map.keys())}

    # Create a temporary ef_data dict for compatibility with existing helper functions
    ef_data = {deme: ef_data_deme}

    # Plot cases
    plot_observed_cases(axs[0], ef_data, deme, label_fontsize=label_fontsize)

    # Plot variant counts histogram
    plot_variant_counts(axs[1], ef_data, deme, filtered_color_map, label_fontsize=label_fontsize)

    # Plot observed frequencies
    plot_observed_freqs(axs[2], ef_data, deme, filtered_color_map, label_fontsize=label_fontsize)
    
    # Set tick label font sizes for all axes
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

    # Add x-axis label
    axs[-1].set_xlabel("Date", fontsize=label_fontsize)

    # Create a legend
    patches = [matplotlib.patches.Patch(color=c, label=l) for l, c in filtered_color_map.items()]
    legend = fig.legend(patches, list(filtered_color_map.keys()), ncol=10, bbox_to_anchor=(0.5, -0.10), 
                       loc="lower center", title="Variant", fontsize=legend_fontsize, title_fontsize=legend_title_fontsize)  
    legend.get_frame().set_linewidth(2.)
    legend.get_frame().set_edgecolor("k")
    fig.tight_layout()
    
    plt.show()
    

def plot_dynamics(
    ef_data: Dict[str, Any], 
    freq: pd.DataFrame, 
    ga: pd.DataFrame, 
    fitness: pd.DataFrame, 
    model: str, 
    deme: str, 
    color_map: Dict[str, str], 
    p: int = 50, 
    sep: int = 1825, 
    pivot_idx: Optional[List[int]] = None, 
    save_path: Optional[str] = None, 
    pivot_date: Optional[str] = None
) -> None:
    """ Plot the dynamics of a single deme.

    Parameters:
    ---------------
        ef_data (dict): Dictionary of evofr data
        freq (pd.DataFrame): Observed and predicted frequencies dataframe
        ga (pd.DataFrame): Inferred growth advantages dataframe
        fitness (pd.DataFrame): Fitness dataframe
        model (str): Model name
        deme (str): Deme name
        color_map (dict): Dictionary of colors to use for variants
        p (int): Percentile to plot
        sep (int): Number of days between date separators
        pivot_idx (list): Index of pivot dates for vertical line drawing
        save_path (str): Path to save figure
        pivot_date (str): Pivot date string for the title

    Returns:
    ---------------
        None
    """
    # Setup figure and grid with shared x-axis
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(30, 15), sharex=True)
    
    # Add a title
    if pivot_date:
        fig.suptitle(f"{model} inference for {deme} deme on {pivot_date}", fontsize=30)
    else:
        fig.suptitle(f"{model} inference for {deme} deme", fontsize=30)
    
    # Subset color map to only include variants present in the deme
    available_variants = ef_data[deme].var_names
    color_map = {k: v for k, v in color_map.items() if k in available_variants}

    # Sort color map by variant name
    color_map = {k: color_map[k] for k in sorted(color_map.keys())}
    # Timestamps for dates
    dates = ef.data.expand_dates(ef_data[deme].dates, T_forecast=0)

    # Plot case counts
    plot_observed_cases(axs[0], ef_data, deme)
    
    # Plot frequencies
    plot_frequencies(axs[1], ef_data, freq, deme, model, color_map, p=p, pivot_idx=pivot_idx)

    # Plot growth advantages
    ga = ga.query(f"variant in {available_variants}")
    plot_rt(axs[2], ga, deme, model, color_map, p=p, pivot_idx=pivot_idx)

    # Plot fitness
    # Subset fitness dataframe to only include variants present in the available_variants
    fitness = fitness.query(f"variant in {available_variants}")
    plot_fitness(axs[3], fitness, deme, color_map, pivot_idx=pivot_idx)
    axs[3].set_xticks(np.arange(0, len(dates), sep))
    axs[3].set_xticklabels(axs[3].get_xticks(), rotation=45)

    # Adjust axis visibility
    for ax in axs[:-1]:  # Hide x-axis labels on all but the last subplot
        ax.label_outer()

    # Add x-axis label
    axs[-1].set_xlabel("Date")

    # Create a legend
    patches = [Patch(color=c, label=l) for l, c in color_map.items()]
    legend = fig.legend(patches, list(color_map.keys()), ncol=10, bbox_to_anchor=(0.5, -0.15), 
                       loc="lower center", title="Variant")  
    legend.get_frame().set_linewidth(2.)
    legend.get_frame().set_edgecolor("k")
    fig.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def get_analysis_window(analysis_date: str, build: str, fitness_df: pd.DataFrame, data_path: str = "../data/", results_path: str = "../results/") -> tuple:
    """ Create a subset of data for a given analysis window.

    Parameters
    ---------------
        analysis_date : str
            Analysis date of interest
        build : str
            Build name where data resides
        fitness_df : pd.DataFrame
            Fitness dataframe with assigned datetimes
        data_path : str
            Path to sequence and case count data
        results_path : str
            Path to rt estimates

    Returns
    ---------------
        tuple: (evo_dict, freqs_df, rts_df, fitness_df)
    """
    data_path = f"{data_path}{build}/time-stamped/{analysis_date}/"
    
    # Read in variant and case count data
    seqs_df = pd.read_csv(f"{data_path}/seq_counts.tsv", sep="\t")
    cases_df = pd.read_csv(f"{data_path}/case_counts.tsv", sep="\t")

    # Create evofr data object
    evo_dict = {}
    for deme in seqs_df['country'].unique():
        deme_seqs = seqs_df[seqs_df['country'] == deme]
        deme_cases = cases_df[cases_df['country'] == deme]
        evo_dict[deme] = ef.CaseFrequencyData(raw_cases=deme_cases, raw_seq=deme_seqs, pivot='other')
    
    # Read in frequency and Rt inference data for all models and locations
    freq_paths = glob.glob(f"{results_path}{build}/estimates/*/freq_*{analysis_date}.tsv")
    rt_paths = glob.glob(f"{results_path}{build}/estimates/*/rt_*{analysis_date}.tsv")
    freqs_df = read_estimates(freq_paths)
    rts_df = read_estimates(rt_paths)
    rts_df = rts_df.groupby(['date', 'pivot_date', 'location', 'variant', 'model']).mean().reset_index()

    # Now subset fitness data to only include data within the analysis window
    min_date = min(freqs_df['date'])
    max_date = max(freqs_df['date'])
    fitness_df = fitness_df.query(f"date >= '{min_date}' and date <= '{max_date}'")

    # Add time index to dataframes
    freq_dates = set(pd.to_datetime(freqs_df['date']).tolist())
    try:
        missing_dates = freq_dates.difference(evo_dict['north'].dates)
        all_dates = evo_dict['north'].dates + list(missing_dates)
    except KeyError:
        missing_dates = freq_dates.difference(evo_dict['south'].dates)
        all_dates = evo_dict['south'].dates + list(missing_dates)
    all_dates.sort()

    # Perform date mapping
    date_mapping = {date: t for t, date in enumerate(sorted(freq_dates))}
    freqs_df['t'] = freqs_df['date'].map(date_mapping)
    rts_df['t'] = rts_df['date'].map(date_mapping)
    fitness_df.drop(columns=['t'], inplace=True)
    fitness_df['t'] = fitness_df['date'].map(date_mapping)

    return (evo_dict, freqs_df, rts_df, fitness_df)


def print_window_stats(evo_dict: dict, deme:str) -> None:
    """ Print statistics for a given analysis window.

    Parameters
    ---------------
        evo_dict : dict
            Dictionary of evofr data
        deme : str
            Demographic unit to analyze

    Returns
    ---------------
        None
    """
    var_counts = np.nansum(evo_dict[deme].seq_counts, axis=0)
    case_counts = np.nansum(evo_dict[deme].cases)
    print(f"Total cases: {case_counts}")
    print(f"Total sequences: {var_counts.sum()}")
    print(f"Variant counts: {var_counts}")


def plot_analysis_window(pivot_date:str, location: str, model: str, build: str, 
                       pruned_variant_fitness_df: pd.DataFrame, color_map: dict) -> None:
    """ Plot analysis window for a given location and model.

    Parameters
    ---------------
        pivot_date : str
            Pivot date for analysis window
        location : str
            Location of interest
        model : str
            Model name
        build : str
            Build name
        pruned_variant_fitness_df : pd.DataFrame
            Pruned fitness dataframe
        color_map : dict
            Dictionary mapping variant names to colors

    Returns
    ---------------
        None
    """
    evo_dict, small_freqs_df, small_rt_df, small_fitness_df = get_analysis_window(pivot_date, build, pruned_variant_fitness_df)
    # Print window stats
    print_window_stats(evo_dict, location)

    # Get t value for pivot date
    pivot_idx = small_freqs_df.query(f"date == '{pivot_date}'")['t'].values[0]
    plot_dynamics(evo_dict, small_freqs_df, small_rt_df, small_fitness_df, model, location, color_map, p=50, sep=30, pivot_idx=[pivot_idx], pivot_date=pivot_date)


def plot_observed_dynamics_window(pivot_date: str, location: str, build: str, 
                       pruned_variant_fitness_df: pd.DataFrame, color_map: dict, fig_size: tuple = (30,15)) -> None:
    """ Plot observed dynamics for a given location and analysis window.
    
    This function serves as a wrapper for plot_observed_dynamics(), creating the necessary
    data structures internally. It reads the evofr data for the specified pivot date and
    filters the fitness dataframe to the appropriate time window.

    Parameters:
    ---------------
        pivot_date : str
            Pivot date for analysis window (YYYY-MM-DD format)
        location : str
            Location of interest
        build : str
            Build name where data resides
        pruned_variant_fitness_df : pd.DataFrame
            Pruned fitness dataframe with assigned datetimes
        color_map : dict
            Dictionary mapping variant names to colors
        fig_size : tuple
            Size of the figure (default is (30, 15))

    Returns:
    ---------------
        None
    """
    # Use the existing get_analysis_window function to create data structures
    evo_dict, _, _, small_fitness_df = get_analysis_window(pivot_date, build, pruned_variant_fitness_df)
    
    # Print window stats for information
    print_window_stats(evo_dict, location)
    
    # Call the main plot_observed_dynamics function with the prepared data
    plot_observed_dynamics(evo_dict, small_fitness_df, location, color_map, figsize=fig_size, pivot_date=pivot_date)


def plot_analysis_window_with_variant_counts(pivot_date:str, location: str, model: str, build: str, 
                       color_map: dict, pruned_variant_fitness_df: pd.DataFrame = None, plot_fitness: bool = False,
                       title_fontsize: int = 18, label_fontsize: int = 14, tick_labelsize: int = 12,
                       legend_fontsize: int = 12, legend_title_fontsize: int = 14) -> None:
    """ Plot analysis window for a given location and model with variant counts histogram.
    
    This creates a four or five-panel plot including:
    1. Case counts
    2. Variant sequence counts (stacked histogram)
    3. Variant frequencies
    4. Growth advantages
    5. Fitness (optional, controlled by plot_fitness parameter)

    Parameters:
    ---------------
        pivot_date : str
            Pivot date for analysis window
        location : str
            Location of interest
        model : str
            Model name
        build : str
            Build name
        color_map : dict
            Dictionary mapping variant names to colors
        pruned_variant_fitness_df : pd.DataFrame, optional
            Pruned fitness dataframe. Required only if plot_fitness=True. Defaults to None.
        plot_fitness : bool, optional
            Whether to include the fitness panel. Defaults to False.
        title_fontsize : int, optional
            Font size for the figure title. Defaults to 18.
        label_fontsize : int, optional
            Font size for axis labels. Defaults to 14.
        tick_labelsize : int, optional
            Font size for tick labels. Defaults to 12.
        legend_fontsize : int, optional
            Font size for legend text. Defaults to 12.
        legend_title_fontsize : int, optional
            Font size for legend title. Defaults to 14.

    Returns
    ---------------
        None
    """
    # Check if fitness plotting is requested but fitness_df is not provided
    if plot_fitness and pruned_variant_fitness_df is None:
        raise ValueError("pruned_variant_fitness_df is required when plot_fitness=True")
    
    # Get analysis window data
    if pruned_variant_fitness_df is not None:
        evo_dict, small_freqs_df, small_rt_df, small_fitness_df = get_analysis_window(pivot_date, build, pruned_variant_fitness_df)
    else:
        # Create a dummy fitness_df just for get_analysis_window
        # We won't use it since plot_fitness=False
        dummy_fitness_df = pd.DataFrame(columns=['date', 'location', 'variant', 'fitness', 'seasonal_fitness', 't'])
        evo_dict, small_freqs_df, small_rt_df, small_fitness_df = get_analysis_window(pivot_date, build, dummy_fitness_df)
    # Print window stats
    print_window_stats(evo_dict, location)

    # Get t value for pivot date
    pivot_idx = small_freqs_df.query(f"date == '{pivot_date}'")['t'].values[0]
    
    # Setup figure and grid with shared x-axis
    num_panels = 5 if plot_fitness else 4
    fig_height = 18 if plot_fitness else 15
    fig, axs = plt.subplots(nrows=num_panels, ncols=1, figsize=(30, fig_height), sharex=True)
    
    # Add a title
    fig.suptitle(f"{model} inference for {location} deme on {pivot_date}", fontsize=title_fontsize)
    
    # Subset color map to only include variants present in the deme
    available_variants = evo_dict[location].var_names
    filtered_color_map = {k: v for k, v in color_map.items() if k in available_variants}

    # Sort color map by variant name
    filtered_color_map = {k: filtered_color_map[k] for k in sorted(filtered_color_map.keys())}
    
    # Timestamps for dates
    dates = ef.data.expand_dates(evo_dict[location].dates, T_forecast=0)

    # Plot case counts
    plot_observed_cases(axs[0], evo_dict, location, pivot_idx=[pivot_idx], label_fontsize=label_fontsize)
    
    # Plot variant counts histogram
    plot_variant_counts(axs[1], evo_dict, location, filtered_color_map, pivot_idx=[pivot_idx], label_fontsize=label_fontsize)
    
    # Plot frequencies
    plot_frequencies(axs[2], evo_dict, small_freqs_df, location, model, filtered_color_map, p=50, pivot_idx=[pivot_idx], label_fontsize=label_fontsize)

    # Plot growth advantages
    ga = small_rt_df.query(f"variant in {list(available_variants)}")
    plot_rt(axs[3], ga, location, model, filtered_color_map, p=50, pivot_idx=[pivot_idx], label_fontsize=label_fontsize)

    # Plot fitness (only if requested)
    if plot_fitness:
        # Subset fitness dataframe to only include variants present in the available_variants
        fitness = small_fitness_df.query(f"variant in {list(available_variants)}")
        plot_fitness(axs[4], fitness, location, filtered_color_map, pivot_idx=[pivot_idx], label_fontsize=label_fontsize)
        axs[4].set_xticks(np.arange(0, len(dates), 30))
        axs[4].set_xticklabels(axs[4].get_xticks(), rotation=45)

    # Adjust axis visibility and font sizes
    for ax in axs[:-1]:  # Hide x-axis labels on all but the last subplot
        ax.label_outer()
    
    # Set tick label font sizes for all axes
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

    # Add x-axis label
    axs[-1].set_xlabel("Date", fontsize=label_fontsize)

    # Create a legend with updated styling to match plot_growth_rate_dynamics
    patches = [Patch(color=c, label=l) for l, c in filtered_color_map.items()]
    legend = fig.legend(patches, list(filtered_color_map.keys()), ncol=10, bbox_to_anchor=(0.5, -0.10), 
                       loc="lower center", title="Variant", fontsize=legend_fontsize, title_fontsize=legend_title_fontsize)  
    legend.get_frame().set_linewidth(2.0)
    legend.get_frame().set_edgecolor("k")
    fig.tight_layout()
    
    plt.show()


def plot_smoothed_variant_counts(
    ax: plt.Axes,
    seqs_df: pd.DataFrame,
    location: str,
    color_map: Dict[str, str],
    analysis_date: Optional[str] = None,
    plot_legend: bool = False,
    max_variants: Optional[int] = None
) -> None:
    """Plot original and smoothed variant sequence counts for a specific location.

    This function plots both the original sequence counts ('sequences' column) and
    the smoothed sequence counts ('smoothed_sequences' column) from the provided DataFrame
    for a specific location. Each variant is plotted as separate lines, colored according
    to the 'color_map'. If 'analysis_date' is provided, a vertical dashed line will be drawn
    at the date corresponding to this value.

    Parameters:
    -----------
        ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
        seqs_df (pd.DataFrame): A DataFrame that must contain 'country', 'variant', 'date',
                                'sequences', and 'smoothed_sequences' columns.
        location (str): The location/country to plot data for.
        color_map (Dict[str, str]): A dictionary mapping variant names to hex color codes.
        analysis_date (Optional[str]): A date string (e.g., YYYY-MM-DD). If provided,
                                       a vertical dashed line will be drawn at this date.
                                       Defaults to None.
        plot_legend (bool): If True, a legend is added below the plot.
                            Defaults to False.
        max_variants (Optional[int]): Maximum number of variants to plot, prioritizing 
                                      those with the highest sequence counts.
                                      If None, all variants are plotted. Defaults to None.

    Raises:
    -------
        ValueError: If the 'seqs_df' DataFrame does not contain the required columns
                    or if no data is found for the specified location.
    """
    # Make a copy to avoid modifying the input DataFrame
    seqs_df_processed = seqs_df.copy()
    
    # Check required columns
    required_cols = ['country', 'variant', 'date', 'sequences', 'smoothed_sequences']
    for col in required_cols:
        if col not in seqs_df_processed.columns:
            raise ValueError(f"Input DataFrame `seqs_df` must contain a '{col}' column.")
    
    # Filter for the specified location
    location_data = seqs_df_processed[seqs_df_processed['country'] == location]
    if len(location_data) == 0:
        raise ValueError(f"No data found for location '{location}'")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(location_data['date']):
        location_data['date'] = pd.to_datetime(location_data['date'])
    
    # Add 't' column for consistent time handling across plotting functions
    if 't' not in location_data.columns:
        # Normalize dates before creating mapping to ensure consistency
        location_data['date_normalized_for_t_mapping'] = location_data['date'].dt.normalize()
        unique_dates = sorted(location_data['date_normalized_for_t_mapping'].unique())
        date_to_t_mapping = {date_val: i for i, date_val in enumerate(unique_dates)}
        location_data['t'] = location_data['date_normalized_for_t_mapping'].map(date_to_t_mapping)
        location_data.drop(columns=['date_normalized_for_t_mapping'], inplace=True)
    
    # Get variants for this location, optionally limit by count
    if max_variants is not None:
        # Get the variants with the highest total sequence counts
        top_variants = location_data.groupby('variant')['sequences'].sum().sort_values(
            ascending=False).head(max_variants).index
        location_data = location_data[location_data['variant'].isin(top_variants)]
    
    # Get unique variants
    variants = location_data['variant'].unique()
    
    # Create an integer-keyed version of the color map for integer variant IDs
    int_color_map = {}
    for k, v in color_map.items():
        try:
            int_key = int(k)
            int_color_map[int_key] = v
        except (ValueError, TypeError):
            # Skip keys that aren't convertible to integers
            pass
    
    # Create a consolidated color map
    combined_color_map = {**color_map, **int_color_map}
    
    # Dictionary to track which variants were successfully plotted with a color
    plotted_variants = {}
    
    # Plot each variant
    for variant in variants:
        # Filter data for this variant
        variant_data = location_data[location_data['variant'] == variant].sort_values('t')
        
        # Get variant as string for label
        variant_str = str(variant)
        
        # Try to find color for this variant
        variant_color = None
        if variant_str in combined_color_map:
            variant_color = combined_color_map[variant_str]
        elif variant in combined_color_map:
            variant_color = combined_color_map[variant]
        
        if variant_color is None:
            # No color found for this variant, use a default color
            # This uses the matplotlib default color cycle
            variant_color = None
        
        # Plot original sequence counts (circles)
        ax.plot(
            variant_data['t'], 
            variant_data['sequences'], 
            'o-', 
            color=variant_color,
            alpha=0.6,
            linewidth=2.0,
            markersize=5
        )
        
        # Plot smoothed sequence counts (solid line)
        ax.plot(
            variant_data['t'], 
            variant_data['smoothed_sequences'], 
            '-', 
            color=variant_color,
            linewidth=3.0
        )
        
        # Track plotted variants with their colors for legend
        if variant_color is not None:
            plotted_variants[variant_str] = variant_color
    
    # Plot vertical line for analysis date if provided
    if analysis_date is not None:
        try:
            analysis_dt = pd.to_datetime(analysis_date).normalize()
            # Look for the corresponding t value in the location_data
            matching_rows = location_data[location_data['date'].dt.normalize() == analysis_dt]
            if not matching_rows.empty:
                analysis_t = matching_rows['t'].iloc[0]
                ax.axvline(x=analysis_t, color='black', linestyle='--', lw=2.0)
            else:
                print(f"Warning: analysis_date '{analysis_date}' not found in data. No vertical line drawn.")
        except Exception as e:
            print(f"Warning: Could not process analysis_date '{analysis_date}' for vertical line: {e}")
    
    # Set labels and formatting
    ax.set_ylabel("Sequence Count")
    ax.set_xlabel("")  # Usually the x-label will be set by the multi-panel plot
    #ax.grid(alpha=0.3)
    
    # Add legend if requested
    if plot_legend and plotted_variants:
        fig = ax.get_figure()
        # Create patches for legend
        patches = [Patch(color=color, label=variant) for variant, color in plotted_variants.items()]
        
        # Sort by variant label for consistent ordering
        sorted_patches = sorted(patches, key=lambda patch: patch.get_label())
        
        legend_obj = fig.legend(
            sorted_patches, 
            [patch.get_label() for patch in sorted_patches],
            ncol=min(10, len(patches)),
            bbox_to_anchor=(0.5, -0.15),
            loc="lower center",
            title="Variant"
        )
        legend_obj.get_frame().set_linewidth(2.0)
        legend_obj.get_frame().set_edgecolor("k")


def plot_r_data(
    ax: plt.Axes,
    growth_rates_df: pd.DataFrame,
    location: str,
    color_map: Dict[str, str],
    analysis_date: Optional[str] = None,
    plot_legend: bool = False,
    max_variants: Optional[int] = None
) -> None:
    """Plot empirically-derived growth rates (r_data) for different variants.

    This function plots the 'growth_rate_r_data' column from the provided DataFrame
    against the 't' column (which is derived from 'date') for a specific location. 
    Each variant is plotted as a separate line, colored according to the 'color_map'. 
    If 'analysis_date' is provided, a vertical dashed line will be drawn at this date.

    Parameters:
    -----------
        ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
        growth_rates_df (pd.DataFrame): A DataFrame that must contain 'country', 'variant', 
                                       'date', and 'growth_rate_r_data' columns.
        location (str): The location/country to plot data for.
        color_map (Dict[str, str]): A dictionary mapping variant names to hex color codes.
        analysis_date (Optional[str]): A date string (e.g., YYYY-MM-DD). If provided,
                                      a vertical dashed line will be drawn at this date.
                                      Defaults to None.
        plot_legend (bool): If True, a legend is added below the plot.
                           Defaults to False.
        max_variants (Optional[int]): Maximum number of variants to plot, prioritizing 
                                     those with the highest sequence counts.
                                     If None, all variants are plotted. Defaults to None.

    Raises:
    -------
        ValueError: If the 'growth_rates_df' DataFrame does not contain the required columns
                   or if no data is found for the specified location.
    """
    # Make a copy to avoid modifying the input DataFrame
    df_processed = growth_rates_df.copy()
    
    # Check required columns
    required_cols = ['country', 'variant', 'date', 'growth_rate_r_data']
    for col in required_cols:
        if col not in df_processed.columns:
            raise ValueError(f"Input DataFrame `growth_rates_df` must contain a '{col}' column.")
    
    # Filter for the specified location
    location_data = df_processed[df_processed['country'] == location]
    if len(location_data) == 0:
        raise ValueError(f"No data found for location '{location}'")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(location_data['date']):
        location_data['date'] = pd.to_datetime(location_data['date'])
    
    # Add 't' column for consistent time handling across plotting functions
    if 't' not in location_data.columns:
        # Normalize dates before creating mapping to ensure consistency
        location_data['date_normalized_for_t_mapping'] = location_data['date'].dt.normalize()
        unique_dates = sorted(location_data['date_normalized_for_t_mapping'].unique())
        date_to_t_mapping = {date_val: i for i, date_val in enumerate(unique_dates)}
        location_data['t'] = location_data['date_normalized_for_t_mapping'].map(date_to_t_mapping)
        location_data.drop(columns=['date_normalized_for_t_mapping'], inplace=True)
    
    # Get variants for this location, optionally limit by count
    if max_variants is not None and 'sequences' in location_data.columns:
        # Get the variants with the highest total sequence counts
        top_variants = location_data.groupby('variant')['sequences'].sum().sort_values(
            ascending=False).head(max_variants).index
        location_data = location_data[location_data['variant'].isin(top_variants)]
    
    # Get unique variants
    variants = location_data['variant'].unique()
    
    # Create an integer-keyed version of the color map for integer variant IDs
    int_color_map = {}
    for k, v in color_map.items():
        try:
            int_key = int(k)
            int_color_map[int_key] = v
        except (ValueError, TypeError):
            # Skip keys that aren't convertible to integers
            pass
    
    # Create a consolidated color map
    combined_color_map = {**color_map, **int_color_map}
    
    # Dictionary to track which variants were successfully plotted with a color
    plotted_variants = {}
    
    # Plot each variant
    for variant in variants:
        # Filter data for this variant
        variant_data = location_data[location_data['variant'] == variant].sort_values('t')
        
        # Skip if no valid growth rate data for this variant
        if variant_data['growth_rate_r_data'].isna().all():
            continue
            
        # Get variant as string for label
        variant_str = str(variant)
        
        # Try to find color for this variant
        variant_color = None
        if variant_str in combined_color_map:
            variant_color = combined_color_map[variant_str]
        elif variant in combined_color_map:
            variant_color = combined_color_map[variant]
        
        if variant_color is None:
            # No color found for this variant, use a default color
            variant_color = None
        
        # Plot the growth rates
        ax.plot(
            variant_data['t'], 
            variant_data['growth_rate_r_data'], 
            'o-',  # Circles with connecting lines
            color=variant_color,
            linewidth=2.5,
            markersize=5,
            alpha=0.8
        )
        
        # Track plotted variants with their colors for legend
        if variant_color is not None:
            plotted_variants[variant_str] = variant_color
    
    # Plot vertical line for analysis date if provided
    if analysis_date is not None:
        try:
            analysis_dt = pd.to_datetime(analysis_date).normalize()
            # Look for the corresponding t value in the location_data
            matching_rows = location_data[location_data['date'].dt.normalize() == analysis_dt]
            if not matching_rows.empty:
                analysis_t = matching_rows['t'].iloc[0]
                ax.axvline(x=analysis_t, color='black', linestyle='--', lw=2.0)
            else:
                print(f"Warning: analysis_date '{analysis_date}' not found in data. No vertical line drawn.")
        except Exception as e:
            print(f"Warning: Could not process analysis_date '{analysis_date}' for vertical line: {e}")
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0.0, color='red', linestyle='--', lw=1.5)
    
    # Set labels and formatting
    ax.set_ylabel("$r_{\\text{data}}$")
    ax.set_xlabel("")  # Usually the x-label will be set by the multi-panel plot
    #ax.grid(alpha=0.3)
    
    # Calculate and set appropriate y-axis limits based on data range if there's data to plot
    valid_data = location_data['growth_rate_r_data'].dropna()
    if not valid_data.empty:
        min_y = valid_data.min()
        max_y = valid_data.max()
        
        # Add some padding (5% of the range)
        if min_y != max_y:  # Avoid division by zero if all values are the same
            padding = 0.05 * (max_y - min_y)
            min_y -= padding
            max_y += padding
        else:
            # If all values are the same, add a small absolute padding
            min_y -= 0.1
            max_y += 0.1
            
        ax.set_ylim(min_y, max_y)
    
    # Add legend if requested
    if plot_legend and plotted_variants:
        fig = ax.get_figure()
        # Create patches for legend
        patches = [Patch(color=color, label=variant) for variant, color in plotted_variants.items()]
        
        # Sort by variant label for consistent ordering
        sorted_patches = sorted(patches, key=lambda patch: patch.get_label())
        
        legend_obj = fig.legend(
            sorted_patches, 
            [patch.get_label() for patch in sorted_patches],
            ncol=min(10, len(patches)),
            bbox_to_anchor=(0.5, -0.15),
            loc="lower center",
            title="Variant"
        )
        legend_obj.get_frame().set_linewidth(2.0)
        legend_obj.get_frame().set_edgecolor("k")

def plot_variant_incidence(
    growth_rates_df: pd.DataFrame,
    location: str,
    color_map: Dict[str, str],
    model: Optional[str] = None,
    analysis_date: Optional[str] = None,
    max_variants: Optional[int] = None,
    figsize: tuple = (20, 10),
    save_path: Optional[str] = None
) -> None:
    """Plot variant incidence (case counts * variant frequency) over time.
    
    This creates a two-panel plot showing:
    1. Smoothed variant incidence
    2. Total case counts for reference
    
    Parameters:
    -----------
        growth_rates_df (pd.DataFrame): A consolidated DataFrame containing:
                                       - 'country' and 'variant' columns for filtering
                                       - 'date' column for x-axis plotting
                                       - 'variant_incidence_smoothed' for smoothed incidence plot (required)
                                       - 'cases' for total case counts plot
        location (str): Location (country) to plot data for.
        color_map (Dict[str, str]): Dictionary mapping variant names to hex color codes.
        model (Optional[str]): Model name, used for title. Defaults to None.
        analysis_date (Optional[str]): Date string (e.g., YYYY-MM-DD) to draw a vertical line at.
                                      Defaults to None.
        max_variants (Optional[int]): Maximum number of variants to plot, prioritizing 
                                     those with the highest incidence.
                                     If None, all variants are plotted. Defaults to None.
        figsize (tuple): Figure size (width, height) in inches. Defaults to (20, 10).
        save_path (Optional[str]): Path to save the figure to. If None, figure is not saved.
                                  Defaults to None.
                                  
    Returns:
    --------
        None
    """
    # Make a copy to avoid modifying input DataFrame
    df_processed = growth_rates_df.copy()
    
    # Ensure required columns exist
    required_cols = ['country', 'variant', 'date', 'variant_incidence_smoothed', 'cases']
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")
    
    # Filter for the specified location
    location_data = df_processed[df_processed['country'] == location]
    if len(location_data) == 0:
        raise ValueError(f"No data found for location '{location}'")
    
    # Ensure 'date' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(location_data['date']):
        try:
            location_data['date'] = pd.to_datetime(location_data['date'])
        except Exception as e:
            raise ValueError(f"Could not convert 'date' column to datetime: {e}")
    
    # Add 't' column if it doesn't exist
    if 't' not in location_data.columns:
        location_data['date_normalized_for_t_mapping'] = location_data['date'].dt.normalize()
        unique_dates = sorted(location_data['date_normalized_for_t_mapping'].unique())
        date_to_t_mapping = {date_val: i for i, date_val in enumerate(unique_dates)}
        location_data['t'] = location_data['date_normalized_for_t_mapping'].map(date_to_t_mapping)
        location_data.drop(columns=['date_normalized_for_t_mapping'], inplace=True)
    
    # Convert analysis_date to t-value if provided
    analysis_t = None
    if analysis_date is not None:
        try:
            analysis_dt = pd.to_datetime(analysis_date).normalize()
            matching_rows = location_data[location_data['date'].dt.normalize() == analysis_dt]
            if not matching_rows.empty:
                analysis_t = matching_rows['t'].iloc[0]
        except Exception as e:
            print(f"Warning: Could not convert analysis_date '{analysis_date}' to t-value: {e}")
    
    # Optionally limit to top variants by incidence
    if max_variants is not None:
        top_variants = location_data.groupby('variant')['variant_incidence_smoothed'].sum().sort_values(
            ascending=False).head(max_variants).index
        location_data = location_data[location_data['variant'].isin(top_variants)]
    
    # Setup figure and grid with shared x-axis
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
    
    # Add a title
    title = f"Variant incidence dynamics for"
    
    # If model is provided, use it; otherwise, try to get the first unique model from the dataframe
    if model:
        title += f" {model} model"
    elif 'model' in location_data.columns and not location_data['model'].isna().all():
        first_model = location_data['model'].dropna().unique()[0] if len(location_data['model'].dropna().unique()) > 0 else "Unknown"
        title += f" {first_model} model"
    else:
        title += " model"
    
    # Add location
    title += f" in {location} deme"
    
    # Add analysis date if provided
    if analysis_date:
        title += f" for {analysis_date}"
    elif 'date' in location_data.columns:
        first_date = location_data['date'].sort_values().iloc[0]
        if pd.api.types.is_datetime64_any_dtype(first_date):
            first_date = first_date.strftime('%Y-%m-%d')
        title += f" for {first_date}"
    
    fig.suptitle(title, fontsize=20)
    
    # Create an integer-keyed version of the color map for integer variant IDs
    int_color_map = {}
    for k, v in color_map.items():
        try:
            int_key = int(k)
            int_color_map[int_key] = v
        except (ValueError, TypeError):
            pass
    
    # Create a consolidated color map that works with both string and integer keys
    combined_color_map = {**color_map, **int_color_map}
    
    # Plot each variant's incidence
    for variant in location_data['variant'].unique():
        # Filter data for this variant
        variant_data = location_data[location_data['variant'] == variant].sort_values('t')
        
        # Skip if no data
        if len(variant_data) == 0:
            continue
            
        # Get variant as string for label and color
        variant_str = str(variant)
        
        # Try to find color for this variant
        variant_color = None
        if variant_str in combined_color_map:
            variant_color = combined_color_map[variant_str]
        elif variant in combined_color_map:
            variant_color = combined_color_map[variant]
        
        # Plot smoothed variant incidence
        valid_smoothed_data = variant_data[~variant_data['variant_incidence_smoothed'].isna()]
        if len(valid_smoothed_data) > 0:
            axs[0].plot(
                valid_smoothed_data['t'], 
                valid_smoothed_data['variant_incidence_smoothed'], 
                '-', 
                color=variant_color,
                linewidth=3.0,
                label=f'Variant {variant_str}'
            )
    
    # Plot total case counts in the second panel
    case_data = location_data[['t', 'cases']].drop_duplicates().sort_values('t')
    axs[1].plot(
        case_data['t'], 
        case_data['cases'], 
        'k-', 
        linewidth=3.0,
        alpha=0.7
    )
    
    # Plot vertical line for analysis date if converted successfully
    if analysis_t is not None:
        for ax in axs:
            ax.axvline(x=analysis_t, color='black', linestyle='--', lw=2.0)
    
    # Set labels and formatting
    axs[0].set_ylabel("Variant Incidence\n(Cases × Frequency)")
    axs[1].set_ylabel("Total Cases")
    
    # Set x-ticks using date information
    if len(location_data) > 0:
        # Create mapping from t to date for x-tick labels
        t_values = sorted(location_data['t'].unique())
        t_to_date = {}
        for t in t_values:
            matching_dates = location_data[location_data['t'] == t]['date']
            if not matching_dates.empty:
                t_to_date[t] = matching_dates.iloc[0]
        
        # Select a reasonable number of ticks
        if len(t_values) > 10:
            tick_indices = np.linspace(0, len(t_values) - 1, 10, dtype=int)
            t_ticks = [t_values[i] for i in tick_indices]
        else:
            t_ticks = t_values
        
        # Format dates for tick labels
        date_labels = []
        for t in t_ticks:
            if t in t_to_date:
                date_str = t_to_date[t].strftime('%Y-%m-%d')
                date_labels.append(date_str)
            else:
                date_labels.append('')
        
        # Set ticks on x-axis of bottom subplot
        axs[1].set_xticks(t_ticks)
        axs[1].set_xticklabels(date_labels, rotation=45)
    
    # Add x-axis label to bottom subplot
    axs[1].set_xlabel("Date")
    
    # Get unique variants that are present in the data and color map
    all_variants = set()
    
    # Add variants from the consolidated dataframe
    location_data_filtered = growth_rates_df[growth_rates_df['country'] == location]
    if max_variants is not None:
        top_variants = location_data_filtered.groupby('variant')['variant_incidence_smoothed'].sum().sort_values(
            ascending=False).head(max_variants).index
        all_variants.update(top_variants)
    else:
        all_variants.update(location_data_filtered['variant'].unique())
    
    # Filter color map to only include variants present in the data
    filtered_color_map = {}
    for variant in all_variants:
        variant_str = str(variant)
        if variant_str in color_map:
            filtered_color_map[variant_str] = color_map[variant_str]
        elif variant in color_map:
            filtered_color_map[variant_str] = color_map[variant]
    
    # Create a legend if we have variants with colors
    if filtered_color_map:
        # Create patches for legend
        patches = [Patch(color=color, label=variant) for variant, color in filtered_color_map.items()]
        
        # Sort by variant label for consistent ordering
        sorted_patches = sorted(patches, key=lambda patch: patch.get_label())
        
        # Add legend at the bottom of the figure
        legend_obj = fig.legend(
            sorted_patches, 
            [patch.get_label() for patch in sorted_patches],
            ncol=min(10, len(sorted_patches)),
            bbox_to_anchor=(0.5, -0.05),
            loc="lower center",
            title="Variant"
        )
        legend_obj.get_frame().set_linewidth(2.0)
        legend_obj.get_frame().set_edgecolor("k")
    
    # Tight layout for better spacing
    fig.tight_layout()
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show the figure
    plt.show()


def plot_growth_rate_dynamics(
    growth_rates_df: pd.DataFrame,
    location: str,
    color_map: Dict[str, str],
    r_data_col: str = 'growth_rate_r_data',
    r_model_col: str = 'growth_rate_r',
    model: Optional[str] = None,
    analysis_date: Optional[str] = None,
    max_variants: Optional[int] = None,
    figsize: tuple = (20, 15),
    save_path: Optional[str] = None,
    connect_gaps: bool = False,
    min_segment_length: Optional[int] = None,
    plot_incidence: bool = True
) -> None:
    """Plot a four-panel figure showing sequence counts, variant frequencies/incidence, and growth rates for variants in a location.
    
    This creates a four-panel plot including:
    1. Original and smoothed sequence counts
    2. Variant incidence (smoothed) OR variant frequencies (original and smoothed) - controlled by plot_incidence parameter
    3. Empirical growth rates (r_data)
    4. Model-derived growth rates (r_model)
    
    All plots use a consistent time coordinate system ('t') derived from dates for 
    consistency across panels. The x-axis displays formatted dates, but internally
    the function uses integer time indices to ensure alignment between different 
    data sources.
    
    Parameters:
    -----------
        growth_rates_df (pd.DataFrame): A consolidated DataFrame containing:
                                       - 'country' and 'variant' columns for filtering
                                       - 'date' column for x-axis plotting (automatically 
                                          converted to 't' internally)
                                       - 'sequences' and 'smoothed_sequences' for variant counts plot
                                       - 'variant_incidence_smoothed' for incidence plot (required if plot_incidence=True)
                                       - 'variant_frequency' and 'variant_frequency_smoothed' for frequency plot (required if plot_incidence=False)
                                       - 'growth_rate_r_data' for empirical growth rates plot
                                       - 'growth_rate_r' for model growth rates plot
        location (str): Location (country) to plot data for.
        color_map (Dict[str, str]): Dictionary mapping variant names to hex color codes.
        model (Optional[str]): Model name, used for title. Defaults to None.
        analysis_date (Optional[str]): Date string (e.g., YYYY-MM-DD) to draw a vertical line at.
                                      This date is automatically converted to its corresponding
                                      't' value. Defaults to None.
        max_variants (Optional[int]): Maximum number of variants to plot, prioritizing 
                                     those with the highest sequence counts.
                                     If None, all variants are plotted. Defaults to None.
        figsize (tuple): Figure size (width, height) in inches. Defaults to (20, 15).
        save_path (Optional[str]): Path to save the figure to. If None, figure is not saved.
                                  Defaults to None.
        connect_gaps (bool): If True, connects all non-NaN points in r_data plots, 
                            ignoring gaps. If False, gaps are shown where data is missing.
                            Defaults to False.
        min_segment_length (Optional[int]): Minimum number of consecutive non-NaN points 
                                           required to plot a segment. Segments shorter than 
                                           this are discarded. If None, all segments are plotted.
                                           Only applies when connect_gaps=False. Defaults to None.
        plot_incidence (bool): If True, plots variant incidence (smoothed) in panel 2.
                              If False, plots variant frequencies (original and smoothed) in panel 2.
                              Defaults to True.
                                  
    Returns:
    --------
        None
    """
    # Make a copy to avoid modifying input DataFrame
    df_processed = growth_rates_df.copy()
    
    # Ensure required columns exist
    base_required_cols = ['country', 'variant', 'date', 'sequences', 'smoothed_sequences', r_data_col, r_model_col]
    if plot_incidence:
        required_cols = base_required_cols + ['variant_incidence_smoothed']
    else:
        required_cols = base_required_cols + ['variant_frequency', 'variant_frequency_smoothed']
    
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")
    
    # Filter for the specified location
    location_data = df_processed[df_processed['country'] == location]
    if len(location_data) == 0:
        raise ValueError(f"No data found for location '{location}'")
    
    # Ensure 'date' column is datetime
    if not pd.api.types.is_datetime64_any_dtype(location_data['date']):
        try:
            location_data['date'] = pd.to_datetime(location_data['date'])
        except Exception as e:
            raise ValueError(f"Could not convert 'date' column to datetime: {e}")
    
    # Add 't' column if it doesn't exist (using the same approach as in plot_r_model)
    if 't' not in location_data.columns:
        # Normalize dates before creating mapping to ensure consistency
        location_data['date_normalized_for_t_mapping'] = location_data['date'].dt.normalize()
        unique_dates = sorted(location_data['date_normalized_for_t_mapping'].unique())
        date_to_t_mapping = {date_val: i for i, date_val in enumerate(unique_dates)}
        location_data['t'] = location_data['date_normalized_for_t_mapping'].map(date_to_t_mapping)
        location_data.drop(columns=['date_normalized_for_t_mapping'], inplace=True)
    
    # Convert analysis_date to t-value if provided
    analysis_t = None
    if analysis_date is not None:
        try:
            analysis_dt = pd.to_datetime(analysis_date).normalize()
            matching_rows = location_data[location_data['date'].dt.normalize() == analysis_dt]
            if not matching_rows.empty:
                analysis_t = matching_rows['t'].iloc[0]
        except Exception as e:
            print(f"Warning: Could not convert analysis_date '{analysis_date}' to t-value: {e}")
    
    # Optionally limit to top variants by count
    if max_variants is not None:
        top_variants = location_data.groupby('variant')['sequences'].sum().sort_values(
            ascending=False).head(max_variants).index
        location_data = location_data[location_data['variant'].isin(top_variants)]
    
    # Setup figure and grid with shared x-axis
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=figsize, sharex=True)
    
    # Add a title with required format
    title = f"Variant growth rate dynamics for"
    
    # If model is provided, use it; otherwise, try to get the first unique model from the dataframe
    if model:
        title += f" {model} model"
    elif 'model' in location_data.columns and not location_data['model'].isna().all():
        first_model = location_data['model'].dropna().unique()[0] if len(location_data['model'].dropna().unique()) > 0 else "Unknown"
        title += f" {first_model} model"
    else:
        title += " model"
    
    # Add location
    title += f" in {location} deme"
    
    # Add analysis date if provided, otherwise try to get the first unique date from the dataframe
    if analysis_date:
        title += f" for {analysis_date}"
    elif 'date' in location_data.columns:
        first_date = location_data['date'].sort_values().iloc[0]
        if pd.api.types.is_datetime64_any_dtype(first_date):
            first_date = first_date.strftime('%Y-%m-%d')
        title += f" for {first_date}"
    
    fig.suptitle(title, fontsize=18)
    
    # Create an integer-keyed version of the color map for integer variant IDs
    int_color_map = {}
    for k, v in color_map.items():
        try:
            int_key = int(k)
            int_color_map[int_key] = v
        except (ValueError, TypeError):
            # Skip keys that aren't convertible to integers
            pass
    
    # Create a consolidated color map that works with both string and integer keys
    combined_color_map = {**color_map, **int_color_map}
    
    # Dictionary to track which variants were successfully plotted with a color
    plotted_variants = {}
    
    # Lists to track actually plotted values for y-axis range calculation
    plotted_r_data_values = []
    plotted_r_model_values = []
    
    # Plot each variant's sequence counts
    for variant in location_data['variant'].unique():
        # Filter data for this variant
        variant_data = location_data[location_data['variant'] == variant].sort_values('t')
        
        # Skip if no data
        if len(variant_data) == 0:
            continue
            
        # Get variant as string for label and color
        variant_str = str(variant)
        
        # Try to find color for this variant
        variant_color = None
        if variant_str in combined_color_map:
            variant_color = combined_color_map[variant_str]
        elif variant in combined_color_map:
            variant_color = combined_color_map[variant]
        
        # Plot original sequence counts (circles)
        axs[0].plot(
            variant_data['t'], 
            variant_data['sequences'], 
            'o-', 
            color=variant_color,
            alpha=0.6,
            linewidth=2.0,
            markersize=5
        )
        
        # Plot smoothed sequence counts (solid line)
        axs[0].plot(
            variant_data['t'], 
            variant_data['smoothed_sequences'], 
            '-', 
            color=variant_color,
            linewidth=3.0
        )
        
        # Plot variant incidence or frequencies based on flag
        if plot_incidence:
            # Plot variant incidence (smoothed) without filtering
            valid_incidence_data = variant_data[~variant_data['variant_incidence_smoothed'].isna()]
            if len(valid_incidence_data) > 0:
                axs[1].plot(
                    valid_incidence_data['t'], 
                    valid_incidence_data['variant_incidence_smoothed'], 
                    '-', 
                    color=variant_color,
                    linewidth=3.0
                )
        else:
            # Plot variant frequencies - always plot observed frequencies
            valid_freq_data = variant_data[~variant_data['variant_frequency'].isna()]
            if len(valid_freq_data) > 0:
                axs[1].plot(
                    valid_freq_data['t'], 
                    valid_freq_data['variant_frequency'], 
                    'o', 
                    color=variant_color,
                    alpha=0.6,
                    markersize=5
                )
            
            # Plot smoothed frequencies if available
            if 'variant_frequency_smoothed' in variant_data.columns:
                valid_smoothed_freq_data = variant_data[~variant_data['variant_frequency_smoothed'].isna()]
                if len(valid_smoothed_freq_data) > 0:
                    axs[1].plot(
                        valid_smoothed_freq_data['t'], 
                        valid_smoothed_freq_data['variant_frequency_smoothed'], 
                        '-', 
                        color=variant_color,
                        linewidth=3.0
                    )
        
        # Plot r_data - filter using the same logic as evaluate_growth_rate_performance
        valid_r_data = variant_data[~variant_data[r_data_col].isna()]
        if len(valid_r_data) > 0:
            # Apply segment filtering if specified
            if min_segment_length is not None:
                # Create a temporary DataFrame for this variant to apply filtering
                temp_df = variant_data.copy()
                temp_df['variant'] = variant  # Ensure variant column exists for filtering
                filtered_variant_data = filter_growth_rates(
                    temp_df, 
                    r_data_col=r_data_col, 
                    connect_gaps=connect_gaps, 
                    min_segment_length=min_segment_length
                )
                valid_r_data = filtered_variant_data[~filtered_variant_data[r_data_col].isna()]
            
            if len(valid_r_data) > 0:
                axs[2].plot(
                    valid_r_data['t'], 
                    valid_r_data[r_data_col], 
                    'o-',  # Circles with connecting lines
                    color=variant_color,
                    linewidth=2.5,
                    markersize=5,
                    alpha=0.8
                )
                # Track the plotted values
                plotted_r_data_values.extend(valid_r_data[r_data_col].dropna().tolist())
        
        # Plot r_model using the same filtered data for consistency
        if len(valid_r_data) > 0 and not valid_r_data[r_model_col].isna().all():
            # Filter out points where r_model is also NaN
            valid_model_data = valid_r_data[~valid_r_data[r_model_col].isna()]
            if len(valid_model_data) > 0:
                axs[3].plot(
                    valid_model_data['t'], 
                    valid_model_data[r_model_col], 
                    'o-',  # Circles with connecting lines
                    color=variant_color,
                    linewidth=2.5,
                    markersize=5,
                    alpha=0.8
                )
                # Track the plotted values
                plotted_r_model_values.extend(valid_model_data[r_model_col].dropna().tolist())
        
        # Track plotted variants with their colors for legend
        if variant_color is not None:
            plotted_variants[variant_str] = variant_color
    
    # Plot vertical line for analysis date if converted successfully
    if analysis_t is not None:
        for ax in axs:
            ax.axvline(x=analysis_t, color='black', linestyle='--', lw=2.0)
    
    # Add horizontal lines at y=0 for growth rate plots
    axs[2].axhline(y=0.0, color='red', linestyle='--', lw=1.5)  # r_data
    axs[3].axhline(y=0.0, color='red', linestyle='--', lw=1.5)  # r_model
    
    # Set labels and formatting
    axs[0].set_ylabel("Sequence Count", fontsize=14)
    axs[1].set_ylabel("Variant Incidence" if plot_incidence else "Variant Frequency", fontsize=14)
    axs[2].set_ylabel("$r_{\\text{data}}$", fontsize=14)
    axs[3].set_ylabel("$r_{\\text{model}}$", fontsize=14)
    
    # Set tick label font sizes for all axes
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid to all plots
    # for ax in axs:
    #     ax.grid(alpha=0.3)
    
    # Set x-ticks using date information
    if len(location_data) > 0:
        # Create mapping from t to date for x-tick labels
        t_values = sorted(location_data['t'].unique())
        t_to_date = {}
        for t in t_values:
            matching_dates = location_data[location_data['t'] == t]['date']
            if not matching_dates.empty:
                t_to_date[t] = matching_dates.iloc[0]
        
        # Select a reasonable number of ticks
        if len(t_values) > 10:
            tick_indices = np.linspace(0, len(t_values) - 1, 10, dtype=int)
            t_ticks = [t_values[i] for i in tick_indices]
        else:
            t_ticks = t_values
        
        # Format dates for tick labels
        date_labels = []
        for t in t_ticks:
            if t in t_to_date:
                date_str = t_to_date[t].strftime('%Y-%m-%d')
                date_labels.append(date_str)
            else:
                date_labels.append('')
        
        # Set ticks on x-axis of bottom subplot
        axs[3].set_xticks(t_ticks)
        axs[3].set_xticklabels(date_labels, rotation=45, fontsize=12)
    
    # Add x-axis label to bottom subplot
    axs[3].set_xlabel("Date", fontsize=14)
    
    # Calculate and set synchronized y-axis limits based on actually plotted values
    if plotted_r_data_values or plotted_r_model_values:
        # Combine all plotted values
        all_plotted_values = plotted_r_data_values + plotted_r_model_values
        
        if all_plotted_values:
            overall_min = min(all_plotted_values)
            overall_max = max(all_plotted_values)
            
            # No padding - use exact range
            # Just ensure we don't have identical min/max
            if overall_min == overall_max:
                # If all values are the same, add a tiny margin
                overall_min -= 0.002
                overall_max += 0.002
            
            # Ensure 0 is included in the range if it's close
            if overall_min > -0.001 and overall_min < 0:
                overall_min = 0
            if overall_max < 0.001 and overall_max > 0:
                overall_max = 0
                
            # Apply the same limits to both growth rate plots
            axs[2].set_ylim(overall_min, overall_max)  # r_data
            axs[3].set_ylim(overall_min, overall_max)  # r_model
    
    # Get unique variants that are present in the data and color map
    all_variants = set()
    
    # Add variants from the consolidated dataframe
    location_data = growth_rates_df[growth_rates_df['country'] == location]
    if max_variants is not None:
        top_variants = location_data.groupby('variant')['sequences'].sum().sort_values(
            ascending=False).head(max_variants).index
        all_variants.update(top_variants)
    else:
        all_variants.update(location_data['variant'].unique())
    
    # Filter color map to only include variants present in the data
    filtered_color_map = {}
    for variant in all_variants:
        variant_str = str(variant)
        if variant_str in color_map:
            filtered_color_map[variant_str] = color_map[variant_str]
        elif variant in color_map:
            filtered_color_map[variant_str] = color_map[variant]
    
    # Create a legend if we have variants with colors
    if filtered_color_map:
        # Create patches for legend
        patches = [Patch(color=color, label=variant) for variant, color in filtered_color_map.items()]
        
        # Sort by variant label for consistent ordering
        sorted_patches = sorted(patches, key=lambda patch: patch.get_label())
        
        # Add legend at the bottom of the figure
        legend_obj = fig.legend(
            sorted_patches, 
            [patch.get_label() for patch in sorted_patches],
            ncol=min(10, len(sorted_patches)),
            bbox_to_anchor=(0.5, -0.05),
            loc="lower center",
            title="Variant",
            fontsize=12,
            title_fontsize=14
        )
        legend_obj.get_frame().set_linewidth(2.0)
        legend_obj.get_frame().set_edgecolor("k")
    
    # Tight layout for better spacing
    fig.tight_layout()
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    
    # Show the figure
    plt.show()

def plot_top_variant_correlations(
    growth_rates_df: pd.DataFrame, 
    color_map: Dict[str, str], 
    location: str = 'tropics', 
    n: int = 4, 
    min_points: int = 3, 
    r_model_col: str = 'median_r', 
    r_data_col: str = 'growth_rate_r_data',
    figsize: tuple = (16, 14) 
) -> None:
    """
    Plot the top n variants with the highest correlation between model growth rate and empirical growth rate.
    
    Parameters:
    -----------
    growth_rates_df : pd.DataFrame
        DataFrame containing growth rate data with columns 'country', 'variant', 
        'growth_rate_r' (model), and 'growth_rate_r_data' (empirical).
    color_map : dict
        Dictionary mapping variant names to colors.
    location : str, default='tropics'
        The location to filter the data by (e.g., 'tropics').
    n : int, default=4
        The number of top variants to plot based on correlation.
    min_points : int, default=3
        Minimum number of data points required for a variant to be considered.
    r_model_col : str, default='growth_rate_r'
        The column name for the model growth rate.
    r_data_col : str, default='growth_rate_r_data'
        The column name for the empirical growth rate.
    figsize : tuple, default=(16, 14)
        Size of the figure to create.
    
    Returns:
    --------
    None
    """
    #fig, axes = plt.subplots(2, 2, figsize=figsize)
    growth_rates_df = growth_rates_df.copy()
    growth_rates_df['variant'] = growth_rates_df['variant'].astype(str)
    top_variants = get_top_variants(growth_rates_df, location, n, min_points, r_model_col, r_data_col)
    
    if not top_variants:
        print(f"No variants found with at least {min_points} data points in {location}.")
        return
    else:
        # Create subplots for the top variants (2 columns per row)
        num_variants = len(top_variants)
        num_rows = (num_variants + 1) // 2  # Two plots per row
        if num_rows == 0:
            num_rows = 1   
        # Create a grid of subplots
        if num_variants == 1:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
        elif num_variants == 2:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        elif num_variants == 3:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
            axes[3].axis('off')
        else:
            # For more than 3 variants, create a grid of subplots
            if num_rows * 2 < num_variants:
                num_rows = (num_variants + 1) // 2
            if num_rows * 2 == num_variants:
                fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
            else:
                num_rows += 1
                fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
    
    for i, (variant, corr, p_value, n_points) in enumerate(top_variants):
        if i >= len(axes):
            break
            
        # Get data for this variant (variant is already a string from get_top_variants)
        variant_data = growth_rates_df[
            (growth_rates_df['country'] == 'tropics') & 
            (growth_rates_df['variant'] == variant)
        ].dropna(subset=[r_data_col, r_model_col])
        
        color = color_map[int(variant)]
        
        # Create scatter plot
        axes[i].scatter(
            variant_data[r_model_col], 
            variant_data[r_data_col],
            color=color, s=80, alpha=0.7, edgecolor='black', linewidth=0.5
        )
        
        # Add reference line (y=x)
        min_val = min(variant_data[r_model_col].min(), variant_data[r_data_col].min())
        max_val = max(variant_data[r_model_col].max(), variant_data[r_data_col].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Add zero lines
        axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set title and labels
        axes[i].set_title(rf"Variant {variant}: $r$ = {corr:.3f}, p = {p_value:.3e}, n = {n_points}")
        axes[i].set_xlabel(r"$r_{\text{model}}$")
        axes[i].set_ylabel(r"$r_{\text{data}}$")
    
    plt.xlabel(r"$r_{\text{model}}$")
    plt.ylabel(r"$r_{\text{data}}$")
    fig.suptitle(f'Top {n} Variants Correlation in {location}')
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_smoothed_frequencies(growth_rates_df: pd.DataFrame, location: str, pivot_date: str,
                              color_map: Dict[str, str], fig_size: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot smoothed variant frequencies over time.
    
    Parameters
    ----------
    growth_rates_df : pd.DataFrame
        DataFrame containing growth rates and variant frequencies. Needs to have columns:
        - 'week_id': The week identifier.
        - 'variant_frequency': The frequency of the variant.
        - 'variant_frequency_smoothed': The smoothed frequency of the variant.
        - 'variant': The variant identifier.
    location : str
        The location for which the data is being plotted.
    pivot_date : str
        The date to pivot the data around, used in the plot title.
    color_map : Dict[str, str]
        Dictionary mapping variant names to colors.
    fig_size : Tuple[int, int], optional
        Size of the figure to create, by default (10, 6).

    Returns
    -------
    None
    -----------
    Displays a plot of variant frequencies over time.
    -----------
    
    """
    # Create a figure
    plt.figure(figsize=fig_size)

    # Get unique variants for plotting
    variants = growth_rates_df['variant'].unique()

    # Plot each variant with its own color
    for variant in variants:
        variant_data = growth_rates_df[growth_rates_df['variant'] == variant]
        
        # Convert variant to string for color mapping
        variant_str = str(variant)
        
        # Plot scatter points for the variant frequency
        plt.scatter(
            variant_data['week_id'], 
            variant_data['variant_frequency'],
            color=color_map[variant_str],
            label=f'Variant {variant}',
            alpha=0.7,
            s=50
        )
        
        # Plot smoothed line
        valid_data = variant_data.dropna(subset=['variant_frequency_smoothed'])
        if not valid_data.empty:
            plt.plot(
                valid_data['week_id'], 
                valid_data['variant_frequency_smoothed'],
                color=color_map[variant_str],
                linestyle='-',
                linewidth=2
            )

    # Customize the plot
    plt.xlabel('Week ID', fontsize=12)
    plt.ylabel('Variant Frequency', fontsize=12)
    plt.title(f'Variant Frequencies Over Time - {location} ({pivot_date})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Create a custom legend that doesn't include duplicated entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(), 
        by_label.keys(), 
        loc='center left', 
        bbox_to_anchor=(1, 0.5),
        ncol=1
    )

    plt.tight_layout()
    plt.show()