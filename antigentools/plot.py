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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from antigentools.utils import read_estimates
# Mute warnings
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional, Union

def get_distinct_colors(n):
    """Generates n distinct colors in hex format.

    Parameters:
    ---------------
        n (int): Number of colors to generate

    Returns:
    ---------------
        list: List of hex colors
    """
    cmap = plt.get_cmap('hsv')  # Use a colormap like 'hsv' or 'rainbow'
    colors = [cmap(i / n) for i in range(n)]
    hex_colors = [plt.cm.colors.to_hex(c) for c in colors]
    # Shuffle colors to avoid adjacent colors being too similar
    random.shuffle(hex_colors)
    return hex_colors


def make_color_map(v_names: list, palette: list) -> dict:
    """ Create a color map for variants.

    Parameters:
    ---------------
        v_names (list): List of variant names
        palette (list): List of colors to use for variants

    Returns:
    ---------------
        dict: A dictionary mapping variant names to colors
    """
    v_colors = palette[:len(v_names)]
    color_map = dict(zip(v_names, v_colors))
    return color_map


def plot_observed_cases(ax, flu_data, deme, pivot_idx=None):
    """ Plot observed cases.

    Parameters:
    ---------------
        ax (matplotlib.axes.Axes): Axis to plot on
        flu_data (dict): Dictionary of `evofr` data
        deme (str): Deme to plot
        pivot_idx (list): List of pivot indices to draw vertical lines at

    Returns:
    ---------------
        None
    """
    t = np.arange(0, flu_data[deme].cases.shape[0])
    ax.bar(t, flu_data[deme].cases, width=5.0, color='black', alpha=0.5)
    if pivot_idx:
        for idx in pivot_idx:
            ax.axvline(x=idx, color='black', linestyle='--', lw=2.0)
    ax.set_ylabel("Cases")


def plot_observed_freqs(ax, ef_data, deme, color_map):
    """ Plot observed frequencies of variants.
    
    Parameters:
    --------------
        ax (matplotlib.pyplot.Axes): Axes object
        ef_data (Dict): Dictionary of `evofr` data
        deme (str): Demographic unit
        color_map (dict): Dictionary mapping variant names to colors
    
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
    ax.set_ylabel("Frequency")


def plot_frequencies(ax, ef_data, freqs, deme, model, color_map, p=50, pivot_idx=None):
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
    ax.set_ylabel("Frequency")


def plot_rt(ax, inferred_rt, deme, model, color_map, p=95, pivot_idx=None):
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
    ax.set_ylabel(r"$R_t$")
    
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


def plot_fitness(ax, fitness_df, deme, color_map, pivot_idx=None):
    """ Plot simulated fitness of variants.

    Parameters:
    ---------------
        ax (matplotlib.axes.Axes): Axis to plot on
        fitness_df (pd.DataFrame): Fitness dataframe
        deme (str): Deme name
        color_map (dict): Dictionary of colors to use for variants
        pivot_idx (list): List of pivot indices

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
    ax.set_ylabel("Fitness")


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


def plot_variant_counts_histogram(ax, ef_data, deme, color_map):
    """ Plot histogram of variant sequence counts.

    Parameters:
    ---------------
        ax (matplotlib.axes.Axes): Axis to plot on
        ef_data (dict): Dictionary of `evofr` data
        deme (str): Deme to plot
        color_map (dict): Dictionary mapping variant names to colors

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
    
    for variant in range(n_variant):
        v_name = v_names[variant]
        # Plot sequence counts for this variant on top of previous variants
        ax.bar(t, seq_counts[:, variant], bottom=bottom, width=5.0, 
               color=color_map[v_name], label=v_name, alpha=0.8, edgecolor='black', linewidth=0.5)
        # Update the bottom for the next variant
        bottom += seq_counts[:, variant]
    
    ax.set_ylabel("Sampled Sequences")
    ax.set_xlabel("")


def plot_dynamics(ef_data, freq, ga, fitness, model, deme, color_map, p=50, sep=1825, pivot_idx=None, save_path=None, pivot_date=None):
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


def get_analysis_window(analysis_date: str, build: str, fitness_df: pd.DataFrame) -> tuple:
    """ Create a subset of data for a given analysis window.

    Parameters
    ---------------
        analysis_date : str
            Analysis date of interest
        build : str
            Build name where data resides
        fitness_df : pd.DataFrame
            Fitness dataframe with assigned datetimes

    Returns
    ---------------
        tuple: (evo_dict, freqs_df, rts_df, fitness_df)
    """
    data_path = f"../data/{build}/time-stamped/{analysis_date}/"
    
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
    freq_paths = glob.glob(f"../results/{build}/estimates/*/freq_*{analysis_date}.tsv")
    rt_paths = glob.glob(f"../results/{build}/estimates/*/rt_*{analysis_date}.tsv")
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


def plot_analysis_window_with_variant_counts(pivot_date:str, location: str, model: str, build: str, 
                       pruned_variant_fitness_df: pd.DataFrame, color_map: dict) -> None:
    """ Plot analysis window for a given location and model with variant counts histogram.
    
    This creates a five-panel plot including:
    1. Case counts
    2. Variant sequence counts (stacked histogram)
    3. Variant frequencies
    4. Growth advantages
    5. Fitness

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
    
    # Setup figure and grid with shared x-axis
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(30, 18), sharex=True)
    
    # Add a title
    fig.suptitle(f"{model} inference for {location} deme on {pivot_date}", fontsize=30)
    
    # Subset color map to only include variants present in the deme
    available_variants = evo_dict[location].var_names
    filtered_color_map = {k: v for k, v in color_map.items() if k in available_variants}

    # Sort color map by variant name
    filtered_color_map = {k: filtered_color_map[k] for k in sorted(filtered_color_map.keys())}
    
    # Timestamps for dates
    dates = ef.data.expand_dates(evo_dict[location].dates, T_forecast=0)

    # Plot case counts
    plot_observed_cases(axs[0], evo_dict, location, pivot_idx=[pivot_idx])
    
    # Plot variant counts histogram
    plot_variant_counts_histogram(axs[1], evo_dict, location, filtered_color_map)
    
    # Plot frequencies
    plot_frequencies(axs[2], evo_dict, small_freqs_df, location, model, filtered_color_map, p=50, pivot_idx=[pivot_idx])

    # Plot growth advantages
    ga = small_rt_df.query(f"variant in {list(available_variants)}")
    plot_rt(axs[3], ga, location, model, filtered_color_map, p=50, pivot_idx=[pivot_idx])

    # Plot fitness
    # Subset fitness dataframe to only include variants present in the available_variants
    fitness = small_fitness_df.query(f"variant in {list(available_variants)}")
    plot_fitness(axs[4], fitness, location, filtered_color_map, pivot_idx=[pivot_idx])
    axs[4].set_xticks(np.arange(0, len(dates), 30))
    axs[4].set_xticklabels(axs[4].get_xticks(), rotation=45)

    # Adjust axis visibility
    for ax in axs[:-1]:  # Hide x-axis labels on all but the last subplot
        ax.label_outer()

    # Add x-axis label
    axs[-1].set_xlabel("Date")

    # Create a legend
    patches = [Patch(color=c, label=l) for l, c in filtered_color_map.items()]
    legend = fig.legend(patches, list(filtered_color_map.keys()), ncol=10, bbox_to_anchor=(0.5, -0.15), 
                       loc="lower center", title="Variant")  
    legend.get_frame().set_linewidth(2.)
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
    ax.grid(alpha=0.3)
    
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
    ax.grid(alpha=0.3)
    
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
            
        # Ensure 0 is included in the range
        if min_y > 0:
            min_y = -0.05
        if max_y < 0:
            max_y = 0.05
            
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

def plot_growth_rate_dynamics(
    growth_rates_df: pd.DataFrame,
    location: str,
    color_map: Dict[str, str],
    model: Optional[str] = None,
    analysis_date: Optional[str] = None,
    max_variants: Optional[int] = None,
    figsize: tuple = (20, 15),
    save_path: Optional[str] = None
) -> None:
    """Plot a three-panel figure showing sequence counts and growth rates for variants in a location.
    
    This creates a three-panel plot including:
    1. Original and smoothed sequence counts
    2. Empirical growth rates (r_data)
    3. Model-derived growth rates (r_model)
    
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
                                  
    Returns:
    --------
        None
    """
    # Make a copy to avoid modifying input DataFrame
    df_processed = growth_rates_df.copy()
    
    # Ensure required columns exist
    required_cols = ['country', 'variant', 'date', 'sequences', 'smoothed_sequences', 'growth_rate_r_data', 'growth_rate_r']
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
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)
    
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
    
    fig.suptitle(title, fontsize=20)
    
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
        
        # Skip if no valid growth rate data for this variant
        if not variant_data['growth_rate_r_data'].isna().all():
            # Plot empirical growth rates (r_data)
            axs[1].plot(
                variant_data['t'], 
                variant_data['growth_rate_r_data'], 
                'o-',  # Circles with connecting lines
                color=variant_color,
                linewidth=2.5,
                markersize=5,
                alpha=0.8
            )
        
        # Skip if no valid model growth rate data for this variant
        if not variant_data['growth_rate_r'].isna().all():
            # Plot model growth rates (r_model)
            axs[2].plot(
                variant_data['t'], 
                variant_data['growth_rate_r'], 
                lw=4.5,
                color=variant_color
            )
        
        # Track plotted variants with their colors for legend
        if variant_color is not None:
            plotted_variants[variant_str] = variant_color
    
    # Plot vertical line for analysis date if converted successfully
    if analysis_t is not None:
        for ax in axs:
            ax.axvline(x=analysis_t, color='black', linestyle='--', lw=2.0)
    
    # Add horizontal lines at y=0 for growth rate plots
    axs[1].axhline(y=0.0, color='red', linestyle='--', lw=1.5)  # r_data
    axs[2].axhline(y=0.0, color='red', linestyle='--', lw=1.5)  # r_model
    
    # Set labels and formatting
    axs[0].set_ylabel("Sequence Count")
    axs[1].set_ylabel("$r_{\\text{data}}$")
    axs[2].set_ylabel("$r_{\\text{model}}$")
    
    # Add grid to all plots
    for ax in axs:
        ax.grid(alpha=0.3)
    
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
        axs[2].set_xticks(t_ticks)
        axs[2].set_xticklabels(date_labels, rotation=45)
    
    # Add x-axis label to bottom subplot
    axs[2].set_xlabel("Date")
    
    # Calculate and set appropriate y-axis limits for the growth rate plots
    for i, col in [(1, 'growth_rate_r_data'), (2, 'growth_rate_r')]:
        valid_data = location_data[col].dropna()
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
                
            # Ensure 0 is included in the range
            if min_y > 0:
                min_y = -0.05
            if max_y < 0:
                max_y = 0.05
                
            axs[i].set_ylim(min_y, max_y)
    
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