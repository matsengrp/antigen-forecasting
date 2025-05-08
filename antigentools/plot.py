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
from matplotlib.patches import Patch
from antigentools.utils import read_estimates
# Mute warnings
import warnings
warnings.filterwarnings('ignore')


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
    
    # Grab predicted frequencies of interest
    deme_ga = inferred_rt.query(f'location == "{deme}" and model == "{model}"')
    for variant in range(n_variant):
        v_name = v_names[variant]
        variant_df = deme_ga.query(f'variant == {v_name}')
        ax.plot(variant_df['t'], variant_df['median_R'], lw=5.5, label=v_name, color=color_map[v_name])
        ax.fill_between(variant_df['t'], variant_df[f'R_lower_{p}'], variant_df[f'R_upper_{p}'], color=color_map[v_name], alpha=0.5)
    
    if pivot_idx:
        for idx in pivot_idx:
            ax.axvline(x=idx, color='black', linestyle='--', lw=2.0)
    ax.axhline(y=1.0, color='red', linestyle='--', lw=1.5)
    ax.set_xlabel('')
    ax.set_ylabel(r"$R_t$")


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