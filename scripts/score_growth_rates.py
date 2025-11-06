"""
Comprehensive growth rate analysis and scoring script.

This script evaluates growth rate predictions from model outputs and exports detailed data.
It processes rt_*.tsv files to calculate comprehensive metrics and exports growth rate data
for downstream analysis and visualization.

Usage:
    python score_growth_rates.py --config configs/benchmark_config.yaml --build flu-simulated-150k-samples --output-dir results/flu-simulated-150k-samples/

Required Arguments:
    --config        Path to the configuration file.
    --build         Build name (e.g. 'flu-simulated-150k-samples').
    --output-dir    Directory to save the analysis outputs.

Output:
    - window_growth_rates.csv        Window-level metrics
    - variant_growth_rates.csv       Variant-level metrics  
    - vi_convergence_diagnostics.csv Convergence diagnostics
    - growth_rates/{MODEL}/growth_rates_{location}_{pivot_date}.tsv

Dependencies:
    - Requires Python 3.x
    - pandas, numpy, pyyaml, scipy
    - antigentools package

Author: Zorian Thornton (@zorian15)
Date: 2025-04-08
"""

import pandas as pd
import numpy as np
import os
import yaml
import argparse
import glob
import json
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# Import from antigentools
from antigentools.utils import get_deme_stats
from antigentools.analysis import (
    get_filtered_growth_rates_df,
    evaluate_growth_rate_performance,
    calculate_variant_mae
)

# Static variables for consistent growth rate filtering
CONNECT_GAPS = True  # Whether to connect gaps in the growth rate data when plotting
MIN_SEGMENT_LENGTH = 3  # Minimum segment length to trust growth rate calculations
MIN_SEQUENCE_COUNT = 10  # Minimum smoothed sequence count per variant
MIN_VARIANT_FREQUENCY = 0.01  # Minimum variant frequency to consider
EPSILON = 1e-3  # Tolerance threshold for overestimation rate calculations
MIN_TOTAL_SEQUENCES = 300  # Minimum total sequences per window (set to None to disable)
CONVERGENCE_THRESHOLD = 0.5  # Threshold for convergence diagnostics


def export_growth_rates_data(
    growth_rates_df: pd.DataFrame, 
    output_dir: str, 
    model: str, 
    location: str, 
    pivot_date: str
) -> None:
    """
    Export processed growth rates data to TSV files.
    
    Creates detailed growth rate files that include model estimates, 
    confidence intervals, frequencies, sequence counts, and filtering status.
    
    Parameters:
    -----------
    growth_rates_df : pd.DataFrame
        DataFrame containing processed growth rate data
    output_dir : str
        Base output directory
    model : str
        Model name (e.g., 'FGA', 'GARW')
    location : str
        Location name (e.g., 'north', 'tropics', 'south')
    pivot_date : str
        Pivot date in YYYY-MM-DD format
    """
    # Create output directory structure
    model_dir = Path(output_dir) / "growth-rates" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename following existing convention
    filename = f"growth_rates_{location}_{pivot_date}.tsv"
    filepath = model_dir / filename
    
    # Save to TSV file
    growth_rates_df.to_csv(filepath, sep="\t", index=False)
    print(f"Exported growth rates data: {filepath}")


def process_all_model_results(config: Dict, build: str, output_dir: str) -> None:
    """
    Discover and process all rt_*.tsv files for comprehensive growth rate analysis.
    
    This function follows the notebook logic exactly:
    1. Loops through all rt_*.tsv files
    2. Filters windows by MIN_TOTAL_SEQUENCES
    3. Calculates growth rates using get_filtered_growth_rates_df
    4. Evaluates performance metrics
    5. Calculates variant-level MAE
    6. Loads VI convergence diagnostics
    7. Exports all results
    
    Parameters:
    -----------
    config : Dict
        Configuration dictionary with models, locations, dates
    build : str
        Build name to process
    output_dir : str
        Output directory for results
    """
    # Initialize result dictionaries
    results_dict = {
        'pivot_date': [],
        'model': [],
        'location': [],
        'correlation': [],
        'mae': [],
        'rmse': [],
        'r2': [],
        'n_seqs': [],
        'n_cases': [],
        'seq_entropy': [],
        'case_entropy': [],
        'seq_entropy_norm': [],
        'case_entropy_norm': []
    }

    variant_results_dict = {
        'variant': [],
        'pivot_date': [],
        'model': [],
        'location': [],
        'mae': [],
        'normalized_mae': [],
        'max_r_data': [],
        'correlation': [],
        'r2': [],
        'n_points': [],
        'total_sequences': [],
        'total_smoothed_sequences': [],
        'mean_variant_frequency': [],
        'mean_smoothed_variant_frequency': [],
        'max_variant_frequency': [],
        'max_smoothed_variant_frequency': []
    }

    # New convergence diagnostics dictionary
    diagnostics_dict = {
        'pivot_date': [],
        'model': [],
        'location': [],
        'inference_method': [],
        'iterations': [],
        'learning_rate': [],
        'num_samples': [],
        'num_iterations': [],
        'initial_loss': [],
        'final_loss': [],
        'min_loss': [],
        'total_improvement': [],
        'converged': [],
        'relative_change': [],
        'threshold': [],
        'window': [],
        'final_iteration': []
    }

    # Tracking variables for filtering
    total_windows = 0
    skipped_low_sequences = 0
    processed_windows = 0

    # Get all rt paths
    rt_paths = glob.glob(f"results/{build}/estimates/*/rt_*.tsv")
    rt_paths.sort()
    
    print(f"Looking for rt files in: results/{build}/estimates/*/rt_*.tsv")
    print(f"Found {len(rt_paths)} rt files")

    for path in rt_paths:
        parts = path.split('/')
        model = parts[-2]  # Model is the parent directory name
        filename = parts[-1]  # e.g., rt_tropics_2040-10-01.tsv
        location = filename.split('_')[1]  # e.g., 'tropics', 'north', 'south'
        pivot_date = filename.split("_")[2].split(".")[0]  # e.g., '2040-10-01'

        if model not in ['FGA', 'GARW']:
            continue
        
        total_windows += 1
        print(f"Processing: {model}_{location}_{pivot_date}")

        try:
            # Get summary stats first (needed for sequence count filtering)
            window_stats_dict = get_deme_stats(pivot_date, location, build, data_dir="data")
            
            # Apply sequence count threshold filter
            if MIN_TOTAL_SEQUENCES is not None:
                total_seqs = window_stats_dict['seq_counts']
                if total_seqs < MIN_TOTAL_SEQUENCES:
                    skipped_low_sequences += 1
                    print(f"Skipping {model}_{location}_{pivot_date}: only {total_seqs} sequences (< {MIN_TOTAL_SEQUENCES})")
                    continue
            
            # Calculate growth rates
            growth_rates_df = get_filtered_growth_rates_df(
                build, 
                model, 
                location, 
                pivot_date, 
                spline_smoothing_factor=1.0, 
                spline_order=3, 
                min_sequence_count=MIN_SEQUENCE_COUNT, 
                min_variant_frequency=MIN_VARIANT_FREQUENCY,
                data_path="data/"
            )
            
            # Export growth rates data
            export_growth_rates_data(
                growth_rates_df, output_dir, model, location, pivot_date
            )
            
            # Evaluate performance with consistent filtering
            eval_results = evaluate_growth_rate_performance(
                growth_rates_df, 
                overestimation_tol=EPSILON,
                connect_gaps=CONNECT_GAPS,
                min_segment_length=MIN_SEGMENT_LENGTH,
                min_sequence_count=MIN_SEQUENCE_COUNT,
                min_variant_frequency=MIN_VARIANT_FREQUENCY
            )
            
            # Calculate variant MAE with consistent filtering
            variant_mae_df = calculate_variant_mae(
                growth_rates_df, 
                overestimation_tol=1e-3,
                min_sequence_count=MIN_SEQUENCE_COUNT,
                min_variant_frequency=MIN_VARIANT_FREQUENCY
            )
            
            # Load VI convergence diagnostics if available
            diagnostics_path = f"results/{build}/convergence-diagnostics/{model}_{location}_{pivot_date}_vi_diagnostics.json"
            try:
                with open(diagnostics_path, 'r') as f:
                    diag_data = json.load(f)
                    
                # Extract data from the JSON structure
                diagnostics_dict['pivot_date'].append(pivot_date)
                diagnostics_dict['model'].append(model)
                diagnostics_dict['location'].append(location)
                diagnostics_dict['inference_method'].append(diag_data.get('inference_method', None))
                
                # Inference settings
                inf_settings = diag_data.get('inference_settings', {})
                diagnostics_dict['iterations'].append(inf_settings.get('iterations', None))
                diagnostics_dict['learning_rate'].append(inf_settings.get('learning_rate', None))
                diagnostics_dict['num_samples'].append(inf_settings.get('num_samples', None))
                
                # ELBO trajectory
                elbo_traj = diag_data.get('convergence_diagnostics', {}).get('elbo_trajectory', {})
                diagnostics_dict['num_iterations'].append(elbo_traj.get('num_iterations', None))
                diagnostics_dict['initial_loss'].append(elbo_traj.get('initial_loss', None))
                diagnostics_dict['final_loss'].append(elbo_traj.get('final_loss', None))
                diagnostics_dict['min_loss'].append(elbo_traj.get('min_loss', None))
                diagnostics_dict['total_improvement'].append(elbo_traj.get('total_improvement', None))
                
                # Convergence status
                conv_status = diag_data.get('convergence_diagnostics', {}).get('convergence', {})
                diagnostics_dict['converged'].append(conv_status.get('converged', None))
                diagnostics_dict['relative_change'].append(conv_status.get('relative_change', None))
                diagnostics_dict['threshold'].append(conv_status.get('threshold', None))
                diagnostics_dict['window'].append(conv_status.get('window', None))
                diagnostics_dict['final_iteration'].append(conv_status.get('final_iteration', None))
                
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                # Append None values to maintain alignment for missing diagnostics
                for key in ['pivot_date', 'model', 'location', 'inference_method', 'iterations', 
                           'learning_rate', 'num_samples', 'num_iterations', 'initial_loss', 
                           'final_loss', 'min_loss', 'total_improvement', 'converged', 
                           'relative_change', 'threshold', 'window', 'final_iteration']:
                    if key in ['pivot_date', 'model', 'location']:
                        diagnostics_dict[key].append(eval(key))
                    else:
                        diagnostics_dict[key].append(None)
            
            # Append variant results to the dictionary
            for _, row in variant_mae_df.iterrows():
                variant_results_dict['variant'].append(row['variant'])
                variant_results_dict['pivot_date'].append(pivot_date)
                variant_results_dict['model'].append(model)
                variant_results_dict['location'].append(location)
                variant_results_dict['mae'].append(row['mae'])
                variant_results_dict['normalized_mae'].append(row['normalized_mae'])
                variant_results_dict['max_r_data'].append(row['max_r_data'])
                variant_results_dict['correlation'].append(row['correlation'])
                variant_results_dict['r2'].append(row['r2'])
                variant_results_dict['n_points'].append(row['n_points'])
                variant_results_dict['total_sequences'].append(row['total_sequences'])
                variant_results_dict['total_smoothed_sequences'].append(row['total_smoothed_sequences'])
                variant_results_dict['mean_variant_frequency'].append(row['mean_variant_frequency'])
                variant_results_dict['mean_smoothed_variant_frequency'].append(row['mean_smoothed_variant_frequency'])
                variant_results_dict['max_variant_frequency'].append(row['max_variant_frequency'])
                variant_results_dict['max_smoothed_variant_frequency'].append(row['max_smoothed_variant_frequency'])
            
            # Now append the results to the results dictionary
            results_dict['pivot_date'].append(pivot_date)
            results_dict['model'].append(model)
            results_dict['location'].append(location)
            results_dict['correlation'].append(eval_results['correlation'])
            results_dict['mae'].append(eval_results['mae'])
            results_dict['rmse'].append(eval_results['rmse'])
            results_dict['r2'].append(eval_results['r2'])
            results_dict['n_seqs'].append(window_stats_dict['seq_counts'])
            results_dict['n_cases'].append(window_stats_dict['case_counts'])
            results_dict['seq_entropy'].append(window_stats_dict['seq_entropy'])
            results_dict['case_entropy'].append(window_stats_dict['case_entropy'])
            results_dict['seq_entropy_norm'].append(window_stats_dict['seq_norm_entropy'])
            results_dict['case_entropy_norm'].append(window_stats_dict['case_norm_entropy'])
            
            processed_windows += 1
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results_dict)
    variant_results_df = pd.DataFrame(variant_results_dict)
    diagnostics_df = pd.DataFrame(diagnostics_dict)

    # Print summary statistics
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Total analysis windows considered: {total_windows}")
    if MIN_TOTAL_SEQUENCES is not None:
        print(f"Windows skipped (< {MIN_TOTAL_SEQUENCES} sequences): {skipped_low_sequences}")
        print(f"Windows after sequence filter: {total_windows - skipped_low_sequences}")
    print(f"Windows successfully processed: {processed_windows}")
    print(f"Windows with errors: {total_windows - skipped_low_sequences - processed_windows}")

    print(f"\nDataFrame shapes:")
    print(f"  results_df: {results_df.shape}")
    print(f"  variant_results_df: {variant_results_df.shape}")
    print(f"  diagnostics_df: {diagnostics_df.shape}")

    if len(diagnostics_df) > 0:
        # Change convergence flag based on if relative_change is less than threshold 
        diagnostics_df['converged'] = diagnostics_df['relative_change'].apply(
            lambda x: False if x is None else (x <= CONVERGENCE_THRESHOLD)
        )
        print(f"\nConvergence diagnostics:")
        print(f"  Available for {diagnostics_df['converged'].notna().sum()} runs")
        print(f"  Models with convergence data: {dict(diagnostics_df['model'].value_counts())}")
        if diagnostics_df['converged'].notna().sum() > 0:
            print(f"  Convergence rate: {diagnostics_df['converged'].mean():.2%} (of runs with data)")

    if MIN_TOTAL_SEQUENCES is not None:
        print(f"\nSequence count filtering:")
        print(f"  Minimum total sequences required: {MIN_TOTAL_SEQUENCES}")
        if len(results_df) > 0:
            print(f"  Remaining sequence count range: {results_df['n_seqs'].min()}-{results_df['n_seqs'].max()}")
    else:
        print(f"\nNo sequence count filtering applied")
        if len(results_df) > 0:
            print(f"  Sequence count range: {results_df['n_seqs'].min()}-{results_df['n_seqs'].max()}")

    # Save results to CSV files
    window_growth_rates_df_path = Path(output_dir) / "window_growth_rates.tsv"
    variant_results_df_path = Path(output_dir) / "variant_growth_rates.tsv"
    diagnostics_df_path = Path(output_dir) / "vi_convergence_diagnostics.tsv"
    
    results_df.to_csv(window_growth_rates_df_path, index=False, sep='\t')
    variant_results_df.to_csv(variant_results_df_path, index=False, sep='\t')
    diagnostics_df.to_csv(diagnostics_df_path, index=False, sep='\t')
    
    print(f"\nResults saved to:")
    print(f"  {window_growth_rates_df_path}")
    print(f"  {variant_results_df_path}")
    print(f"  {diagnostics_df_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive growth rate analysis and scoring.')
    
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to the configuration file.')
    parser.add_argument('--build', type=str, required=True,
                       help='Build name (e.g., flu-simulated-150k-samples).')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save the analysis outputs.')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process all model results
    process_all_model_results(config, args.build, args.output_dir)