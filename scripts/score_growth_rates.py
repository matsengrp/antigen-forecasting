"""
Comprehensive growth rate analysis and scoring script.

This script evaluates growth rate predictions from model outputs using a modular,
configurable approach. It extends the antigentools.scoring module with growth 
rate-specific functionality including RT file discovery, type-safe result collection,
and flexible YAML-based configuration.

Features:
- Modular architecture extending antigentools.scoring
- Type-safe dataclasses with validation
- YAML-based configuration system
- Comprehensive logging and error handling
- Backwards-compatible output formats

Usage:
    python score_growth_rates.py --config configs/growth_rate_config.yaml --build flu-simulated-150k-samples --output-dir results/flu-simulated-150k-samples/

Required Arguments:
    --config        Path to YAML configuration file with growth rate settings
    --build         Build name (e.g. 'flu-simulated-150k-samples')
    --output-dir    Directory to save analysis outputs
    --verbose       Enable verbose logging (optional)

Configuration Format (YAML):
    growth_rate:
      min_sequence_count: 10
      min_variant_frequency: 0.01
      min_total_sequences: 300
      epsilon: 0.001
      # ... additional parameters
    
    convergence:
      threshold: 0.5
      required_models: [FGA, GARW]

Output Files:
    - window_growth_rates.tsv        Window-level metrics  
    - variant_growth_rates.tsv       Variant-level metrics
    - vi_convergence_diagnostics.tsv Convergence diagnostics
    - growth-rates/{MODEL}/growth_rates_{location}_{pivot_date}.tsv

Dependencies:
    - antigentools package with extended scoring module
    - Standard scientific Python stack (pandas, numpy, scipy)

Author: Zorian Thornton (@zorian15)
Date: 2025-04-08 (Refactored: 2025-09-10)
"""

import pandas as pd
import numpy as np
import argparse
import json
import logging
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

# Import from the extended scoring module
from antigentools.scoring import (
    load_config, parse_growth_rate_config,
    GrowthRateConfig, ConvergenceConfig,
    discover_rt_files, load_growth_rates, load_convergence_diagnostics,
    WindowResult, VariantResult, ConvergenceDiagnostic,
    GrowthRateResultsCollector
)


logger = logging.getLogger(__name__)


def export_growth_rates_data(
    growth_rates_df: pd.DataFrame, 
    output_dir: Path, 
    model: str, 
    location: str, 
    pivot_date: str
) -> None:
    """
    Export processed growth rates data to TSV files.
    
    Creates detailed growth rate files that include model estimates, 
    confidence intervals, frequencies, sequence counts, and filtering status.
    
    Parameters
    ----------
    growth_rates_df : pd.DataFrame
        DataFrame containing processed growth rate data
    output_dir : Path
        Base output directory
    model : str
        Model name (e.g., 'FGA', 'GARW')
    location : str
        Location name (e.g., 'north', 'tropics', 'south')
    pivot_date : str
        Pivot date in YYYY-MM-DD format
    """
    # Create output directory structure
    model_dir = output_dir / "growth-rates" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename following existing convention
    filename = f"growth_rates_{location}_{pivot_date}.tsv"
    filepath = model_dir / filename
    
    # Save to TSV file
    growth_rates_df.to_csv(filepath, sep="\t", index=False)
    logger.info(f"Exported growth rates data: {filepath}")


def process_rt_window(
    rt_file,
    growth_config: GrowthRateConfig,
    conv_config: ConvergenceConfig,
    build: str,
    base_dir: str = "."
) -> Tuple[Optional[WindowResult], List[VariantResult], Optional[ConvergenceDiagnostic]]:
    """
    Process a single RT window for growth rate analysis.
    
    Parameters
    ----------
    rt_file : RTFile
        RT file metadata
    growth_config : GrowthRateConfig
        Growth rate configuration
    conv_config : ConvergenceConfig
        Convergence configuration
    build : str
        Build name
    base_dir : str
        Base directory
        
    Returns
    -------
    Tuple[Optional[WindowResult], List[VariantResult], Optional[ConvergenceDiagnostic]]
        Window result, variant results, and convergence diagnostic
    """
    model, location, pivot_date = rt_file.model, rt_file.location, rt_file.pivot_date
    
    logger.info(f"Processing: {model}_{location}_{pivot_date}")
    
    try:
        # Get summary stats first (needed for sequence count filtering)
        window_stats_dict = get_deme_stats(pivot_date, location, build, data_dir="data")
        
        # Apply sequence count threshold filter
        if growth_config.min_total_sequences is not None:
            total_seqs = window_stats_dict['seq_counts']
            if total_seqs < growth_config.min_total_sequences:
                logger.info(f"Skipping {model}_{location}_{pivot_date}: only {total_seqs} sequences (< {growth_config.min_total_sequences})")
                return None, [], None
        
        # Calculate growth rates using our loader
        growth_rates_df = get_filtered_growth_rates_df(
            build, 
            model, 
            location, 
            pivot_date, 
            spline_smoothing_factor=growth_config.spline_smoothing_factor, 
            spline_order=growth_config.spline_order, 
            min_sequence_count=growth_config.min_sequence_count, 
            min_variant_frequency=growth_config.min_variant_frequency,
            skip_first_n_points=growth_config.skip_first_n_points,
            use_freqs=growth_config.use_freqs,
            use_smoothed_incidence=growth_config.use_smoothed_incidence,
            data_path="data/"
        )
        
        # Evaluate performance with configuration
        eval_results = evaluate_growth_rate_performance(
            growth_rates_df, 
            overestimation_tol=growth_config.epsilon,
            connect_gaps=growth_config.connect_gaps,
            min_segment_length=growth_config.min_segment_length,
            min_sequence_count=growth_config.min_sequence_count,
            min_variant_frequency=growth_config.min_variant_frequency
        )
        
        # Calculate variant MAE with configuration
        variant_mae_df = calculate_variant_mae(
            growth_rates_df, 
            overestimation_tol=growth_config.epsilon,
            min_sequence_count=growth_config.min_sequence_count,
            min_variant_frequency=growth_config.min_variant_frequency
        )
        
        # Create window result
        window_result = WindowResult(
            pivot_date=pivot_date,
            model=model,
            location=location,
            correlation=eval_results['correlation'],
            mae=eval_results['mae'],
            rmse=eval_results['rmse'],
            sign_disagreement_rate=eval_results['sign_disagreement_rate'],
            overestimation_rate=eval_results['overestimation_rate'],
            n_seqs=window_stats_dict['seq_counts'],
            n_cases=window_stats_dict['case_counts'],
            seq_entropy=window_stats_dict['seq_entropy'],
            case_entropy=window_stats_dict['case_entropy'],
            seq_entropy_norm=window_stats_dict['seq_norm_entropy'],
            case_entropy_norm=window_stats_dict['case_norm_entropy']
        )
        
        # Create variant results
        variant_results = []
        for _, row in variant_mae_df.iterrows():
            variant_result = VariantResult(
                variant=row['variant'],
                pivot_date=pivot_date,
                model=model,
                location=location,
                mae=row['mae'],
                normalized_mae=row['normalized_mae'],
                max_r_data=row['max_r_data'],
                correlation=row['correlation'],
                sign_disagreement_rate=row['sign_disagreement_rate'],
                overestimation_rate=row['overestimation_rate'],
                n_points=row['n_points'],
                total_sequences=row['total_sequences'],
                total_smoothed_sequences=row['total_smoothed_sequences'],
                mean_variant_frequency=row['mean_variant_frequency'],
                mean_smoothed_variant_frequency=row['mean_smoothed_variant_frequency'],
                max_variant_frequency=row['max_variant_frequency'],
                max_smoothed_variant_frequency=row['max_smoothed_variant_frequency']
            )
            variant_results.append(variant_result)
        
        # Load convergence diagnostics if configured
        convergence_diagnostic = None
        if conv_config.check_diagnostics and model in conv_config.required_models:
            diagnostics_path = Path(f"results/{build}/convergence-diagnostics/{model}_{location}_{pivot_date}_vi_diagnostics.json")
            diag_data = load_convergence_diagnostics(diagnostics_path)
            
            if diag_data is not None:
                # Extract convergence data
                inf_settings = diag_data.get('inference_settings', {})
                elbo_traj = diag_data.get('convergence_diagnostics', {}).get('elbo_trajectory', {})
                conv_status = diag_data.get('convergence_diagnostics', {}).get('convergence', {})
                
                # Determine convergence status
                relative_change = conv_status.get('relative_change')
                converged = None
                if relative_change is not None:
                    converged = relative_change <= conv_config.threshold
                
                convergence_diagnostic = ConvergenceDiagnostic(
                    pivot_date=pivot_date,
                    model=model,
                    location=location,
                    inference_method=diag_data.get('inference_method'),
                    iterations=inf_settings.get('iterations'),
                    learning_rate=inf_settings.get('learning_rate'),
                    num_samples=inf_settings.get('num_samples'),
                    num_iterations=elbo_traj.get('num_iterations'),
                    initial_loss=elbo_traj.get('initial_loss'),
                    final_loss=elbo_traj.get('final_loss'),
                    min_loss=elbo_traj.get('min_loss'),
                    total_improvement=elbo_traj.get('total_improvement'),
                    converged=converged,
                    relative_change=relative_change,
                    threshold=conv_status.get('threshold'),
                    window=conv_status.get('window'),
                    final_iteration=conv_status.get('final_iteration')
                )
        
        return window_result, variant_results, convergence_diagnostic
        
    except Exception as e:
        logger.error(f"Error processing {model}_{location}_{pivot_date}: {e}")
        return None, [], None


def process_all_model_results(
    growth_config: GrowthRateConfig,
    conv_config: ConvergenceConfig, 
    build: str, 
    output_dir: Path,
    base_dir: str = "."
) -> GrowthRateResultsCollector:
    """
    Discover and process all rt_*.tsv files for comprehensive growth rate analysis.
    
    Uses modular components from antigentools.scoring to process RT files,
    apply configuration-based filtering, and collect results in type-safe dataclasses.
    
    Parameters
    ----------
    growth_config : GrowthRateConfig
        Growth rate analysis configuration
    conv_config : ConvergenceConfig
        Convergence diagnostics configuration
    build : str
        Build name to process
    output_dir : Path
        Output directory for results
    base_dir : str
        Base directory for file discovery
        
    Returns
    -------
    GrowthRateResultsCollector
        Collected results from all processed windows
    """
    # Discover RT files using our modular system
    models = conv_config.required_models
    rt_files = discover_rt_files(build, models, base_dir=base_dir)
    
    logger.info(f"Discovered {len(rt_files)} RT files for models: {models}")
    
    # Initialize results collector
    collector = GrowthRateResultsCollector()
    
    # Track processing statistics
    total_windows = len(rt_files)
    skipped_windows = 0
    processed_windows = 0
    
    # Process each RT file
    for i, rt_file in enumerate(rt_files, 1):
        logger.info(f"Progress: {i}/{total_windows}")
        
        # Process the window
        window_result, variant_results, convergence_diagnostic = process_rt_window(
            rt_file, growth_config, conv_config, build, base_dir
        )
        
        # Collect results
        if window_result is not None:
            collector.add_window_result(window_result)
            processed_windows += 1
            
            # Export individual growth rates data
            growth_rates_df = get_filtered_growth_rates_df(
                build, rt_file.model, rt_file.location, rt_file.pivot_date, 
                spline_smoothing_factor=growth_config.spline_smoothing_factor, 
                spline_order=growth_config.spline_order, 
                min_sequence_count=growth_config.min_sequence_count, 
                min_variant_frequency=growth_config.min_variant_frequency,
                skip_first_n_points=growth_config.skip_first_n_points,
                use_freqs=growth_config.use_freqs,
                use_smoothed_incidence=growth_config.use_smoothed_incidence,
                data_path="data/"
            )
            export_growth_rates_data(
                growth_rates_df, output_dir, rt_file.model, rt_file.location, rt_file.pivot_date
            )
        else:
            skipped_windows += 1
        
        # Add variant results
        for variant_result in variant_results:
            collector.add_variant_result(variant_result)
        
        # Add convergence diagnostic
        if convergence_diagnostic is not None:
            collector.add_convergence_diagnostic(convergence_diagnostic)
    
    # Log summary statistics
    logger.info("=== ANALYSIS SUMMARY ===")
    logger.info(f"Total analysis windows considered: {total_windows}")
    logger.info(f"Windows skipped (sequence filter): {skipped_windows}")
    logger.info(f"Windows successfully processed: {processed_windows}")
    
    stats = collector.summary_stats()
    logger.info(f"Final results: {stats['total_windows']} windows, {stats['total_variants']} variants, {stats['total_diagnostics']} diagnostics")
    
    if stats['total_diagnostics'] > 0:
        convergence_count = sum(1 for d in collector.convergence_diagnostics if d.converged)
        total_with_data = sum(1 for d in collector.convergence_diagnostics if d.converged is not None)
        if total_with_data > 0:
            logger.info(f"Convergence rate: {convergence_count/total_with_data:.2%} ({convergence_count}/{total_with_data})")
    
    # Convert to DataFrames and save
    window_df, variant_df, diagnostics_df = collector.to_dataframes()
    
    # Save results
    window_path = output_dir / "window_growth_rates.tsv"
    variant_path = output_dir / "variant_growth_rates.tsv"
    diagnostics_path = output_dir / "vi_convergence_diagnostics.tsv"
    
    window_df.to_csv(window_path, index=False, sep='\t')
    variant_df.to_csv(variant_path, index=False, sep='\t')
    diagnostics_df.to_csv(diagnostics_path, index=False, sep='\t')
    
    logger.info("Results saved to:")
    logger.info(f"  {window_path}")
    logger.info(f"  {variant_path}")
    logger.info(f"  {diagnostics_path}")
    
    return collector


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Quiet noisy modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Comprehensive growth rate analysis and scoring.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to the configuration file.')
    parser.add_argument('--build', type=str, required=True,
                       help='Build name (e.g., flu-simulated-150k-samples).')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save the analysis outputs.')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging.')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Convert paths
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and parse configuration
    try:
        config_dict = load_config(config_path)
        growth_config, conv_config = parse_growth_rate_config(config_dict)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Log configuration
    logger.info("Growth Rate Configuration:")
    logger.info(f"  Min sequence count: {growth_config.min_sequence_count}")
    logger.info(f"  Min variant frequency: {growth_config.min_variant_frequency}")
    logger.info(f"  Min total sequences: {growth_config.min_total_sequences}")
    logger.info(f"  Epsilon: {growth_config.epsilon}")
    logger.info(f"  Convergence threshold: {conv_config.threshold}")
    
    # Process all model results
    try:
        collector = process_all_model_results(growth_config, conv_config, args.build, output_dir)
        logger.info("Analysis completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())