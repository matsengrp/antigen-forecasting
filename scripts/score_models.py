#!/usr/bin/env python
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
    python score_models.py --config config.yaml --truth-set truth_set.tsv \\
                          --estimates-path estimates --output-path scores.tsv \\
                          [--verbose] [--log-file log.txt]

Required Arguments:
    --config            Path to the configuration file with scoring parameters
    --truth-set         Path to the truth set of sequences
    --estimates-path    Path to the model estimates
    --output-path       Path to save the output scores

Optional Arguments:
    --verbose           Enable debug-level logging
    --log-file          Path to save log messages to file (in addition to console)

Configuration (in YAML):
    main:
      estimation_dates: [2027-01-01, 2027-02-01, ...]
      locations: [north, south, tropics]
      models: [MLR, FGA, GARW, NAIVE]
    
    scoring:
      min_frequency_threshold: 0.01  # Minimum frequency to include (1%)
      active_window_days: 90         # Look back window for active variants
      min_sequences: 10              # Minimum sequences for a variant
      min_observations: 3            # Minimum observations in window
      handle_missing_smoothed: true  # Filter missing smoothed frequencies
      smoothing_window: 7            # Window size for frequency smoothing

Output:
    - scores.tsv: Model scores with filtering applied

Dependencies:
    - Python 3.x with: pandas, numpy, pyyaml, scipy

Author: Zorian Thornton (@zorian15)
Date: 2025-02-06
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import sys

# Import from the new modular structure
from antigentools.scoring import (
    load_data, load_truthset,
    filter_active_variants,
    merge_truth_pred, calculate_errors
)
from antigentools.scoring.config import load_config, parse_config

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None):
    """
    Set up logging configuration for console and file output.
    
    Parameters
    ----------
    verbose : bool, default=False
        Enable debug-level logging
    log_file : Optional[Path], default=None
        Path to log file. If None, logs only to console
    """
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set logging level
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(level)
    logging.root.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
        
        # Log that file logging is enabled
        logger.info(f"Logging to file: {log_file}")
    
    # Set levels for noisy modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute model scores with variant filtering.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default="../configs/benchmark_config.yaml",
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--truth-set', 
        type=str,
        required=True,
        help='Path to the truth set of sequences.'
    )
    parser.add_argument(
        '--estimates-path', 
        type=str,
        required=True,
        help='Path to the estimates directory.'
    )
    parser.add_argument(
        '--output-path', 
        type=str,
        required=True,
        help='Path to save the output scores.'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging.'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file. If not specified, logs only to console.'
    )
    
    return parser.parse_args()


def load_model_predictions(
    filepath: Path,
    model: str,
    location: str,
    pivot_date: str
) -> Optional[pd.DataFrame]:
    """
    Load model predictions with fallback to CSV format.
    
    Parameters
    ----------
    filepath : Path
        Base path to predictions file
    model : str
        Model name
    location : str
        Location name
    pivot_date : str
        Analysis date
        
    Returns
    -------
    Optional[pd.DataFrame]
        Loaded predictions or None if not found
    """
    # Try TSV first
    tsv_path = filepath.with_suffix('.tsv')
    if tsv_path.exists():
        try:
            return load_data(tsv_path, sep='\t')
        except Exception as e:
            logger.error(f"Error loading {tsv_path}: {e}")
            return None
    
    # Try CSV as fallback
    csv_path = filepath.with_suffix('.csv')
    if csv_path.exists():
        try:
            return load_data(csv_path, sep=',')
        except Exception as e:
            logger.error(f"Error loading {csv_path}: {e}")
            return None
    
    logger.warning(
        f"No {model} predictions found for {location} on {pivot_date}"
    )
    return None


def process_model_location_date(
    model: str,
    location: str,
    pivot_date: str,
    location_truth: pd.DataFrame,
    estimates_path: Path,
    main_config,
    scoring_config
) -> Optional[pd.DataFrame]:
    """
    Process scoring for a single model/location/date combination.
    
    Parameters
    ----------
    model : str
        Model name
    location : str
        Location name
    pivot_date : str
        Analysis date
    location_truth : pd.DataFrame
        Truth data for the location
    estimates_path : Path
        Base path to estimates
    main_config : MainConfig
        Main configuration
    scoring_config : ScoringConfig
        Scoring configuration
        
    Returns
    -------
    Optional[pd.DataFrame]
        Error dataframe or None if no valid data
    """
    # Construct file path
    filepath = estimates_path / model / f"freq_{location}_{pivot_date}"
    
    # Load predictions
    raw_pred = load_model_predictions(filepath, model, location, pivot_date)
    if raw_pred is None:
        return None
    
    # Merge with truth data
    merged = merge_truth_pred(raw_pred, location_truth)
    
    # Track filtering progress
    original_size = len(merged)
    logger.info(f"\nProcessing {model} {location} {pivot_date}:")
    logger.info(f"  Initial merge: {original_size} rows")
    
    # Apply active variant filtering if configured
    if scoring_config.active_window_days is not None:
        merged = filter_active_variants(
            merged,
            pivot_date=pivot_date,
            lookback_days=scoring_config.active_window_days,
            min_observations=scoring_config.min_observations,
            min_sequences=scoring_config.min_sequences
        )
        active_filtered_size = len(merged)
        logger.info(f"  Active variant filter: {original_size} → {active_filtered_size} rows")
    
    # Calculate errors with additional filtering
    error_df = calculate_errors(
        merged,
        pivot_date,
        country=location,
        model=model,
        min_freq_threshold=scoring_config.min_frequency_threshold,
        handle_missing_smoothed=scoring_config.handle_missing_smoothed
    )
    
    if error_df is None:
        logger.warning(f"  No valid data after all filtering")
        return None
    
    final_size = len(error_df)
    logger.info(f"  Final size after all filters: {final_size} rows")
    
    return error_df


def main(args):
    """Main execution function."""
    # Setup logging
    log_file_path = Path(args.log_file) if args.log_file else None
    setup_logging(verbose=args.verbose, log_file=log_file_path)
    
    # Convert paths
    config_path = Path(args.config)
    truth_path = Path(args.truth_set)
    estimates_path = Path(args.estimates_path)
    output_path = Path(args.output_path)
    
    # Load and parse configuration
    try:
        config_dict = load_config(config_path)
        main_config, scoring_config = parse_config(config_dict)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("Scoring Configuration:")
    logger.info(f"  Min frequency threshold: {scoring_config.min_frequency_threshold}")
    logger.info(f"  Active window days: {scoring_config.active_window_days}")
    logger.info(f"  Min sequences: {scoring_config.min_sequences}")
    logger.info(f"  Min observations: {scoring_config.min_observations}")
    logger.info(f"  Handle missing smoothed: {scoring_config.handle_missing_smoothed}")
    logger.info(f"  Smoothing window: {scoring_config.smoothing_window}")
    logger.info("=" * 60)
    
    # Load truth set
    try:
        logger.info(f"\nLoading truth set from {truth_path}")
        truth_set = load_truthset(truth_path)
    except Exception as e:
        logger.error(f"Error loading truth set: {e}")
        sys.exit(1)
    
    # Process all model/location/date combinations
    score_df_list = []
    total_combinations = len(main_config.models) * len(main_config.locations) * len(main_config.estimation_dates)
    processed = 0
    
    for model in main_config.models:
        for location in main_config.locations:
            # Filter truth set to location
            location_truth = truth_set[truth_set["country"] == location]
            
            if len(location_truth) == 0:
                logger.warning(f"{location} not found in truth set")
                continue
            
            for pivot_date in main_config.estimation_dates:
                processed += 1
                logger.info(f"\nProgress: {processed}/{total_combinations}")
                
                error_df = process_model_location_date(
                    model, location, pivot_date,
                    location_truth, estimates_path,
                    main_config, scoring_config
                )
                
                if error_df is not None:
                    score_df_list.append(error_df)
    
    # Check if we have any results
    if not score_df_list:
        logger.error("No scores computed for any model/location/date combination")
        sys.exit(1)
    
    # Combine all scores
    logger.info(f"\nCombining {len(score_df_list)} score dataframes")
    score_df = pd.concat(score_df_list, ignore_index=True)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(output_path, sep="\t", index=False)
    
    # Log summary statistics
    logger.info("=" * 60)
    logger.info("Scoring Summary:")
    logger.info(f"  Total scores: {len(score_df)}")
    logger.info(f"  Models: {score_df['model'].nunique()}")
    logger.info(f"  Locations: {score_df['country'].nunique()}")
    logger.info(f"  Analysis dates: {score_df['pivot_date'].nunique()}")
    logger.info(f"  Variants: {score_df['variant'].nunique()}")
    logger.info(f"  Output saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)