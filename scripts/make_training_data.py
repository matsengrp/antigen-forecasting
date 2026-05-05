#!/usr/bin/env python3
"""
make_training_data.py

This script processes variant and case count data to generate training windows for forecasting.
It reads in sequence and case count data, splits them into predefined time windows, and saves
the processed datasets to an output directory. Additionally, it generates a configuration YAML
file specifying estimation dates, locations, and models.

Usage:
    python make_training_data.py -s sequences.tsv -c cases.tsv -o output_dir [--window-size 365] [--buffer-size 0] [--config-path configs/benchmark_config.yaml] [-n] [--deme_population_size 30000000]

Required Arguments:
    -s, --sequences        Path to the variant counts file (TSV format).
    -c, --cases            Path to the case counts file (TSV format).
    -o, --output           Path to the output directory where processed files will be saved.

Optional Arguments:
    --window-size          Size of the training window in days (default: 365).
    --buffer-size          Number of days to remove from the training set before the analysis date (default: 0).
    --config-path          Path to save the configuration YAML file (default: 'configs/benchmark_config.yaml').

Processing Steps:
    1. Load variant and case count data from TSV files.
    2. Identify analysis dates (April 1 and October 1 of each year).
    3. Generate training windows using specified start and end dates.
    4. Save processed training windows to the output directory.
    5. Generate and save a configuration YAML file.

Outputs:
    - Processed sequence counts (`seq_counts.tsv`) and case counts (`case_counts.tsv`) per training window.
    - A configuration file (`configs/benchmark_config.yaml`) defining estimation dates, locations, and models.

Dependencies:
    - Python 3.x
    - pandas
    - numpy
    - PyYAML (for YAML configuration)

Author: Zorian Thornton (@zorian15)
Date: 2025-02-06
"""
# Package imports
import argparse
import pandas as pd
import os
import yaml

def generate_config_yaml(locations: list, dates: list, models: list, out_path: str) -> None:
    """ Generate a configuration file for the training data.
    
    Parameters
    ----------
    locations : list
        The list of locations to include in the configuration file.
    dates : list
        The list of dates to include in the configuration file.
    models : list
        The list of models to include in the configuration file.
    out_path : str
        The path to save the configuration file.
    """
    # Define the configuration dictionary
    config = {
        'main': {  
            'estimation_dates': dates,
            'locations': locations,
            'models': models
        }
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration file saved to {out_path}.")

def get_analysis_start_date(analysis_date: pd.Timestamp, n_days: int=180) -> pd.Timestamp:
    """
    Get the start date for the analysis -- reach back in the past n_days.

    Parameters
    ----------
    analysis_date : pd.Timestamp
        The date of the analysis.
    n_days : int
        The number of days to reach back in the past.

    Returns
    -------
    pd.Timestamp
        The start date for the analysis.
    """
    start_date = analysis_date - pd.Timedelta(days=n_days)
    return start_date

def get_analysis_end_date(analysis_date: pd.Timestamp, n_days: int=180) -> pd.Timestamp:
    """ Obtain end date of an analysis period by removing n_days from the analysis_date (bias correction).

    Parameters
    ------------
    analysis_date : pd.Timestamp
        Analysis date.
    n_days : int
        Number of days to remove from the analysis_date.

    Returns
    -----------
    end_date : pd.Timestamp
        End date of the analysis period (i.e., last day of samples to include in the analysis folder).
    """
    end_date = analysis_date - pd.Timedelta(days=n_days)
    return end_date

def main(args) -> None:
    """ Main function to run the data preparation script."""
    seqs_path = args.sequences
    cases_path = args.cases
    out_dir = args.output
    window_size = args.window_size
    buffer_size = args.buffer_size
    config_path = args.config_path

    # Load the data
    seqs_df = pd.read_csv(seqs_path, sep='\t')
    cases_df = pd.read_csv(cases_path, sep='\t')

    # Grab min and max dates
    min_date = seqs_df['date'].min()
    max_date = seqs_df['date'].max()

    all_dates = pd.date_range(min_date, max_date, freq='D')

    # Analysis dates are first day of April and October
    analysis_dates = all_dates[( (all_dates.month == 4) | (all_dates.month == 10) )& (all_dates.day == 1) ]

    # Dump the configuration file
    locations = seqs_df['country'].unique().tolist()
    dates = analysis_dates.strftime('%Y-%m-%d').tolist()
    models = ["NAIVE", "MLR", "FGA", "GARW"]
    generate_config_yaml(locations, dates, models, config_path)

    seqs_df['date'] = pd.to_datetime(seqs_df['date'])
    cases_df['date'] = pd.to_datetime(cases_df['date'])

    # Loop through the analysis dates and dump the data; collect manifest rows.
    manifest_rows = []
    for timepoint in analysis_dates:
        # Get the start and end dates for the analysis
        start_date = get_analysis_start_date(timepoint, n_days=window_size)
        end_date = get_analysis_end_date(timepoint, n_days=buffer_size)

        # Filter the data to the analysis window
        seqs_window = seqs_df[(seqs_df['date'] >= start_date) & (seqs_df['date'] <= end_date)]
        cases_window = cases_df[(cases_df['date'] >= start_date) & (cases_df['date'] <= end_date)]

        # Save the data
        if len(seqs_window) > 0:
            analysis_date = timepoint.strftime('%Y-%m-%d')
            out_path = os.path.join(out_dir, analysis_date)

            os.makedirs(out_path, exist_ok=True)

            seq_path = os.path.join(out_path, 'seq_counts.tsv')
            case_path = os.path.join(out_path, 'case_counts.tsv')
            seqs_window.to_csv(seq_path, sep='\t', index=False)
            cases_window.to_csv(case_path, sep='\t', index=False)
            manifest_rows.append({
                'date': analysis_date,
                'seq_counts_path': seq_path,
                'case_counts_path': case_path,
            })

    # An empty manifest must not appear on disk: a header-only TSV would still
    # satisfy run_pipeline.py's sentinel check on re-run and silently skip
    # legitimately incomplete state.
    if not manifest_rows:
        raise RuntimeError(
            f"No analysis dates produced training windows from {seqs_path}; "
            f"expected at least one April-1 / October-1 date inside the input range"
        )

    # Write MANIFEST.tsv last so its presence proves all per-date dirs are complete.
    # Used as the run_pipeline.py sentinel for this step.
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, 'MANIFEST.tsv')
    pd.DataFrame(
        manifest_rows, columns=['date', 'seq_counts_path', 'case_counts_path']
    ).to_csv(manifest_path, sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split variant and count data into training windows.")
    
    # Required arguments
    parser.add_argument("-s", "--sequences", type=str, help="Path to the variant counts file.")
    parser.add_argument("-c", "--cases", type=str, help="Path to the case counts file.")
    parser.add_argument("-o", "--output", type=str, help="Path to the output directory.")

    # Optional arguments
    parser.add_argument("--window-size", type=int, default=365, help="Size of the training window in days.")
    parser.add_argument("--buffer-size", type=int, default=0, help="Number of days to remove from the training set before the analysis date.")
    parser.add_argument("--config-path", type=str, default='configs/benchmark_config.yaml', help="Path to dump the training configuration file.")

    
    # Parse the arguments and run program
    args = parser.parse_args()
    main(args)