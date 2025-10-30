#!/usr/bin/env python3
"""
prep_antigen_data.py

This script processes antigen-prime simulated data for variant frequency forecasting. 
It reads in observed "tips" and "cases" data, processes them, and outputs cleaned datasets 
for forecasting.

Usage:
    python prep_antigen_data.py -t tips.csv -c cases.csv -o output_dir -sd 2025-01-01 -v variant [-n] [--deme_population_size 30000000]

Required Arguments:
    -t, --tips          Path to a variant-assigned tips file (CSV format).
    -c, --cases         Path to the antigen-prime raw summary/cases file (CSV format).
    -o, --output        Output directory for preprocessed sequence and case count files.

Optional Arguments:
    -sd, --start_date   Start date for the analysis (default: 2025-01-01).
    -v, --variant-col-name  Name of the column containing the variant information (default: 'variant').
    -n, --normalize_cases   Normalize case counts to report cases per 100k hosts (default: False).
    --deme_population_size  Population size of the deme (default: 30,000,000).

Output:
    - seq_counts.tsv    Processed sequence count data, containing columns: `date`, `country`, `variant`, `sequences`.
    - case_counts.tsv   Processed case count data, containing columns: `date`, `country`, `cases`.

Dependencies:
    - Requires Python 3.x
    - pandas
    - numpy

Author: Zorian Thornton (@zorian15)
Date: 2025-02-06
"""

# Package imports
import datetime
import os
import argparse

import pandas as pd
import numpy as np
import glob as glob
from pandas.tseries.offsets import Day, BDay

def normalize_case_counts(cases_df: pd.DataFrame, deme_population_size: int = 30_000_000) -> pd.DataFrame:
    """ Normalize case counts to report cases per 100k hosts.

    Parameters
    ----------
    cases_df : pd.DataFrame
        The cases dataframe to be normalized.
    deme_population_size : int
        The population size of the deme.

    Returns
    -------
    cases_df : pd.DataFrame
        The normalized cases dataframe where reported cases are per 100k hosts.
    """
    normalized_df = cases_df.copy()
    normalized_df['northCases'] = ((normalized_df['northCases'] / deme_population_size) * 100_000).astype(int)
    normalized_df['southCases'] = ((normalized_df['southCases'] / deme_population_size) * 100_000).astype(int)
    normalized_df['tropicsCases'] = ((normalized_df['tropicsCases'] / deme_population_size) * 100_000).astype(int)

    return normalized_df

def prep_tips_df(tips_df: pd.DataFrame, start_date: pd.Timestamp, variant_col: str = 'variant') -> pd.DataFrame:
    """ Prepare the tips dataframe for the forecasting analysis.
    
    Parameters
    ----------
    tips_df : pd.DataFrame
        The tips dataframe to be prepared.
    start_date : pd.Timestamp
        The start date for the analysis.
    variant_col : str
        The name of the column in tips_df containing the variant information
        
    Returns
    -------
    variant_counts_df : pd.DataFrame
        The prepared tips dataframe with columns: `date`, `country`, `variant`, and `sequences.
    """
    if 'country' not in tips_df.columns:
        demes = {0:"north", 1:"tropics", 2:"south"}
        tips_df['country'] = tips_df['location'].map(demes)
    
    # Convert the 'year' column to datetime format
    tips_df['date'] = start_date + pd.to_timedelta(tips_df['year'] * 365.25, unit='D')

    # Adjust dates to the last day of the week
    tips_df['date'] = tips_df['date'] + pd.offsets.Week(weekday=6)

    # Format the new date column to '%Y-%m-%d'
    tips_df['date'] = tips_df['date'].dt.strftime('%Y-%m-%d')

    variant_counts_df = tips_df.groupby(['date', 'country', variant_col]).size().reset_index(name='sequences')
    
    # Rename the 'variant' column to 'variant'
    variant_counts_df = variant_counts_df.rename(columns={variant_col: 'variant'})
    
    return variant_counts_df

def prep_cases_df(cases_df: pd.DataFrame, start_date: pd.Timestamp, normalize_cases: bool = False, deme_population_size: int = 30_000_000) -> pd.DataFrame:
    """ Prepare the cases dataframe for the forecasting analysis.
    
    Parameters
    ----------
    cases_df : pd.DataFrame
        The cases dataframe to be prepared.
    start_date : pd.Timestamp
        The start date for the analysis.
    normalize_cases : bool
        Whether to normalize case counts to report cases per 100k hosts.
    deme_population_size : int
        The population size of the deme for normalization.

    Returns
    -------
    case_counts_df : pd.DataFrame
        The prepared cases dataframe with columns: `date`, `country`, and `cases`.
    """
    if 'year' not in cases_df.columns and 'date' in cases_df.columns:
        cases_df.rename(columns={'date': 'year'}, inplace=True)
    # Normalize case counts if specified
    if normalize_cases:
        print("Normalizing case counts to report cases per 100k hosts...")
        cases_df = normalize_case_counts(cases_df, deme_population_size)
    # Round dates for smoother date assignments
    cases_df['rounded_year'] = cases_df['year'].apply(lambda x: round(x,3))

    # Convert the floating point decimals to days and then add to the reference date
    cases_df['date'] = start_date + pd.to_timedelta(cases_df['rounded_year'] * 365.25, unit='D')

    # Convert datetime objects to the desired format (yyyy-mm-dd)
    cases_df['date'] = cases_df['date'].dt.strftime('%Y-%m-%d')
    cases_df['date'] = pd.to_datetime(cases_df['date'])
    
    # Melt the DataFrame
    cases_melted = pd.melt(cases_df, id_vars=['date'], value_vars=['northCases', 'southCases', 'tropicsCases'], var_name='country', value_name='cases')

    # Replace 'Cases' in country names with an empty string
    cases_melted['country'] = cases_melted['country'].str.replace('Cases', '')

    case_counts_df = cases_melted.groupby(['date', 'country']).sum().reset_index()

    return case_counts_df

def main(args) -> None:
    """ Main function to run the data preparation script.
    """
    # Get arguments
    tips_path = args.tips
    cases_path = args.cases
    output_dir = args.output
    start_date = pd.Timestamp(args.start_date)
    variant_col = args.variant_col_name
    normalize_cases = args.normalize_cases
    deme_population_size = args.deme_population_size

    # Load the tips and cases data (auto-detect separator based on file extension)
    tips_sep = '\t' if tips_path.endswith('.tsv') else ','
    cases_sep = '\t' if cases_path.endswith('.tsv') else ','
    tips_df = pd.read_csv(tips_path, sep=tips_sep)
    cases_df = pd.read_csv(cases_path, sep=cases_sep)
    
    # Prepare the sequence and case count dataframes
    variant_counts_df = prep_tips_df(tips_df, start_date, variant_col)
    case_counts_df = prep_cases_df(cases_df, start_date, normalize_cases, deme_population_size)

    # Save the prepared dataframes
    os.makedirs(output_dir, exist_ok=True)
    variant_counts_df.to_csv(os.path.join(output_dir, 'seq_counts.tsv'), sep='\t', index=False)
    case_counts_df.to_csv(os.path.join(output_dir, 'case_counts.tsv'), sep='\t', index=False)
    


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Prepare antigen-prime simulated data for variant frequency forecasting.")
    
    # Required arguments: observed tips and cases files
    parser.add_argument("-t", "--tips", type=str, help="Path to the tips file.")
    parser.add_argument("-c", "--cases", type=str, help="Path to the cases file.")
    parser.add_argument("-o", "--output", type=str, help="Path to the output directory for prepped sequence and case counts.")

    # Optional arguments
    parser.add_argument("-sd", "--start_date", type=str, default='2025-01-01' ,help="Start date for the analysis.")
    parser.add_argument("-v", "--variant-col-name", type=str, default='variant', help="Name of the column containing the variant information.")
    parser.add_argument("-n", "--normalize_cases", action='store_true', help="Normalize case counts to report cases per 100k hosts.")
    parser.add_argument("--deme_population_size", type=int, default=30_000_000, help="Population size of the deme.")
    
    # Parse arguments and run the program
    args = parser.parse_args()
    main(args)

    # Print success message
    print(f"Data preparation complete, output saved to {args.output}.")