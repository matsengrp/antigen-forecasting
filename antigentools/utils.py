import glob
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress, gamma

def read_estimates(paths: list) -> pd.DataFrame:
    """ Read evofr forecasting estimates from a list of paths.

    Parses the pivot date from the path to add to the dataframe.

    Parameters:
    ---------------
        paths (list): List of paths to read

    Returns:
    ---------------
        pd.DataFrame: Dataframe of estimates
    """
    dfs = []
    for path in paths:
        pivot_date = path.split("/")[-1].split("_")[-1].split(".")[0]
        df = pd.read_csv(path, sep="\t")
        df['pivot_date'] = pivot_date
        dfs.append(df)
    return pd.concat(dfs)


def map_dates(df, reverse_mapping: bool=False):
    """ Map dates in dataframe to time index.

    Parameters:
    ---------------
        df (pd.DataFrame): Dataframe with column 'date'
        reverse_mapping (bool): If True, return a mapping from index to date

    Returns:
    ---------------
        dict: Dictionary mapping dates to time indices (or vice versa)
    """
    date_mapping = {date: t for t, date in enumerate(sorted(df['date'].unique()))}
    if reverse_mapping:
        date_mapping = {value: key for key, value in date_mapping.items()}
    return date_mapping


def filter_to_lifespan(df: pd.DataFrame, variant: str, deme: str, lifespan: tuple, time_col_name: str='t') -> pd.DataFrame:
    """ Pre-process fitness dataframe to set all fitness values outside of the lifespan of a variant to NaN.

    Parameters:
    ---------------
        df (pd.DataFrame): Dataframe with columns 'time', 'fitness', 'variant'
        variant (str): Variant name
        deme (str): Deme/location name
        lifespan (tuple): Tuple of start and end dates
        time_col_name (str): Name of time column

    Returns:
    ---------------
        pd.DataFrame: Filtered dataframe
    """
    start, end = lifespan
    # Set fitness values of variant outside of lifespan to NaN
    var_df = df.query(f"variant == {variant} and location == '{deme}'")
    # Set fitness values outside of lifespan to NaN
    var_df.loc[(var_df[time_col_name] < start) | (var_df[time_col_name] > end), 'fitness'] = np.nan
    var_df.loc[(var_df[time_col_name] < start) | (var_df[time_col_name] > end), 'seasonal_fitness'] = np.nan

    return var_df


def prune_fitness_dataframe(fitness_df: pd.DataFrame, lifespan_df: pd.DataFrame, var_col_name: str='variant', 
                           loc_col_name: str='location', time_col_name: str='t') -> pd.DataFrame:
    """ Prune fitness dataframe to only include data within lifespan.

    Parameters:
    ---------------
        fitness_df (pd.DataFrame): Dataframe with columns 'time', 'fitness', 'variant'
        lifespan_df (pd.DataFrame): Dataframe with columns 'variant', 'birth', 'death'
        var_col_name (str): Name of variant column
        loc_col_name (str): Name of location column
        time_col_name (str): Name of time column

    Returns:
    ---------------
        pruned_df (pd.DataFrame): Dataframe
    """
    variant_dfs = []
    for variant in fitness_df[var_col_name].unique():
        for deme in fitness_df[loc_col_name].unique():
            # Get lifespan of variant
            lifespan = lifespan_df.query(f"variant == {variant} and location == '{deme}'")[['birth', 'death']].values[0]
            # Prune dataframe
            pruned_df = filter_to_lifespan(fitness_df, variant, deme, lifespan, time_col_name)
            variant_dfs.append(pruned_df)
    pruned_df = pd.concat(variant_dfs)
    return pruned_df

