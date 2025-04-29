"""
Calculate variant fitness.

This script calculates the fitness of each variant in the tips dataframe over time. 
The fitness is calculated based on the average risk of infection of the variant's centroid and the immune memories of the hosts. 
The script loads the tips dataframe, immune memory file, and calculates the fitness of each variant at each timepoint and deme. 
The script also calculates the centroid of each variant and saves it to a file if the `--centroid-output` argument is provided.

Usage:
    python calc_variant_fitness.py -t tips.csv -i immune_memory.txt -o variant_fitness.csv [--centroid-output variant_centroids.csv] [--burnin 10.0] [--variant-col variant]

Required Arguments:
    -t, --tips          Path to the variant-assigned tips dataframe.
    -i, --immunity      Path to the immune memory file.
    -o, --output        Path to save the variant fitness dataframe.

Optional Arguments:
    --centroid-output    Path to save the variant centroid dataframe.
    --burnin            Burn-in period in years (default: 10.0).
    --variant-col       Column name of the variant in the tips dataframe (default: 'variant').

Output:
    - variant_fitness.csv       Variant fitness dataframe containing columns: `variant`, `time`, `deme`, `fitness`.
    - variant_centroids.csv     Variant centroid dataframe containing columns: `variant`, `ag1`, `ag2`, `birth`, `death`.

Dependencies:
    - Requires Python 3.x
    - pandas
    - numpy
    - joblib

Author: Zorian Thornton (@zorian15)
Date: 2025-02-06
"""
## Module imports
import argparse
import json
import os
import time
import glob as glob
from typing import Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Default params used in antigen simulations -- used for calculating contact rates in fitness calculations
ANTIGEN_PARAMS = {
    'delta_t': 0.01,
    'beta': 0.36,
    'between_deme_prob': 0.001,
}

## Helper functions
def get_closest_virus_distance(virus: pd.DataFrame, immune_memory: list) -> float:
    """
    Given a set of antigenic coordinates for a virus and those in an immune memory, return the distance of the closest virus in the immune memory.

    Parameters
    ----------
    virus : pd.DataFrame
        The antigenic coordinates of the virus.
    immune_memory : list
        A list of np.arrays representing the antigenic coordinates of viruses in the immune memory.

    Returns
    -------
    dist : float
        The euclidean distance of the closest virus in the immune memory.
    """
    if len(immune_memory) == 0:
        return float('inf')

    # Stack the immune memory coordinates for vectorized distance calculation
    immune_memory_np = np.vstack(immune_memory)
    virus_coords = virus[['ag1', 'ag2']].to_numpy()

    # Calculate euclidean distances and return the minimum
    distances = np.linalg.norm(immune_memory_np - virus_coords, axis=1)
    
    return np.min(distances)

def infection_risk(distance: float, smith_conversion: float = 0.07, homologous_immunity: float = 0.95) -> float:
    """
    Calculate the risk of infection given an antigenic distance, smith conversion multiplier, and homologous immunity.
    
    Parameters
    ----------
    distance : float
        The antigenic distance between the virus and the host.
    smith_conversion : float
        The smith conversion multiplier for cross-immunity.
    homologous_immunity : float
        The immunity raised against an antigenically identical virus.
    """
    if distance == np.inf:
        return 1.0
    min_risk = 1.0 - homologous_immunity
    risk = distance * smith_conversion
    risk = max(min_risk, risk)
    return min(1.0, risk)

def get_variant_centroid(variant_df: pd.DataFrame, variant: str, col_name: str = 'variant') -> dict:
    """
    Given a variant and a dataframe of viruses, return the centroid of the variant.

    Parameters
    ----------
    variant_df : pd.DataFrame
        The dataframe of viruses.
    variant : str
        The variant to find the centroid of.
    col_name : str
        The column name of the variant in the dataframe.

    Returns
    -------
    dict
        A dictionary with variant:centroid key-value pair.
    """
    if 'year' in variant_df.columns:
        variant_df.rename(columns={'year': 'date'}, inplace=True)
    subset_df = variant_df[variant_df[col_name] == variant]
    min_date = subset_df['date'].min()
    max_date = subset_df['date'].max()
    centroid = subset_df[['ag1', 'ag2']].mean().values
    return {col_name: variant, 'ag1': centroid[0], 'ag2': centroid[1], 'birth': min_date, 'death': max_date}

def parse_host(host_string: str) -> list:
    """
    Parse the host string into a numpy array.

    Parameters
    ----------
    host_string : str
        The string representation of the host.

    Returns
    -------
    host_memory : list
        The host as a list of coordinates.
    """
    host_memory = []
    # Check for naive host
    if host_string == '\n':
        return host_memory
    host_string = host_string.strip()
    # If there are multiple entries, split them [this means we have ')()']
    if host_string.count(')(') > 0:
        host_string = host_string.split(')(')
        # Remove the parentheses and split the string on commas
        host_string = [host.replace('(', '').replace(')', '') for host in host_string]
        host_memory = [host.split(',') for host in host_string]
        # Make entries in sublists floats
        host_memory = [[float(coord) for coord in infection] for infection in host_memory]
    else:
        # Remove leading and trailing parentheses
        host_string = host_string[1:-1]
        host_memory = host_string.split(',')
        # Make entries floats
        host_memory = [[float(coord) for coord in host_memory]]
    return host_memory

def load_memories(memory_path: str) -> tuple:
    """
    Load the immune memory file and return a dictionary of host memories.

    Parameters
    ----------
    memory_path : str
        The path to the immune memory file.

    Returns
        -------
        tuple : (dict, dict, dict)
            A tuple of host memories, seasonality values, and fraction of susceptible hosts.
    """
    host_memories = {}
    seasonality_values = {}
    frac_susceptibles = {}
    with open(memory_path, "r") as f:
        for line in f:
            if 'date' in line:
                date = line.split()[1]
                host_memories[date] = {}
                seasonality_values[date] = {}
                frac_susceptibles[date] = {}
            elif 'seasonality' in line:
                    seasonality = line.split()[1]
            elif 'fracSusceptible' in line:
                frac_s = line.split()[1]
            elif 'contactRate' in line:
                continue
            else:
                deme, sample = line.split(sep=":")
                if deme not in host_memories[date]:
                    host_memories[date][deme] = []
                    seasonality_values[date][deme] = float(seasonality)
                    frac_susceptibles[date][deme] = float(frac_s)
                memory = parse_host(sample)
                host_memories[date][deme].append(memory)
    return (host_memories, seasonality_values, frac_susceptibles)

def process_variant(variant: str, variant_df: pd.DataFrame, var_col_name: str, host_memories_list: list, seasonality: float, frac_s: float, float_time: float, deme: str, antigen_params: dict = ANTIGEN_PARAMS) -> dict:
    """
    Process a variant and return the fitness of the variant in the given deme.

    Parameters
    ----------
    variant : str
        The variant to process.
    variant_df : pd.DataFrame
        The dataframe of viruses.
    var_col_name : str
        The column name of the variant in the dataframe.
    host_memories_list : list
        A list of host memories.
    seasonality : float
        The reported seasonality value of the deme.
    frac_s : float
        The fraction of susceptible hosts in the deme.
    float_time : float
        The time to calculate the fitness at.
    deme : str
        The deme to calculate the fitness in.
    antigen_params : dict
        A dictionary of parameters used in antigen simulations. 
        Required keys: `delta_t`, `beta`, and `between_deme_prob`.

    Returns
    -------
    dict
        A dictionary with the variant, time, deme, and fitness.
    """
    # Get centroid of variant
    centroid = variant_df[variant_df[var_col_name] == variant]
    
    # Check to see if virus exists at least half a year into the future
    if (centroid['birth'] > float_time).all():
        return {var_col_name: variant, 'time': float_time, 'deme': deme, 'fitness': np.NaN, 'seasonal_fitness': np.NaN, 'frac_susceptible': frac_s, 'seasonality': seasonality}
    
    # Vectorize the computation for all host memories at once
    distances = [get_closest_virus_distance(centroid, host) for host in host_memories_list]
    risks = np.array([infection_risk(distance) for distance in distances])
    average_infection_risk = risks.mean()
    seasonal_fitness = average_infection_risk * seasonality * frac_s
    
    return {var_col_name: variant, 'time': float_time, 'deme': deme, 'fitness': average_infection_risk, 'seasonal_fitness': seasonal_fitness, 'frac_susceptible': frac_s, 'seasonality': seasonality}

def calculate_variant_fitness(variant_df: pd.DataFrame, variant_col_name: str, host_memories: dict, seasonality_values: dict, frac_S_dict: dict, time: str, deme: str) -> pd.DataFrame:
    """
    Calculate the fitness of each variant in the dataframe given the host memories.

    Parameters
    ----------
    variant_df : pd.DataFrame
        The dataframe of viruses.
    variant_col_name : str
        The column name of the variant in the dataframe.
    host_memories : dict
        A dictionary of host memories.
    seasonality_values : dict
        A dictionary of deme-wise seasonality values.
    frac_S_dict : dict
        A dictionary of deme-wise fraction of susceptible hosts.
    time : str
        The time to calculate the fitness at.
    deme : str
        The deme to calculate the fitness in.

    Returns
    -------
    pd.DataFrame
        A dataframe of variants and their fitnesses across demes and time.
    """
    # Convert time to float once
    float_time = float(time) / 365.25

    # Pre-load host memories list for this time and deme
    host_memories_list = host_memories[time][deme]

    # Grab the seasonality value for this deme
    seasonality = seasonality_values[time][deme]

    # Grab the fraction of susceptible hosts for this deme
    frac_s = frac_S_dict[time][deme]

    # Parallel processing of variants
    results = Parallel(n_jobs=-1)(delayed(process_variant)(variant, variant_df, variant_col_name, host_memories_list, seasonality, frac_s, float_time, deme)
                                  for variant in variant_df[variant_col_name].unique())
    
    # Convert the list of results to a DataFrame
    return pd.DataFrame(results)

def remove_burnin(fitness_df: pd.DataFrame, burnin_period: float = 10.0):
    """ Remove burn-in from fitness data

    Parameters:
    ---------------
        fitness_df (pd.DataFrame): Fitness data
        burnin_period (float): Burn-in period in years
    
    Returns:
    ---------------
        pd.DataFrame: Fitness data without burn-in days
    """
    # Convert time to float
    fitness_df["time"] = fitness_df["time"].astype(float)
    # Remove burn-in of 10 years
    fitness_df = fitness_df[fitness_df["time"] >= burnin_period]
    fitness_df = fitness_df.copy()
    # Shift time to start from 0
    fitness_df.loc[:, "time"] = fitness_df["time"] - burnin_period
    return fitness_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate variant fitness over time.")
    parser.add_argument("-t", "--tips", help="Path to the variant-assigned tips dataframe.")
    parser.add_argument("-i", "--immunity", help="Path to the immune memory file.")
    parser.add_argument("-o", "--output", help="Path to save the variant fitness dataframe.")
    parser.add_argument("--centroid-output", required=False, help="Path to save the variant centroid dataframe.")
    parser.add_argument("--burnin", required=False, type=float, default=10.0, help="Burn-in period in years.")
    parser.add_argument("--variant-col", required=False, type=str, default="variant", help="Column name of the variant in the tips dataframe.")
    args = parser.parse_args()
    variant_col = args.variant_col
    # Load the tips dataframe
    print("Loading tips dataframe...")
    tips_df = pd.read_csv(args.tips)

    # Load the immune memory
    print("Loading immune memories from hosts...")
    host_memories, seasonality_values, frac_susceptibles = load_memories(args.immunity)

    # Get variant centroids and write to file if args.centroid_output is provided
    print("Calculating variant centroids...")
    variant_centroids = [get_variant_centroid(tips_df, variant, variant_col) for variant in tips_df[args.variant_col].unique()]
    variant_centroids_df = pd.DataFrame(variant_centroids)
    if args.centroid_output:
        print(f"Writing variant centroids to {args.centroid_output}...")
        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(args.centroid_output), exist_ok=True)
        variant_centroids_df.to_csv(args.centroid_output, index=False)
    
    # Calculate variant fitness
    print("Calculating variant fitness...")
    fitness_dfs = []
    n_timepoints = len(host_memories.keys())
    n_demes = len(host_memories['0.00'].keys())
    n_total = n_timepoints * n_demes
    counter = 0
    for timepoint in host_memories.keys():
        start_time = time.time()
        for deme in host_memories[timepoint].keys():
            fitness_df = calculate_variant_fitness(variant_centroids_df, variant_col, host_memories, seasonality_values, frac_susceptibles, timepoint, deme)
            fitness_dfs.append(fitness_df)
        end_time = time.time()
        counter += 1
        print(f'Done with {counter}/{n_timepoints} timepoints. Time taken: {end_time - start_time:.2f} seconds.')
    fitness_df = pd.concat(fitness_dfs)
    print("Removing burn-in...")
    n_burn_in = args.burnin
    fitness_df = remove_burnin(fitness_df, n_burn_in)
    print(f"Writing variant fitness to {args.output}...")
    fitness_df.to_csv(args.output, index=False)