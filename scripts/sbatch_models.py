#!/usr/bin/env python3
"""
sbatch_models.py

This script generates SLURM scripts for submitting forecasting model inference jobs to the cluster.
It reads in a YAML configuration file containing model parameters and generates all possible combinations
of model configurations. It then creates a SLURM script for each model configuration and submits the job
to the cluster.

Usage:
    python sbatch_models.py -b build -c config

Required Arguments:
    -b, --build         Name of the build of the desired analysis.
    -c, --config        Path to the YAML configuration file.

Output:
    - SLURM scripts for each model configuration submitted to the cluster.
    - Output files from the model fitting script (i.e., frequency, rt/ga inference).

Dependencies:
    - Requires Python 3.x
    - pandas
    - numpy
    - pyyaml

Author: Zorian Thornton (@zorian15)
Date: 2025-02-06
"""

import argparse
import yaml
import itertools
import os

def load_config(yaml_path: str) -> dict:
    """Load the YAML config file and return the parsed dictionary.

    Parameters
    ----------
    yaml_path : str
        The path to the YAML configuration file.

    Returns
    -------
    config : dict
        The parsed configuration dictionary.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_model_configs(config: dict) -> list:
    """Generate all possible combinations of model configurations.
    
    Parameters
    ----------
    config : dict
        The configuration dictionary containing the model parameters.

    Returns
    -------
    model_configs : list
        A list of dictionaries containing the model configurations.
    """
    if 'main' in config:
        config = config['main']
    categories = config.keys()  # Assuming your YAML has categories as top-level keys
    values = [config[category] for category in categories]  # Extract values for each category
    model_configs = list(itertools.product(*values))  # Generate combinations
    return model_configs

def create_slurm_script(build: str, analysis_date: str, country: str, model_type: str) -> str:
    """Create a Slurm batch script for this job configuration.

    Parameters
    ----------
    build : str
        Name of the build of the desired analysis.
    analysis_date : str
        Date of the analysis.
    country : str
        Name of the country to fit the model to.
    model_type : str
        Name of the model to fit.

    Returns
    -------
    slurm_script_path : str
        The path to the saved Slurm script.
    """
    # Define paths
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_dir = f"data/{build}/time-stamped/{analysis_date}/"
    slurm_dir = f"results/{build}/slurm/"
    output_dir = f"results/{build}/"

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(slurm_dir, exist_ok=True)

    # Get path for script
    script_path = os.path.abspath("scripts/run_model.py")
    # Slurm script template
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=run_model_{model_type}_{country}_{analysis_date}
#SBATCH --output={slurm_dir}/{model_type}_{country}_{analysis_date}.out
#SBATCH --error={slurm_dir}/{model_type}_{country}_{analysis_date}.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

# Activate virtual environment (if needed)
# mamba activate antigen

# Run the model fitting script
python {script_path} --data_path {data_dir} --country {country} --model {model_type} --output_dir {output_dir}
    """

    # Save Slurm script
    slurm_script_path = os.path.join(slurm_dir, f"job_{model_type}_{country}_{analysis_date}.sh")
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script)
    return slurm_script_path


def main(args) -> None:
    """Main function to generate SLURM scripts for forecasting model inference."""
    # Load arguments
    build = args.build
    config_path = args.config

    # Load the configuration file
    config = load_config(config_path)

    # Generate model configurations
    model_configs = generate_model_configs(config)

    # Create SLURM scripts for each model configuration
    for model_config in model_configs:
        analysis_date, country, model_type = model_config
        
        slurm_script_path = create_slurm_script(build, analysis_date, country, model_type)
        # Submit the job to the cluster
        os.system(f"sbatch {slurm_script_path}") 
        print(f"SLURM script submitted for {model_type} model in {country} on {analysis_date}.")

    print("All SLURM scripts submitted successfully.")



if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Submit forecasting model inference jobs via SLURM.")

    # Required arguments
    parser.add_argument("-b", "--build", type=str, help="Name of the build of the desired analysis.")
    parser.add_argument("-c", "--config", type=str, help="Path to the YAML configuration file.")
    
    # Parse arguments and run the program
    args = parser.parse_args()
    main(args)