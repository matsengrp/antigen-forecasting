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
import json
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
        A list of tuples containing (analysis_date, country, model_type).
    """
    if 'main' in config:
        config = config['main']
    
    # Extract the core parameters for job generation (exclude model_config)
    estimation_dates = config.get('estimation_dates', [])
    locations = config.get('locations', [])
    models = config.get('models', [])
    
    # Handle both simple model list and model-specific config format
    if models and isinstance(models[0], dict):
        # Model-specific configs format: extract just the model names
        model_names = [model.get('name') for model in models if model.get('name')]
    else:
        # Simple model list format
        model_names = models
    
    # Generate combinations
    model_configs = list(itertools.product(estimation_dates, locations, model_names))
    return model_configs

def create_slurm_script(build: str, analysis_date: str, country: str, model_type: str, model_config: str = None) -> str:
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
    model_config : str, optional
        Path to JSON model configuration file.

    Returns
    -------
    slurm_script_path : str
        The path to the saved Slurm script.
    """
    # Define paths
    data_dir = f"data/{build}/time-stamped/{analysis_date}/"
    slurm_dir = f"results/{build}/slurm/"
    output_dir = f"results/{build}/"

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(slurm_dir, exist_ok=True)

    # Get path for script
    script_path = os.path.abspath("scripts/run_model.py")
    
    # Build command with optional config
    cmd = f"python {script_path} --data_path {data_dir} --country {country} --model {model_type} --output_dir {output_dir}"
    if model_config:
        cmd += f" --config {model_config}"
    
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
{cmd}
    """

    # Save Slurm script
    slurm_script_path = os.path.join(slurm_dir, f"job_{model_type}_{country}_{analysis_date}.sh")
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script)
    return slurm_script_path

def validate_sbatch_config(config: dict) -> None:
    """Validate sbatch YAML config before job submission.
    
    Parameters
    ----------
    config : dict
        The parsed YAML configuration dictionary.
        
    Raises
    ------
    FileNotFoundError
        If specified model config files don't exist.
    """
    main_config = config.get('main', {})
    
    # Check for single model_config that applies to all models
    if 'model_config' in main_config:
        model_config_path = main_config['model_config']
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found: {model_config_path}")
        
        # Load and validate JSON structure
        try:
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
            print(f"Using shared model config '{model_config_path}' for all models")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in model config file {model_config_path}: {e}")
    
    # Check for model-specific configs
    models = main_config.get('models', [])
    if models and isinstance(models[0], dict):
        # Model-specific configs format
        for model_entry in models:
            if isinstance(model_entry, dict):
                model_name = model_entry.get('name')
                model_cfg_path = model_entry.get('config')
                if model_cfg_path:
                    if not os.path.exists(model_cfg_path):
                        raise FileNotFoundError(f"Config for {model_name} not found: {model_cfg_path}")
                    try:
                        with open(model_cfg_path, 'r') as f:
                            json.load(f)
                        print(f"Validated model-specific config for {model_name}: {model_cfg_path}")
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in {model_name} config file {model_cfg_path}: {e}")

def extract_model_config(config: dict, model_type: str) -> str | None:
    """Extract the appropriate model config path for a given model type.
    
    Parameters
    ----------
    config : dict
        The parsed YAML configuration dictionary.
    model_type : str
        The model type (e.g., 'MLR', 'FGA', 'GARW').
        
    Returns
    -------
    str | None
        Path to the model config file, or None if no config specified.
    """
    main_config = config.get('main', {})
    
    # Check for single model_config that applies to all models
    if 'model_config' in main_config:
        return main_config['model_config']
    
    # Check for model-specific configs
    models = main_config.get('models', [])
    if models and isinstance(models[0], dict):
        # Model-specific configs format
        for model_entry in models:
            if isinstance(model_entry, dict):
                if model_entry.get('name') == model_type:
                    return model_entry.get('config')
    
    return None

def main(args) -> None:
    """Main function to generate SLURM scripts for forecasting model inference."""
    # Load arguments
    build = args.build
    config_path = args.config

    # Load the configuration file
    config = load_config(config_path)
    
    # Validate configuration before proceeding
    try:
        validate_sbatch_config(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration validation failed: {e}")
        return

    # Generate model configurations
    model_configs = generate_model_configs(config)

    # Create SLURM scripts for each model configuration
    for model_config in model_configs:
        analysis_date, country, model_type = model_config
        
        # Extract model config for this specific model type
        model_config_path = extract_model_config(config, model_type)
        
        slurm_script_path = create_slurm_script(build, analysis_date, country, model_type, model_config_path)
        # Submit the job to the cluster
        os.system(f"sbatch {slurm_script_path}") 
        
        if model_config_path:
            print(f"SLURM script submitted for {model_type} model in {country} on {analysis_date} with config: {model_config_path}")
        else:
            print(f"SLURM script submitted for {model_type} model in {country} on {analysis_date} (using defaults)")

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