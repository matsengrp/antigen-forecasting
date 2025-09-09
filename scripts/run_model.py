#!/usr/bin/env python3
"""
Variant Frequency Forecasting Script

This script performs forecasting of variant frequencies using different modeling approaches, 
including Multinomial Logistic Regression (MLR), Fixed Growth Advantage (FGA), 
and Growth Advantage with Random Walk (GARW). It processes input sequence and case count data,
fits the specified model, and generates forecasts.

Usage:
    python run_model.py --data_path <path> --country <country> --model <model_type> --output_dir <output_directory> --forecast_L <forecast_length> --seed_L <seed_length>

Required Arguments:
    -d, --data_path      Path to the data directory.
    -c, --country        Country to fit the model to.
    -m, --model          Model to fit to the data.
    -o, --output_dir     Directory to save the model results.

Optional Arguments:
    --model_args         Additional arguments to pass to the model.
    --forecast_L         Number of days to forecast (default: 180).
    --seed_L             Number of days to seed the forecast with (default: 14).

Output:
    - freq_<country>_<date>.tsv: Forecasted and nowcasted frequencies for the specified country.
    - ga_<country>_<date>.tsv: Growth advantage of each variant relative to a reference variant.
    - rt_<country>_<date>.tsv: Time-varying Rt estimates for the specified country.
                     
Dependencies:
    - argparse
    - os
    - pandas
    - jax.numpy
    - evofr
    - numpyro
    - pathlib
    - matplotlib
    - numpy

Author: Zorian Thornton (@zorian15)
Date: 2025-02-06
"""
# Import packages
import argparse
import json
import os
import shutil
import pandas as pd
import jax.numpy as jnp
import evofr as ef
from evofr.infer import InferSVI, InferFullRank, InferMAP
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal
from pathlib import Path
from jax import vmap
import jax.numpy as jnp
from jax.nn import softmax
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries.offsets import Day, BDay
import sys
import os
sys.path.append('..')  # Adjust path to import antigentools
from antigentools.utils import save_vi_convergence_diagnostics, naive_forecast, naive_forecast_full_window

def load_data(data_path: str, country: str) -> pd.DataFrame:
    """ Load variant or case count data from a path, and filter to the desired location.

    Parameters
    ----------
    data_path : str
        The directory containing the count data.
    country : str
        The location to filter the data to.

    Returns
    -------
    count_df : pd.DataFrame
        The location-specific count data.
    """
    counts_df = pd.read_csv(data_path, sep='\t')
    counts_df = counts_df[counts_df['country'] == country]
    
    return counts_df

def make_variant_frequency_data(seq_counts: pd.DataFrame, v_names: list, case_counts: pd.DataFrame=None) -> ef.VariantFrequencies | ef.CaseFrequencyData:
    """ Create a VariantFrequencies object from sequence and case count data.

    Parameters
    ----------
    seq_counts : pd.DataFrame
        The sequence count data.
    case_counts : pd.DataFrame
        The case count data.
    v_names : list
        The names of the variants.

    Returns
    -------
    variant_freqs : ef.VariantFrequencies
        The variant frequency data.
    """
    if case_counts is None:
        variant_freqs = ef.VariantFrequencies(raw_seq=seq_counts, var_names=v_names)
    else:
        variant_freqs = ef.CaseFrequencyData(raw_seq=seq_counts, raw_cases=case_counts, var_names=v_names)
    
    return variant_freqs

def forecast_frequencies(samples: dict, model: ef.models.MultinomialLogisticRegression, forecast_L: int=30):
    """
    Use MLR posterior betas to forecast using MLR model.

    Parameters
    ----------
    samples : dict
        Samples from the MCMC run.
    model : evofr.models.MultinomialLogisticRegression
        MLR model used for forecasting.
    forecast_L : int
        Number of days to forecast.
    
    Returns
    -------
    jnp.array
        Forecasted frequencies.
    """
    
    # Making feature matrix for forecasting
    last_T = samples["freq"].shape[1]
    X = model.make_ols_feature(start=last_T, stop=last_T + forecast_L)
    
    # Posterior beta
    beta = jnp.array(samples["beta"])
    
    # Matrix multiplication by sample
    dot_by_sample = vmap(jnp.dot, in_axes=(None, 0), out_axes=0)
    logits = dot_by_sample(X, beta) # Logit frequencies by variant
    return softmax(logits, axis=-1)

def save_freq(samples: dict, variant_data: ef.VariantFrequencies, ps: list, analysis_date: str, country:str, model_name: str, filepath:str, num_nowcast_days:int = None) -> None:
    """
    Save posterior frequencies to a file.

    Parameters
    ----------
    samples : dict
        Samples from the MCMC run.
    variant_data : ef.VariantFrequencies | ef.CaseFrequencyData
        The variant frequency data.
    ps : list
        List of quantiles to sample from the posterior.
    analysis_date : str
        Date of the analysis.
    country : str
        Country to save the frequencies for.
    model_name : str
        Name of the model used for forecasting.
    filepath : str
        Path to save the frequencies file.
    num_nowcast_days : int
        Number of days to nowcast.

    Returns
    -------
    None
    """
    # Get nowcast frequencies
    freq_now = pd.DataFrame(ef.get_freq(samples, variant_data, ps, name=country, forecast = False))
    if num_nowcast_days:
        nowcast_dates = variant_data.dates[-num_nowcast_days:]
    else:
        nowcast_dates = variant_data.dates    
    freq_now = freq_now[freq_now['date'].isin(nowcast_dates)]
    
    # Get forecasted frequencies
    freq_fr = pd.DataFrame(ef.get_freq(samples, variant_data, ps, name=country, forecast = True))
    
    # Merge and export file
    freq_merged = pd.concat([freq_now, freq_fr])
    freq_merged = freq_merged.rename(columns = {'median_freq':'median_freq_nowcast'}, inplace = False)
    if model_name is not None:
        freq_merged['model'] = model_name
    freq_merged['analysis_date'] = analysis_date
    freq_merged.to_csv(f'{filepath}/freq_{country}_{analysis_date}.tsv', sep="\t", index = False)
    return None

def get_fixed_growth_advantage(samples: dict, variant_data: ef.VariantFrequencies, ps: list, country: str, date: str, filepath: str, ref_variant: str=None, model_name: str=None) -> None:
    """
    Get the growth advantage of each variant relative to a reference variant.

    Parameters
    ----------
    samples : dict
        Samples from the MCMC run.
    variant_data : ef.VariantFrequencies | ef.CaseFrequencyData
        The variant frequency data.
    ps : list
        List of quantiles to sample from the posterior.
    country : str
        Country to save the frequencies for.
    date : str
        Date of the analysis.
    filepath : str
        Path to save the frequencies file.
    ref_variant : str
        Reference variant to compare growth advantages to.
    model_name : str
        Name of the model used for forecasting.

    Returns
    -------
    None
    """
    growth_adv = pd.DataFrame(ef.posterior.get_growth_advantage(samples, variant_data, ps, name=country, rel_to=None))
    growth_adv['analysis_date'] = date
    if model_name is not None:
        growth_adv['model'] = model_name
    growth_adv.to_csv(f'{filepath}/ga_{country}_{date}.tsv', sep="\t", index = False, header=True)
    return None

def get_time_varying_rt(samples: dict, variant_data: ef.VariantFrequencies, ps: list, country: str, date: str, filepath: str, model_name: str=None):
    """ Get time-varying Rt estimates from posterior and dump to file.

    Parameters
    ----------
    samples : dict
        Samples from the MCMC run.
    variant_data : evofr.data.VariantData
        Variant data object.
    ps : list
        List of quantiles to compute.
    country : str
        Name of the deme/location to consider.
    date : str
        Date of the analysis.
    site : str
        The summary statistic to compute.
    filepath : str
        Path to dump the file.
    """
    rt = pd.DataFrame(ef.posterior.get_site_by_variant(samples, variant_data, ps, name=country, site='R', forecast=False))
    rt_forecast = pd.DataFrame(ef.posterior.get_site_by_variant(samples, variant_data, ps, name=country, site='R', forecast=False))
    # Grab growth rates as well
    growth_rates = pd.DataFrame(ef.posterior.get_site_by_variant(samples, variant_data, ps, name=country, site='r', forecast=False))
    growth_rates_forecast = pd.DataFrame(ef.posterior.get_site_by_variant(samples, variant_data, ps, name=country, site='r', forecast=False))
    rt = pd.concat([rt, rt_forecast, growth_rates, growth_rates_forecast], axis=1)
    if model_name is not None:
        rt['model'] = model_name
    rt["analysis_date"] = date
    rt.to_csv(f'{filepath}/rt_{country}_{date}.tsv', sep="\t", index = False, header = True)
    return None

def get_mlr_rt(samples: dict, variant_data: ef.VariantFrequencies, ps: list, country: str, date: str, filepath: str, model_name: str="MLR", tau: float=3.0) -> None:
    """
    Calculate and save effective reproductive number (Rt) from MLR model parameters.
    
    For MLR models, Rt is calculated as exp(β_v * τ), where β_v is the growth rate
    parameter and τ is the fixed generation time.

    Parameters
    ----------
    samples : dict
        Samples from the posterior.
    variant_data : ef.VariantFrequencies
        The variant frequency data.
    ps : list
        List of quantiles to compute.
    country : str
        Name of the location to consider.
    date : str
        Date of the analysis.
    filepath : str
        Path to save the output file.
    model_name : str, optional
        Name of the model used for forecasting, defaults to "MLR".
    tau : float, optional
        Fixed generation time, default is 3.0.

    Returns
    -------
    None
    """
    # Get variant names
    v_names = variant_data.var_names
    
    # Get beta values from MLR model
    betas = samples["beta"]
    
    # Calculate Rt for each variant and sample
    rt_values = {}
    for i, v in enumerate(v_names):
        # Beta values are stored as [sample, variant]
        beta_v = betas[:, i]
        # Calculate Rt = exp(β_v * τ)
        rt_v = np.exp(beta_v * tau)
        
        # Store in format matching other Rt functions
        for q_idx, q in enumerate(ps):
            q_label = f"{int(q*100)}"
            
            # Calculate quantiles
            lower = np.quantile(rt_v, (1-q)/2)
            median = np.median(rt_v)
            upper = np.quantile(rt_v, 1-(1-q)/2)
            
            # Store values in dictionary with proper keys
            if v not in rt_values:
                rt_values[v] = {}
            
            rt_values[v][f"R_lower_{q_label}"] = lower
            rt_values[v]["median_R"] = median
            rt_values[v][f"R_upper_{q_label}"] = upper
    
    # Convert to dataframe format similar to other Rt functions
    rt_rows = []
    for v in v_names:
        for day in variant_data.dates:
            row = {
                "variant": v,
                "date": day,
                "location": country,
                "median_R": rt_values[v]["median_R"],
            }
            
            # Add quantile values
            for q in ps:
                q_label = f"{int(q*100)}"
                row[f"R_lower_{q_label}"] = rt_values[v][f"R_lower_{q_label}"]
                row[f"R_upper_{q_label}"] = rt_values[v][f"R_upper_{q_label}"]
            
            rt_rows.append(row)
    
    # Create dataframe and add model name and analysis date
    rt_df = pd.DataFrame(rt_rows)
    rt_df['model'] = model_name
    rt_df["analysis_date"] = date
    
    # Save to file
    rt_df.to_csv(f'{filepath}/rt_{country}_{date}.tsv', sep="\t", index=False, header=True)
    return None



def load_config(config_path: str) -> dict:
    """Load and validate JSON configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.
        
    Returns
    -------
    config : dict
        Parsed configuration dictionary, empty if file doesn't exist.
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from: {config_path}")
            return config
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON config file: {e}")
            return {}
    elif config_path:
        print(f"Warning: Config file not found: {config_path}")
    return {}

def get_config_value(config: dict, keys: list, default):
    """Safely get nested configuration value with default fallback.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    keys : list
        List of keys to traverse (e.g., ['inference', 'settings', 'iters']).
    default : any
        Default value to return if key path doesn't exist.
        
    Returns
    -------
    any
        Configuration value or default.
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

def validate_config_for_model(config: dict, model_type: str) -> dict:
    """Validate that config contains appropriate sections for the specified model.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    model_type : str
        Model type (MLR, FGA, GARW, NAIVE).
        
    Returns
    -------
    config : dict
        Validated configuration dictionary.
    """
    if not config:
        return config
    
    print(f"Validating config for model type: {model_type}")
    
    # Check if model-specific config exists
    if 'model_specific' in config and model_type in config['model_specific']:
        model_config = config['model_specific'][model_type]
        print(f"Found model-specific config for {model_type}")
        
        if model_type == "MLR":
            # MLR requires tau parameter
            if 'tau' not in model_config:
                print(f"Warning: MLR config missing 'tau' parameter, using default")
        
        elif model_type in ["FGA", "GARW"]:
            # FGA/GARW require likelihood configs
            required_keys = ['case_likelihood', 'seq_likelihood']
            for key in required_keys:
                if key not in model_config:
                    print(f"Warning: {model_type} config missing '{key}', using default")
    else:
        print(f"No model-specific config found for {model_type}, using defaults")
    
    return config

def main(args) -> None:
    # Load arguments
    data_path = args.data_path
    country = args.country
    model_type = args.model
    output_dir = args.output_dir
    forecast_L = args.forecast_L
    seed_L = args.seed_L
    
    # Load and validate configuration
    config = load_config(args.config) if args.config else {}
    config = validate_config_for_model(config, model_type)
    
    # Get seed_L from config (with command line override)
    seed_L = get_config_value(config, ['seed_L'], args.seed_L)
    # Get forecast_L from config (with command line override)
    forecast_L = get_config_value(config, ['forecast_L'], args.forecast_L)
    
    # Configuration logging and reproduction
    if args.config:
        print(f"Using config file: {args.config}")
        print(f"Model type: {model_type}")
        
        # Log which config sections are being used
        if 'model_specific' in config and model_type in config['model_specific']:
            print(f"Found model-specific config for {model_type}")
        else:
            print(f"Using default parameters for {model_type}")
    
    ps = [0.95, 0.8, 0.5]

    # Load data
    analysis_date = data_path.split("/")[-2]
    print(f"Loading data from {data_path}...")
    seq_counts = load_data(data_path + "seq_counts.tsv", country)
    if model_type in ["NAIVE", "MLR"]:
        case_counts = None
    else:
        case_counts = load_data(data_path + "case_counts.tsv", country)
    v_names = seq_counts['variant'].unique().tolist()
    
    # Check if there is data available for training.
    if (len(seq_counts) == 0) or (seq_counts.variant.nunique() == 1):
        print("No sequence data available for this location.")
        return None
    
    # Create the variant frequency data
    print("Creating variant frequency data...")
    variant_data = make_variant_frequency_data(seq_counts, v_names, case_counts)

    # Define basis function and inference method from config
    basis_order = get_config_value(config, ['basis_function', 'parameters', 'order'], 4)
    basis_k = get_config_value(config, ['basis_function', 'parameters', 'k'], 10)
    basis_fn = ef.Spline(order=basis_order, k=basis_k)
    
    # Get inference settings from config
    iters = get_config_value(config, ['inference', 'settings', 'iters'], 50_000)
    lr = get_config_value(config, ['inference', 'settings', 'lr'], 0.01)
    num_samples = get_config_value(config, ['inference', 'settings', 'num_samples'], 500)
    inference_method = ef.InferFullRank(iters=iters, lr=lr, num_samples=num_samples)
    
    # Create delay padding embeddings for each variant from config
    gen_mean = get_config_value(config, ['generation_time', 'parameters', 'mean'], 3.0)
    gen_std = get_config_value(config, ['generation_time', 'parameters', 'std'], 1.2)
    gen = ef.pad_delays(
        [
            ef.discretise_gamma(mn=gen_mean, std=gen_std) for _ in v_names
        ]
    )
    
    delay_mean = get_config_value(config, ['delay_distribution', 'parameters', 'mean'], 3.1)
    delay_std = get_config_value(config, ['delay_distribution', 'parameters', 'std'], 1.0)
    delays = ef.pad_delays([ef.discretise_lognorm(mn=delay_mean, std=delay_std)])
    
    # Setup the model with config-driven parameters
    if model_type == "MLR":
        tau = get_config_value(config, ['model_specific', 'MLR', 'tau'], 3.0)
        model = ef.MultinomialLogisticRegression(tau=tau)
    elif model_type == "FGA":
        # Get FGA-specific parameters from config
        ga_prior = get_config_value(config, ['model_specific', 'FGA', 'ga_prior'], 0.1)
        case_conc = get_config_value(config, ['model_specific', 'FGA', 'case_likelihood', 'concentration'], 0.05)
        seq_conc = get_config_value(config, ['model_specific', 'FGA', 'seq_likelihood', 'concentration'], 100)
        
        model = ef.RenewalModel(
            gen, delays, seed_L=seed_L, forecast_L=forecast_L,
            RLik=ef.FixedGA(ga_prior), # Likelihood on effective reproduction number
            CLik=ef.ZINegBinomCases(case_conc), # Case counts likelihood
            SLik = ef.DirMultinomialSeq(seq_conc), # Sequence counts likelihood
            v_names=v_names,
            basis_fn=basis_fn
        )
    elif model_type == "GARW":
        # Get GARW-specific parameters from config
        ga_prior_mean = get_config_value(config, ['model_specific', 'GARW', 'ga_prior_mean'], 0.1)
        ga_prior_std = get_config_value(config, ['model_specific', 'GARW', 'ga_prior_std'], 0.01)
        prior_family = get_config_value(config, ['model_specific', 'GARW', 'prior_family'], 'Normal')
        case_conc = get_config_value(config, ['model_specific', 'GARW', 'case_likelihood', 'concentration'], 0.05)
        seq_conc = get_config_value(config, ['model_specific', 'GARW', 'seq_likelihood', 'concentration'], 100)
        
        model = ef.RenewalModel(
            gen, delays, seed_L=seed_L, forecast_L=forecast_L,
            RLik=ef.GARW(ga_prior_mean, ga_prior_std, prior_family=prior_family), 
            CLik=ef.ZINegBinomCases(case_conc), # Case counts likelihood
            SLik = ef.DirMultinomialSeq(seq_conc), # Sequence counts likelihood
            v_names=v_names,
            basis_fn=basis_fn
        )
    elif model_type == "NAIVE":
        print("Running naive forecast...")
        # Check for full window configuration
        use_full_window = get_config_value(config, ['model_specific', 'NAIVE', 'full_window'], False)
        if use_full_window:
            print("Using full training window coverage for NAIVE model...")
        else:
            print("Using traditional 60-day window for NAIVE model...")
    else:
        print(f"Model type {model_type} not recognized.")
        return None
        

    # Fit model
    print("Fitting model...")
    filepath = f"{output_dir}/estimates/{model_type}"
    os.makedirs(filepath, exist_ok=True)
    if model_type == "NAIVE":
        # Check for full window configuration (command line flag overrides config)
        use_full_window = getattr(args, 'naive_full_window', False) or get_config_value(config, ['model_specific', 'NAIVE', 'full_window'], False)
        
        # Get NAIVE-specific parameters from config
        n_days_to_average = get_config_value(config, ['model_specific', 'NAIVE', 'n_days_to_average'], 7)
        
        if use_full_window:
            # Always use all available training data (training_window=None)
            model_posterior = naive_forecast_full_window(
                seq_counts, 
                pivot=analysis_date,
                n_days_to_average=n_days_to_average,
                forecast_period=forecast_L,
                training_window=None
            )
        else:
            # Use original implementation for backward compatibility
            # For original function, use forecast_L as the period parameter
            model_posterior = naive_forecast(
                seq_counts, 
                pivot=analysis_date, 
                n_days_to_average=n_days_to_average, 
                period=forecast_L
            )
        
        model_posterior['model'] = model_type

        print("Saving naive results to file...")
        model_posterior.to_csv(f'{filepath}/freq_{country}_{analysis_date}.tsv', sep="\t", index = False)
        return None
    # For all other models, try fitting with evofr.
    try:
        model_posterior = inference_method.fit(model, variant_data)
    except:
        print("Model fitting failed.")
        return None
    
    # Save VI convergence diagnostics if using VI inference
    if isinstance(inference_method, (InferSVI, InferFullRank, InferMAP)):
        print("Saving VI convergence diagnostics...")
        try:
            # Extract inference settings
            inference_settings = {
                'iterations': getattr(inference_method, 'iters', None),
                'learning_rate': getattr(inference_method, 'lr', None),
                'num_samples': getattr(inference_method, 'num_samples', None)
            }
            
            # Save diagnostics
            diagnostics_path = save_vi_convergence_diagnostics(
                posterior=model_posterior,
                model_name=model_type,
                location=country,
                analysis_date=analysis_date,
                inference_method=inference_method.__class__.__name__,
                inference_settings=inference_settings,
                output_dir=f"{output_dir}/convergence-diagnostics"
            )
            print(f"VI diagnostics saved to: {diagnostics_path}")
        except Exception as e:
            print(f"Warning: Failed to save VI convergence diagnostics: {e}")
    
    # Copy config to output directory for reproducibility
    if args.config:
        try:
            config_copy_path = f"{output_dir}/config_{model_type}_{country}_{analysis_date}.json"
            shutil.copy2(args.config, config_copy_path)
            print(f"Config copied to: {config_copy_path}")
        except Exception as e:
            print(f"Warning: Failed to copy config file: {e}")
    
    print("Sampling posterior and saving to file...")
    if model_type == "MLR":
        model_posterior.samples['freq_forecast'] = forecast_frequencies(model_posterior.samples, model, forecast_L=forecast_L)
        get_fixed_growth_advantage(model_posterior.samples, variant_data, ps, country, analysis_date, filepath, model_name=model_type)
        get_mlr_rt(model_posterior.samples, variant_data, ps, country, analysis_date, filepath, model_name=model_type)
    elif model_type in ["FGA", "GARW"]:
        get_time_varying_rt(model_posterior.samples, variant_data, ps, country, analysis_date, filepath, model_name=model_type)
    else:
        print(f"Model type {model_type} not recognized.")
        return None
    # Save frequencies
    save_freq(model_posterior.samples, variant_data, ps, analysis_date, country, model_type, filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a specified forecasting model.")

    # Required arguments
    parser.add_argument("-d", "--data_path", type=str, help="Path to the data directory.")
    parser.add_argument("-c", "--country", type=str, help="Country to fit the model to.")
    parser.add_argument("-m", "--model", type=str, help="Model to fit to the data.")
    parser.add_argument("-o", "--output_dir", type=str, help="Directory to save the model results.")

    # Optional arguments
    parser.add_argument("--forecast_L", type=int, default=365, help="Number of days to forecast.")
    parser.add_argument("--seed_L", type=int, default=14, help="Number of days to seed the forecast with.")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file with model parameters.")
    parser.add_argument("--naive-full-window", action="store_true", help="Use full training window for NAIVE model (overrides config).")
    


    args = parser.parse_args()
    main(args)