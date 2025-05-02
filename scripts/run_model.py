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
import os
import pandas as pd
import jax.numpy as jnp
import evofr as ef
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal
from pathlib import Path
from jax import vmap
import jax.numpy as jnp
from jax.nn import softmax
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries.offsets import Day, BDay

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
    rt = pd.concat([rt, rt_forecast])
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

def naive_forecast(seq_count_date, pivot, n_days_to_average=7, period=30):
    """
    Naive forecast of the frequency of a variant.

    Parameters
    ----------
    seq_count_date : pd.DataFrame
        Sequence count data.
    pivot : str
        Pivot/forecasting date
    n_days_to_average : int
        Number of days to average counts over for forecasting.
    period : int
        Number of days to forecast.

    Returns
    -------
    pd.DataFrame
        Forecasted frequencies.
    """
    # Define dates for forecasting and nowcasting
    back_date = pd.to_datetime(pivot) - Day(period)
    forecast_dates = pd.to_datetime(pd.unique(pd.date_range(start=pivot, periods=period, freq='D'))).astype(str)
    nowcast_dates = pd.to_datetime(pd.unique(pd.date_range(start=back_date, periods=period, freq='D'))).astype(str)

    # Define prediction period for nowcasting and forecasting
    pred_dates = forecast_dates.union(nowcast_dates)

    # Compute frequency of variants for the demes using seq_count_data
    seq_count_date['total_seq'] = seq_count_date.groupby(['date', 'country'])['sequences'].transform('sum')
    seq_count_date['freq'] = seq_count_date['sequences'] / seq_count_date['total_seq']

    # Add prediction dates to date column for each country and variant
    sc_s = []
    for d in pred_dates:
        recent_dates = pd.Series(pd.to_datetime(seq_count_date[seq_count_date.date < d].date).unique()).nlargest(n_days_to_average).astype(str)

        # Get mean frequency of variants for the last 7 days
        seq_count_mean = seq_count_date[seq_count_date.date.isin(recent_dates)].groupby(["variant", "country"])["freq"].mean().reset_index()

        sc_ = seq_count_mean.copy()
        
        # Adding dates column
        sc_["date"] = d
        sc_s.append(sc_)

    sc = pd.concat(sc_s).sort_values(by=["country", "variant", "date"])
    
    # Adding nowcast and forecast columns
    sc['median_freq_nowcast'] = sc['freq']
    sc['median_freq_forecast'] = sc['freq']
    
    # Matching dates for nowcast and forecast
    sc.loc[sc.date.isin(forecast_dates),'median_freq_nowcast'] = np.nan
    sc.loc[sc.date.isin(nowcast_dates),'median_freq_forecast'] = np.nan
    return sc.reset_index(drop=True)

def main(args) -> None:
    # Load arguments
    data_path = args.data_path
    country = args.country
    model_type = args.model
    output_dir = args.output_dir
    model_args = args.model_args
    forecast_L = args.forecast_L
    seed_L = args.seed_L
    # TODO: Implement model_args
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

    # Define basis function and inference method.
    basis_fn = ef.Spline(order=4, k=10)
    # TODO: Allow these to change with model args.
    inference_method = ef.InferFullRank(iters=50_000, lr=0.01, num_samples=500)
    
    # Create delay padding embeddings for each variant
    gen = ef.pad_delays(
        [
            ef.discretise_gamma(mn=3.0, std=1.2) for _ in v_names
        ]
    )
    delays = ef.pad_delays([ef.discretise_lognorm(mn=3.1, std=1.0)])
    
    # Setup the model
    if model_type == "MLR":
        model = ef.MultinomialLogisticRegression(tau=3.0)
    elif model_type == "FGA":
        model = ef.RenewalModel(
            gen, delays, seed_L=seed_L, forecast_L=forecast_L,
            RLik=ef.FixedGA(0.1), # Likelihood on effective reproduction number
            CLik=ef.ZINegBinomCases(0.05), # Case counts likelihood
            SLik = ef.DirMultinomialSeq(100), # Sequence counts likelihood
            v_names=v_names,
            basis_fn=basis_fn
        )
    elif model_type == "GARW":
        model = ef.RenewalModel(
            gen, delays, seed_L=seed_L, forecast_L=forecast_L,
            RLik=ef.GARW(0.1,0.01, prior_family='Normal'), # Likelihood on effective reproduction number (GARW depend on R and gen time
            CLik=ef.ZINegBinomCases(0.05), # Case counts likelihood
            SLik = ef.DirMultinomialSeq(100), # Sequence counts likelihood
            v_names=v_names,
            basis_fn=basis_fn
        )
    elif model_type == "NAIVE":
        print("Running naive forecast...")
    else:
        print(f"Model type {model_type} not recognized.")
        return None
        

    # Fit model
    print("Fitting model...")
    filepath = f"{output_dir}/{model_type}"
    os.makedirs(filepath, exist_ok=True)
    if model_type == "NAIVE":
        model_posterior = naive_forecast(seq_counts, pivot=analysis_date, n_days_to_average=7, period=forecast_L)
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
    parser.add_argument("--model_args", type=str, help="Additional arguments to pass to the model.")
    parser.add_argument("--forecast_L", type=int, default=366, help="Number of days to forecast.")
    parser.add_argument("--seed_L", type=int, default=14, help="Number of days to seed the forecast with.")
    


    args = parser.parse_args()
    main(args)