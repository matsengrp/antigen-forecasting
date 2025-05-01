# Forecasting with `antigen-prime` simulated data

This repository contains a suite of Python scripts designed to prepare, analyze, and forecast variant frequency data using `antigen-prime` and `evofr`. 
These scripts enable you to process antigen data, calculate variant fitness, create training datasets, run forecasting models, and evaluate model performance.

## Overview

The workflow typically follows these steps:

1. Prepare antigen data using `prep_antigen_data.py`
2. Calculate variant fitness with `calc_variant_fitness.py`
3. Create training datasets with `make_training_data.py`
4. Run forecasting models using either `run_model.py` directly or `sbatch_models.py` for SLURM-based environments
5. Score and evaluate models with `score_models.py`

## Installation

```bash
# Clone the repository
git clone https://github.com/matsengrp/antigen-forecasting.git
cd antigen-forecasting

# Install dependencies with mamba
mamba env create -f environment.yaml

# Activate the environment
mamba activate antigen

# Install the package
pip install .
```

## Usage

### 1. Prepare Antigen Data

The `prep_antigen_data.py` script prepares antigen-prime simulated data for variant frequency forecasting.

```bash
python prep_antigen_data.py -t path/to/tips.csv -c path/to/cases.csv -o output/directory
```

#### Options:

- `-t, --tips` - Path to the tips file
- `-c, --cases` - Path to the cases file
- `-o, --output` - Path to the output directory for prepped sequence and case counts
- `-sd, --start_date` - Start date for the analysis
- `-v, --variant-col-name` - Name of the column containing the variant information
- `-n, --normalize_cases` - Normalize case counts to report cases per 100k hosts
- `--deme_population_size` - Population size of the deme

#### Example:

```bash
python prep_antigen_data.py -t data/antigen_h3n2_sim/antigen-outputs/tips.csv -c data/antigen_h3n2_sim/antigen-outputs/cases.csv -o data/antigen_h3n2_sim/time-stamped/truth/ -sd 2022-01-01 -v variant_name --normalize_cases
```

### 2. Calculate Variant Fitness

The `calc_variant_fitness.py` script calculates variant fitness over time.

```bash
python calc_variant_fitness.py -t path/to/tips.csv -i path/to/immunity.csv -o output/fitness.csv
```

#### Options:

- `-t, --tips` - Path to the variant-assigned tips dataframe
- `-i, --immunity` - Path to the immune memory text file
- `-o, --output` - Path to save the variant fitness dataframe
- `--centroid-output` - Path to save the variant centroid dataframe
- `--burnin` - Burn-in period in years
- `--variant-col` - Column name of the variant in the tips dataframe

#### Example:

```bash
python calc_variant_fitness.py -t data/antigen_h3n2_sim/antigen-outputs/tips.csv -i data/antigen_h3n2_sim/antigen-outputs/immunity.txt -o data/antigen_h3n2_sim/antigen-outputs/variant_fitness.csv --burnin 1 --variant-col variant_name
```

### 3. Create Training Data

The `make_training_data.py` script splits variant and count data into training windows.

```bash
python make_training_data.py -s path/to/sequences.csv -c path/to/cases.csv -o output/directory
```

#### Options:

- `-s, --sequences` - Path to the variant counts file
- `-c, --cases` - Path to the case counts file
- `-o, --output` - Path to the output directory
- `--window-size` - Size of the training window in days
- `--buffer-size` - Number of days to remove from the training set before the analysis date
- `--config-path` - Path to dump the training configuration file

#### Example:

```bash
python make_training_data.py -s data/antigen_h3n2_sim/time-stamped/truth/sequences.csv -c data/antigen_h3n2_sim/time-stamped/truth/cases.csv -o data/antigen_h3n2_sim/training/ --window-size 90 --buffer-size 14 --config-path config/antigen_h3n2_sim/training_config.yaml
```

### 4. Run Forecasting Models

#### Option A: Run a single model

The `run_model.py` script fits a specified forecasting model.

```bash
python run_model.py -d data/directory -c country_name -m model_name -o output/directory
```

#### Options:

- `-d, --data_path` - Path to the data directory
- `-c, --country` - Country to fit the model to
- `-m, --model` - Model to fit to the data
- `-o, --output_dir` - Directory to save the model results
- `--model_args` - Additional arguments to pass to the model
- `--forecast_L` - Number of days to forecast
- `--seed_L` - Number of days to seed the forecast with

#### Example:

```bash
python run_model.py -d data/antigen_h3n2_sim/time-stamped/2025-10-01/ -c north -m FGA -o results/antigen_h3n2_sim/estimates/ --forecast_L 28 --seed_L 7
```

#### Option B: Run multiple models via SLURM

The `sbatch_models.py` script submits forecasting model inference jobs via SLURM.

```bash
python sbatch_models.py -b build_name -c path/to/config.yaml
```

#### Options:

- `-b, --build` - Name of the build of the desired analysis
- `-c, --config` - Path to the YAML configuration file

#### Example:

```bash
python sbatch_models.py -b antigen_h3n2_sim -c config/model_config.yaml
```

### 5. Evaluate Model Performance

The `score_models.py` script computes model scores.

```bash
python score_models.py --config path/to/config.yaml --truth-set path/to/truth.csv --estimates-path path/to/estimates/ --output-path results/scores/
```

#### Options:

- `--config` - Path to the configuration file
- `--truth-set` - Path to the truth set of sequences
- `--estimates-path` - Path to the estimates
- `--output-path` - Path to save the output

#### Example:

```bash
python score_models.py --config config/antigen_h3n2_sim/scoring_config.yaml --truth-set data/antigen_h3n2_sim/time-stamped/truth/true_values.csv --estimates-path results/antigen_h3n2_sim/estimates/ --output-path results/antigen_h3n2_sim/scores/
```

## Complete Pipeline Example

Here's an example of running the complete pipeline:

```bash
# 1. Prepare antigen data
python prep_antigen_data.py -t data/raw/tips.csv -c data/raw/cases.csv -o data/prepped/ -sd 2022-01-01 -v variant_name

# 2. Calculate variant fitness
python calc_variant_fitness.py -t data/prepped/tips.csv -i data/immunity.csv -o data/prepped/variant_fitness.csv

# 3. Create training data
python make_training_data.py -s data/prepped/sequences.csv -c data/prepped/cases.csv -o data/training/ --window-size 90

# 4. Run models (individual or batch)
python run_model.py -d data/training/ -c US -m RandomForest -o results/forecasts/
# Or with SLURM
python sbatch_models.py -b antigen_h3n2_sim -c config/model_config.yaml

# 5. Score models
python score_models.py --config config/scoring_config.yaml --truth-set data/true_values.csv --estimates-path results/forecasts/ --output-path results/scores/
```

## Notes

- Make sure all required input files are in the specified formats
- For SLURM-based job submission, ensure your SLURM environment is properly configured
- Check the generated configuration files to verify settings before running models

## License

[Your license information here]
