# Forecasting with `antigen-prime` simulated data

This repository contains code, scripts, analysis notebooks, and data for forecasting variant frequencies using simulated pathogen evolution data from `antigen-prime` and growth advantage models from `evofr`.

## Overview

The workflow typically follows these steps:

1. Parse `antigen-prime` simulation outputs using `parse_sim_outputs.py`
2. Prepare antigen data using `prep_antigen_data.py`
3. Assign variants using `assign_all_variants.py`
4. Create training datasets with `make_training_data.py`
5. Run forecasting models using either `run_model.py` directly or `sbatch_models.py` for SLURM-based environments
6. Score and evaluate models:
   - `score_models.py` for frequency estimates
   - `score_growth_rates.py` for growth rate estimates

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
pip install -e .
```

## Data

The primary dataset used in this analysis is in `data/flu-final/`, which contains:
- `time-stamped/` - Sequence and case counts at bi-annual time points (April 1 and October 1)
- `antigen-outputs/` - Processed simulation outputs
- `variant-assignment/` - Variant assignment results from phylogenetic and sequence-based methods
- `tips_with_variants.tsv` - Tip data with assigned variants

## Usage

All scripts are in the `scripts/` directory.

### 1. Parse Simulation Outputs

The `parse_sim_outputs.py` script parses raw `antigen-prime` simulation outputs (`run-out.tips`) into pipeline-ready CSV/FASTA artifacts. Replaces the manual notebook step previously used for this purpose.

```bash
python scripts/parse_sim_outputs.py --sim-path path/to/run_N/ --output-dir data/build/antigen-outputs/
```

#### Options:

- `--sim-path` - Path to a `run_N/` directory under `antigen-experiments/`
- `--output-dir` - Directory to write parsed outputs (typically `data/<build>/antigen-outputs/`)

#### Outputs:

- `tips.csv` - Full parsed tips with `country` column added (no dedup)
- `unique_tips.csv` - Deduplicated on `name` then `nucleotideSequence` (in this exact order)
- `unique_sequences.fasta` - FASTA of unique sequences
- `out_timeseries.csv` - Symlink to `run_N/out_timeseries.csv` (copy fallback)

#### Example:

```bash
python scripts/parse_sim_outputs.py \
    --sim-path antigen-experiments/experiments/<exp>/<param-set>/run_0/ \
    --output-dir data/flu-final/antigen-outputs/
```

### 2. Prepare Antigen Data

The `prep_antigen_data.py` script prepares antigen-prime simulated data for variant frequency forecasting.

```bash
python scripts/prep_antigen_data.py -t path/to/tips.csv -c path/to/cases.csv -o output/directory
```

#### Options:

- `-t, --tips` - Path to the tips file
- `-c, --cases` - Path to the cases file
- `-o, --output` - Path to the output directory for prepped sequence and case counts
- `-sd, --start_date` - Start date for the analysis
- `-v, --variant-col-name` - Name of the column containing the variant information
- `-m, --variant-mapping` - Path to variant mapping file (joins on nucleotideSequence)
- `-n, --normalize_cases` - Normalize case counts to report cases per 100k hosts
- `--deme_population_size` - Population size of the deme

#### Example:

```bash
python scripts/prep_antigen_data.py \
    -t data/antigen_h3n2_sim/antigen-outputs/tips.csv \
    -c data/antigen_h3n2_sim/antigen-outputs/cases.csv \
    -o data/antigen_h3n2_sim/time-stamped/truth/ \
    -sd 2022-01-01 -v variant_name --normalize_cases
```

### 3. Assign Variants

The `assign_all_variants.py` script assigns variant labels using three methods: antigenic (k-means), sequence-based (t-SNE), and phylogenetic (Augur clades).

```bash
python scripts/assign_all_variants.py \
    --tips data/build/tips.tsv \
    --fasta data/build/sequences.fasta \
    --output data/build/tips_with_variants.tsv \
    --work-dir data/build/variant_assignment/
```

#### Options:

- `--tips` - Path to the tips TSV file
- `--fasta` - Path to the sequences FASTA file
- `--output` - Path to save the combined tips DataFrame with variant columns
- `--work-dir` - Working directory for intermediate files

### 4. Create Training Data

The `make_training_data.py` script splits variant and count data into training windows.

```bash
python scripts/make_training_data.py -s path/to/sequences.csv -c path/to/cases.csv -o output/directory
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
python scripts/make_training_data.py \
    -s data/antigen_h3n2_sim/time-stamped/truth/sequences.csv \
    -c data/antigen_h3n2_sim/time-stamped/truth/cases.csv \
    -o data/antigen_h3n2_sim/training/ \
    --window-size 90 --buffer-size 14 \
    --config-path configs/training_config.yaml
```

### 5. Run Forecasting Models

#### Option A: Run a single model

The `run_model.py` script fits a specified forecasting model.

```bash
python scripts/run_model.py -d data/directory -c country_name -m model_name -o output/directory
```

#### Options:

- `-d, --data_path` - Path to the data directory
- `-c, --country` - Country/deme to fit the model to
- `-m, --model` - Model to fit (e.g., FGA, NAIVE)
- `-o, --output_dir` - Directory to save the model results
- `--forecast_L` - Number of days to forecast
- `--seed_L` - Number of days to seed the forecast with
- `--config` - Path to JSON configuration file with model parameters
- `--naive-full-window` - Use full training window for NAIVE model (overrides config)

#### Example:

```bash
python scripts/run_model.py \
    -d data/antigen_h3n2_sim/time-stamped/2025-10-01/ \
    -c north -m FGA \
    -o results/antigen_h3n2_sim/estimates/ \
    --forecast_L 28 --seed_L 7
```

#### Option B: Run multiple models via SLURM

The `sbatch_models.py` script submits forecasting model inference jobs via SLURM.

```bash
python scripts/sbatch_models.py -b build_name -c configs/benchmark_config.yaml
```

#### Options:

- `-b, --build` - Name of the build of the desired analysis
- `-c, --config` - Path to the YAML configuration file

#### Example:

```bash
python scripts/sbatch_models.py -b antigen_h3n2_sim -c configs/benchmark_config.yaml
```

### 6. Evaluate Model Performance

#### Score frequency predictions

The `score_models.py` script computes frequency prediction scores with variant filtering.

```bash
python scripts/score_models.py \
    --config configs/benchmark_config.yaml \
    --truth-set path/to/truth.csv \
    --estimates-path path/to/estimates/ \
    --output-path results/scores/
```

#### Options:

- `--config` - Path to the configuration file with scoring parameters
- `--truth-set` - Path to the truth set of sequences
- `--estimates-path` - Path to the model estimates directory
- `--output-path` - Path to save the output scores
- `--verbose` - Enable verbose logging
- `--log-file` - Path to log file (optional)

#### Score growth rate predictions

The `score_growth_rates.py` script performs comprehensive growth rate analysis.

```bash
python scripts/score_growth_rates.py \
    --config configs/benchmark_config.yaml \
    --build build_name \
    --output-dir results/growth_rates/
```

#### Options:

- `--config` - Path to the configuration file
- `--build` - Build name (e.g., flu-simulated-150k-samples)
- `--output-dir` - Directory to save the analysis outputs

## Complete Pipeline Example

Here's an example of running the complete pipeline:

```bash
# 1. Parse antigen-prime simulation outputs
python scripts/parse_sim_outputs.py \
    --sim-path antigen-experiments/experiments/<exp>/<param-set>/run_0/ \
    --output-dir data/antigen_h3n2_sim/antigen-outputs/

# 2. Prepare antigen data
python scripts/prep_antigen_data.py \
    -t data/antigen_h3n2_sim/antigen-outputs/tips.csv \
    -c data/antigen_h3n2_sim/antigen-outputs/out_timeseries.csv \
    -o data/prepped/ -sd 2022-01-01 -v variant_name

# 3. Assign variants
python scripts/assign_all_variants.py \
    --tips data/prepped/tips.tsv \
    --fasta data/antigen_h3n2_sim/antigen-outputs/unique_sequences.fasta \
    --output data/prepped/tips_with_variants.tsv \
    --work-dir data/prepped/variant_assignment/

# 4. Create training data
python scripts/make_training_data.py \
    -s data/prepped/seq_counts.tsv -c data/prepped/case_counts.tsv \
    -o data/training/ --window-size 90

# 5. Run models (individual or batch)
python scripts/run_model.py \
    -d data/training/2025-10-01/ -c north -m FGA \
    -o results/estimates/
# Or with SLURM
python scripts/sbatch_models.py -b antigen_h3n2_sim -c configs/benchmark_config.yaml

# 6. Score frequency predictions
python scripts/score_models.py \
    --config configs/benchmark_config.yaml \
    --truth-set data/prepped/seq_counts.tsv \
    --estimates-path results/estimates/ \
    --output-path results/scores/

# 7. Score growth rate predictions
python scripts/score_growth_rates.py \
    --config configs/benchmark_config.yaml \
    --build antigen_h3n2_sim \
    --output-dir results/growth_rates/
```

## Notes

- Make sure all required input files are in the specified formats
- For SLURM-based job submission, ensure your SLURM environment is properly configured
- Check the generated configuration files to verify settings before running models
- For programmatic path construction across pipeline stages, use `antigentools.paths.SimulationPaths` (`from_sim_path` for batch runs, `from_build` for single-dataset runs like `flu-final`) — it is the single source of truth for where files live

## Clustering Evaluation Metrics

When comparing different clustering methods (e.g., antigenic, sequence-based, phylowave) for variant assignment, we evaluate them based on two main goals:

**Goal 1 (Internal Evaluation)**: Assess how well cluster assignments explain the variance in the fitness distribution while penalizing the number of clusters used.
We treat this as an ANOVA-like problem where cluster assignments are categorical predictors of fitness. 
Each cluster predicts fitness using its mean fitness value, and we evaluate both the variance explained (R²) and a version that penalizes methods for the number of clusters defined in the time window (adj-R²).

**Goal 2 (External Evaluation)**: When considering the antigenic clusters as ground truth, evaluate how well the sequence-based clusters recover the same cluster structure.
This measures whether alternative clustering methods (which may use different feature spaces) successfully identify the same underlying variant groupings.

### Metrics Summary

| Metric                                  | Type     | Range                     | Interpretation                                                                                                         | Why We're Using It                                                                                                                                                |
| --------------------------------------- | -------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R²** (R-squared)                      | Internal | [0, 1], higher is better  | Proportion of fitness variance explained by cluster assignments. 0 = no explanatory power, 1 = perfect prediction.     | **Goal 1**: Directly measures how well cluster assignments explain fitness variation. Complements BIC by showing raw predictive power without complexity penalty. |
| **Adj-R²** (Adjusted R²)                | Internal | (-∞, 1], higher is better | R² adjusted for number of clusters. Penalizes adding more clusters. Can be negative if model is very poor.             | **Goal 1**: Alternative to BIC that also penalizes complexity. Useful for comparing methods with different cluster counts.                                        |
| **NMI** (Normalized Mutual Information) | External | [0, 1], higher is better  | Information-theoretic measure of cluster agreement. 0 = independent, 1 = identical clustering. Adjusts for chance.     | **Goal 2**: Measures how much information about antigenic clusters is captured by sequence/phylowave clusters. Symmetric and normalized.                          |
| **ARI** (Adjusted Rand Index)           | External | [-1, 1], higher is better | Measures pairwise agreement adjusted for chance. 0 = random labeling, 1 = perfect match. Negative = worse than random. | **Goal 2**: Gold standard for comparing clusterings. Adjusts for chance agreement and doesn't assume cluster size balance.                                        |

### Usage Notes:

**Internal Metrics (Goal 1)**: We report BIC since it directly optimizes the fitness prediction vs. complexity trade-off by penalizing the number of inferred clusters. 
R² and Adj-R² provide interpretable supplementary information about prediction quality. 
The approach treats clusters as categorical variables in an ANOVA-like framework where each cluster's mean fitness serves as the predicted value for all members of that cluster.

**External Metrics (Goal 2)**: ARI is known as the most robust single metric for cluster agreement. 
NMI provides an information-theoretic perspective. 

## License

MIT License - see [LICENSE](LICENSE) for details.
