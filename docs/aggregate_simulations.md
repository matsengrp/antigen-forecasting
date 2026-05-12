# Aggregating Simulation Results

`scripts/aggregate_simulations.py` assembles a per-simulation summary table (`sim_stats.csv`)
for all completed runs in a batch by joining four data sources per simulation.

## Output

`sim_stats.csv` is written to `experiments/<experiment>/sim_stats.csv` by default.
One row per simulation. Frozen column order:

| Column(s) | Source |
|-----------|--------|
| `path`, `run`, `param_set`, `run_id` | Identifiers derived from the build string |
| `<param cols, sorted>` | Union of `parameters.yml` and param-set directory name |
| `n_unique_sequences` | Row count of `data/<build>/antigen-outputs/unique_tips.csv` |
| `<summary stat cols>` | All parameter/value pairs from `run-out.summary` (pivoted wide) |
| `antigenic_movement_per_year` | Sliding-window antigenic drift (from `unique_tips.csv`) |
| `trunk_*`, `side_branch_*` | Epitope/non-epitope mutation counts from `run-out.branches` |

## Usage

```bash
python scripts/aggregate_simulations.py \
    --experiments-root /path/to/antigen-experiments/experiments/ \
    --experiment 2026-01-06-mutation-bug-fix-runs \
    --data-root /path/to/antigen-forecasting/data/ \
    --batch-name 2026-04-29-all-sims \
    --output /path/to/antigen-experiments/experiments/2026-01-06-mutation-bug-fix-runs/sim_stats.csv
```

`--output` is optional and defaults to `<experiments-root>/<experiment>/sim_stats.csv`.

## Parameter cross-validation

Parameters are read from two sources and symmetrically validated:

| Case | Behaviour |
|------|-----------|
| Same key, same value | OK — included in output |
| Same key, different values | **Fail loudly** (names the sim, key, and both values) |
| Key only in directory name | **Fail loudly** (signals a stale rename) |
| Key only in `parameters.yml` | Warn but continue |

## Missing data handling

- `run-out.summary` missing → warn and skip the simulation
- `run-out.branches` missing → warn and set all branch columns to `NaN`
- No complete simulations found → fail with a clear error

## Running after the pipeline

This script should be invoked once after `run_all_simulations.py` (or individual
`run_pipeline.py` runs) have completed for the batch:

```bash
# 1. Run all simulations
python scripts/run_all_simulations.py --batch-name 2026-04-29-all-sims ...

# 2. Aggregate simulation stats
python scripts/aggregate_simulations.py --batch-name 2026-04-29-all-sims ...
```
