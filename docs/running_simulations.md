# Running Simulations

`scripts/run_all_simulations.py` is the batch entry point for processing all simulations
under a given experiment / parameter-set directory. It discovers simulation directories,
skips already-complete runs, then either executes `run_pipeline.py` locally (with
optional parallelism) or writes SLURM array-job artifacts and exits.

---

## Usage

```bash
python scripts/run_all_simulations.py \
    --experiments-root PATH \
    --experiment EXP \
    --param-set PARAM_SET \
    --batch-name BATCH \
    --config configs/pipeline_config.yaml \
    --mode [local | slurm] \
    [--max-parallel N] \
    [--slurm-config configs/slurm_config.yaml]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--experiments-root` | Root of the `antigen-experiments/experiments/` directory |
| `--experiment` | Experiment directory name (e.g. `2026-01-06-mutation-bug-fix-runs`) |
| `--param-set` | Parameter-set subdirectory name |
| `--batch-name` | Human-supplied batch name; namespaces outputs under `results/` |
| `--config` | Path to `pipeline_config.yaml` |
| `--mode` | `local` or `slurm` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-parallel` | `1` | Max concurrent subprocesses in local mode |
| `--slurm-config` | `None` | Path to `slurm_config.yaml`; uses built-in defaults if omitted |

---

## Modes

### Local mode

Runs `run_pipeline.py` for each incomplete simulation as a subprocess.

- `--max-parallel 1` (default): serial execution, equivalent to a manual loop.
- `--max-parallel N` (N > 1): uses `ProcessPoolExecutor`; automatically passes
  `--set-thread-caps` to each subprocess to prevent BLAS oversubscription
  (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `JAX_PLATFORMS=cpu`
  are set inside each worker).

```bash
# Serial
python scripts/run_all_simulations.py \
    --experiments-root /data/antigen-experiments/experiments \
    --experiment 2026-01-06-mutation-bug-fix-runs \
    --param-set nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0 \
    --batch-name my-batch \
    --config configs/pipeline_config.yaml \
    --mode local

# Parallel (4 workers)
python scripts/run_all_simulations.py \
    ... \
    --mode local \
    --max-parallel 4
```

### SLURM mode

Writes `sim_list.txt` and `submit_array.sh` under
`results/<batch-name>/slurm_submission_<timestamp>/` then exits without calling `sbatch`.

```bash
python scripts/run_all_simulations.py \
    --experiments-root /data/antigen-experiments/experiments \
    --experiment 2026-01-06-mutation-bug-fix-runs \
    --param-set nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0 \
    --batch-name my-batch \
    --config configs/pipeline_config.yaml \
    --mode slurm \
    --slurm-config configs/slurm_config.yaml

# Then submit manually:
sbatch results/my-batch/slurm_submission_<timestamp>/submit_array.sh
```

---

## Simulation Discovery

Globs `<experiments-root>/<experiment>/<param-set>/run_*/` for directories and
validates each contains both required input files before enqueuing:

- `output/run-out.tips`
- `out_timeseries.csv`

Discovery never reads from `data/` or `results/`.

---

## Skip Logic

A simulation is considered **complete** — and skipped — only when **both** sentinel
files exist under `results/<batch-name>/<param-set>__<run-id>/`:

- `scores.tsv`
- `growth_rate_scores.tsv`

One skipped-simulation log line is printed per skipped sim. Resubmitting after partial
completion produces a shorter `sim_list.txt` covering only the remaining sims.

---

## SLURM Configuration

`configs/slurm_config.yaml` controls the generated SLURM script. Edit it before
submitting on a new cluster.

```yaml
slurm:
  partition: campus-new
  time: "08:00:00"
  mem_gb: 32
  cpus_per_task: 4
  max_concurrent: 20
  conda_env: antigen
  log_dir: logs/slurm/  # relative to project_root
  project_root: /fh/fast/matsen_e/<yourname>/antigen-forecasting  # update before submitting
  python_bin: python  # override if conda activate does not put python on PATH
```

| Key | Description |
|-----|-------------|
| `partition` | SLURM partition name |
| `time` | Wall-clock time limit per task |
| `mem_gb` | Memory per task in GB |
| `cpus_per_task` | CPUs per array task |
| `max_concurrent` | Max simultaneously running array tasks |
| `conda_env` | Conda environment to activate |
| `log_dir` | Directory for `.out`/`.err` logs (relative to `project_root`) |
| `project_root` | Absolute path to the repo on the cluster |
| `python_bin` | Python executable to use |

If `--slurm-config` is omitted, built-in defaults are used (repo-local paths — almost
certainly wrong on the cluster; always pass the config for cluster runs).

---

## Outputs

### Local mode

Per-simulation outputs land in the standard locations written by `run_pipeline.py`:

```
data/<build>/
results/<batch-name>/<param-set>__<run-id>/scores.tsv
results/<batch-name>/<param-set>__<run-id>/growth_rate_scores.tsv
```

### SLURM mode

```
results/<batch-name>/slurm_submission_<timestamp>/
    sim_list.txt        # one sim path per line (1-indexed, matches array task IDs)
    submit_array.sh     # executable SLURM array script
```

`sim_stats.csv` is written separately by `aggregate_simulations.py` and never by this script.
