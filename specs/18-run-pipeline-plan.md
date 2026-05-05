# Issue #18 — `run_pipeline.py` single-simulation orchestrator

Implementation plan for the single-simulation orchestrator that runs all seven
pipeline steps in DAG order with idempotent re-runs via per-step sentinel files.

References:
- `specs/analysis-pipeline.md` (Issue 2)
- GitHub issue #18

## Files to create

### `configs/pipeline_config.yaml`
New config file. Schema:
```yaml
pipeline:
  experiments_root: ../antigen-experiments/experiments/
  data_root: data/
  results_root: results/
  window_size: 365
  buffer_size: 0
  models: [FGA, GARW]
  locations: [north, south, tropics]
  forecast_L: 180
  seed_L: 14
  numpyro_seed: 42
```

### `scripts/run_pipeline.py`
Top-level functions/classes:

- `def load_pipeline_config(path: Path) -> dict`
  - Input: yaml path
  - Returns: validated dict (all required keys present)
  - Example: `load_pipeline_config(Path("configs/pipeline_config.yaml"))` → `{"pipeline": {...}}`

- `@dataclass class StepResult`
  - Fields: `name: str`, `status: Literal["ok", "skipped", "failed"]`, `start_time: str`, `end_time: str`, `duration_sec: float`, `command: str`

- `@dataclass class Step`
  - Fields: `name: str`, `sentinels: list[Path]`, `force_clean: Callable[[], None]`, `run: Callable[[], None]` (raises on failure), `command_str: str`

- `def is_complete(step: Step) -> bool` — all sentinels exist
- `def execute_step(step: Step, dry_run: bool, force: bool) -> StepResult`
- `def write_pipeline_log(results: list[StepResult], path: Path) -> None` — TSV columns `step, start_time, end_time, duration_sec, status, command`

- `def build_steps(paths: SimulationPaths, cfg: dict, skip_va: bool, skip_fc: bool, benchmark_cfg_path: Path) -> list[Step]`
  - Wires the seven steps using existing subscript CLIs (no signature changes).
  - `--skip-variant-assignment` ⇒ pass `--fast` to `assign_all_variants.py` (still runs antigenic k-means baseline; skips sequence/phylogenetic).
  - `--skip-forecasting` ⇒ omit `run_model`, `score_models`, `score_growth_rates`.

- `def run_run_model_step(paths, cfg, env) -> None`
  - Iterates (date × location × model) from `cfg["pipeline"]["models"]`, `["locations"]`, and `make_training_data`'s manifest dates.
  - Spawns `run_model.py` per combo via `subprocess.run`.
  - On full success, writes `paths.estimates_manifest` with columns `date, location, model, estimate_path, convergence_status`.

- `def main(argv: Sequence[str] | None = None) -> int`
  - argparse:
    - `--sim-path` (required, Path)
    - mutex required group: `--batch-name STR | --build STR`
    - `--config configs/pipeline_config.yaml` (required)
    - `--skip-variant-assignment` (flag)
    - `--skip-forecasting` (flag)
    - `--dry-run` (flag)
    - `--force` (flag)
  - Constructs `SimulationPaths` via `from_sim_path` (batch) or `from_build` (single).
  - If env var `ANTIGEN_PIPELINE_PARALLEL=1` is set by parent runner, sets BLAS thread caps to 1.

CLI examples:
```
python scripts/run_pipeline.py --sim-path … --build flu-final --config configs/pipeline_config.yaml --dry-run
python scripts/run_pipeline.py --sim-path … --batch-name 2026-04-29-all-sims --config configs/pipeline_config.yaml --force
```

### `antigentools/tests/test_run_pipeline.py`
- `--dry-run` prints all 7 steps; subprocess never invoked (mock `subprocess.run`).
- Mutex: both `--build` and `--batch-name` → exit 2.
- Mutex: neither → exit 2.
- Sentinel skip: pre-create sentinels for steps 1–3 → those rows in `pipeline.log` show `skipped`; downstream steps run.
- `--force`: pre-create sentinels → all steps still execute, sentinels removed pre-step.
- `pipeline.log` schema column check.

### `antigentools/tests/test_assign_all_variants_dedup.py`
Test that `assign_all_variants.py` raises `AssertionError` when `--tips` has duplicates on `name` or `nucleotideSequence`.

### `antigentools/tests/test_make_training_data_manifest.py`
Test that `make_training_data.py` writes `MANIFEST.tsv` with one row per per-date subdir.

## Files to modify

### `scripts/assign_all_variants.py`
After `tips_df = pd.read_csv(args.tips, ...)`, add:
```python
n_name_dupes = len(tips_df) - tips_df["name"].nunique()
assert n_name_dupes == 0, (
    f"--tips must be deduplicated on 'name' (got {n_name_dupes} dupes); "
    f"pass unique_tips.csv from parse_sim_outputs.py, not the full tips.csv"
)
n_seq_dupes = len(tips_df) - tips_df["nucleotideSequence"].nunique()
assert n_seq_dupes == 0, (
    f"--tips must be deduplicated on 'nucleotideSequence' (got {n_seq_dupes} dupes)"
)
```
Remove the now-redundant internal `drop_duplicates(['nucleotideSequence'])`.

### `scripts/make_training_data.py`
After the `for timepoint in analysis_dates:` loop, write `MANIFEST.tsv` to `out_dir`:
columns `date, seq_counts_path, case_counts_path`, one row per analysis date that
produced output. Write last so its presence guarantees all per-date dirs are
complete.

### `scripts/score_growth_rates.py`
Rename the summary output `window_growth_rates.tsv` → `growth_rate_scores.tsv` to
match the spec / `SimulationPaths.growth_rate_scores`. Update the corresponding
`docs/empirical_growth_rates.md` row and the path in
`notebooks/manuscript-figure-5-growth-rate-errors.ipynb`.

## Decisions

- **Subprocess vs in-process**: subprocess for every step. Matches the "exact
  subprocess command" log column, isolates JAX state, and `--force` recovers
  cleanly from prior crashes.
- **`--skip-variant-assignment`**: passes `--fast` to `assign_all_variants.py`
  → antigenic (k-means on `ag1, ag2`) baseline still runs; sequence (t-SNE) and
  phylogenetic (augur) are skipped.
- **`run_model` MANIFEST**: written by the orchestrator after all (date,
  location, model) subprocesses succeed. `run_model.py` itself is unchanged.
- **`make_training_data` MANIFEST**: written by the subscript itself.
- **Truth-set for `score_models`**: `paths.seq_counts` (full unwindowed counts).
- **Regression-anchor bootstrap**: deferred to a follow-up PR. The
  score-equivalence acceptance criterion is not validated in this PR.
- **`--config` is required** (not optional as bracketed in the issue example) —
  `numpyro_seed`, `models`, `locations` are sourced from it.

## Out of scope

- `run_all_simulations.py` (Issue #19, depends on this PR).
- SLURM submission (Issue #20).
- Result aggregation (Issue #21).
- Regression-anchor regeneration (deferred follow-up).
