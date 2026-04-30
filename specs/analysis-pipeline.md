# Analysis Pipeline Specification

## Overview

This spec describes the work needed to scale the antigen-forecasting analysis pipeline
from a single hard-coded simulation dataset (`flu-final`) to a parameterized, batch-capable
pipeline that can process many simulation datasets on a SLURM cluster.
The pipeline has two major analysis tracks — variant assignment and frequency forecasting —
and this spec covers both.

---

## Background

### Current state

The pipeline currently processes one simulation dataset (`data/flu-final/`) via a sequence
of manually invoked scripts:

1. *(notebook step)* Parse `run-out.tips` into a CSV, deduplicate on `name` then
   `nucleotideSequence`, write `unique_tips.csv` to `antigen-outputs/`.
   This step needs to be a proper script.
2. `prep_antigen_data.py` — processes the full tips CSV and `out_timeseries.csv` into
   `seq_counts.tsv` and `case_counts.tsv`.
3. `assign_all_variants.py` — assigns sequences to variants via three methods:
   antigenic k-means, sequence t-SNE, and phylogenetic clades.
   Reads `unique_tips.csv` (not the full tips file).
4. `make_training_data.py` — slices the sequence and case counts into per-estimation-date
   training windows and generates a `benchmark_config.yaml`.
5. `run_model.py` / `sbatch_models.py` — fits FGA/GARW forecasting models per
   (date, location, model) combination.
6. `score_models.py` — computes frequency-prediction scores (R², Adj-R²) against truth.
7. `score_growth_rates.py` — computes growth-rate scores (NMI, ARI).

Each script currently expects specific paths or infers them relative to the project root.
There is no top-level entry point that runs all steps for a given simulation.

**Tips file usage** (read this carefully — the two tracks consume different files):

- **Variant assignment** (`assign_all_variants.py`) operates on `unique_tips.csv`
  only — the deduplicated frame (~4.4k rows, unique on `name` then
  `nucleotideSequence`). The k-means / t-SNE / phylogenetic methods would be
  meaningless on the full tips because duplicate sequences would dominate the
  clustering. The script writes `tips_with_variants.tsv` (one row per unique
  sequence, variant labels attached).
- **Forecasting prep** (`prep_antigen_data.py`) operates on the full `tips.csv`
  (~150k rows, no dedup) joined to `tips_with_variants.tsv` via `--variant_mapping`
  on `nucleotideSequence`. The full tips are needed to count sequences over time;
  the join propagates each unique sequence's pre-computed variant label to all of
  its full-tips rows. Without the join, `prep_antigen_data.py`'s
  `groupby(['date', 'country', 'variant'])` crashes because parsed tips have no
  `variant` column.

The new parsing script (Issue 1) writes both `tips.csv` and `unique_tips.csv` so all
downstream scripts read from `data/<build>/antigen-outputs/` (or
`data/<build>/tips_with_variants.tsv` for the variant mapping) and never reach back
into `antigen-experiments`. **Variant assignment is never recomputed on the full
tips** — it happens once, on unique tips, and the labels are reused via join.

### Data layout

Raw antigen-prime outputs (in `antigen-experiments`):

```
experiments/<experiment-date>/
    sim_stats.csv                # per-experiment summary stats; rows keyed by
                                 # (run, epitopeAcceptance, nonEpitopeAcceptance, ...)
    <param-set>/
        parameters.yml           # canonical parameter values
        run_N/
            output/
                run-out.tips     # ~150k rows; full simulation tips (no extension)
                run-out.fasta    # full nucleotide sequences
                run-out.branches
                run-out.range
                run-out.summary
                run-out.trees
            out_timeseries.csv   # case counts; lives directly in run_N/
```

Note: `sim_stats.csv` lives at the experiment level (one file per experiment, indexed
by run + parameter values), not inside each `run_N/`.

Processed outputs written by the pipeline (in `antigen-forecasting`):

```
data/<build>/
    antigen-outputs/
        tips.csv                 # full parsed tips from run-out.tips (no dedup)
        unique_tips.csv          # deduplicated: name first, then nucleotideSequence
        unique_sequences.fasta   # FASTA of sequences in unique_tips.csv
        out_timeseries.csv       # copy of run_N/out_timeseries.csv
    variant-assignment/          # outputs of assign_all_variants.py
        metadata.tsv
        phylogenetic/
        sequence/
    time-stamped/                # outputs of make_training_data.py
        <YYYY-MM-DD>/
            seq_counts.tsv
            case_counts.tsv
        .complete                # sentinel file written after all dates emitted
    seq_counts.tsv               # full (un-windowed) counts from prep_antigen_data.py
    case_counts.tsv
    tips_with_variants.tsv       # unified variant-assignment output
    auspice/
    ref_HA.fasta

results/<build>/
    estimates/                   # per-(date, location, model) frequency estimates
    growth-rates/
    convergence-diagnostics/
    slurm/                       # SLURM job logs
    fitness_variance.tsv
    variant_growth_rates.tsv
    window_growth_rates.tsv
    vi_convergence_diagnostics.tsv
    scores.tsv                   # written by score_models.py
    growth_rate_scores.tsv       # written by score_growth_rates.py
    pipeline.log                 # TSV step timings (see Issue 2)

results/<batch-name>/aggregate.tsv   # batch-level aggregate (Issue 5)
```

**Build name convention:**

A simulation is identified by its path in `antigen-experiments`.
The `build` is an opaque path fragment used for `data/<build>/` and `results/<build>/`.

- Single-dataset case: `build` is a human-chosen name, e.g. `flu-final`.
- Batch case: `build = <batch-name>/<sim-id>` where `sim-id = <param-set>__<run-id>`,
  e.g. `2026-04-29-all-sims/nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0__run_0`.

The `<batch-name>` is human-supplied at submission time (only free-form input to
`run_all_simulations.py`).
The `<sim-id>` is auto-derived from `sim_path` via `SimulationPaths.from_sim_path`.

```
data/<build>/
results/<build>/
results/<batch-name>/aggregate.tsv
```

No script needs to know whether it is running in single or batch mode — the build string
is opaque to them.

### CLI contract (applies to all scripts)

Every pipeline script takes **resolved paths** as CLI arguments, not a `--build` flag.
`SimulationPaths` is an internal helper used by `run_pipeline.py` to construct those
paths from `--sim-path` + `--batch-name` (or `--build` directly for the single-dataset
case) and forward them to subscripts.
This keeps subscript CLIs unchanged from their current per-path form (e.g.
`prep_antigen_data.py -t TIPS -c CASES -o OUT`) and makes individual scripts trivially
testable in isolation.

### Remote simulation inventory

On the remote machine (Fred Hutch cluster), simulation outputs live in `antigen-experiments`
under `experiments/<experiment-date>/<param-set>/run_N/`.
Parameter set names encode simulation parameters directly, e.g.
`nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0`.
The goal is to run the full pipeline on every `run_N` independently, collect per-run
scores, and aggregate them for comparison across parameter sets and experiments.

---

## Scope

**In scope:**
- Parameterizing all scripts so any simulation name can be passed in.
- A single-simulation orchestration entry point (`run_pipeline.py`).
- A multi-simulation runner that calls `run_pipeline.py` once per simulation.
- A SLURM array-job submission script for batch processing on the cluster.
- A result aggregation script that collects scores across simulations.

**Out of scope:**
- Changes to the statistical models themselves (`evofr`, `numpyro` internals).
- Modifying how individual simulation outputs are generated by `antigen-prime`.
- Visualization or notebook updates (separate concern, but `aggregate.tsv` schema is
  frozen as a downstream contract — see Issue 5).

---

## Design

### Issue 1 — Add `parse_sim_outputs.py` and `SimulationPaths`

**Part A: `scripts/parse_sim_outputs.py`**

Raw antigen-prime output files (`run-out.tips`, `run-out.fasta`) use a bare filename
convention with no extension and an antigen-prime-specific format.
Converting them to properly named `.csv`/`.tsv` files is currently done manually inside
`analysis.ipynb` in each experiment folder.
This script replaces that notebook step, making the conversion reproducible and
scriptable as part of the pipeline.
Producing named `.csv`/`.tsv` outputs ensures that `prep_antigen_data.py`'s
extension-based separator detection works correctly downstream.

Before implementing, read `antigentools/antigen_reader.py` to understand the
`AntigenReader` interface (specifically `write_tips_to_fasta()`) and
`antigentools/__init__.py` to confirm what is exported.
Import as `from antigentools.antigen_reader import AntigenReader`.

**CLI:**

```
python scripts/parse_sim_outputs.py \
    --sim-path experiments/<experiment>/<param-set>/run_N \
    --output-dir data/<build>/antigen-outputs/
```

**Steps** (reference: `analysis.ipynb` in each experiment folder; verified dedup order
is `name` then `nucleotideSequence`):

1. Read `run-out.tips` with `pd.read_csv(path, sep=",")`.
   Assert that expected columns are present: `name`, `year`, `location`,
   `nucleotideSequence`, `ag1`, `ag2`.
2. Add `country` column by mapping `location` → `{0: "north", 1: "tropics", 2: "south"}`.
   This mapping is pinned: it matches the existing `analysis.ipynb` reference notebooks
   in every experiment folder verbatim. Do not vary it.
3. Write the full (non-deduplicated) frame as `tips.csv`.
   This is the file `prep_antigen_data.py` will consume via `-t`.
4. Build `unique_tips_df` by sequential deduplication, in this exact order:
   ```python
   unique_tips_df = tips_df.drop_duplicates(subset=["name"])
   unique_tips_df = unique_tips_df.drop_duplicates(subset=["nucleotideSequence"])
   ```
5. Write `unique_tips.csv`.
6. Write `unique_sequences.fasta` using `AntigenReader.write_tips_to_fasta()` from
   `unique_tips_df`.
7. Symlink `out_timeseries.csv` from `run_N/` into the output directory (same name).
   The source is read-only and a copy of a ~150k-row CSV per run wastes cluster disk.
   Fall back to a copy on filesystems that do not support symlinks (e.g., some
   network mounts) — detect by `try: os.symlink(...) except OSError: shutil.copy(...)`.

Do not add a `variant` column — variant assignment is handled downstream by
`assign_all_variants.py`, which writes `tips_with_variants.tsv`.
`prep_antigen_data.py` joins variant assignments via its `--variant_mapping` argument.

Assert `len(unique_tips_df) < len(tips_df)`.
Log row counts dropped at each deduplication step.

**Part B: `antigentools/paths.py` — `SimulationPaths` dataclass**

Centralize path logic in a single dataclass so path construction is testable and DRY.
A simulation is identified by its `sim_path` in `antigen-experiments`; the build string
is derived from that path plus a human-supplied `batch_name`.

```python
# antigentools/paths.py
@dataclass
class SimulationPaths:
    sim_path: Path        # path to run_N/ in antigen-experiments
    data_root: Path       # antigen-forecasting/data/
    results_root: Path    # antigen-forecasting/results/
    build: str            # opaque path fragment, e.g. "flu-final" or
                          # "2026-04-29-all-sims/nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0__run_0"

    @classmethod
    def from_sim_path(
        cls,
        sim_path: Path,
        data_root: Path,
        results_root: Path,
        batch_name: str,
    ) -> "SimulationPaths":
        """Derive sim-id from sim_path; combine with batch_name to form build.

        sim_id = f"{sim_path.parent.name}__{sim_path.name}"   # double underscore separator
        build  = f"{batch_name}/{sim_id}"

        Preconditions (asserted at construction):
        - "__" must NOT appear in sim_path.parent.name or sim_path.name. The "__"
          separator between param-set and run-id is the parsing anchor and must be
          unique within the build string. Violations raise ValueError with a message
          naming the offending segment.
        - sim_path.parent.name must NOT be empty.
        - batch_name must NOT contain "/" (it would corrupt the data/<build>/ split).
        """

    @classmethod
    def from_build(
        cls,
        build: str,
        sim_path: Path,
        data_root: Path,
        results_root: Path,
    ) -> "SimulationPaths":
        """Single-dataset constructor. build is passed through verbatim (e.g. 'flu-final').
        sim_path is still required because raw_* properties need it.
        """

    # Raw antigen-prime outputs (read-only)
    @property
    def raw_tips(self) -> Path: ...        # sim_path / "output" / "run-out.tips"
    @property
    def raw_fasta(self) -> Path: ...       # sim_path / "output" / "run-out.fasta"
    @property
    def raw_timeseries(self) -> Path: ...  # sim_path / "out_timeseries.csv"
    @property
    def raw_parameters_yml(self) -> Path:  # sim_path.parent / "parameters.yml"
        ...

    # Parsed outputs in data/<build>/antigen-outputs/ (Issue 1)
    @property
    def antigen_outputs(self) -> Path: ...      # data_root / build / "antigen-outputs"
    @property
    def tips(self) -> Path: ...                 # antigen_outputs / "tips.csv"
    @property
    def unique_tips(self) -> Path: ...          # antigen_outputs / "unique_tips.csv"
    @property
    def unique_sequences(self) -> Path: ...     # antigen_outputs / "unique_sequences.fasta"
    @property
    def timeseries(self) -> Path: ...           # antigen_outputs / "out_timeseries.csv"

    # Per-build derived outputs in data/<build>/
    @property
    def variant_assignment(self) -> Path: ...   # data_root / build / "variant-assignment"
    @property
    def tips_with_variants(self) -> Path: ...   # data_root / build / "tips_with_variants.tsv"
    @property
    def seq_counts(self) -> Path: ...           # data_root / build / "seq_counts.tsv"
    @property
    def case_counts(self) -> Path: ...          # data_root / build / "case_counts.tsv"
    @property
    def time_stamped(self) -> Path: ...         # data_root / build / "time-stamped"
    @property
    def time_stamped_manifest(self) -> Path:    # time_stamped / "MANIFEST.tsv"  (sentinel)
        ...

    # Model outputs in results/<build>/
    @property
    def results(self) -> Path: ...              # results_root / build
    @property
    def estimates(self) -> Path: ...            # results / "estimates"
    @property
    def estimates_manifest(self) -> Path:       # estimates / "MANIFEST.tsv"  (sentinel)
        ...
    @property
    def growth_rates(self) -> Path: ...         # results / "growth-rates"
    @property
    def scores(self) -> Path: ...               # results / "scores.tsv"
    @property
    def growth_rate_scores(self) -> Path: ...   # results / "growth_rate_scores.tsv"
    @property
    def pipeline_log(self) -> Path: ...         # results / "pipeline.log"
```

Each script that needs `SimulationPaths` constructs one from its CLI args at startup.
Subscripts (`prep_antigen_data.py`, etc.) keep their existing per-path CLIs;
`run_pipeline.py` is the only consumer that builds a `SimulationPaths` and forwards its
properties to subscripts.

`from_sim_path` and `from_build` should both assert that `sim_path` exists; per-property
existence checks are the caller's responsibility (we want path objects to be
constructible even when downstream files do not exist yet).

**Acceptance criteria:**
- `python scripts/parse_sim_outputs.py --sim-path <run_0> --output-dir data/flu-final/antigen-outputs/`
  produces `unique_tips.csv` whose rows match the existing committed
  `data/flu-final/antigen-outputs/unique_tips.csv` on:
  - row count, and
  - the **set of `(name, nucleotideSequence)` tuples**
    (`set(zip(new["name"], new["nucleotideSequence"])) == set(zip(...committed))`).
  This catches the case where dedup-on-name picks a different representative row
  while the `name` set alone is unchanged.
  Byte-for-byte equality is not required — column ordering and float formatting may
  differ from the manually produced reference.
- `unique_sequences.fasta` contains the same set of `(name → sequence)` pairs as the
  committed reference (compared as a dict, not as raw text — record order may differ).
- `tips.csv` row count equals the row count of `run-out.tips`.
- `python scripts/prep_antigen_data.py -t data/flu-final/antigen-outputs/tips.csv -c
  data/flu-final/antigen-outputs/out_timeseries.csv ...` produces output equivalent
  to the current invocation (column-set + per-column tolerance check on numerics).
- `SimulationPaths` has unit tests in `antigentools/tests/test_paths.py` covering:
  - Single-dataset case (`from_build("flu-final", ...)`).
  - Batch case with multi-underscore param-set
    (`from_sim_path("...experiments/2026-01-06/nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0/run_0", batch_name="2026-04-29-all-sims")`)
    yielding `build == "2026-04-29-all-sims/nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0__run_0"`.
  - All path properties resolve to the expected concrete paths.
- `parse_sim_outputs.py` has unit tests in `antigentools/tests/test_parse_sim_outputs.py`
  with a fixture containing intentional `name` and `nucleotideSequence` collisions to
  pin the dedup order.

---

### Issue 2 — Single-simulation orchestration script

Create `scripts/run_pipeline.py` that runs all seven pipeline steps in order for one
simulation, short-circuiting at any step whose sentinel output already exists
(idempotent re-runs).

**CLI:**

```
python scripts/run_pipeline.py \
    --sim-path experiments/<experiment>/<param-set>/run_N \
    (--batch-name <batch-name> | --build <build>) \
    --config configs/pipeline_config.yaml \
    [--skip-variant-assignment]
    [--skip-forecasting]
    [--dry-run]
    [--force]                               # ignore sentinels and rerun all steps
```

`--sim-path` is always required (subscripts need raw inputs).
`--batch-name` and `--build` form a **mutually exclusive required group** (argparse
`add_mutually_exclusive_group(required=True)`).
With `--batch-name`, `build` is derived as `<batch-name>/<sim-id>` via
`SimulationPaths.from_sim_path`.
With `--build`, the literal string is used (single-dataset mode, e.g. `flu-final`).
Passing both or neither must error before any work begins.

**Step DAG** (edge labels show the file each step consumes from its parent):

```
parse_sim_outputs
        │  unique_tips.csv
        ▼
assign_all_variants
        │  tips_with_variants.tsv
        ▼
prep_antigen_data ◄── tips.csv (full, from parse_sim_outputs)
        │  seq_counts.tsv, case_counts.tsv
        ▼
make_training_data
        │  time-stamped/<date>/{seq,case}_counts.tsv
        ▼
   run_model
        │  estimates/, growth-rates/
   ┌────┴────┐
   ▼         ▼
score_   score_
models   growth_rates
```

`prep_antigen_data` is the only step with two parents: `tips.csv` (full) from
`parse_sim_outputs` and `tips_with_variants.tsv` (variant labels keyed on
`nucleotideSequence`) from `assign_all_variants`. The join propagates labels; it
does not re-assign variants.

- `parse_sim_outputs` is the root, writing `tips.csv`, `unique_tips.csv`,
  `unique_sequences.fasta`, and copying `out_timeseries.csv`.
- `assign_all_variants` reads `unique_tips.csv` and writes `tips_with_variants.tsv`.
- `prep_antigen_data` reads `tips.csv` plus `tips_with_variants.tsv` (via
  `--variant_mapping`) and writes `seq_counts.tsv` and `case_counts.tsv`.
  It depends on `assign_all_variants` because the parsed tips have no `variant` column
  on their own.
- `make_training_data` reads `seq_counts.tsv` and `case_counts.tsv` and slices into
  per-date training windows.
- `run_model` is internally parallelized across (date, location, model).
- `score_models` and `score_growth_rates` are independent siblings consuming
  `run_model` outputs and may run concurrently.

Run steps sequentially in DAG order:
`parse_sim_outputs → assign_all_variants → prep_antigen_data → make_training_data →
run_model → {score_models, score_growth_rates}`.
The two scoring scripts are independent and may be parallelized in a follow-up; the
initial implementation runs them serially.

**Idempotency sentinels (per step):**

| Step | Sentinel (skip if exists) |
|---|---|
| `parse_sim_outputs` | `paths.unique_tips` |
| `assign_all_variants` | `paths.tips_with_variants` |
| `prep_antigen_data` | `paths.seq_counts` AND `paths.case_counts` (both required) |
| `make_training_data` | `paths.time_stamped / "MANIFEST.tsv"` (see script-modifications below) |
| `run_model` | `paths.estimates / "MANIFEST.tsv"` (see script-modifications below) |
| `score_models` | `paths.scores` |
| `score_growth_rates` | `paths.growth_rate_scores` |

`--force` removes the affected output directory or sentinel file before invoking each
step (e.g., `rm -rf time-stamped/`, `rm -f scores.tsv`) so partial outputs from a prior
run with different parameters do not contaminate the new run.
`--force` does not delete inputs (raw `run-out.tips`, `data/<build>/antigen-outputs/`).

**Required modifications to existing scripts (sub-tasks of Issue 2):**

The first item is a defensive contract; the next two turn currently-implicit
completion signals into explicit, testable sentinels.
The manifests double as audit logs of what each step actually produced.

1. `scripts/assign_all_variants.py`: assert at script entry, immediately after loading
   `--tips`, that the input is already deduplicated on both keys. The script's
   internal `drop_duplicates(['nucleotideSequence'])` (single column) silently
   diverges from the canonical `name → nucleotideSequence` two-step dedup if it is
   ever fed the full `tips.csv` instead of `unique_tips.csv` — different
   representative rows survive on `name` collisions, no exception raised. The
   assertion closes that path-of-least-resistance failure mode:
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
   The internal `drop_duplicates` call may then be removed (the assertion makes it
   provably a no-op) or kept as belt-and-suspenders — implementer's choice.
2. `scripts/make_training_data.py`: after all per-date subdirectories are written,
   emit `time-stamped/MANIFEST.tsv` with columns `date`, `seq_counts_path`,
   `case_counts_path`, one row per date.
   Write the manifest **last**, so its presence proves all prior writes finished.
3. `scripts/run_model.py` (or `sbatch_models.py`, whichever is the entry point used by
   `run_pipeline.py`): after the final (date, location, model) job completes, emit
   `estimates/MANIFEST.tsv` with columns `date`, `location`, `model`, `estimate_path`,
   `convergence_status`, one row per fitted model.
   When `run_model.py` is parallelized, the manifest is written by the orchestrator
   that joins on completion of all child jobs, not by individual child jobs.

`MANIFEST.tsv` is preferred over hidden-file markers (`.complete`) so the sentinels
appear in standard listings and `tar` archives without special flags.

**Other requirements:**
- After each step completes, assert its sentinel(s) exist on disk; fail loudly otherwise.
- Write `paths.pipeline_log` as a TSV with columns:
  `step`, `start_time` (ISO 8601), `end_time`, `duration_sec`, `status` (`ok` |
  `skipped` | `failed`), `command` (the exact subprocess command run, or empty for
  skipped steps).
- `--dry-run` prints the planned step sequence (with skip decisions based on existing
  sentinels) and exits without executing.
- When `--max-parallel > 1` is in effect via the parent (Issue 3), `run_pipeline.py`
  itself sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `JAX_PLATFORMS=cpu` in subprocess env to prevent BLAS oversubscription.
  When run standalone (no parent runner), it does not modify thread env vars.

**New config file: `configs/pipeline_config.yaml`**

A single config covers both pipeline parameters and environment-specific roots.
Per-environment files (`configs/local.yaml`, `configs/remote.yaml`) are concrete
instantiations of this schema; there is no separate `local.yaml` / `remote.yaml` schema.

```yaml
pipeline:
  experiments_root: /path/to/antigen-experiments/experiments/
  data_root: data/
  results_root: results/
  window_size: 365
  buffer_size: 0
  models: [FGA, GARW]
  locations: [north, south, tropics]
  forecast_L: 180
  seed_L: 14
  numpyro_seed: 42        # frozen RNG seed for VI; required for score equivalence
```

CLI flags (`--experiments-root`, etc., on `run_all_simulations.py`) override config
values when both are present.

**Acceptance criteria:**
- `python scripts/run_pipeline.py --sim-path <flu-final-sim> --build flu-final --dry-run`
  prints all seven steps with correct skip decisions.
- A full run on `flu-final` produces `scores.tsv` whose numeric columns match the
  regression-anchor `results/flu-final/scores.tsv` to `np.allclose(rtol=1e-3, atol=1e-4)`
  (VI is stochastic; tolerance acknowledges library-version drift even with a frozen
  seed).
  Non-numeric columns must match exactly.
- **Regression anchor provenance:** before this acceptance test runs, regenerate
  `results/flu-final/scores.tsv` once with `numpyro_seed: 42` (the value pinned in
  `pipeline_config.yaml`) and commit it as the new anchor. The currently committed
  file may have been produced with an unspecified seed; relying on it as-is would
  make this criterion fail despite a correct implementation. This regeneration is a
  one-time bootstrap step done by the Issue 2 implementer before submitting the PR.
- Re-running an already-complete pipeline marks every step `skipped` in `pipeline.log`
  and exits within a few seconds.
- `--force` reruns every step and overwrites sentinels without error.

---

### Issue 3 — Multi-simulation runner

Create `scripts/run_all_simulations.py` that discovers simulation directories in
`antigen-experiments` and invokes `run_pipeline.py` for each, either sequentially
(local mode) or by submitting SLURM jobs (cluster mode).

**CLI:**

```
python scripts/run_all_simulations.py \
    --experiments-root /fh/fast/matsen_e/zthornto/antigen-experiments/experiments/ \
    --experiment 2026-01-06-mutation-bug-fix-runs \
    --param-set nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0 \
    --batch-name 2026-04-29-all-sims \
    --config configs/pipeline_config.yaml \
    --mode [local | slurm] \
    [--max-parallel 4] \
    [--slurm-config configs/slurm_config.yaml]
```

**Discovery:** glob `<experiments-root>/<experiment>/<param-set>/run_*/` (matching
directories only) and assert each contains `output/run-out.tips` and `out_timeseries.csv`
before enqueuing.
Discovery operates entirely within `antigen-experiments`, never `data/`.

`--experiment` and `--param-set` are both **required, single-valued** arguments per
invocation. Sweeping multiple param-sets within one experiment is done by invoking
`run_all_simulations.py` once per param-set (typically scripted by a shell loop).
This keeps `sim_list.txt` and `aggregate.tsv` scoped to a single param-set, which
matches how the cluster-side aggregation is consumed downstream.

**Skip logic:** a simulation is considered complete if
`results/<batch-name>/<sim-id>/scores.tsv` AND
`results/<batch-name>/<sim-id>/growth_rate_scores.tsv` both exist.
Print one line per skipped sim.
Resubmissions therefore naturally narrow to incomplete sims.

**Local mode:**
- `--max-parallel N`: spawn at most N concurrent `run_pipeline.py` subprocesses
  (e.g., via `concurrent.futures.ProcessPoolExecutor`).
- For `N > 1`, set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `JAX_PLATFORMS=cpu` in the subprocess env to prevent BLAS oversubscription.
- For `N == 1`, env is unchanged.

**Concurrent-write safety:**
- Per-sim outputs (`data/<build>/`, `results/<build>/`, `pipeline.log`) are isolated
  by construction — different `<sim-id>` segments — so concurrent `run_pipeline.py`
  instances never share a write target.
- `results/<batch-name>/aggregate.tsv` is written **only** by `aggregate_results.py`
  (Issue 5), which is invoked once after the batch completes, never by
  `run_pipeline.py`. `run_all_simulations.py` does not invoke aggregation.
- `results/<batch-name>/slurm_submission_<timestamp>/` is per-invocation (timestamped),
  so resubmissions never clobber prior submission artifacts.

**SLURM mode:** see Issue 4.

**Acceptance criteria (agent-verifiable):**
- `--mode local --max-parallel 1` runs simulations serially and is equivalent to
  calling `run_pipeline.py` in a loop.
- `--mode slurm` writes `sim_list.txt` and `submit_array.sh` under
  `results/<batch-name>/slurm_submission_<timestamp>/` and exits without error
  (no `sbatch` invocation; cluster access is not assumed during agent runs).
- Skipping logic: pre-creating empty `scores.tsv` and `growth_rate_scores.tsv` for one
  sim causes that sim to be omitted from `sim_list.txt`.

**Acceptance criteria (manual, on cluster):**
- `--mode slurm` successfully submits the array job on the Fred Hutch cluster.
- Resubmitting after partial completion produces a shorter `sim_list.txt`.

---

### Issue 4 — SLURM array job submission

Create `configs/slurm_config.yaml` and extend `run_all_simulations.py` to generate and
submit a SLURM array job where each array index maps to one simulation.
CPU-only for now — JAX/numpyro use CPU; GPU can be added later if the per-sim wall-clock
on `flu-final` (record this baseline number when Issue 2 lands) becomes a bottleneck
for typical batches (~50 sims).

**Config:**

```yaml
# configs/slurm_config.yaml
slurm:
  partition: campus-new          # Fred Hutch general CPU partition
  time: "08:00:00"
  mem_gb: 32
  cpus_per_task: 4
  max_concurrent: 20             # max simultaneous array tasks (%N suffix)
  conda_env: antigen-forecasting
  log_dir: logs/slurm/           # relative to project_root
  project_root: /fh/fast/matsen_e/zthornto/antigen-forecasting
  python_bin: python             # leave default unless conda activate is non-standard
```

All paths in the generated SBATCH script are interpolated from this config — no
hardcoded paths in the generator.

**Generated `submit_array.sh` template:**

```bash
#!/bin/bash
#SBATCH --job-name=antigen-pipeline
#SBATCH --array=1-{N_SIMS}%{MAX_CONCURRENT}
#SBATCH --output={project_root}/{log_dir}/%A_%a.out
#SBATCH --error={project_root}/{log_dir}/%A_%a.err
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={cpus_per_task}

# 1-indexed array; sed line N == array task N
SIM_PATH=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {sim_list_file})
source activate {conda_env}
export OMP_NUM_THREADS={cpus_per_task}
export MKL_NUM_THREADS={cpus_per_task}
export OPENBLAS_NUM_THREADS={cpus_per_task}
export JAX_PLATFORMS=cpu
{python_bin} {project_root}/scripts/run_pipeline.py \
    --sim-path "$SIM_PATH" \
    --batch-name {batch_name} \
    --config {config}
```

Notes:
- `--array=1-N%M` with `sed -n "${{SLURM_ARRAY_TASK_ID}}p"` gives a clean 1-to-1
  mapping; no `+1` arithmetic required.
- Inside SLURM, each task gets its full `cpus_per_task` — set thread caps to
  `cpus_per_task`, not 1 (the parallelism budget is per-task here, unlike local mode).
- `sim_list.txt` lists one absolute `run_N/` path per line, written by
  `run_all_simulations.py` before submission.
- After regenerating `sim_list.txt` to skip already-complete sims, `N_SIMS` shrinks
  accordingly — partial-failure recovery is just "rerun `--mode slurm`".

**Acceptance criteria (agent-verifiable):**
- `--mode slurm` writes `sim_list.txt` and `submit_array.sh` under
  `results/<batch-name>/slurm_submission_<timestamp>/`.
- The generated `submit_array.sh`:
  - is syntactically valid bash (`bash -n submit_array.sh` exits 0);
  - contains the expected `#SBATCH` headers (assert via string inspection);
  - contains no template placeholders (`{...}`) in the rendered output.
- `sim_list.txt` line count equals `N_SIMS` in the `--array` directive.
- The first and last lines of `sim_list.txt` (i.e., `sed -n "1p"` and
  `sed -n "${N_SIMS}p"`) are non-empty absolute paths that exist on disk and contain
  `output/run-out.tips`. This catches off-by-one errors in the `--array=1-N` range.

**Acceptance criteria (manual, on cluster):**
- `sbatch --test-only submit_array.sh` validates SLURM syntax and resource requests
  without queuing.
- A real array submission completes without scheduler errors.
- Resubmitting after partial completion produces a new array covering only the
  remaining incomplete simulations.

---

### Issue 5 — Result aggregation across simulations

Create `scripts/aggregate_results.py` that assembles a per-simulation summary table by
joining four data sources for each completed simulation in a batch:

1. **Parameter metadata**, parsed two ways and cross-validated:
   - From the `param-set` directory name
     (`nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0` → `{"nonEpitopeAcceptance": 0.1,
     "epitopeAcceptance": 1.0}`).
   - From `<param-set>/parameters.yml`.
   - Symmetric validation:
     - **Value mismatch on a shared key** → fail loudly, naming the simulation, the
       key, and both values.
     - **Key present only in directory name** → fail loudly. The directory name is
       supposed to summarize the YAML, so a name-only key signals a stale rename.
     - **Key present only in YAML** → warn but do not fail. The YAML may include
       parameters the namer chose to elide; this is informational.
   - The merged result (YAML ∪ directory name, with shared keys reconciled) is what
     gets emitted as parsed parameter columns.
2. **Simulation summary statistics** from `experiments/<experiment-date>/sim_stats.csv`
   (per-experiment, NOT per-run).
   Filter rows by matching `(run, epitopeAcceptance, nonEpitopeAcceptance, ...)` to the
   simulation's parameters; assert exactly one row matches per sim.
   Columns of interest: `diversity`, `tmrca`, `antigenic_movement_per_year`,
   `antigenicDiversity`, `trunk_epitope_mutations`,
   `trunk_epitope_to_non-epitope_ratio` (see `analysis.ipynb` for the full list).
3. **Unique sequence count**: `len(pd.read_csv(unique_tips.csv))` from
   `data/<build>/antigen-outputs/unique_tips.csv`.
4. **Model scores and convergence diagnostics** from
   `results/<build>/scores.tsv`, `results/<build>/growth_rate_scores.tsv`, and
   `results/<build>/vi_convergence_diagnostics.tsv`.

**CLI:**

```
python scripts/aggregate_results.py \
    --experiments-root /fh/fast/matsen_e/zthornto/antigen-experiments/experiments/ \
    --data-root /fh/fast/matsen_e/zthornto/antigen-forecasting/data/ \
    --results-root /fh/fast/matsen_e/zthornto/antigen-forecasting/results/ \
    --batch-name 2026-04-29-all-sims \
    --output results/2026-04-29-all-sims/aggregate.tsv
```

**Output schema (one row per simulation; column order frozen):**

```
build, batch_name, experiment, param_set, run_id,
<parsed parameter columns, lower_camelCase keys from param-set name>,
n_unique_sequences,
sim_<column> for every column from sim_stats.csv,
freq_<column> for every column from scores.tsv,
growth_<column> for every column from growth_rate_scores.tsv,
vi_<column> for every column from vi_convergence_diagnostics.tsv,
```

The source-prefix scheme (`sim_`, `freq_`, `growth_`, `vi_`) eliminates name collisions
across the four sources.
Columns that already start with the prefix are left unchanged (no double-prefixing).

**Join discipline:**
- Each source must produce exactly one row per `build`.
- After each merge, assert `len(joined) == len(builds_with_complete_results)`.
  No silent drops.

**Missing data handling:**
- Print a warning (not a failure) for any `build` whose `scores.tsv` is missing — likely
  still running; exclude from output.
- Assert at least one complete simulation was found; otherwise fail with a clear error.

**Acceptance criteria:**
- All four data sources are joined on `build` with no silent drops; per-merge cardinality
  assertions documented above pass.
- Parsed parameter columns match the values in `parameters.yml` for at least one
  spot-checked simulation (covered by parameter cross-validation in step 1).
- Output filename is `aggregate.tsv` (no `aggregate_scores.tsv` variant; pick one
  everywhere).
- The output TSV loads cleanly into a notebook (`pd.read_csv(..., sep="\t")`) with no
  mixed-type columns.

---

## Issue filing order

Dependency structure:

```
Issue 1 (parse_sim_outputs + SimulationPaths)
    └── Issue 2 (run_pipeline.py)
            ├── Issue 3 (run_all_simulations.py)
            │       └── Issue 4 (SLURM submission)
            └── Issue 5 (aggregation)         # parallel with Issue 3
```

Issues 4 and 5 are independent and can be worked in parallel once Issue 3 is complete.
File these as GitHub issues via `gh issue create`.
Issues 1 and 2 are the critical path and must be verified locally on `flu-final` before
any remote work begins.

---

## Deployment

### Environments

Both environments instantiate the same `pipeline_config.yaml` schema:

```yaml
# configs/local.yaml — development against flu-final reference
pipeline:
  experiments_root: ../antigen-experiments/experiments/   # for raw flu-final inputs
  data_root: data/
  results_root: results/
  window_size: 365
  ...

# configs/remote.yaml — Fred Hutch cluster
pipeline:
  experiments_root: /fh/fast/matsen_e/zthornto/antigen-experiments/experiments/
  data_root: /fh/fast/matsen_e/zthornto/antigen-forecasting/data/
  results_root: /fh/fast/matsen_e/zthornto/antigen-forecasting/results/
  window_size: 365
  ...
```

CLI flags on `run_all_simulations.py` override config values; the config supplies
defaults so command lines stay short.

### Cluster batch invocation

```bash
python scripts/run_all_simulations.py \
    --experiment 2026-01-06-mutation-bug-fix-runs \
    --param-set nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0 \
    --batch-name 2026-04-29-all-sims \
    --config configs/remote.yaml \
    --slurm-config configs/slurm_config.yaml \
    --mode slurm
```

Results land in `results/2026-04-29-all-sims/<sim-id>/` and the aggregated table at
`results/2026-04-29-all-sims/aggregate.tsv`.

### Reference before implementing Issue 1

Before writing `parse_sim_outputs.py`, read the `analysis.ipynb` in the most recent
experiment folder (e.g., `experiments/2026-01-06-mutation-bug-fix-runs/analysis.ipynb`)
to confirm the dedup logic.
The verified order (already pinned in Issue 1, step 4) is:
`drop_duplicates(["name"])` then `drop_duplicates(["nucleotideSequence"])`.

---

## Testing strategy

| Layer | What | Where |
|---|---|---|
| Unit | `SimulationPaths` path construction (single + batch) | `antigentools/tests/test_paths.py` |
| Unit | `parse_sim_outputs.py` dedup logic with collision fixture | `antigentools/tests/test_parse_sim_outputs.py` |
| Unit | Config loading and validation | `antigentools/tests/test_config.py` |
| Integration | Single-sim pipeline on `flu-final` | `python scripts/run_pipeline.py --sim-path ... --build flu-final` |
| Integration | Score equivalence (`np.allclose(rtol=1e-3, atol=1e-4)`) | Assert against checked-in `results/flu-final/scores.tsv` |
| Integration | Idempotent re-run (all steps `skipped`) | Run twice; second run pipeline.log shows all `skipped` |
| Generation | SLURM script syntactic validity | `bash -n submit_array.sh` |
| Remote smoke test | `run_pipeline.py` on one remote sim | SSH, `--dry-run` first, then full run |
| Remote SLURM | Generated array script on cluster | `sbatch --test-only submit_array.sh` |
| Full batch | All remote simulations | SLURM array after smoke test passes |

---

## Unresolved questions

- `sim_stats.csv` columns: full column list still TBD beyond the listed core stats.
  Default: include all columns with `sim_` prefix; revisit if `aggregate.tsv` becomes
  unwieldy.
- `assign_all_variants.py` and `make_training_data.py` CLI signatures: not changed by
  this spec, but Issue 2 must verify their existing CLIs accept resolvable per-path
  args. Read both before starting Issue 2.
- VI seed plumbing: confirm `evofr` exposes a seed hook reachable from
  `run_model.py`. If not, the seed must be threaded via `numpyro.set_host_device_count`
  or `jax.random.PRNGKey` at script entry. Verify before Issue 2 implementation.
- Does the existing committed `results/flu-final/scores.tsv` need to be regenerated
  with `numpyro_seed: 42` as the new regression anchor, or is it already seed-42?
  (Issue 2 acceptance criteria currently mandate regeneration as a bootstrap step;
  drop that step if the existing file is already seed-42.)
