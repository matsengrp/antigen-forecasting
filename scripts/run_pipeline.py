"""Run the full single-simulation analysis pipeline for one build.

Orchestrates the seven pipeline steps in DAG order with idempotent re-runs
gated by per-step sentinel files. Subscript CLIs are unchanged; this script is
the only consumer that constructs a ``SimulationPaths`` and forwards its
properties to subscripts.

Step order (sequential):
    parse_sim_outputs
        -> assign_all_variants
            -> prep_antigen_data       (also reads tips.csv from parse_sim_outputs)
                -> make_training_data
                    -> run_model       (fans out over date x location x model)
                        -> score_models
                        -> score_growth_rates

CLI:
    python scripts/run_pipeline.py \\
        --sim-path PATH \\
        (--batch-name STR | --build STR) \\
        --config configs/pipeline_config.yaml \\
        [--skip-variant-assignment] [--skip-forecasting] [--dry-run] [--force] \\
        [--set-thread-caps]

``--batch-name`` and ``--build`` form a mutually exclusive required group.
``--skip-variant-assignment`` runs only the antigenic (k-means on ag1, ag2)
baseline, skipping the t-SNE and phylogenetic variant pipelines.
``--set-thread-caps`` pins ``OMP_NUM_THREADS=MKL_NUM_THREADS=
OPENBLAS_NUM_THREADS=1`` and ``JAX_PLATFORMS=cpu`` in subprocess env (parent
runners with ``--max-parallel > 1`` should pass this).

Design References:
- PRIMARY: specs/analysis-pipeline.md (Issue 2)

Implementation Status: WIP
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import yaml

from antigentools.paths import SimulationPaths


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "experiments_root",
    "data_root",
    "results_root",
    "window_size",
    "buffer_size",
    "models",
    "locations",
    "forecast_L",
    "seed_L",
    "numpyro_seed",
)

PIPELINE_LOG_COLUMNS: tuple[str, ...] = (
    "step",
    "start_time",
    "end_time",
    "duration_sec",
    "status",
    "command",
)


@dataclass
class Step:
    """One pipeline step.

    A step is uniquely identified by ``name`` and considered complete when every
    path in ``sentinels`` exists on disk. ``runner`` does the actual work
    (``subprocess.run`` for single-shot steps; an ad-hoc closure for steps that
    fan out, like ``run_model``). ``clean_paths`` are removed before re-running
    on ``--force``; defaulting to ``sentinels`` covers the simple cases.

    Attributes:
        name: Stable identifier; appears in ``pipeline.log``.
        sentinels: Output paths that must all exist for the step to be skipped.
        runner: Callable that performs the work and raises on failure.
        command_str: Human-readable command rendered in ``pipeline.log``.
        clean_paths: Paths removed by ``--force`` before invoking ``runner``;
            falls back to ``sentinels`` when empty.
    """

    name: str
    sentinels: list[Path]
    runner: Callable[[], None]
    command_str: str
    clean_paths: list[Path] = field(default_factory=list)


@dataclass
class StepResult:
    """One row in ``pipeline.log``."""

    name: str
    start_time: str
    end_time: str
    duration_sec: float
    status: str
    command: str


def load_pipeline_config(path: Path) -> dict:
    """Load and validate a pipeline config YAML.

    Args:
        path: Path to the YAML file (e.g. ``configs/pipeline_config.yaml``).

    Returns:
        The ``pipeline:`` mapping with all required keys present.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is missing the ``pipeline`` block or any
            required key.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    with open(path) as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict) or "pipeline" not in loaded:
        raise ValueError(
            f"{path} must define a top-level 'pipeline:' block; got keys "
            f"{list(loaded.keys()) if isinstance(loaded, dict) else type(loaded).__name__}"
        )
    cfg = loaded["pipeline"]
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path} missing required pipeline keys: {missing}")
    return cfg


def is_complete(step: Step) -> bool:
    """Return True if every sentinel path for ``step`` exists."""
    return all(p.exists() for p in step.sentinels)


def _force_clean(step: Step) -> None:
    """Remove ``step.clean_paths`` (or ``step.sentinels`` if empty)."""
    targets = step.clean_paths or step.sentinels
    for target in targets:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def execute_step(step: Step, dry_run: bool, force: bool) -> StepResult:
    """Run one pipeline step honoring sentinel skip and ``--force`` semantics.

    Decision order: dry-run reports the intent without doing anything; ``--force``
    cleans pre-existing outputs first; otherwise an existing sentinel set means
    the step is skipped. After execution, every sentinel must exist or the step
    is reported as failed.

    Args:
        step: The step to execute.
        dry_run: If True, log the planned action and return without running.
        force: If True, remove existing sentinels/clean_paths before running.

    Returns:
        A ``StepResult`` for logging.
    """
    start_iso = _now_iso()
    t0 = time.monotonic()

    if dry_run:
        planned = "skipped" if (is_complete(step) and not force) else "ok"
        logger.info("[dry-run] %s -> %s (%s)", step.name, planned, step.command_str)
        return StepResult(
            name=step.name,
            start_time=start_iso,
            end_time=start_iso,
            duration_sec=0.0,
            status=planned,
            command=step.command_str,
        )

    if force:
        _force_clean(step)

    if is_complete(step):
        logger.info("Skipping %s (sentinels exist)", step.name)
        return StepResult(
            name=step.name,
            start_time=start_iso,
            end_time=start_iso,
            duration_sec=0.0,
            status="skipped",
            command="",
        )

    logger.info("Running %s: %s", step.name, step.command_str)
    try:
        step.runner()
    except Exception as exc:
        end_iso = _now_iso()
        logger.error("Step %s failed: %s", step.name, exc)
        return StepResult(
            name=step.name,
            start_time=start_iso,
            end_time=end_iso,
            duration_sec=time.monotonic() - t0,
            status="failed",
            command=step.command_str,
        )

    missing = [p for p in step.sentinels if not p.exists()]
    if missing:
        end_iso = _now_iso()
        logger.error("Step %s completed but sentinels missing: %s", step.name, missing)
        return StepResult(
            name=step.name,
            start_time=start_iso,
            end_time=end_iso,
            duration_sec=time.monotonic() - t0,
            status="failed",
            command=step.command_str,
        )

    end_iso = _now_iso()
    return StepResult(
        name=step.name,
        start_time=start_iso,
        end_time=end_iso,
        duration_sec=time.monotonic() - t0,
        status="ok",
        command=step.command_str,
    )


def write_pipeline_log(results: list[StepResult], log_path: Path) -> None:
    """Write ``results`` to ``log_path`` as a TSV with the spec's columns."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "step": r.name,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "duration_sec": f"{r.duration_sec:.3f}",
            "status": r.status,
            "command": r.command,
        }
        for r in results
    ]
    pd.DataFrame(rows, columns=list(PIPELINE_LOG_COLUMNS)).to_csv(
        log_path, sep="\t", index=False
    )


def _run_subprocess(cmd: list[str], env: dict[str, str]) -> None:
    """Thin ``subprocess.run`` wrapper that surfaces failures via ``check=True``."""
    subprocess.run(cmd, check=True, env=env)


def _build_thread_capped_env(set_caps: bool) -> dict[str, str]:
    """Return a copy of ``os.environ`` with optional BLAS/JAX thread caps applied."""
    env = dict(os.environ)
    if set_caps:
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["JAX_PLATFORMS"] = "cpu"
    return env


def _read_estimation_dates(time_stamped_manifest: Path) -> list[str]:
    """Return the list of analysis dates from ``time-stamped/MANIFEST.tsv``."""
    df = pd.read_csv(time_stamped_manifest, sep="\t")
    return df["date"].astype(str).tolist()


def _read_convergence_status(
    diagnostics_dir: Path, model: str, location: str, date: str
) -> str:
    """Look up convergence status for one (model, location, date) combo.

    Reads ``{diagnostics_dir}/{model}_{location}_{date}_vi_diagnostics.json`` and
    extracts ``convergence_diagnostics.convergence.converged`` (or
    ``relative_change`` thresholded at 0.5 if ``converged`` is missing).

    Args:
        diagnostics_dir: ``results/<build>/convergence-diagnostics/``.
        model: Model name (e.g. ``GARW``).
        location: Country / deme name.
        date: Analysis date in ``YYYY-MM-DD``.

    Returns:
        ``"converged"``, ``"not_converged"``, or ``"unknown"`` if the diagnostics
        file is missing or malformed.
    """
    diag_path = diagnostics_dir / f"{model}_{location}_{date}_vi_diagnostics.json"
    if not diag_path.exists():
        return "unknown"
    try:
        with open(diag_path) as f:
            diag = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "unknown"
    conv = diag.get("convergence_diagnostics", {}).get("convergence", {})
    converged = conv.get("converged")
    if converged is None:
        rel = conv.get("relative_change")
        if rel is None:
            return "unknown"
        converged = rel <= 0.5
    return "converged" if converged else "not_converged"


def _make_run_model_runner(
    paths: SimulationPaths,
    cfg: dict,
    benchmark_cfg_path: Path,
    env: dict[str, str],
) -> Callable[[], None]:
    """Build the runner closure for the ``run_model`` step.

    Iterates the (date x location x model) grid from the time-stamped manifest
    and the pipeline config, invokes ``scripts/run_model.py`` per combo, then
    writes ``estimates/MANIFEST.tsv`` summarizing every run.
    """

    def runner() -> None:
        dates = _read_estimation_dates(paths.time_stamped_manifest)
        locations: list[str] = list(cfg["locations"])
        models: list[str] = list(cfg["models"])
        forecast_l = cfg["forecast_L"]
        seed_l = cfg["seed_L"]

        manifest_rows: list[dict[str, str]] = []
        for date in dates:
            for location in locations:
                for model in models:
                    data_path = f"{paths.time_stamped / date}/"
                    cmd = [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / "run_model.py"),
                        "--data_path",
                        data_path,
                        "--country",
                        location,
                        "--model",
                        model,
                        "--output_dir",
                        str(paths.results),
                        "--forecast_L",
                        str(forecast_l),
                        "--seed_L",
                        str(seed_l),
                        "--config",
                        str(benchmark_cfg_path),
                    ]
                    _run_subprocess(cmd, env)
                    estimate_path = (
                        paths.estimates / model / f"freq_{location}_{date}.tsv"
                    )
                    manifest_rows.append(
                        {
                            "date": date,
                            "location": location,
                            "model": model,
                            "estimate_path": str(estimate_path),
                            "convergence_status": _read_convergence_status(
                                paths.results / "convergence-diagnostics",
                                model,
                                location,
                                date,
                            ),
                        }
                    )

        paths.estimates.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            manifest_rows,
            columns=[
                "date",
                "location",
                "model",
                "estimate_path",
                "convergence_status",
            ],
        ).to_csv(paths.estimates_manifest, sep="\t", index=False)

    return runner


def build_steps(
    paths: SimulationPaths,
    cfg: dict,
    benchmark_cfg_path: Path,
    skip_variant_assignment: bool,
    skip_forecasting: bool,
    env: dict[str, str],
) -> list[Step]:
    """Build the ordered step list for one simulation.

    Args:
        paths: The simulation's ``SimulationPaths`` (build, roots, properties).
        cfg: ``pipeline:`` block from the loaded config.
        benchmark_cfg_path: Path where ``make_training_data.py`` will write
            ``benchmark_config.yaml`` and from which ``run_model.py`` /
            ``score_models.py`` will read it.
        skip_variant_assignment: If True, pass ``--fast`` to
            ``assign_all_variants.py`` (antigenic baseline only).
        skip_forecasting: If True, omit the four forecasting steps.
        env: Subprocess environment (BLAS thread caps applied or not).

    Returns:
        An ordered list of ``Step`` instances. Skipped tracks are simply
        omitted from the list.
    """
    steps: list[Step] = []

    parse_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "parse_sim_outputs.py"),
        "--sim-path",
        str(paths.sim_path),
        "--output-dir",
        str(paths.antigen_outputs),
    ]
    steps.append(
        Step(
            name="parse_sim_outputs",
            sentinels=[paths.unique_tips],
            runner=lambda: _run_subprocess(parse_cmd, env),
            command_str=" ".join(parse_cmd),
            clean_paths=[paths.antigen_outputs],
        )
    )

    assign_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "assign_all_variants.py"),
        "--tips",
        str(paths.unique_tips),
        "--fasta",
        str(paths.unique_sequences),
        "--output",
        str(paths.tips_with_variants),
        "--work-dir",
        str(paths.variant_assignment),
    ]
    if skip_variant_assignment:
        assign_cmd.append("--fast")
    steps.append(
        Step(
            name="assign_all_variants",
            sentinels=[paths.tips_with_variants],
            runner=lambda: _run_subprocess(assign_cmd, env),
            command_str=" ".join(assign_cmd),
            clean_paths=[paths.tips_with_variants, paths.variant_assignment],
        )
    )

    prep_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prep_antigen_data.py"),
        "-t",
        str(paths.tips),
        "-c",
        str(paths.timeseries),
        "-o",
        str(paths.data_root / paths.build),
        "-m",
        str(paths.tips_with_variants),
    ]
    steps.append(
        Step(
            name="prep_antigen_data",
            sentinels=[paths.seq_counts, paths.case_counts],
            runner=lambda: _run_subprocess(prep_cmd, env),
            command_str=" ".join(prep_cmd),
        )
    )

    if skip_forecasting:
        return steps

    train_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "make_training_data.py"),
        "-s",
        str(paths.seq_counts),
        "-c",
        str(paths.case_counts),
        "-o",
        str(paths.time_stamped),
        "--window-size",
        str(cfg["window_size"]),
        "--buffer-size",
        str(cfg["buffer_size"]),
        "--config-path",
        str(benchmark_cfg_path),
    ]
    steps.append(
        Step(
            name="make_training_data",
            sentinels=[paths.time_stamped_manifest],
            runner=lambda: _run_subprocess(train_cmd, env),
            command_str=" ".join(train_cmd),
            clean_paths=[paths.time_stamped],
        )
    )

    run_model_runner = _make_run_model_runner(paths, cfg, benchmark_cfg_path, env)
    steps.append(
        Step(
            name="run_model",
            sentinels=[paths.estimates_manifest],
            runner=run_model_runner,
            command_str=(
                f"run_model.py over {len(cfg['models'])} models x "
                f"{len(cfg['locations'])} locations x N dates"
            ),
            clean_paths=[paths.estimates],
        )
    )

    score_models_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "score_models.py"),
        "--config",
        str(benchmark_cfg_path),
        "--truth-set",
        str(paths.seq_counts),
        "--estimates-path",
        str(paths.estimates),
        "--output-path",
        str(paths.scores),
    ]
    steps.append(
        Step(
            name="score_models",
            sentinels=[paths.scores],
            runner=lambda: _run_subprocess(score_models_cmd, env),
            command_str=" ".join(score_models_cmd),
        )
    )

    score_growth_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "score_growth_rates.py"),
        "--config",
        str(benchmark_cfg_path),
        "--build",
        paths.build,
        "--output-dir",
        str(paths.results),
    ]
    steps.append(
        Step(
            name="score_growth_rates",
            sentinels=[paths.growth_rate_scores],
            runner=lambda: _run_subprocess(score_growth_cmd, env),
            command_str=" ".join(score_growth_cmd),
        )
    )

    return steps


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full single-simulation analysis pipeline."
    )
    parser.add_argument(
        "--sim-path",
        required=True,
        type=Path,
        help="Path to run_N/ in antigen-experiments.",
    )
    build_group = parser.add_mutually_exclusive_group(required=True)
    build_group.add_argument(
        "--batch-name",
        type=str,
        help="Batch name for batch runs; build is derived as <batch>/<sim-id>.",
    )
    build_group.add_argument(
        "--build",
        type=str,
        help="Verbatim build string (single-dataset mode, e.g. 'flu-final').",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to pipeline_config.yaml.",
    )
    parser.add_argument(
        "--skip-variant-assignment",
        action="store_true",
        help=(
            "Run only the antigenic k-means baseline; skip sequence-embedding "
            "and phylogenetic variant assignment."
        ),
    )
    parser.add_argument(
        "--skip-forecasting",
        action="store_true",
        help="Skip make_training_data, run_model, score_models, score_growth_rates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned step sequence with skip decisions; do not execute.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore sentinels and rerun every step; removes affected outputs first.",
    )
    parser.add_argument(
        "--set-thread-caps",
        action="store_true",
        help=(
            "Pin OMP/MKL/OpenBLAS to 1 thread and JAX to CPU in subprocess env. "
            "Parent runners with --max-parallel > 1 should pass this."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse args and run the pipeline; write ``results/<build>/pipeline.log``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    cfg = load_pipeline_config(args.config)

    data_root = Path(cfg["data_root"])
    results_root = Path(cfg["results_root"])
    if args.batch_name is not None:
        paths = SimulationPaths.from_sim_path(
            sim_path=args.sim_path,
            data_root=data_root,
            results_root=results_root,
            batch_name=args.batch_name,
        )
    else:
        paths = SimulationPaths.from_build(
            build=args.build,
            sim_path=args.sim_path,
            data_root=data_root,
            results_root=results_root,
        )

    benchmark_cfg_path = paths.results / "benchmark_config.yaml"
    paths.results.mkdir(parents=True, exist_ok=True)

    env = _build_thread_capped_env(args.set_thread_caps)
    steps = build_steps(
        paths=paths,
        cfg=cfg,
        benchmark_cfg_path=benchmark_cfg_path,
        skip_variant_assignment=args.skip_variant_assignment,
        skip_forecasting=args.skip_forecasting,
        env=env,
    )

    results: list[StepResult] = []
    failed = False
    for step in steps:
        result = execute_step(step, dry_run=args.dry_run, force=args.force)
        results.append(result)
        if result.status == "failed":
            failed = True
            break

    if not args.dry_run:
        write_pipeline_log(results, paths.pipeline_log)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
