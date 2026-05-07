"""Discover and run the full multi-simulation batch pipeline.

Globs simulation directories under a param-set, applies skip logic for already-complete
simulations, then either invokes ``run_pipeline.py`` per sim locally (with optional
parallelism via ``ProcessPoolExecutor``) or writes SLURM array-job artifacts and exits.

Design References:
- PRIMARY: specs/analysis-pipeline.md (Issue 3)

Implementation Status: 🚧 WIP
Last Design Review: 2026-05-07
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import stat
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_SLURM_CONFIG: dict = {
    "partition": "campus-new",
    "time": "08:00:00",
    "mem_gb": 32,
    "cpus_per_task": 4,
    "max_concurrent": 20,
    "conda_env": "antigen-forecasting",
    "log_dir": "logs/slurm/",
    "project_root": str(REPO_ROOT),
    "python_bin": "python",
}


def discover_sim_paths(
    experiments_root: Path,
    experiment: str,
    param_set: str,
) -> list[Path]:
    """Discover and validate simulation directories under a param-set.

    Globs ``<experiments_root>/<experiment>/<param_set>/run_*/`` for directories
    and asserts each contains both required input files before enqueuing.
    Discovery never touches ``data/`` or ``results/``.

    Args:
        experiments_root: Root of the antigen-experiments directory.
        experiment: Experiment name (e.g. ``"2026-01-06-mutation-bug-fix-runs"``).
        param_set: Parameter-set subdirectory name.

    Returns:
        Sorted list of valid ``run_N/`` paths.

    Raises:
        ValueError: If no ``run_*/`` directories are found under the param-set.
        FileNotFoundError: If any sim dir is missing ``output/run-out.tips`` or
            ``out_timeseries.csv``.
    """
    base = experiments_root / experiment / param_set
    candidates = sorted(p for p in base.glob("run_*/") if p.is_dir())
    if not candidates:
        raise ValueError(
            f"No run_*/ directories found under {base}; check --experiment and "
            f"--param-set values"
        )
    for sim_path in candidates:
        tips = sim_path / "output" / "run-out.tips"
        timeseries = sim_path / "out_timeseries.csv"
        if not tips.exists():
            raise FileNotFoundError(
                f"Missing required file output/run-out.tips in {sim_path}"
            )
        if not timeseries.exists():
            raise FileNotFoundError(
                f"Missing required file out_timeseries.csv in {sim_path}"
            )
    return candidates


def _sim_id_from_path(sim_path: Path) -> str:
    """Derive ``<param_set>__<run_id>`` from a sim path."""
    return f"{sim_path.parent.name}__{sim_path.name}"


def is_sim_complete(
    sim_path: Path,
    results_root: Path,
    batch_name: str,
) -> bool:
    """Return True if both completion sentinels for a simulation exist.

    A simulation is complete only when ``scores.tsv`` AND
    ``growth_rate_scores.tsv`` both exist under
    ``results/<batch_name>/<sim_id>/``.

    Args:
        sim_path: Path to ``run_N/`` (used to derive ``sim_id``).
        results_root: Root results directory.
        batch_name: Human-supplied batch name.

    Returns:
        True iff both sentinel files exist.
    """
    sim_id = _sim_id_from_path(sim_path)
    sim_results = results_root / batch_name / sim_id
    return (sim_results / "scores.tsv").exists() and (
        sim_results / "growth_rate_scores.tsv"
    ).exists()


def build_run_pipeline_cmd(
    sim_path: Path,
    batch_name: str,
    config_path: Path,
    set_thread_caps: bool,
) -> list[str]:
    """Build the argv list for invoking ``run_pipeline.py``.

    Args:
        sim_path: Path to ``run_N/`` in antigen-experiments.
        batch_name: Batch name forwarded as ``--batch-name``.
        config_path: Path to ``pipeline_config.yaml``.
        set_thread_caps: If True, append ``--set-thread-caps``.

    Returns:
        Full argv list suitable for ``subprocess.run``.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_pipeline.py"),
        "--sim-path",
        str(sim_path),
        "--batch-name",
        batch_name,
        "--config",
        str(config_path),
    ]
    if set_thread_caps:
        cmd.append("--set-thread-caps")
    return cmd


def _run_one_sim(cmd: list[str]) -> None:
    """Invoke one ``run_pipeline.py`` subprocess.

    Module-level so it is picklable for ``ProcessPoolExecutor``.

    Args:
        cmd: Argv list for ``run_pipeline.py``.

    Raises:
        subprocess.CalledProcessError: If the subprocess exits non-zero.
    """
    subprocess.run(cmd, check=True)


def run_local(
    sim_paths: list[Path],
    batch_name: str,
    config_path: Path,
    results_root: Path,
    max_parallel: int,
) -> None:
    """Run simulations locally, sequentially or in parallel.

    For ``max_parallel == 1``: runs sims serially; subprocess env is unchanged.
    For ``max_parallel > 1``: uses ``ProcessPoolExecutor`` and passes
    ``--set-thread-caps`` to each subprocess to prevent BLAS oversubscription.

    Args:
        sim_paths: Incomplete simulation paths to process.
        batch_name: Batch name forwarded to ``run_pipeline.py``.
        config_path: Path to ``pipeline_config.yaml``.
        results_root: Root results directory (unused here; kept for interface
            symmetry with ``write_slurm_artifacts``).
        max_parallel: Maximum concurrent subprocesses; 1 for serial execution.
    """
    set_thread_caps = max_parallel > 1
    cmds = [
        build_run_pipeline_cmd(p, batch_name, config_path, set_thread_caps)
        for p in sim_paths
    ]
    if max_parallel == 1:
        for cmd in cmds:
            _run_one_sim(cmd)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel) as ex:
            futures = [ex.submit(_run_one_sim, cmd) for cmd in cmds]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()


def load_slurm_config(path: Path | None) -> dict:
    """Load a SLURM config YAML, falling back to built-in defaults.

    Args:
        path: Optional path to ``slurm_config.yaml``; None uses defaults.

    Returns:
        Dict containing all SLURM configuration values (the ``slurm:`` block).

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        ValueError: If ``path`` exists but lacks the ``slurm:`` top-level key.
    """
    if path is None:
        return dict(_DEFAULT_SLURM_CONFIG)
    if not path.exists():
        raise FileNotFoundError(f"SLURM config not found: {path}")
    with open(path) as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict) or "slurm" not in loaded:
        raise ValueError(
            f"{path} must define a top-level 'slurm:' block; got "
            f"{list(loaded.keys()) if isinstance(loaded, dict) else type(loaded).__name__}"
        )
    return loaded["slurm"]


def write_slurm_artifacts(
    sim_paths: list[Path],
    batch_name: str,
    results_root: Path,
    config_path: Path,
    slurm_config_path: Path | None,
) -> Path:
    """Write ``sim_list.txt`` and ``submit_array.sh`` for a SLURM array job.

    Creates ``results/<batch_name>/slurm_submission_<timestamp>/`` and writes
    both files. Does not invoke ``sbatch``.

    ``submit_array.sh`` uses ``SLURM_ARRAY_TASK_ID`` to index 1-based into
    ``sim_list.txt`` via ``sed``, so array 1-N maps cleanly to lines 1-N with no
    off-by-one arithmetic.

    Args:
        sim_paths: Incomplete simulation paths; each becomes one line in
            ``sim_list.txt`` (1-indexed; line N == array task N).
        batch_name: Batch name interpolated into the SLURM script.
        results_root: Root results directory; submission dir is created here.
        config_path: Path to ``pipeline_config.yaml``; forwarded to each task.
        slurm_config_path: Optional path to ``slurm_config.yaml``; None uses
            built-in defaults.

    Returns:
        Path to the created submission directory.
    """
    slurm_cfg = load_slurm_config(slurm_config_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    submission_dir = results_root / batch_name / f"slurm_submission_{timestamp}"
    submission_dir.mkdir(parents=True, exist_ok=True)

    sim_list_path = submission_dir / "sim_list.txt"
    sim_list_path.write_text("\n".join(str(p) for p in sim_paths) + "\n")

    n_sims = len(sim_paths)
    cpus = slurm_cfg["cpus_per_task"]

    script = textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name=antigen-pipeline
        #SBATCH --array=1-{n_sims}%{slurm_cfg["max_concurrent"]}
        #SBATCH --output={slurm_cfg["project_root"]}/{slurm_cfg["log_dir"]}/%A_%a.out
        #SBATCH --error={slurm_cfg["project_root"]}/{slurm_cfg["log_dir"]}/%A_%a.err
        #SBATCH --partition={slurm_cfg["partition"]}
        #SBATCH --time={slurm_cfg["time"]}
        #SBATCH --mem={slurm_cfg["mem_gb"]}G
        #SBATCH --cpus-per-task={cpus}

        # 1-indexed; sed line N == SLURM_ARRAY_TASK_ID N.
        SIM_PATH=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {sim_list_path})
        source activate {slurm_cfg["conda_env"]}
        export OMP_NUM_THREADS={cpus}
        export MKL_NUM_THREADS={cpus}
        export OPENBLAS_NUM_THREADS={cpus}
        export JAX_PLATFORMS=cpu
        {slurm_cfg["python_bin"]} {slurm_cfg["project_root"]}/scripts/run_pipeline.py \\
            --sim-path "$SIM_PATH" \\
            --batch-name {batch_name} \\
            --config {config_path}
        """
    )

    submit_path = submission_dir / "submit_array.sh"
    submit_path.write_text(script)
    submit_path.chmod(submit_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    logger.info("SLURM artifacts written to %s", submission_dir)
    logger.info("  sim_list.txt: %d simulation(s)", n_sims)
    logger.info(
        "  submit_array.sh: array 1-%d%%%d", n_sims, slurm_cfg["max_concurrent"]
    )

    return submission_dir


def _load_results_root(config_path: Path) -> Path:
    """Extract ``results_root`` from a ``pipeline_config.yaml``.

    Args:
        config_path: Path to ``pipeline_config.yaml``.

    Returns:
        Resolved ``Path`` for ``results_root``.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the config lacks the expected structure or ``results_root`` key.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    with open(config_path) as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict) or "pipeline" not in loaded:
        raise ValueError(f"{config_path} must define a top-level 'pipeline:' block")
    cfg = loaded["pipeline"]
    if "results_root" not in cfg:
        raise ValueError(f"{config_path} missing required key: results_root")
    return Path(cfg["results_root"])


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover simulations in antigen-experiments and invoke run_pipeline.py "
            "for each, either locally or by writing SLURM array-job artifacts."
        )
    )
    parser.add_argument(
        "--experiments-root",
        required=True,
        type=Path,
        help="Root of antigen-experiments/experiments/.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=str,
        help="Experiment directory name (e.g. 2026-01-06-mutation-bug-fix-runs).",
    )
    parser.add_argument(
        "--param-set",
        required=True,
        type=str,
        help="Parameter-set subdirectory name.",
    )
    parser.add_argument(
        "--batch-name",
        required=True,
        type=str,
        help="Human-supplied batch name; used to namespace results/.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to pipeline_config.yaml.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["local", "slurm"],
        help=(
            "Execution mode: 'local' runs subprocesses; 'slurm' writes SLURM "
            "artifacts and exits without invoking sbatch."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum concurrent subprocesses in local mode (default: 1).",
    )
    parser.add_argument(
        "--slurm-config",
        type=Path,
        default=None,
        help="Path to slurm_config.yaml (slurm mode only; uses defaults if omitted).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Discover sims, skip completed ones, and run locally or write SLURM artifacts."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    results_root = _load_results_root(args.config)

    all_sims = discover_sim_paths(
        args.experiments_root, args.experiment, args.param_set
    )
    logger.info(
        "Discovered %d simulation(s) under %s/%s",
        len(all_sims),
        args.experiment,
        args.param_set,
    )

    pending: list[Path] = []
    for sim_path in all_sims:
        if is_sim_complete(sim_path, results_root, args.batch_name):
            logger.info("Skipping complete sim: %s", sim_path)
        else:
            pending.append(sim_path)

    n_skipped = len(all_sims) - len(pending)
    logger.info("%d sim(s) pending; %d skipped", len(pending), n_skipped)

    if not pending:
        logger.info("All simulations complete. Nothing to do.")
        return

    if args.mode == "local":
        run_local(
            sim_paths=pending,
            batch_name=args.batch_name,
            config_path=args.config,
            results_root=results_root,
            max_parallel=args.max_parallel,
        )
    else:
        submission_dir = write_slurm_artifacts(
            sim_paths=pending,
            batch_name=args.batch_name,
            results_root=results_root,
            config_path=args.config,
            slurm_config_path=args.slurm_config,
        )
        print(f"SLURM artifacts written to: {submission_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
