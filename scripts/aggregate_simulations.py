"""Assemble per-simulation summary table (sim_stats.csv) for a completed batch.

Joins four data sources per simulation: parameter metadata (cross-validated between
directory name and ``parameters.yml``), ``run-out.summary`` stats, unique sequence
count from ``unique_tips.csv``, and branch-level mutation counts from
``run-out.branches``.  Writes one row per simulation to ``sim_stats.csv``.

Design References:
- PRIMARY: specs/analysis-pipeline.md (Issue 5)

Implementation Status: 🚧 WIP
Last Design Review: 2026-05-08
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from antigentools.antigen_reader import (
    calculate_antigenic_movement_per_year,
    count_branch_mutations,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter parsing and cross-validation
# ---------------------------------------------------------------------------


def _try_numeric(value: str) -> int | float | str:
    """Cast a string to int, then float, or leave as str if neither applies."""
    try:
        int_val = int(value)
        # Guard against floats that happen to be whole numbers
        if str(int_val) == value:
            return int_val
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def parse_params_from_name(param_set: str) -> dict[str, int | float | str]:
    """Parse simulation parameters from a param-set directory name.

    Parameter values must not contain underscores; underscores are the
    key/value separator.

    Args:
        param_set: Directory name encoding alternating key/value pairs separated
            by ``_``, e.g. ``"nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0"``.

    Returns:
        Dict mapping parameter names to cast values.

    Raises:
        ValueError: If the token count after splitting on ``_`` is odd (not even
            key/value pairs).

    Example:
        >>> parse_params_from_name("nonEpitopeAcceptance_0.1_epitopeAcceptance_1.0")
        {'nonEpitopeAcceptance': 0.1, 'epitopeAcceptance': 1.0}
    """
    tokens = param_set.split("_")
    if len(tokens) % 2 != 0:
        raise ValueError(
            f"param_set {param_set!r} has an odd number of '_'-separated tokens "
            f"({len(tokens)}); expected alternating key/value pairs"
        )
    return {tokens[i]: _try_numeric(tokens[i + 1]) for i in range(0, len(tokens), 2)}


def load_params_from_yaml(params_yml: Path) -> dict[str, Any]:
    """Load simulation parameters from a parameters.yml file.

    Args:
        params_yml: Path to ``parameters.yml``.

    Returns:
        Flat dict of parameter names to values.

    Raises:
        FileNotFoundError: If ``params_yml`` does not exist.
    """
    if not params_yml.exists():
        raise FileNotFoundError(f"parameters.yml not found: {params_yml}")
    with open(params_yml) as f:
        return yaml.safe_load(f) or {}


def validate_and_merge_params(
    name_params: dict[str, Any],
    yaml_params: dict[str, Any],
    sim_id: str,
) -> dict[str, Any]:
    """Cross-validate and merge params from directory name and parameters.yml.

    Args:
        name_params: Parameters parsed from the param-set directory name.
        yaml_params: Parameters loaded from ``parameters.yml``.
        sim_id: Simulation identifier used in error messages.

    Returns:
        Union dict (YAML ∪ name params) with shared keys reconciled.

    Raises:
        ValueError: If a key appears in both sources with mismatched values, or
            if a key appears only in the directory name (signals a stale rename).
    """
    for key, name_val in name_params.items():
        if key not in yaml_params:
            raise ValueError(
                f"sim {sim_id}: key {key!r} present in dir name but not in "
                f"parameters.yml (stale rename?)"
            )
        yaml_val = yaml_params[key]
        if name_val != yaml_val:
            raise ValueError(
                f"sim {sim_id}: key {key!r} mismatch: dir={name_val!r} yaml={yaml_val!r}"
            )

    for key in yaml_params:
        if key not in name_params:
            warnings.warn(
                f"sim {sim_id}: key {key!r} present in parameters.yml only (informational)",
                UserWarning,
                stacklevel=2,
            )

    return {**yaml_params, **name_params}


# ---------------------------------------------------------------------------
# Per-simulation data loading
# ---------------------------------------------------------------------------


def read_summary_stats(summary_path: Path) -> dict[str, int | float | str]:
    """Read a ``run-out.summary`` file and return stats as a wide dict.

    The file is tab-separated with the first row as a comment/header, followed
    by ``parameter\\tvalue`` rows.

    Args:
        summary_path: Path to ``run-out.summary``.

    Returns:
        Dict mapping parameter names to cast numeric or string values.

    Raises:
        FileNotFoundError: If ``summary_path`` does not exist.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"run-out.summary not found: {summary_path}")
    df = pd.read_csv(summary_path, sep="\t", skiprows=1, names=["parameter", "value"])
    return {str(row.parameter): _try_numeric(str(row.value)) for row in df.itertuples()}


def load_tips(data_root: Path, build: str) -> pd.DataFrame:
    """Load unique tips for a simulation.

    Args:
        data_root: Root of processed data directory.
        build: Build string (e.g. ``"2026-04-29-all-sims/param__run_0"``).

    Returns:
        DataFrame from ``unique_tips.csv`` with at least ``year``, ``ag1``, ``ag2``.

    Raises:
        FileNotFoundError: If ``unique_tips.csv`` does not exist.
    """
    tips_path = data_root / build / "antigen-outputs" / "unique_tips.csv"
    if not tips_path.exists():
        raise FileNotFoundError(f"unique_tips.csv not found: {tips_path}")
    return pd.read_csv(tips_path)


# ---------------------------------------------------------------------------
# Build discovery
# ---------------------------------------------------------------------------


def discover_builds(data_root: Path, batch_name: str) -> list[str]:
    """Discover processed simulation builds under a batch.

    Args:
        data_root: Root of processed data directory.
        batch_name: Batch name (e.g. ``"2026-04-29-all-sims"``).

    Returns:
        Sorted list of build strings (``"<batch_name>/<sim_id>"``).

    Raises:
        ValueError: If no builds are found.
    """
    batch_dir = data_root / batch_name
    dirs = sorted(d.parent for d in batch_dir.glob("*/antigen-outputs/") if d.is_dir())
    if not dirs:
        raise ValueError(
            f"No builds found under {batch_dir}; ensure parse_sim_outputs.py has run "
            f"for this batch"
        )
    return [f"{batch_name}/{d.name}" for d in dirs]


# ---------------------------------------------------------------------------
# Frozen column ordering
# ---------------------------------------------------------------------------

_ID_COLS = ["build", "batch_name", "experiment", "path", "run", "param_set", "run_id"]
_BRANCH_COLS = [
    "trunk_epitope_mutations",
    "trunk_non_epitope_mutations",
    "trunk_epitope_to_non-epitope_ratio",
    "side_branch_epitope_mutations",
    "side_branch_non_epitope_mutations",
    "side_branch_epitope_to_non-epitope_ratio",
]


def _freeze_columns(
    row: dict[str, Any],
    merged_param_keys: list[str],
    summary_keys: list[str],
) -> dict[str, Any]:
    """Return a new dict with keys in the frozen column order.

    Order: id cols, sorted param cols, n_unique_sequences,
    summary stats, antigenic_movement_per_year, branch cols.
    """
    ordered: dict[str, Any] = {}
    for col in _ID_COLS:
        ordered[col] = row[col]
    for key in merged_param_keys:
        ordered[key] = row.get(key)
    ordered["n_unique_sequences"] = row["n_unique_sequences"]
    for key in summary_keys:
        ordered[key] = row.get(key)
    ordered["antigenic_movement_per_year"] = row["antigenic_movement_per_year"]
    for col in _BRANCH_COLS:
        ordered[col] = row[col]
    return ordered


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------


def aggregate_simulations(
    experiments_root: Path,
    experiment: str,
    data_root: Path,
    batch_name: str,
) -> pd.DataFrame:
    """Assemble per-simulation summary rows for all builds in a batch.

    Args:
        experiments_root: Root of the antigen-experiments directory.
        experiment: Experiment name (e.g. ``"2026-01-06-mutation-bug-fix-runs"``).
        data_root: Root of processed data (antigen-forecasting/data/).
        batch_name: Human-supplied batch name (e.g. ``"2026-04-29-all-sims"``).

    Returns:
        DataFrame with one row per complete simulation, frozen column order.

    Raises:
        ValueError: If no complete simulations were found.
    """
    builds = discover_builds(data_root, batch_name)
    rows: list[dict[str, Any]] = []
    all_param_keys: set[str] = set()
    all_summary_keys: list[str] = []

    for build in builds:
        sim_id = build.split("/", 1)[1]
        param_set, run_id = sim_id.split("__", 1)
        run_int = int(run_id.removeprefix("run_"))

        sim_path = experiments_root / experiment / param_set / run_id
        params_yml = experiments_root / experiment / param_set / "parameters.yml"

        # Parameter cross-validation
        name_params = parse_params_from_name(param_set)
        yaml_params = load_params_from_yaml(params_yml)
        merged = validate_and_merge_params(name_params, yaml_params, sim_id)
        all_param_keys.update(merged.keys())

        # run-out.summary
        summary_path = sim_path / "output" / "run-out.summary"
        try:
            summary_stats = read_summary_stats(summary_path)
        except FileNotFoundError:
            warnings.warn(
                f"run-out.summary missing for {sim_id} — skipping",
                UserWarning,
                stacklevel=2,
            )
            continue

        if not all_summary_keys:
            all_summary_keys = list(summary_stats.keys())

        # Tips (used for both sequence count and antigenic movement)
        tips_df = load_tips(data_root, build)
        n_unique = len(tips_df)
        antigenic_movement = calculate_antigenic_movement_per_year(tips_df)

        # Branch mutations
        branches_path = sim_path / "output" / "run-out.branches"
        try:
            branch_stats = count_branch_mutations(str(branches_path))
        except FileNotFoundError:
            warnings.warn(
                f"run-out.branches missing for {sim_id} — branch stats will be NaN",
                UserWarning,
                stacklevel=2,
            )
            branch_stats = {col: float("nan") for col in _BRANCH_COLS}

        row: dict[str, Any] = {
            "build": build,
            "batch_name": batch_name,
            "experiment": experiment,
            "path": str(sim_path),
            "run": run_int,
            "param_set": param_set,
            "run_id": run_id,
            **merged,
            "n_unique_sequences": n_unique,
            **summary_stats,
            "antigenic_movement_per_year": antigenic_movement,
            **branch_stats,
        }
        rows.append(row)

    if not rows:
        raise ValueError(
            f"No complete simulations found in batch {batch_name!r} "
            f"(experiment {experiment!r})"
        )

    merged_param_keys = sorted(all_param_keys)
    df = pd.DataFrame(
        [_freeze_columns(r, merged_param_keys, all_summary_keys) for r in rows]
    )
    assert len(df) == len(rows), (
        f"DataFrame row count ({len(df)}) != assembled rows ({len(rows)})"
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and run aggregation."""
    parser = argparse.ArgumentParser(
        description="Assemble sim_stats.csv for a completed batch of simulations."
    )
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    output = args.output or (args.experiments_root / args.experiment / "sim_stats.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    df = aggregate_simulations(
        experiments_root=args.experiments_root,
        experiment=args.experiment,
        data_root=args.data_root,
        batch_name=args.batch_name,
    )
    df.to_csv(output, sep="\t", index=False)
    logger.info("Wrote %d rows to %s", len(df), output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
