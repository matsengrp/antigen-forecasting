"""Select candidate simulation runs from a set of antigen-experiments experiments.

For every experiment matched under ``--experiments-root`` this reuses the
experiment's committed ``sim_stats.csv`` when present, and otherwise generates it
in place via ``antigen-experiments/scripts/summarize_sims.py`` (the single source
of aggregation logic — this script never reimplements it). The per-experiment
tables are concatenated into one combined ``sim_stats.csv``, then filtered to the
runs that clear the candidate thresholds, written as ``candidate_runs.csv``.

``candidate_runs.csv`` has the same schema as the most recent committed version in
antigen-experiments: the canonical ``sim_stats`` columns followed by a trailing
``complete`` column. A run is ``complete`` when it produced summary output (a
non-null ``diversity``); incomplete runs have all-NaN stats and are never
candidates.

Candidate thresholds (all must hold, on complete runs only):
    diversity                            <= 9.0
    tmrca                                <= 6.0
    antigenic_movement_per_year          in [1.0, 2.0]
    trunk_epitope_to_non-epitope_ratio   >= 1.3

Usage (run from the antigen-forecasting repo root; experiments live in a sibling
repo):
    python scripts/find_candidate_runs.py -j 8 \\
        --experiments-root ../antigen-experiments/experiments

    # Restrict to specific experiments (glob patterns, relative to the root):
    python scripts/find_candidate_runs.py 2026-07-04-reviewer-runs

    # Force re-aggregation even where a sim_stats.csv already exists:
    python scripts/find_candidate_runs.py --refresh -j 8
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import operator
import sys
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Default location of antigen-experiments/scripts/summarize_sims.py relative to
# this file: <repo>/scripts/find_candidate_runs.py -> ../../antigen-experiments.
_DEFAULT_SUMMARIZE_SIMS = (
    Path(__file__).resolve().parent.parent.parent
    / "antigen-experiments"
    / "scripts"
    / "summarize_sims.py"
)

# Per-experiment aggregated stats file, produced/consumed by summarize_sims.
SIM_STATS_FILENAME = "sim_stats.csv"

# Candidate thresholds, keyed by the sim_stats column they apply to. Each entry is
# a list of (comparison, bound) pairs; a run qualifies only if every pair holds.
# This is the sole definition of "candidate" — matches analysis.ipynb.
_COMPARATORS: dict[str, Callable[[pd.Series, float], "pd.Series"]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
}
CANDIDATE_FILTERS: dict[str, list[tuple[str, float]]] = {
    "diversity": [("<=", 9.0)],
    "tmrca": [("<=", 6.0)],
    "antigenic_movement_per_year": [(">=", 1.0), ("<=", 2.0)],
    "trunk_epitope_to_non-epitope_ratio": [(">=", 1.3)],
}

# Column whose presence marks a run as having produced summary output.
COMPLETENESS_STAT = "diversity"


def load_summarize_sims(summarize_sims_path: Path):
    """Import ``summarize_sims`` from antigen-experiments as a module.

    The aggregator lives in a sibling repository rather than an installed package,
    so it is loaded by path. Fails loudly if the file is not where expected.
    """
    if not summarize_sims_path.exists():
        raise FileNotFoundError(
            f"summarize_sims.py not found at {summarize_sims_path}. "
            "Pass --summarize-sims to point at "
            "antigen-experiments/scripts/summarize_sims.py."
        )

    module_name = "summarize_sims"
    # summarize_sims parallelises with a ProcessPoolExecutor (n_jobs > 1), whose
    # workers pickle its functions by (module_name, qualname) and re-import the
    # module by name. Loading purely by file path leaves it unimportable by name,
    # so register it on sys.path and in sys.modules before executing it.
    sys.path.insert(0, str(summarize_sims_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, summarize_sims_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load a module spec from {summarize_sims_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_experiments(experiments_root: Path, patterns: Sequence[str]) -> list[Path]:
    """Find experiment directories matching ``patterns``.

    Parameters
    ----------
    experiments_root
        Directory holding one subdirectory per experiment.
    patterns
        Glob patterns evaluated relative to ``experiments_root`` (e.g. ``["*"]``).

    Returns
    -------
    Sorted, de-duplicated directories. Whether each is actually usable (has a
    ``sim_stats.csv`` to reuse or a ``simulations/`` tree to aggregate) is decided
    later, per experiment, by ``resolve_experiment_sim_stats``.
    """
    if not experiments_root.is_dir():
        raise FileNotFoundError(
            f"Experiments root not found or not a directory: {experiments_root}"
        )

    matched: set[Path] = set()
    for pattern in patterns:
        for path in experiments_root.glob(pattern):
            if path.is_dir():
                matched.add(path)

    return sorted(matched)


def resolve_experiment_sim_stats(
    experiment: Path, summarize_sims_module, n_jobs: int, refresh: bool
) -> pd.DataFrame | None:
    """Return one experiment's sim_stats, reusing or generating it as needed.

    Reuses ``<experiment>/sim_stats.csv`` when it exists and ``refresh`` is False.
    Otherwise, if the experiment has a ``simulations/`` tree, aggregates it with
    ``summarize_sims`` and writes the result to that canonical path (so the notebook
    and future runs can reuse it). Returns None for a directory that offers neither
    (skipped with a log line).
    """
    sim_stats_path = experiment / SIM_STATS_FILENAME

    if sim_stats_path.exists() and not refresh:
        logger.info("Reusing existing %s", sim_stats_path)
        return pd.read_csv(sim_stats_path)

    if not (experiment / "simulations").is_dir():
        logger.info(
            "Skipping %s: no %s to reuse and no simulations/ to aggregate.",
            experiment.name,
            SIM_STATS_FILENAME,
        )
        return None

    logger.info("Aggregating %s -> %s", experiment.name, sim_stats_path)
    try:
        df = summarize_sims_module.summarize_sims(
            str(experiment), output_path=str(sim_stats_path), n_jobs=n_jobs
        )
    except summarize_sims_module.BranchSchemaError as error:
        # Legacy experiments predate the current antigen-prime `.branches` schema,
        # which summarize_sims treats as fatal format drift. In a cross-experiment
        # sweep that must not abort the whole run: skip this experiment loudly and
        # continue. (Provide its sim_stats.csv or drop it from the glob to include.)
        logger.error(
            "Skipping %s: incompatible .branches format (%s).",
            experiment.name,
            error,
        )
        return None
    if df.empty:
        logger.warning("No runs found for %s; skipping.", experiment.name)
        return None
    return df


def build_sim_stats(
    experiments: Sequence[Path],
    summarize_sims_module,
    n_jobs: int,
    refresh: bool,
) -> pd.DataFrame:
    """Combine per-experiment sim_stats into one canonical-layout DataFrame.

    Each experiment is resolved (reused or generated) independently and the frames
    are concatenated. The union of columns is taken so experiments with different
    swept parameters coexist; the result is reordered to the canonical sim_stats
    layout.
    """
    frames: list[pd.DataFrame] = []
    for experiment in experiments:
        df = resolve_experiment_sim_stats(
            experiment, summarize_sims_module, n_jobs, refresh
        )
        if df is not None:
            frames.append(df)

    logger.info(
        "Contributed %d / %d experiment director(ies) to sim_stats.",
        len(frames),
        len(experiments),
    )

    if not frames:
        raise RuntimeError(
            "No usable experiments: none had a sim_stats.csv to reuse or a "
            "simulations/ tree to aggregate."
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # A run's path is globally unique (it includes the experiment name), so exact
    # duplicate paths would indicate a double-counted experiment.
    before = len(combined)
    combined = combined.drop_duplicates(subset=["path"], keep="first")
    if len(combined) != before:
        logger.warning("Dropped %d duplicate run paths.", before - len(combined))

    return order_sim_stats_columns(combined)


def order_sim_stats_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to the canonical sim_stats layout.

    Mirrors ``summarize_sims.summarize_sims``: ``path, run`` first, then sorted
    config/summary columns, then sorted branch columns, with ``branches_parse_failed``
    pinned last. Any unexpected columns are kept, placed before the trailing flag.
    """
    meta_cols = ["path", "run"]
    flag_cols = ["branches_parse_failed"]
    config_cols = [
        c
        for c in df.columns
        if c not in meta_cols
        and c not in flag_cols
        and not c.startswith("trunk")
        and not c.startswith("side")
    ]
    branch_cols = [
        c for c in df.columns if c.startswith("trunk") or c.startswith("side")
    ]

    all_ordered = meta_cols + sorted(config_cols) + sorted(branch_cols)
    ordered_cols = [c for c in all_ordered if c in df.columns]
    trailing_flags = [c for c in flag_cols if c in df.columns]
    remaining = [
        c for c in df.columns if c not in ordered_cols and c not in trailing_flags
    ]
    return df[ordered_cols + remaining + trailing_flags]


def select_candidates(sim_stats: pd.DataFrame) -> pd.DataFrame:
    """Filter aggregated stats to candidate runs.

    Adds a ``complete`` column (True when the run produced summary output), keeps
    only complete runs, then applies every threshold in ``CANDIDATE_FILTERS``. The
    returned frame's schema is the sim_stats columns plus a trailing ``complete``.
    """
    for column in (COMPLETENESS_STAT, *CANDIDATE_FILTERS):
        assert column in sim_stats.columns, (
            f"Required column {column!r} missing from sim_stats; "
            f"have {list(sim_stats.columns)}."
        )

    df = sim_stats.copy()
    df["complete"] = df[COMPLETENESS_STAT].notna()

    complete = df[df["complete"]].copy()
    logger.info("Complete runs: %d / %d", int(df["complete"].sum()), len(df))

    mask = pd.Series(True, index=complete.index)
    for column, conditions in CANDIDATE_FILTERS.items():
        series = complete[column]
        for comparison, bound in conditions:
            mask &= _COMPARATORS[comparison](series, bound)

    candidates = complete[mask].copy()
    logger.info("Candidate runs: %d / %d complete", len(candidates), len(complete))
    return candidates


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate antigen-experiments runs and select candidate runs by "
            "diversity, tMRCA, antigenic movement, and trunk epitope ratio."
        )
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        default=["*"],
        help=(
            "Experiment glob patterns relative to --experiments-root "
            "(default: '*', i.e. every experiment)."
        ),
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("../antigen-experiments/experiments"),
        help="Directory containing experiment folders (default: %(default)s).",
    )
    parser.add_argument(
        "--summarize-sims",
        type=Path,
        default=_DEFAULT_SUMMARIZE_SIMS,
        help="Path to antigen-experiments/scripts/summarize_sims.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("candidate_runs.csv"),
        help="Output path for candidate runs (default: %(default)s).",
    )
    parser.add_argument(
        "--sim-stats-output",
        type=Path,
        default=Path("sim_stats.csv"),
        help="Output path for the combined sim_stats table (default: %(default)s).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-aggregate via summarize_sims even where a sim_stats.csv already "
            "exists (rewrites each experiment's sim_stats.csv)."
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers passed to summarize_sims (-1 uses all CPUs).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    summarize_sims_module = load_summarize_sims(args.summarize_sims)

    experiments = discover_experiments(args.experiments_root, args.experiments)
    if not experiments:
        raise SystemExit(
            f"No directories matched {args.experiments} under {args.experiments_root}."
        )
    logger.info("Found %d experiment director(ies).", len(experiments))

    sim_stats = build_sim_stats(
        experiments, summarize_sims_module, args.jobs, args.refresh
    )
    sim_stats.to_csv(args.sim_stats_output, index=False)
    logger.info("Wrote %d rows to %s.", len(sim_stats), args.sim_stats_output)

    candidates = select_candidates(sim_stats)
    candidates.to_csv(args.output, index=False)
    logger.info("Wrote %d candidate runs to %s.", len(candidates), args.output)


if __name__ == "__main__":
    main()
