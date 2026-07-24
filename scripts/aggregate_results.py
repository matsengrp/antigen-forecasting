"""Aggregate per-run sweep results into small, committable tables.

Discovers every ``results/<batch>/<config>__run_N/`` produced by the SLURM sweep,
reduces each run to a handful of small per-run quantities, and concatenates them
(tagged with ``batch, config, run``) into tidy CSVs under
``results/aggregated/<batch>/`` that are safe to commit and push. The heavy
per-run trees (``estimates/`` etc.) are never read wholesale.

Emitted tables:
- ``variant_counts_over_time.csv`` — distinct variants per year bin per method.
- ``method_agreement_nid.csv`` — NID between each pair of assignment methods.
- ``fitness_variance_over_time.csv`` — within-variant fitness variance over time
  (only when a centroid history file is available; see ``--experiments-root`` /
  ``--histories-name``).
- ``growth_rate_scores_all.csv`` — the per-run growth-rate score TSVs stacked.
- ``scores_summary.csv`` — per-run frequency scores summarized to
  ``(model, location, lead)`` means (the raw per-point ``scores.tsv`` is far too
  large to commit, so it is collapsed rather than concatenated).

Progress streams to stdout and to ``--log-file`` so status is visible at a glance
on the HPC (``tail -f`` the log). Runs missing an input are skipped with a logged
reason rather than aborting the batch.

Usage:
    python scripts/aggregate_results.py --batch 2026-07-04-reviewer-runs \\
        --results-root results/ --data-root data/ -j 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from antigentools import variant_agreement as va
from antigentools.analysis import POP_TOTAL_DEMES as analysis_pop_total_demes

logger = logging.getLogger(__name__)

METHOD_COLS: tuple[str, ...] = ("variant_ag", "variant_tsne", "variant_phylo")

# Population-total immune-memory deme label in out.histories.csv, most-preferred
# first: "global" in the current schema, "total" in the older flu-final one. Re-exported
# from antigentools.analysis so this and calc_variant_fitness_variance.py share one
# definition.
POP_TOTAL_DEMES = analysis_pop_total_demes

# scores.tsv is per-forecast-point (model x location x pivot_date x lead x variant
# x date), which concatenated across runs is far too large to commit (~GB). We
# summarize it per run by averaging the metric columns over the high-cardinality
# date/variant/pivot dimensions, keeping only these grouping keys.
SCORE_SUMMARY_KEYS: tuple[str, ...] = ("model", "location", "lead")
SCORE_METRIC_COLS: tuple[str, ...] = (
    "MAE",
    "MSE",
    "loglik",
    "coverage_posterior",
    "coverage_predictive",
)


def setup_logging(verbose: bool, log_file: Path | None) -> None:
    """Configure root logging to stream to stdout and, optionally, a file.

    Mirrors ``scripts/score_models.py`` so the on-screen and on-disk logs share
    one timestamped format.

    Args:
        verbose: Emit DEBUG-level records when True, else INFO.
        log_file: If given, also append records to this file (parent dirs made).
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logging.root.setLevel(level)
    logging.root.addHandler(console_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def discover_runs(results_root: Path, batch: str) -> list[Path]:
    """Return the sorted per-run result directories for a batch.

    Globs ``<results_root>/<batch>/*__run_*/``. The ``__`` separator between the
    config name and the run id is guaranteed unique by
    ``antigentools.paths.SimulationPaths``.

    Args:
        results_root: Root results directory (e.g. ``results/``).
        batch: Batch name that namespaces the sweep's results.

    Returns:
        Sorted list of ``<config>__run_N/`` directories.

    Raises:
        FileNotFoundError: If ``<results_root>/<batch>/`` does not exist.
        ValueError: If no ``*__run_*/`` directories are found.
    """
    base = results_root / batch
    if not base.is_dir():
        raise FileNotFoundError(f"Batch results directory not found: {base}")
    runs = sorted(p for p in base.glob("*__run_*") if p.is_dir())
    if not runs:
        raise ValueError(f"No *__run_*/ directories found under {base}")
    return runs


def parse_run_identity(sim_dir: Path) -> tuple[str, int]:
    """Recover ``(config, run_number)`` from a ``<config>__run_N`` directory name.

    Splits on the last ``__`` (configs may contain single underscores but not the
    ``__`` delimiter).

    Args:
        sim_dir: Path whose name is ``<config>__run_N``.

    Returns:
        ``(config, run_number)``.

    Raises:
        ValueError: If the name lacks the ``__run_<n>`` structure.
    """
    name = sim_dir.name
    if "__" not in name:
        raise ValueError(f"Cannot parse config/run from {name!r} (no '__')")
    config, run_id = name.rsplit("__", 1)
    if not run_id.startswith("run_"):
        raise ValueError(f"Cannot parse run number from {name!r}")
    return config, int(run_id.removeprefix("run_"))


def _tag(df: pd.DataFrame, batch: str, config: str, run: int) -> pd.DataFrame:
    """Return ``df`` with ``batch, config, run`` id columns prepended."""
    df = df.copy()
    df.insert(0, "run", run)
    df.insert(0, "config", config)
    df.insert(0, "batch", batch)
    return df


def process_run(
    sim_dir: Path,
    batch: str,
    data_root: Path,
    experiments_root: Path | None,
    experiment: str,
    histories_name: str | None,
) -> dict:
    """Reduce one run's outputs to tagged per-run DataFrames.

    Every sub-part is attempted independently; a failure in one leaves that piece
    as None and records a note, so a single missing/broken file never discards the
    rest of the run.

    Args:
        sim_dir: The run's ``results/<batch>/<config>__run_N/`` directory.
        batch: Batch name (tagged onto every row).
        data_root: Root data directory holding ``<batch>/<sim_id>/``.
        experiments_root: Root of ``antigen-experiments/experiments/`` for centroid
            histories, or None to skip fitness variance.
        experiment: Experiment folder name under ``experiments_root``.
        histories_name: Filename of the per-run centroid/history file, or None.

    Returns:
        A dict with keys ``sim_id, config, run, counts, nid, variance, gr_scores,
        scores, notes``. The five DataFrame slots are None when unavailable.
    """
    sim_id = sim_dir.name
    config, run = parse_run_identity(sim_dir)
    result: dict = {
        "sim_id": sim_id,
        "config": config,
        "run": run,
        "counts": None,
        "nid": None,
        "variance": None,
        "gr_scores": None,
        "scores": None,
        "notes": [],
    }

    tips_path = data_root / batch / sim_id / "tips_with_variants.tsv"
    tips_df: pd.DataFrame | None = None
    if tips_path.exists():
        try:
            tips_df = pd.read_csv(tips_path, sep="\t")
        except Exception as exc:  # noqa: BLE001 - record and continue.
            result["notes"].append(f"tips unreadable ({exc})")
    else:
        result["notes"].append("no tips_with_variants.tsv")

    if tips_df is not None:
        method_cols = [c for c in METHOD_COLS if c in tips_df.columns]
        counts = va.variant_counts_over_time(tips_df, method_cols)
        if not counts.empty:
            result["counts"] = _tag(counts, batch, config, run)
        nid = va.nid_pairs(tips_df, method_cols)
        if not nid.empty:
            result["nid"] = _tag(nid, batch, config, run)

        if experiments_root is not None and histories_name is not None:
            variance = _compute_fitness_variance(
                tips_df, experiments_root, experiment, config, run, histories_name
            )
            if variance is None:
                result["notes"].append("fitness_variance skipped: no centroid history")
            else:
                result["variance"] = _tag(variance, batch, config, run)

    gr_path = sim_dir / "growth_rate_scores.tsv"
    if gr_path.exists():
        try:
            result["gr_scores"] = _tag(
                pd.read_csv(gr_path, sep="\t"), batch, config, run
            )
        except Exception as exc:  # noqa: BLE001
            result["notes"].append(f"growth_rate_scores unreadable ({exc})")
    else:
        result["notes"].append("no growth_rate_scores.tsv")

    scores_path = sim_dir / "scores.tsv"
    if scores_path.exists():
        try:
            summary = _summarize_scores(pd.read_csv(scores_path, sep="\t"))
            if summary is None or summary.empty:
                result["notes"].append("scores present but not summarizable")
            else:
                result["scores"] = _tag(summary, batch, config, run)
        except Exception as exc:  # noqa: BLE001
            result["notes"].append(f"scores unreadable ({exc})")
    else:
        result["notes"].append("no scores.tsv")

    return result


def _summarize_scores(scores_df: pd.DataFrame) -> pd.DataFrame | None:
    """Collapse a run's per-point scores.tsv to a small committable summary.

    Groups by the low-cardinality keys present among ``SCORE_SUMMARY_KEYS`` and
    averages the metric columns present among ``SCORE_METRIC_COLS`` over the
    date/variant/pivot dimensions, adding an ``n_points`` count.

    Args:
        scores_df: The raw per-forecast-point scores table.

    Returns:
        A summarized DataFrame, or None if no usable keys or metrics are present.
    """
    keys = [k for k in SCORE_SUMMARY_KEYS if k in scores_df.columns]
    metrics = [m for m in SCORE_METRIC_COLS if m in scores_df.columns]
    if not keys or not metrics:
        return None
    grouped = scores_df.groupby(keys, dropna=False)
    summary = grouped[metrics].mean()
    summary["n_points"] = grouped.size()
    return summary.reset_index()


def _compute_fitness_variance(
    tips_df: pd.DataFrame,
    experiments_root: Path,
    experiment: str,
    config: str,
    run: int,
    histories_name: str,
) -> pd.DataFrame | None:
    """Compute within-variant fitness variance over time, or None if unavailable.

    Locates the run's centroid/history file under the raw antigen-experiments tree
    and delegates to ``antigentools.analysis.calc_variance_over_time``. Any missing
    file or schema mismatch returns None (caller logs a skip) rather than raising.

    Args:
        tips_df: The run's tips table (has ``ag1, ag2, year, variant_*``).
        experiments_root: Root of ``antigen-experiments/experiments/``.
        experiment: Experiment folder name.
        config: Sweep-config name.
        run: Run number.
        histories_name: Centroid/history filename within the run directory.

    Returns:
        Long DataFrame from ``calc_variance_over_time`` (``year, method,
        mean_variance, n_variants``), or None.
    """
    from antigentools.analysis import calc_variance_over_time

    history_path = (
        experiments_root
        / experiment
        / "simulations"
        / config
        / f"run_{run}"
        / histories_name
    )
    if not history_path.exists():
        return None
    try:
        histories_df = pd.read_csv(history_path)
        if "deme" in histories_df.columns:
            # The population-total immune-memory centroid is labeled "global" in
            # these histories ("total" in the older flu-final schema).
            demes = set(histories_df["deme"])
            label = next((d for d in POP_TOTAL_DEMES if d in demes), None)
            if label is None:
                logger.debug(
                    "no population-total deme %s in %s (have %s)",
                    POP_TOTAL_DEMES,
                    history_path,
                    sorted(demes),
                )
                return None
            histories_df = histories_df[histories_df["deme"] == label].copy()
        method_cols = [c for c in METHOD_COLS if c in tips_df.columns]
        result = calc_variance_over_time(tips_df, histories_df, method_cols)
        return result if result is not None and not result.empty else None
    except Exception as exc:  # noqa: BLE001 - variance is best-effort.
        logger.debug("fitness variance failed for %s/run_%s: %s", config, run, exc)
        return None


def _concat_and_write(
    frames: list[pd.DataFrame], output_path: Path, label: str
) -> None:
    """Concatenate ``frames`` and write to ``output_path``, logging the row count."""
    if not frames:
        logger.info("  %s: no rows (0 runs contributed); not written", label)
        return
    combined = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info("  %s: wrote %d rows -> %s", label, len(combined), output_path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=str, help="Batch name.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root results directory (default: results/).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory holding tips_with_variants.tsv (default: data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write aggregated CSVs (default: results/aggregated/<batch>/).",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="Root of antigen-experiments/experiments/ (enables fitness variance).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment folder for centroid histories (default: --batch value).",
    )
    parser.add_argument(
        "--histories-name",
        type=str,
        default=None,
        help="Per-run centroid/history filename (enables fitness variance).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path (default: <output-dir>/aggregate.log).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="DEBUG-level logging."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Discover runs, reduce each, and write the aggregated committable CSVs."""
    args = _parse_args(argv)

    output_dir = args.output_dir or (args.results_root / "aggregated" / args.batch)
    log_file = args.log_file or (output_dir / "aggregate.log")
    setup_logging(args.verbose, log_file)

    experiment = args.experiment or args.batch
    runs = discover_runs(args.results_root, args.batch)
    logger.info(
        "Aggregating batch %s: %d run(s) discovered, -j=%d, output -> %s",
        args.batch,
        len(runs),
        args.jobs,
        output_dir,
    )
    if args.experiments_root is None or args.histories_name is None:
        logger.info(
            "Fitness variance disabled (need both --experiments-root and "
            "--histories-name)."
        )

    t0 = time.monotonic()
    results: list[dict] = []
    skipped: list[tuple[str, str]] = []
    total = len(runs)

    def _record(index: int, res: dict) -> None:
        note = "; ".join(res["notes"]) if res["notes"] else "ok"
        produced = [
            k
            for k in ("counts", "nid", "variance", "gr_scores", "scores")
            if res[k] is not None
        ]
        if not produced:
            skipped.append((res["sim_id"], note))
            logger.warning("[%d/%d] %s -> SKIP (%s)", index, total, res["sim_id"], note)
        else:
            logger.info(
                "[%d/%d] %s -> ok [%s]%s",
                index,
                total,
                res["sim_id"],
                ",".join(produced),
                "" if note == "ok" else f" ({note})",
            )
        results.append(res)

    if args.jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = {
                ex.submit(
                    process_run,
                    sim_dir,
                    args.batch,
                    args.data_root,
                    args.experiments_root,
                    experiment,
                    args.histories_name,
                ): sim_dir
                for sim_dir in runs
            }
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                _record(i, future.result())
    else:
        for i, sim_dir in enumerate(runs, 1):
            res = process_run(
                sim_dir,
                args.batch,
                args.data_root,
                args.experiments_root,
                experiment,
                args.histories_name,
            )
            _record(i, res)

    logger.info("Writing aggregated tables to %s", output_dir)
    _concat_and_write(
        [r["counts"] for r in results if r["counts"] is not None],
        output_dir / "variant_counts_over_time.csv",
        "variant_counts_over_time",
    )
    _concat_and_write(
        [r["nid"] for r in results if r["nid"] is not None],
        output_dir / "method_agreement_nid.csv",
        "method_agreement_nid",
    )
    _concat_and_write(
        [r["variance"] for r in results if r["variance"] is not None],
        output_dir / "fitness_variance_over_time.csv",
        "fitness_variance_over_time",
    )
    _concat_and_write(
        [r["gr_scores"] for r in results if r["gr_scores"] is not None],
        output_dir / "growth_rate_scores_all.csv",
        "growth_rate_scores_all",
    )
    _concat_and_write(
        [r["scores"] for r in results if r["scores"] is not None],
        output_dir / "scores_summary.csv",
        "scores_summary",
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "Done: %d processed, %d skipped, %.1fs elapsed.",
        total - len(skipped),
        len(skipped),
        elapsed,
    )
    for sim_id, reason in skipped:
        logger.info("  skipped %s (%s)", sim_id, reason)


if __name__ == "__main__":
    main(sys.argv[1:])
