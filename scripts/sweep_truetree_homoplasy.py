"""Measure true-genealogy homoplasy and variant confusability across a sweep.

This is the across-run counterpart to ``scripts/plot_truetree_homoplasy.py`` and
the true-genealogy analogue of ``scripts/sweep_mutation_homoplasy.py``. For every
run in a batch it streams the run's ``run-out.branches`` (the exact simulation
genealogy, with parent and child sequences on every edge), reduces it to a
handful of per-run tables, and concatenates them into small committable CSVs
under ``results/aggregated/<batch>/``.

Two per-run inputs live in different trees:

- ``run-out.branches`` is a raw antigen output, under
  ``<experiments_root>/<batch>/simulations/<config>/run_N/output/``.
- ``tips_with_variants.tsv`` (the per-run variant labels) is a pipeline output,
  under ``<data_root>/<batch>/<config>__run_N/``.

The branches file is multi-GB and streamed line by line, so per-run memory stays
near half a gigabyte and the sweep runs as one ``-j`` job rather than a SLURM
array (see ``scripts/submit_truetree_sweep.sh``). A run is processed in two
tiers, matching ``sweep_mutation_homoplasy.py``:

1. **Recurrence and origin counts.** Need only ``run-out.branches`` and the
   shared reference and epitope-site list. A failure here aborts the run's row.
2. **Confusability.** Needs the per-run ``tips_with_variants.tsv``; a run missing
   it still contributes tier-1 rows, with the reason recorded in ``notes``.

Emitted tables:
- ``truetree_recurrence_by_run.csv`` -- one row per run.
- ``truetree_origin_counts_by_run.csv`` -- long form, the cumulative fraction of
  substitutions of each site class arising on at most X branches (the ECDF).
- ``truetree_confusability_by_run.csv`` -- long form, per run, method, and
  background-distance bin: the same-substitution and matched-null co-assignment
  rates whose difference is the confusability signal.

Usage:
    python scripts/sweep_truetree_homoplasy.py --batch 2026-07-04-reviewer-runs \\
        --data-root data/ --experiments-root ../antigen-experiments/experiments \\
        --ref-genbank data/flu-final/auspice/ref_HA.gb \\
        --epitope-sites ../antigen-prime/src/main/resources/epitopeSites.txt -j 6
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_truetree_homoplasy import (  # noqa: E402
    BINS,
    HA1_LENGTH,
    METHODS,
    _bin_labels,
    compute_truetree_tables,
    load_epitope_sites,
    load_gene_layouts,
)

logger = logging.getLogger(__name__)

RECURRENCE_NAME = "truetree_recurrence_by_run.csv"
ORIGIN_COUNTS_NAME = "truetree_origin_counts_by_run.csv"
CONFUSABILITY_NAME = "truetree_confusability_by_run.csv"

# Per-run inputs, relative to their respective roots.
BRANCHES_SUFFIX = Path("output") / "run-out.branches"
TIPS_VARIANTS_NAME = "tips_with_variants.tsv"

SITE_CLASSES = [(True, "epitope"), (False, "non_epitope")]


def setup_logging(verbose: bool, log_file: Path | None) -> None:
    """Stream logs to stdout and, optionally, a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_run_identity(sim_dir: Path) -> tuple[str, int]:
    """Recover ``(config, run_number)`` from a ``<config>__run_N`` directory name.

    Splits on the last ``__``, matching ``scripts/aggregate_results.py`` (configs
    may contain single underscores but never the ``__`` delimiter).
    """
    match = re.match(r"^(.+)__run_(\d+)$", sim_dir.name)
    if match is None:
        raise ValueError(
            f"cannot parse config/run from directory name {sim_dir.name!r}"
        )
    return match.group(1), int(match.group(2))


def discover_runs(
    data_root: Path, experiments_root: Path, batch: str
) -> list[tuple[str, int, Path, Path | None]]:
    """Find every run in a batch and locate its branches and variant-label files.

    Globs the *data* side (matching ``sweep_mutation_homoplasy.discover_run_trees``)
    so the run set is the same one the inferred sweep uses, then maps each run to
    its experiments-side ``run-out.branches``.

    Returns a sorted list of ``(config, run, branches_path, tips_path_or_None)``.
    ``tips_path`` is None when the run has no ``tips_with_variants.tsv``.
    """
    base = data_root / batch
    if not base.is_dir():
        raise FileNotFoundError(f"batch data directory not found: {base}")

    found: list[tuple[str, int, Path, Path | None]] = []
    for run_dir in sorted(p for p in base.glob("*__run_*") if p.is_dir()):
        config, run = parse_run_identity(run_dir)
        branches = (
            experiments_root / batch / "simulations" / config / f"run_{run}"
            / BRANCHES_SUFFIX
        )
        tips = run_dir / TIPS_VARIANTS_NAME
        found.append((config, run, branches, tips if tips.is_file() else None))

    if not found:
        raise ValueError(f"no run directories under {base}")
    return found


def _recurrence_row(config, run, occurrence, epitope_sites, stats, notes):
    row: dict[str, Any] = {"config": config, "run": run}
    # Report the unfiltered recurrence (min_progeny == 1); the established-lineage
    # recurrence is derived downstream from the origin-count table, which carries
    # every progeny threshold.
    base = occurrence[occurrence["min_progeny"] == 1]
    for is_epi, name in SITE_CLASSES:
        counts = base[base["is_epitope"] == is_epi]["n_independent_origins"]
        n = len(counts)
        row[f"{name}_n"] = n
        row[f"{name}_n_recurrent"] = int((counts >= 2).sum()) if n else 0
        row[f"{name}_recurrence_rate"] = float((counts >= 2).mean()) if n else np.nan
        row[f"{name}_max_origins"] = int(counts.max()) if n else 0
    row["n_edges"] = stats["n_edges"]
    row["n_tips"] = stats["n_tips"]
    row["n_aa_changing_edges"] = stats["n_changed"]
    row["n_tips_missing_label"] = stats["n_tip_missing"]
    row["notes"] = "; ".join(notes)
    return row


def _origin_count_rows(config, run, occurrence):
    """Long-form origin-count CDF, one block per progeny threshold and site class."""
    rows = []
    for thr in sorted(occurrence["min_progeny"].unique()):
        at_thr = occurrence[occurrence["min_progeny"] == thr]
        for is_epi, name in SITE_CLASSES:
            counts = at_thr[at_thr["is_epitope"] == is_epi][
                "n_independent_origins"
            ].to_numpy()
            n = len(counts)
            if n == 0:
                continue
            for x in range(1, int(counts.max()) + 1):
                rows.append(
                    {
                        "config": config,
                        "run": run,
                        "site_class": name,
                        "min_progeny": int(thr),
                        "x": x,
                        "cumulative_fraction": float(
                            np.count_nonzero(counts <= x) / n
                        ),
                        "n_substitutions": n,
                    }
                )
    return rows


def _confusability_rows(config, run, pairs):
    if pairs.empty:
        return []
    labels = _bin_labels()
    binned = pd.cut(
        pairs["background_distance_aa"], bins=BINS, right=False, labels=labels
    )
    same = pairs[pairs["kind"] == "same substitution"]
    null = pairs[pairs["kind"] == "random origins"]
    same_bin, null_bin = binned[same.index], binned[null.index]
    rows = []
    for method in METHODS:
        for low, lab in zip(BINS[:-1], labels):
            s = same[same_bin == lab][f"same_{method}"]
            r = null[null_bin == lab][f"same_{method}"]
            rows.append(
                {
                    "config": config,
                    "run": run,
                    "method": method,
                    "bg_bin": lab,
                    "bg_bin_low": low,
                    "same_rate": float(s.mean()) if len(s) else np.nan,
                    "null_rate": float(r.mean()) if len(r) else np.nan,
                    "n_same": int(len(s)),
                    "n_null": int(len(r)),
                }
            )
    return rows


def process_run(
    config: str,
    run: int,
    branches_path: Path,
    tips_path: Path | None,
    ref_genbank: Path,
    epitope_sites_path: Path,
) -> tuple[dict, list, list]:
    """Reduce one run to its recurrence, origin-count, and confusability rows.

    Loads the shared reference and epitope list inside the worker so the call is
    picklable for ``ProcessPoolExecutor``. A missing branches file is a tier-1
    failure recorded in ``notes``; a missing variant-label file drops only the
    confusability rows.
    """
    notes: list[str] = []
    if not branches_path.is_file():
        empty = {"config": config, "run": run, "notes": "no run-out.branches"}
        return empty, [], []
    if tips_path is None:
        notes.append("no tips_with_variants.tsv")

    genes = load_gene_layouts(ref_genbank)
    epitope_sites = load_epitope_sites(epitope_sites_path, HA1_LENGTH)
    tables = compute_truetree_tables(branches_path, tips_path, genes, epitope_sites)

    recurrence = _recurrence_row(
        config, run, tables["occurrence"], epitope_sites, tables["stats"], notes
    )
    origin_counts = _origin_count_rows(config, run, tables["occurrence"])
    confusability = _confusability_rows(config, run, tables["pairs"])
    return recurrence, origin_counts, confusability


def _write(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("wrote %s (%d rows) to %s", label, len(df), path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/"))
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("../antigen-experiments/experiments"),
        help="Root holding <batch>/simulations/<config>/run_N/output/run-out.branches.",
    )
    parser.add_argument("--ref-genbank", type=Path, required=True)
    parser.add_argument("--epitope-sites", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/"),
        help="Aggregated CSVs are written under <results-root>/aggregated/<batch>/.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir or (args.results_root / "aggregated" / args.batch)
    log_file = args.log_file or (output_dir / "sweep_truetree_homoplasy.log")
    setup_logging(args.verbose, log_file)

    runs = discover_runs(args.data_root, args.experiments_root, args.batch)
    n_with_branches = sum(1 for _c, _r, b, _t in runs if b.is_file())
    n_with_tips = sum(1 for _c, _r, _b, t in runs if t is not None)
    logger.info(
        "batch %s: %d runs (%d with run-out.branches, %d with variant labels)",
        args.batch,
        len(runs),
        n_with_branches,
        n_with_tips,
    )

    recurrence_rows: list[dict] = []
    origin_count_rows: list[dict] = []
    confusability_rows: list[dict] = []

    def record(index: int, result: tuple[dict, list, list]) -> None:
        rec, occ, conf = result
        recurrence_rows.append(rec)
        origin_count_rows.extend(occ)
        confusability_rows.extend(conf)
        status = "ok" if not rec.get("notes") else f"partial ({rec['notes']})"
        logger.info(
            "[%d/%d] %s__run_%s -> %s", index, len(runs), rec["config"], rec["run"],
            status,
        )

    call_args = [
        (config, run, branches, tips, args.ref_genbank, args.epitope_sites)
        for config, run, branches, tips in runs
    ]
    if args.jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(process_run, *call): call[:2] for call in call_args
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                record(index, future.result())
    else:
        for index, call in enumerate(call_args, start=1):
            record(index, process_run(*call))

    recurrence = pd.DataFrame(recurrence_rows).sort_values(["config", "run"])
    origin_counts = pd.DataFrame(origin_count_rows)
    if not origin_counts.empty:
        origin_counts = origin_counts.sort_values(
            ["config", "run", "min_progeny", "site_class", "x"]
        )
    confusability = pd.DataFrame(confusability_rows)
    if not confusability.empty:
        confusability = confusability.sort_values(
            ["config", "run", "method", "bg_bin_low"]
        )

    _write(recurrence, output_dir / RECURRENCE_NAME, "per-run recurrence")
    _write(origin_counts, output_dir / ORIGIN_COUNTS_NAME, "origin-count ECDF")
    _write(confusability, output_dir / CONFUSABILITY_NAME, "confusability")


if __name__ == "__main__":
    main()
