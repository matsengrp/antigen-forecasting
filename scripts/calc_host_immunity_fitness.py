"""Compute per-sequence risk of infection over time from real host immune histories.

Replaces the population-centroid approximation used by
``scripts/calc_variant_fitness_variance.py``. Instead of one ``(ag1, ag2)`` immune-memory
centroid per year, this scores every unique sequence against a sample of individual hosts'
full immune histories, taking the minimum antigenic distance within each host's memory --
the same rule the antigen simulation itself applies.

A sequence's fitness at time ``t`` is its mean risk of infection across sampled hosts. The
per-sequence output carries both weightings -- ``experienced`` (hosts with immune memory
only) and ``population`` (naive hosts folded in at risk 1.0, using the file's
``naive_fraction``) -- while the variance output uses whichever ``--host-weighting``
selects, so its columns stay identical to
``antigentools.analysis.calc_variance_over_time`` and nothing downstream has to change.

Inputs come straight from an antigen-prime run directory::

    run_N/out.histories.raw.csv   per-host immune histories (~100 MB)
    run_N/out.histories.csv       per-deme centroids, used only by --validate-centroids

The tips file must be deduplicated on ``name`` and ``nucleotideSequence`` -- use
``scripts/parse_sim_outputs.py``, which emits ``unique_tips.csv`` and
``unique_sequences.fasta`` from the run directory's full ``output/run-out.tips``.

Usage:
    python scripts/calc_host_immunity_fitness.py \
        --tips data/<batch>/<config>__run_0/tips_with_variants.tsv \
        --histories-raw <run_dir>/out.histories.raw.csv \
        --variance-output results/<batch>/host_immunity_variance.tsv \
        --risk-output results/<batch>/host_immunity_risk.tsv.gz

Design References:
- PRIMARY: specs/analysis-pipeline.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from antigentools.host_immunity import (
    HOST_WEIGHTINGS,
    YEAR_TOL,
    load_raw_histories,
    risk_of_infection_over_time,
    select_timepoints,
    variance_from_risk,
)

logger = logging.getLogger(__name__)


# The histories snapshot cadence is printHostImmunityStep days; the reviewer runs use 365,
# so the native grid is yearly and no finer delta-t is representable. select_timepoints
# raises with an actionable message if a finer value is requested.
DEFAULT_DELTA_T: float = 1.0
DEFAULT_N_HOSTS: int = 1000
DEFAULT_SEED: int = 42
DEFAULT_SMITH_CONVERSION: float = 0.07
DEFAULT_HOMOLOGOUS_IMMUNITY: float = 0.95
DEFAULT_N_VARIANT_WINDOW: float = 1.0
# Population weighting keeps the variance output a drop-in for the centroid method's
# schema and matches the absolute infection risk the simulation itself averages.
DEFAULT_HOST_WEIGHTING: str = "population"
# Element budget per distance block: 8M float32 values is ~32 MB per temporary, which
# keeps peak memory flat whether 1,000 or 30,000 hosts are sampled.
DEFAULT_MAX_BLOCK_ELEMENTS: int = 8_000_000


def read_tips(tips_path: Path) -> pd.DataFrame:
    """Read a tips table, inferring the separator from the file suffix.

    Args:
        tips_path: Path to a ``.csv`` (comma) or any other suffix (tab) tips table.

    Returns:
        The parsed tips DataFrame.
    """
    separator = "," if tips_path.suffix == ".csv" else "\t"
    return pd.read_csv(tips_path, sep=separator)


def apply_burn_in(
    tips_df: pd.DataFrame, histories_df: pd.DataFrame, burn_in: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the first ``burn_in`` years from both frames and re-zero their year axes.

    Both frames are shifted in the same direction, unlike
    ``scripts/calc_variant_fitness_variance.py``, which shifts tips and histories
    oppositely.

    Args:
        tips_df: Tips table with a ``year`` column.
        histories_df: Raw histories with a ``year`` column.
        burn_in: Years to discard from the start of the simulation.

    Returns:
        Tuple of the filtered, re-zeroed ``(tips_df, histories_df)``.

    Raises:
        ValueError: If the burn-in leaves either frame empty.
    """
    if burn_in <= 0:
        return tips_df, histories_df

    tips_out = tips_df[tips_df["year"] >= burn_in].copy()
    histories_out = histories_df[histories_df["year"] >= burn_in].copy()
    if tips_out.empty or histories_out.empty:
        raise ValueError(
            f"burn_in={burn_in} left {len(tips_out)} tips and {len(histories_out)} "
            f"history rows; nothing to compute"
        )

    tips_out["year"] = tips_out["year"] - burn_in
    histories_out["year"] = histories_out["year"] - burn_in
    logger.info(
        "Burn-in of %s years: %d -> %d tips, %d -> %d history rows.",
        burn_in,
        len(tips_df),
        len(tips_out),
        len(histories_df),
        len(histories_out),
    )
    return tips_out, histories_out


def validate_against_centroids(
    histories_df: pd.DataFrame, centroids_path: Path, timepoints: np.ndarray
) -> None:
    """Cross-check the raw histories against the per-deme centroid summary.

    Two independent checks, both on ``out.histories.csv``:

    1. The distinct host count per ``(year, deme)`` must equal ``experienced_hosts``. This
       confirms the two files describe the same sampled host universe, which is what makes
       the ``naive_fraction`` reconstruction in
       ``antigentools.host_immunity.global_naive_fraction`` valid.
    2. The summary centroid must be the mean of each host's *most recent* infection, per
       ``HostPopulation.getPopulationImmunitySummary``, which calls
       ``getMostRecentInfectionCoordinates``. Reported rather than raised, since only the
       host counts are load-bearing here -- but a large residual means the centroid is not
       what the old fitness method assumed it was.

    Args:
        histories_df: Raw histories from :func:`load_raw_histories`.
        centroids_path: Path to the run's ``out.histories.csv``.
        timepoints: Years to check.

    Raises:
        ValueError: If a ``(year, deme)`` host count disagrees with ``experienced_hosts``.
    """
    centroids = pd.read_csv(centroids_path)
    if "experienced_hosts" not in centroids.columns:
        logger.warning(
            "%s has no 'experienced_hosts' column; skipping host-count check.",
            centroids_path,
        )
        return

    most_recent_errors = []
    all_entry_errors = []

    for timepoint in timepoints:
        year_df = histories_df[
            np.abs(histories_df["year"].to_numpy() - timepoint) <= YEAR_TOL
        ]
        summary = centroids[np.abs(centroids["year"] - timepoint) <= YEAR_TOL]
        for deme, group in year_df.groupby("deme", observed=True):
            row = summary[summary["deme"] == deme]
            if row.empty:
                continue
            expected = int(row["experienced_hosts"].iloc[0])
            observed = int(group["host_id"].nunique())
            if observed != expected:
                raise ValueError(
                    f"year {timepoint}, deme {deme!r}: raw histories hold {observed} "
                    f"distinct hosts but {centroids_path} reports "
                    f"experienced_hosts={expected}"
                )
            latest = group.loc[
                group.groupby("host_id", observed=True)["infection_index"].idxmax()
            ]
            for col in ("ag1", "ag2"):
                reference = float(row[col].iloc[0])
                most_recent_errors.append(abs(latest[col].mean() - reference))
                all_entry_errors.append(abs(group[col].mean() - reference))

    if not most_recent_errors:
        logger.warning("No overlapping (year, deme) rows to validate against centroids.")
        return

    logger.info(
        "Centroid check passed on host counts. Max |centroid - mean of most recent "
        "infections| = %.4g (this is the definition antigen uses); max |centroid - mean "
        "over all memory entries| = %.4g.",
        max(most_recent_errors),
        max(all_entry_errors),
    )


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Build and parse the command-line interface.

    Args:
        argv: Argument list (``None`` -> ``sys.argv[1:]``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-sequence risk of infection over time against sampled hosts' "
            "immune histories, and the resulting within-variant fitness variance."
        )
    )
    parser.add_argument(
        "--tips",
        required=True,
        type=Path,
        help="Deduplicated tips table with name, year, ag1, ag2 and variant_* columns.",
    )
    parser.add_argument(
        "--histories-raw",
        required=True,
        type=Path,
        help="Path to the run's out.histories.raw.csv.",
    )
    parser.add_argument(
        "--variance-output",
        required=True,
        type=Path,
        help="Output TSV of mean within-variant fitness variance over time.",
    )
    parser.add_argument(
        "--risk-output",
        type=Path,
        default=None,
        help="Optional output of per-sequence risk of infection (.tsv or .tsv.gz).",
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=DEFAULT_DELTA_T,
        help=(
            f"Years between evaluated timepoints; must be a multiple of the histories' "
            f"snapshot cadence, which is printHostImmunityStep/365 years "
            f"(default: {DEFAULT_DELTA_T})."
        ),
    )
    parser.add_argument(
        "--n-hosts",
        type=int,
        default=DEFAULT_N_HOSTS,
        help=(
            f"Distinct hosts to sample per timepoint; 0 or less uses every host "
            f"(default: {DEFAULT_N_HOSTS})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for host sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--smith-conversion",
        type=float,
        default=DEFAULT_SMITH_CONVERSION,
        help=(
            f"Factor scaling antigenic distance to infection risk "
            f"(default: {DEFAULT_SMITH_CONVERSION})."
        ),
    )
    parser.add_argument(
        "--homologous-immunity",
        type=float,
        default=DEFAULT_HOMOLOGOUS_IMMUNITY,
        help=(
            f"Immunity against an identical antigen "
            f"(default: {DEFAULT_HOMOLOGOUS_IMMUNITY})."
        ),
    )
    parser.add_argument(
        "--n-variant-window",
        type=float,
        default=DEFAULT_N_VARIANT_WINDOW,
        help=(
            f"Window in years for counting sampled variants, kept at "
            f"{DEFAULT_N_VARIANT_WINDOW} for parity with calc_variance_over_time."
        ),
    )
    parser.add_argument(
        "--host-weighting",
        choices=HOST_WEIGHTINGS,
        default=DEFAULT_HOST_WEIGHTING,
        help=(
            f"Which risk average the variance is taken over: 'population' includes naive "
            f"hosts at risk 1.0, 'experienced' does not "
            f"(default: {DEFAULT_HOST_WEIGHTING})."
        ),
    )
    parser.add_argument(
        "--burn-in",
        type=float,
        default=0.0,
        help="Years to discard from the start of both tips and histories (default: 0).",
    )
    parser.add_argument(
        "--max-block-elements",
        type=int,
        default=DEFAULT_MAX_BLOCK_ELEMENTS,
        help=(
            f"Element budget per distance block, bounding peak memory "
            f"(default: {DEFAULT_MAX_BLOCK_ELEMENTS})."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier for aggregation (defaults to the tips filename stem).",
    )
    parser.add_argument(
        "--validate-centroids",
        type=Path,
        default=None,
        help="Optional out.histories.csv to cross-check host counts and centroids against.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the full risk-of-infection and fitness-variance calculation.

    Args:
        argv: Argument list (``None`` -> ``sys.argv[1:]``).

    Raises:
        ValueError: If the tips table carries no ``variant_*`` columns.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    tips_df = read_tips(args.tips)
    logger.info("Read %d tips from %s.", len(tips_df), args.tips)

    variant_cols = [col for col in tips_df.columns if col.startswith("variant_")]
    if not variant_cols:
        raise ValueError(f"no variant_* columns found in {args.tips}")
    logger.info("Variant assignment columns: %s", variant_cols)

    histories_df = load_raw_histories(args.histories_raw)
    tips_df, histories_df = apply_burn_in(tips_df, histories_df, args.burn_in)

    n_hosts = args.n_hosts if args.n_hosts > 0 else None
    if n_hosts is None:
        logger.info("Using every host at each timepoint (--n-hosts <= 0).")

    if args.validate_centroids is not None:
        timepoints = select_timepoints(histories_df["year"].to_numpy(), args.delta_t)
        validate_against_centroids(histories_df, args.validate_centroids, timepoints)

    risk_df = risk_of_infection_over_time(
        tips_df,
        histories_df,
        delta_t=args.delta_t,
        n_hosts=n_hosts,
        seed=args.seed,
        smith_conversion=args.smith_conversion,
        homologous_immunity=args.homologous_immunity,
        max_block_elements=args.max_block_elements,
    )

    variance_df = variance_from_risk(
        risk_df,
        tips_df,
        variant_cols,
        n_variant_window=args.n_variant_window,
        host_weighting=args.host_weighting,
    )
    logger.info("Variance computed on the %r host weighting.", args.host_weighting)

    run_id = args.run_id or args.tips.stem
    variance_df.insert(0, "run_id", run_id)

    args.variance_output.parent.mkdir(parents=True, exist_ok=True)
    variance_df.to_csv(args.variance_output, sep="\t", index=False)
    logger.info("Wrote %d rows to %s", len(variance_df), args.variance_output)

    if args.risk_output is not None:
        risk_df.insert(0, "run_id", run_id)
        args.risk_output.parent.mkdir(parents=True, exist_ok=True)
        risk_df.to_csv(args.risk_output, sep="\t", index=False)
        logger.info("Wrote %d rows to %s", len(risk_df), args.risk_output)


if __name__ == "__main__":
    main(sys.argv[1:])
