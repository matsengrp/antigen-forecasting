"""Parse antigen-prime ``run-out.tips`` into named CSV outputs for the pipeline.

Replaces the manual notebook step in ``analysis.ipynb`` that reads ``run-out.tips``,
adds a ``country`` column, deduplicates, and emits the artifacts every downstream
pipeline stage consumes.

Outputs (written to ``--output-dir``):
    tips.csv               Full parsed tips, no dedup.
    unique_tips.csv        Dedup name -> nucleotideSequence (in this exact order).
    unique_sequences.fasta FASTA of unique sequences (via AntigenReader).
    out_timeseries.csv     Symlink to run_N/out_timeseries.csv (copy fallback).

Design References:
- PRIMARY: specs/analysis-pipeline.md (Issue 1, Part A)

Implementation Status: WIP
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from antigentools.antigen_reader import AntigenReader


REQUIRED_COLUMNS: tuple[str, ...] = (
    "name",
    "year",
    "location",
    "nucleotideSequence",
    "ag1",
    "ag2",
)

# Pinned to match analysis.ipynb in every experiment folder. Do not vary.
COUNTRY_MAP: dict[int, str] = {0: "north", 1: "tropics", 2: "south"}


logger = logging.getLogger(__name__)


def parse_run_out_tips(tips_path: Path) -> pd.DataFrame:
    """Read ``run-out.tips`` and add the ``country`` column.

    Args:
        tips_path: Path to a ``run-out.tips`` file (CSV format, comma-separated,
            no extension).

    Returns:
        DataFrame with all original columns plus a ``country`` column derived from
        ``location`` via the pinned ``{0: "north", 1: "tropics", 2: "south"}`` map.

    Raises:
        ValueError: If any required column is missing, or if ``location`` contains
            a value outside the pinned map.
    """
    df = pd.read_csv(tips_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{tips_path} missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )

    unknown = set(df["location"].unique()) - set(COUNTRY_MAP.keys())
    if unknown:
        raise ValueError(
            f"{tips_path} contains unknown location values {sorted(unknown)}; "
            f"pinned mapping covers only {sorted(COUNTRY_MAP.keys())}"
        )

    df = df.copy()
    df["country"] = df["location"].map(COUNTRY_MAP)
    return df


def dedup_tips(tips_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate tips on ``name`` first, then ``nucleotideSequence``.

    Order is load-bearing: ``name``-first then ``nucleotideSequence`` matches the
    canonical ``analysis.ipynb`` reference. Reversing the order picks different
    representative rows on collisions, silently corrupting downstream variant
    assignment.

    Args:
        tips_df: Full parsed tips frame (must include ``name`` and
            ``nucleotideSequence`` columns).

    Returns:
        Deduplicated DataFrame (rows are a strict subset of ``tips_df`` rows).
    """
    n_in = len(tips_df)
    after_name = tips_df.drop_duplicates(subset=["name"])
    n_after_name = len(after_name)
    after_seq = after_name.drop_duplicates(subset=["nucleotideSequence"])
    n_after_seq = len(after_seq)

    logger.info(
        "Dedup dropped %d rows on 'name' (%d -> %d), %d more on 'nucleotideSequence' "
        "(%d -> %d).",
        n_in - n_after_name,
        n_in,
        n_after_name,
        n_after_name - n_after_seq,
        n_after_name,
        n_after_seq,
    )
    return after_seq


def link_or_copy(src: Path, dst: Path) -> None:
    """Symlink ``src`` to ``dst``; fall back to a copy on filesystems that don't
    support symlinks.

    The symlink target is resolved to an absolute path so the link works regardless
    of ``dst``'s parent directory. ``os.symlink`` interprets relative ``src`` paths
    relative to ``dst``'s directory, not the caller's CWD, which is the wrong default
    when the caller passed a CWD-relative ``--sim-path``.

    Args:
        src: Source path (must exist).
        dst: Destination path. Parent directory must already exist.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
    """
    if not src.exists():
        raise FileNotFoundError(f"Cannot link/copy missing source: {src}")
    dst.unlink(missing_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy(src, dst)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI args and write all four outputs to ``--output-dir``.

    Args:
        argv: Argument list (``None`` -> ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Parse antigen-prime run-out.tips into pipeline-ready CSV/FASTA outputs."
        )
    )
    parser.add_argument(
        "--sim-path",
        required=True,
        type=Path,
        help="Path to run_N/ directory under antigen-experiments.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write tips.csv, unique_tips.csv, unique_sequences.fasta, "
        "and out_timeseries.csv.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    sim_path: Path = args.sim_path
    output_dir: Path = args.output_dir

    raw_tips = sim_path / "output" / "run-out.tips"
    raw_timeseries = sim_path / "out_timeseries.csv"

    if not raw_tips.exists():
        raise FileNotFoundError(f"run-out.tips not found at {raw_tips}")
    if not raw_timeseries.exists():
        raise FileNotFoundError(f"out_timeseries.csv not found at {raw_timeseries}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading raw tips from %s", raw_tips)
    tips_df = parse_run_out_tips(raw_tips)
    logger.info("Read %d full tips rows.", len(tips_df))

    tips_out = output_dir / "tips.csv"
    tips_df.to_csv(tips_out, index=False)
    logger.info("Wrote %s", tips_out)

    unique_df = dedup_tips(tips_df)
    assert len(unique_df) < len(tips_df), (
        f"dedup did not drop any rows ({len(tips_df)} -> {len(unique_df)}); "
        f"input is unexpectedly already-unique on both 'name' and 'nucleotideSequence'"
    )

    unique_out = output_dir / "unique_tips.csv"
    unique_df.to_csv(unique_out, index=False)
    logger.info("Wrote %s (%d rows)", unique_out, len(unique_df))

    fasta_out = output_dir / "unique_sequences.fasta"
    AntigenReader().write_tips_to_fasta(unique_df, str(fasta_out))
    logger.info("Wrote %s", fasta_out)

    timeseries_out = output_dir / "out_timeseries.csv"
    link_or_copy(raw_timeseries, timeseries_out)
    logger.info("Linked/copied %s -> %s", raw_timeseries, timeseries_out)


if __name__ == "__main__":
    main(sys.argv[1:])
