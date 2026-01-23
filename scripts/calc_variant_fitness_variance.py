#!/usr/bin/env python
"""
Calculate temporal fitness variance for variant assignment methods.

For each timepoint, computes mean within-variant fitness variance using ALL tips
and host immune memory at that time. Outputs long-format TSV for aggregation
across simulation runs.

Usage:
    python scripts/calc_variant_fitness_variance.py \
        --tips data/run1/tips.tsv \
        --histories data/run1/histories.csv \
        --output results/run1/fitness_variance.tsv
"""

import argparse
from pathlib import Path
import pandas as pd
from antigentools.analysis import calc_variance_over_time


def main():
    parser = argparse.ArgumentParser(
        description="Calculate temporal fitness variance for variant assignments."
    )
    parser.add_argument(
        "--tips", "-t",
        required=True,
        help="Path to tips TSV with ag1, ag2, year, and variant_* columns"
    )
    parser.add_argument(
        "--histories", "-i",
        required=True,
        help="Path to histories CSV with year, deme, ag1, ag2 columns"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output TSV path"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier for aggregation (defaults to tips filename stem)"
    )
    parser.add_argument(
        "--burn-in",
        type=float,
        default=0,
        help="Years to discard from start of histories (default: 0)"
    )
    args = parser.parse_args()

    # Load data
    tips_df = pd.read_csv(args.tips, sep="\t")
    histories_df = pd.read_csv(args.histories)

    # Apply burn-in filtering
    if args.burn_in > 0:
        min_year = histories_df['year'].min()
        histories_df = histories_df[histories_df['year'] >= min_year + args.burn_in].copy()
        histories_df['year'] = histories_df['year'] - args.burn_in
        tips_df = tips_df[tips_df['year'] >= -args.burn_in].copy()
        tips_df['year'] = tips_df['year'] + args.burn_in

    # Filter to total deme (population-averaged host memory)
    host_memory_df = histories_df[histories_df['deme'] == 'total'].copy()

    # Auto-detect variant columns
    variant_cols = [col for col in tips_df.columns if col.startswith('variant_')]
    if not variant_cols:
        raise ValueError("No variant_* columns found in tips file")

    # Calculate variance over time
    result_df = calc_variance_over_time(tips_df, host_memory_df, variant_cols)

    # Add run_id
    run_id = args.run_id or Path(args.tips).stem
    result_df.insert(0, 'run_id', run_id)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Write output
    result_df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(result_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
