#!/bin/bash
# Render the manuscript figure notebooks by executing them in place. Run this
# where the paper repo (../antigen-tex) is checked out — most notebooks write
# their PDFs/PNGs into ../antigen-tex/figures/, so this is normally a LOCAL step
# after the aggregated CSVs have been produced (scripts/reproduce_batch.sh) and
# pulled/committed.
#
# Notebooks fall into two groups by what they read:
#
#   cross-run (default): reproducible from committed cross-run outputs alone
#     - manuscript-figure-3-variant-assignment-compare-aggregated
#         reads results/aggregated/<batch>/*.csv + candidate_runs.csv
#     - manuscript-figures-S1-and-S2-sim-summary-stats
#         reads sim_stats.csv (produced by find_candidate_runs.py)
#
#   per-run (--all): single-example-run figures that need per-run inputs under
#   data/<build>/ and results/<build>/ (tips_with_variants.tsv, seq/case_counts,
#   estimates/*/rt_*.tsv, *_growth_rates.tsv, ...). These are git-ignored and
#   produced on the cluster, so they render only where those per-run trees exist
#   locally (a full local run, or synced from HPC):
#     - manuscript-figure-2-simulation-summary
#     - manuscript-figure-4-freq-errors
#     - manuscript-figure-4-growth-rate-inference-plots
#     - manuscript-figure-5-growth-rate-errors
#     - manuscript-figure-6-growth-rate-zoom-ins
#     - manuscript-figureS5-mutation-homoplasy
#
# Each notebook carries its own BATCH constant, so <experiment> here is only for
# display; it does not reparametrize the notebooks.
#
# The script attempts every notebook in the selected group, reports per-notebook
# pass/fail, and exits non-zero if any failed (a failure is usually a missing
# per-run input — sync it from the cluster and re-run).
#
# Usage:
#   scripts/render_figures.sh [<experiment>] [--all] [--timeout SECONDS]
#
# Examples:
#   scripts/render_figures.sh 2026-07-04-reviewer-runs
#   scripts/render_figures.sh 2026-07-04-reviewer-runs --all
#   scripts/render_figures.sh --all --timeout 3600

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
NB_DIR="$REPO_ROOT/notebooks"

EXPERIMENT=""
RENDER_ALL=""
TIMEOUT="1800"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --all) RENDER_ALL=1 ;;
        --timeout) shift; TIMEOUT="${1:?--timeout needs a value}" ;;
        --*) echo "Unrecognized argument: $1" >&2; exit 1 ;;
        *) EXPERIMENT="$1" ;;
    esac
    shift
done

if ! command -v jupyter >/dev/null 2>&1; then
    echo "ERROR: 'jupyter' not found on PATH. Activate the antigen env first." >&2
    exit 1
fi

CROSS_RUN=(
    manuscript-figure-3-variant-assignment-compare-aggregated
    manuscript-figures-S1-and-S2-sim-summary-stats
)
PER_RUN=(
    manuscript-figure-2-simulation-summary
    manuscript-figure-4-freq-errors
    manuscript-figure-4-growth-rate-inference-plots
    manuscript-figure-5-growth-rate-errors
    manuscript-figure-6-growth-rate-zoom-ins
    manuscript-figureS5-mutation-homoplasy
)

NOTEBOOKS=("${CROSS_RUN[@]}")
if [ -n "$RENDER_ALL" ]; then
    NOTEBOOKS+=("${PER_RUN[@]}")
fi

echo "Rendering ${#NOTEBOOKS[@]} notebook(s)${EXPERIMENT:+ for $EXPERIMENT} from $NB_DIR"
[ -z "$RENDER_ALL" ] && echo "(cross-run group only; pass --all to also render the per-run figures)"
echo

cd "$NB_DIR"

FAILED=()
for nb in "${NOTEBOOKS[@]}"; do
    path="$nb.ipynb"
    if [ ! -f "$path" ]; then
        echo "  MISSING  $nb (no such notebook; skipping)"
        FAILED+=("$nb")
        continue
    fi
    echo "  RUN      $nb"
    if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout="$TIMEOUT" "$path" >/dev/null 2>&1; then
        echo "  OK       $nb"
    else
        echo "  FAIL     $nb (likely a missing per-run input — sync from HPC and retry)"
        FAILED+=("$nb")
    fi
done

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "${#FAILED[@]} notebook(s) failed:"
    for nb in "${FAILED[@]}"; do echo "  - $nb"; done
    exit 1
fi
echo "All ${#NOTEBOOKS[@]} notebook(s) rendered. Figures written under ../antigen-tex/figures/."
