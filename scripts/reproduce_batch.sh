#!/bin/bash
# End-to-end reproduction driver (compute phase) for one antigen-experiments
# experiment. This is a thin orchestration layer over the existing scripts; it
# adds nothing to their behavior except chaining them with a SLURM job
# dependency so aggregation runs only after the per-run pipeline array finishes.
#
# It chains three stages:
#   1. find_candidate_runs.py    -> candidate_runs.csv          (local, blocking)
#   2. run_all_simulations.py    -> per-run pipeline SLURM array (variant
#      assignment + FGA/GARW forecasting + scoring; skips already-complete runs)
#   3. submit_aggregation.sh     -> one aggregation job that depends (afterok)
#      on the array, writing results/aggregated/<experiment>/*.csv
#
# By default it STAGES the two SLURM scripts and prints the exact sbatch chain
# for you to inspect/run. Pass --submit to launch the dependency chain itself.
#
# Figure rendering is a SEPARATE, local step (scripts/render_figures.sh), run
# where the ../antigen-tex paper repo is checked out — not part of this driver.
#
# Stage 1 re-aggregates sim_stats from the live simulations/ tree by default
# (--refresh) so candidate screening never misses a newly finished sim; pass
# --no-refresh to reuse the cached sim_stats.csv when nothing new has completed.
#
# Usage:
#   scripts/reproduce_batch.sh <experiment> [max_concurrent] [--host-immunity] [--no-refresh] [--submit]
#
# Examples:
#   scripts/reproduce_batch.sh 2026-07-04-reviewer-runs
#   scripts/reproduce_batch.sh 2026-07-04-reviewer-runs 20 --host-immunity
#   scripts/reproduce_batch.sh 2026-07-04-reviewer-runs 20 --host-immunity --submit
#
# Cluster/resource settings come from configs/slurm_config.yaml (per-run array)
# and the AGG_* environment overrides honored by submit_aggregation.sh.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment> [max_concurrent] [--host-immunity] [--submit]" >&2
    exit 1
fi

EXPERIMENT="$1"
shift

# Resolve repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Batch name namespaces results/<batch>/; default to the experiment name, matching
# submit_experiment.sh and submit_aggregation.sh.
BATCH_NAME="$EXPERIMENT"
EXPERIMENTS_ROOT="$REPO_ROOT/../antigen-experiments/experiments"
PIPELINE_CONFIG="$REPO_ROOT/configs/pipeline_config.yaml"
SLURM_CONFIG="$REPO_ROOT/configs/slurm_config.yaml"

MAX_CONCURRENT=""
HOST_IMMUNITY=""
SUBMIT=""
# Candidate screening reuses a cached sim_stats.csv unless refreshed, so by
# default we re-aggregate it from the live simulations/ tree; otherwise newly
# finished sims are silently missed from candidate_runs.csv. Pass --no-refresh
# to reuse the cache when you know no new sims have completed.
REFRESH="--refresh"
for arg in "$@"; do
    case "$arg" in
        --host-immunity) HOST_IMMUNITY="--host-immunity" ;;
        --submit) SUBMIT=1 ;;
        --no-refresh) REFRESH="" ;;
        ''|*[!0-9]*) echo "Unrecognized argument: $arg" >&2; exit 1 ;;
        *) MAX_CONCURRENT="$arg" ;;
    esac
done

# Run from the repo root so relative paths inside the subscripts resolve.
cd "$REPO_ROOT"

echo "==> Stage 1/3: find candidate runs${REFRESH:+ (refreshing sim_stats from the live tree)}"
python scripts/find_candidate_runs.py "$EXPERIMENT" -j 8 \
    --experiments-root "$EXPERIMENTS_ROOT" ${REFRESH:+$REFRESH}

echo
echo "==> Stage 2/3: stage per-run pipeline array (skips already-complete runs)"
STAGE_CMD=(
    python scripts/run_all_simulations.py
    --experiments-root "$EXPERIMENTS_ROOT"
    --experiment "$EXPERIMENT"
    --all-configs
    --batch-name "$BATCH_NAME"
    --config "$PIPELINE_CONFIG"
    --mode slurm
    --slurm-config "$SLURM_CONFIG"
)
[ -n "$MAX_CONCURRENT" ] && STAGE_CMD+=(--max-concurrent "$MAX_CONCURRENT")

# Capture stdout (the "To submit: sbatch <path>" line) while letting the
# script's logging stream live to the terminal via stderr.
echo "Running: ${STAGE_CMD[*]}"
ARRAY_OUT="$("${STAGE_CMD[@]}")"
printf '%s\n' "$ARRAY_OUT"

# run_all_simulations.py prints "To submit: sbatch <path>" only when there are
# pending (incomplete) runs; when everything is already complete it prints no
# such line. Parse the staged array-script path from that line.
ARRAY_SCRIPT="$(printf '%s\n' "$ARRAY_OUT" | sed -n 's/^To submit: sbatch //p' | tail -n 1)"

echo
echo "==> Stage 3/3: stage aggregation job"
AGG_OUT="$(scripts/submit_aggregation.sh "$EXPERIMENT" ${HOST_IMMUNITY:+$HOST_IMMUNITY})"
printf '%s\n' "$AGG_OUT"
# submit_aggregation.sh prints a "  script:  <path>" line for the staged sbatch.
AGG_SCRIPT="$(printf '%s\n' "$AGG_OUT" | sed -n 's/^[[:space:]]*script:[[:space:]]*//p' | tail -n 1)"

if [ -z "$AGG_SCRIPT" ] || [ ! -f "$AGG_SCRIPT" ]; then
    echo "ERROR: could not locate staged aggregation sbatch script." >&2
    exit 1
fi

echo
echo "======================================================================"
if [ -z "$ARRAY_SCRIPT" ]; then
    echo "All per-run outputs already complete; no pipeline array to submit."
    echo "Aggregation will run standalone (it re-reads whatever is on disk)."
    if [ -n "$SUBMIT" ]; then
        echo "Submitting aggregation (no dependency)..."
        AGG_JID="$(sbatch --parsable "$AGG_SCRIPT")"
        echo "Aggregation job: $AGG_JID"
    else
        echo "To submit aggregation:"
        echo "  sbatch $AGG_SCRIPT"
        echo "Re-run with --submit to launch automatically."
    fi
else
    if [ -n "$SUBMIT" ]; then
        echo "Submitting dependency chain (aggregation waits for the array)..."
        ARRAY_JID="$(sbatch --parsable "$ARRAY_SCRIPT")"
        echo "Pipeline array job: $ARRAY_JID"
        AGG_JID="$(sbatch --parsable --dependency=afterok:"$ARRAY_JID" "$AGG_SCRIPT")"
        echo "Aggregation job:    $AGG_JID (afterok:$ARRAY_JID)"
    else
        echo "Not submitted. To launch the dependency chain:"
        echo "  ARRAY_JID=\$(sbatch --parsable $ARRAY_SCRIPT)"
        echo "  sbatch --parsable --dependency=afterok:\$ARRAY_JID $AGG_SCRIPT"
        echo "Or re-run this command with --submit."
    fi
fi
echo "======================================================================"
echo
echo "After aggregation completes, render figures locally (where ../antigen-tex"
echo "is checked out): scripts/render_figures.sh $EXPERIMENT"
