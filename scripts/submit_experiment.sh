#!/bin/bash
# Ready-to-run wrapper: stage (and optionally submit) the whole-experiment
# forecasting SLURM array for one antigen-experiments experiment.
#
# It runs run_all_simulations.py in --all-configs mode, which discovers every
# run across all sweep cells (<experiment>/simulations/*/run_*), skips runs
# with missing inputs and already-complete sims, and writes a throttled array
# job. By default it only STAGES the artifacts and prints the sbatch command;
# pass --submit to launch immediately.
#
# Usage:
#   scripts/submit_experiment.sh <experiment> [max_concurrent] [--submit]
#
# Examples:
#   scripts/submit_experiment.sh 2026-07-04-reviewer-runs
#   scripts/submit_experiment.sh 2026-07-04-reviewer-runs 20
#   scripts/submit_experiment.sh 2026-07-04-reviewer-runs 20 --submit
#
# Cluster/resource settings (partition, cpus, mem, time, project_root, conda
# env) come from configs/slurm_config.yaml. max_concurrent given here overrides
# the config value for this submission only.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment> [max_concurrent] [--submit]" >&2
    exit 1
fi

EXPERIMENT="$1"
shift

# Resolve repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Batch name namespaces results/<batch>/; default to the experiment name.
BATCH_NAME="$EXPERIMENT"
EXPERIMENTS_ROOT="$REPO_ROOT/../antigen-experiments/experiments"
PIPELINE_CONFIG="$REPO_ROOT/configs/pipeline_config.yaml"
SLURM_CONFIG="$REPO_ROOT/configs/slurm_config.yaml"

MAX_CONCURRENT=""
SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT="--submit" ;;
        ''|*[!0-9]*) echo "Unrecognized argument: $arg" >&2; exit 1 ;;
        *) MAX_CONCURRENT="$arg" ;;
    esac
done

CMD=(
    python "$REPO_ROOT/scripts/run_all_simulations.py"
    --experiments-root "$EXPERIMENTS_ROOT"
    --experiment "$EXPERIMENT"
    --all-configs
    --batch-name "$BATCH_NAME"
    --config "$PIPELINE_CONFIG"
    --mode slurm
    --slurm-config "$SLURM_CONFIG"
)
[ -n "$MAX_CONCURRENT" ] && CMD+=(--max-concurrent "$MAX_CONCURRENT")
[ -n "$SUBMIT" ] && CMD+=("$SUBMIT")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
