#!/bin/bash
# Stage (and optionally submit) a single SLURM job that aggregates one batch's
# per-run results into the committable tables under results/aggregated/<batch>/.
#
# This is the aggregation counterpart to submit_experiment.sh, which stages the
# per-simulation pipeline array. Aggregation is one short job, not an array.
#
# Usage:
#   scripts/submit_aggregation.sh <experiment> [--host-immunity] [--submit]
#
# Examples:
#   scripts/submit_aggregation.sh 2026-07-04-reviewer-runs
#   scripts/submit_aggregation.sh 2026-07-04-reviewer-runs --host-immunity
#   scripts/submit_aggregation.sh 2026-07-04-reviewer-runs --host-immunity --submit
#
# --host-immunity additionally computes fitness variance from each run's
# out.histories.raw.csv (~100 MB per run, ~0.7 GB resident per worker) and emits
# host_immunity_variance_over_time.csv. Without it, only the centroid-based
# fitness_variance_over_time.csv is produced.
#
# Cluster settings (partition, conda env, project_root, log dir) come from
# configs/slurm_config.yaml. Resources are aggregation-specific and NOT taken
# from that file: the per-simulation values there (3 days, 16 cpus) are sized for
# the pipeline, whereas aggregation is minutes of work dominated by reading run
# outputs off shared storage. Override via environment variables:
#   AGG_TIME (default 02:00:00)  AGG_MEM_GB (32)  AGG_CPUS (8)  AGG_JOBS (8)

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment> [--host-immunity] [--submit]" >&2
    exit 1
fi

EXPERIMENT="$1"
shift

# Resolve repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

BATCH_NAME="$EXPERIMENT"
SLURM_CONFIG="$REPO_ROOT/configs/slurm_config.yaml"

HOST_IMMUNITY=""
SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --host-immunity) HOST_IMMUNITY="--host-immunity" ;;
        --submit) SUBMIT=1 ;;
        *) echo "Unrecognized argument: $arg" >&2; exit 1 ;;
    esac
done

# Aggregation-specific resources; see the header note on why these are not read
# from slurm_config.yaml.
AGG_TIME="${AGG_TIME:-02:00:00}"
AGG_MEM_GB="${AGG_MEM_GB:-32}"
AGG_CPUS="${AGG_CPUS:-8}"
AGG_JOBS="${AGG_JOBS:-8}"

if [ ! -f "$SLURM_CONFIG" ]; then
    echo "Missing SLURM config: $SLURM_CONFIG" >&2
    exit 1
fi

# Pull only the cluster-identity keys from the shared config. Fail loudly on a
# missing key rather than silently substituting a default that points at someone
# else's filesystem.
read -r PARTITION CONDA_ENV PROJECT_ROOT LOG_DIR PYTHON_BIN <<EOF
$(python - "$SLURM_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)["slurm"]
required = ("partition", "conda_env", "project_root", "log_dir", "python_bin")
missing = [k for k in required if k not in cfg]
if missing:
    sys.exit(f"slurm_config.yaml missing required key(s): {missing}")
print(" ".join(str(cfg[k]) for k in required))
PY
)
EOF

# Normalize: log_dir carries a trailing slash in the config, and the experiments
# tree is a sibling of the repo rather than a path below it.
LOG_DIR="${LOG_DIR%/}"
EXPERIMENTS_ROOT="$(dirname "$PROJECT_ROOT")/antigen-experiments/experiments"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMIT_DIR="$REPO_ROOT/results/$BATCH_NAME/aggregation_submission_$STAMP"
mkdir -p "$SUBMIT_DIR"
SBATCH_SCRIPT="$SUBMIT_DIR/submit_aggregation.sbatch"

cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=agg_${BATCH_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${AGG_TIME}
#SBATCH --cpus-per-task=${AGG_CPUS}
#SBATCH --mem=${AGG_MEM_GB}G
#SBATCH --output=${PROJECT_ROOT}/${LOG_DIR}/agg_${BATCH_NAME}_%j.out
#SBATCH --error=${PROJECT_ROOT}/${LOG_DIR}/agg_${BATCH_NAME}_%j.err

set -euo pipefail

source activate ${CONDA_ENV}
cd "${PROJECT_ROOT}"

# Aggregation parallelism is process-level (-j); keep each worker's BLAS
# single-threaded so N workers do not each spawn N threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

${PYTHON_BIN} scripts/aggregate_results.py \\
    --batch "${BATCH_NAME}" \\
    --experiments-root "${EXPERIMENTS_ROOT}" \\
    --experiment "${EXPERIMENT}" \\
    --histories-name out.histories.csv \\
    -j ${AGG_JOBS} ${HOST_IMMUNITY}
EOF
chmod +x "$SBATCH_SCRIPT"

mkdir -p "$REPO_ROOT/$LOG_DIR"

echo "Staged aggregation job:"
echo "  script:        $SBATCH_SCRIPT"
echo "  batch:         $BATCH_NAME"
echo "  host immunity: ${HOST_IMMUNITY:-disabled}"
echo "  resources:     ${AGG_CPUS} cpus, ${AGG_MEM_GB}G, ${AGG_TIME}, -j ${AGG_JOBS}"

if [ -n "$SUBMIT" ]; then
    echo "Submitting..."
    sbatch "$SBATCH_SCRIPT"
else
    echo
    echo "Not submitted. To launch:"
    echo "  sbatch $SBATCH_SCRIPT"
    echo "Or re-run this command with --submit."
fi
