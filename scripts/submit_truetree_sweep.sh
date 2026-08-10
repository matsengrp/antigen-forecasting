#!/bin/bash
# Stage (and optionally submit) a single SLURM job that measures true-genealogy
# homoplasy and variant confusability across every run in a batch, writing the
# small committable tables under results/aggregated/<batch>/.
#
# This is the true-genealogy counterpart to submit_homoplasy_sweep.sh. It reads
# each run's run-out.branches (the exact simulation genealogy) rather than the
# reconstructed Auspice tree, so origins are counted without inference error.
#
# It is one -j job, not a SLURM array: each run is ~25 s of streaming and stays
# near half a gigabyte of memory, so the array's per-task isolation is not worth
# its two-step array-then-aggregate complexity. (It also avoids the pipeline
# array template's `source activate`, which fails on non-interactive shells; the
# conda activation below matches submit_homoplasy_sweep.sh.)
#
# Usage:
#   scripts/submit_truetree_sweep.sh <experiment> [--submit]
#
# Examples:
#   scripts/submit_truetree_sweep.sh 2026-07-04-reviewer-runs
#   scripts/submit_truetree_sweep.sh 2026-07-04-reviewer-runs --submit
#
# Cluster settings (partition, conda env, project_root, log dir) come from
# configs/slurm_config.yaml. Resources are sweep-specific and NOT taken from that
# file: the per-simulation values there (3 days, 16 cpus) are sized for the
# pipeline. The branches files are multi-GB and streamed, so this wants more time
# and memory margin than the inferred-tree sweep. Override via environment:
#   SWEEP_TIME (default 03:00:00)  SWEEP_MEM_GB (48)  SWEEP_CPUS (8)  SWEEP_JOBS (6)

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment> [--submit]" >&2
    exit 1
fi

EXPERIMENT="$1"
shift

# Resolve repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

BATCH_NAME="$EXPERIMENT"
SLURM_CONFIG="$REPO_ROOT/configs/slurm_config.yaml"

SUBMIT=""
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT=1 ;;
        *) echo "Unrecognized argument: $arg" >&2; exit 1 ;;
    esac
done

# Sweep-specific resources; see the header note on why these are not read from
# slurm_config.yaml.
SWEEP_TIME="${SWEEP_TIME:-03:00:00}"
SWEEP_MEM_GB="${SWEEP_MEM_GB:-48}"
SWEEP_CPUS="${SWEEP_CPUS:-8}"
SWEEP_JOBS="${SWEEP_JOBS:-6}"

if [ ! -f "$SLURM_CONFIG" ]; then
    echo "Missing SLURM config: $SLURM_CONFIG" >&2
    exit 1
fi

# Pull only the cluster-identity keys from the shared config. Fail loudly on a
# missing key rather than silently substituting a default that points at someone
# else's filesystem.
read -r PARTITION CONDA_ENV PROJECT_ROOT LOG_DIR PYTHON_BIN CONDA_BASE <<EOF
$(python - "$SLURM_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)["slurm"]
required = ("partition", "conda_env", "project_root", "log_dir", "python_bin", "conda_base")
missing = [k for k in required if k not in cfg]
if missing:
    sys.exit(f"slurm_config.yaml missing required key(s): {missing}")
print(" ".join(str(cfg[k]) for k in required))
PY
)
EOF

# log_dir carries a trailing slash in the config; the experiments tree is a
# sibling of the repo rather than a path below it.
LOG_DIR="${LOG_DIR%/}"
EXPERIMENTS_ROOT="$(dirname "$PROJECT_ROOT")/antigen-experiments/experiments"

# The epitope-site list lives in the sibling antigen-prime checkout, and the
# GenBank reference is shared across runs rather than written per run.
EPITOPE_SITES="$(dirname "$PROJECT_ROOT")/antigen-prime/src/main/resources/epitopeSites.txt"
REF_GENBANK="data/flu-final/auspice/ref_HA.gb"

# Fail at STAGING time rather than minutes into the job. Both inputs live outside
# the batch, so a missing one is not caught by the per-run globbing below.
for shared_input in "$EPITOPE_SITES" "$PROJECT_ROOT/$REF_GENBANK"; do
    if [ ! -f "$shared_input" ]; then
        echo "ERROR: missing shared input: $shared_input" >&2
        echo "  (checked from staging; the job resolves it under $PROJECT_ROOT)" >&2
        exit 1
    fi
done

# Warn early if no run has a branches file yet: the sweep would produce empty
# tables. The branches files are raw antigen outputs on the experiments side,
# four levels below simulations/ (<config>/run_N/output/run-out.branches), so the
# depth must reach 4 -- an earlier -maxdepth 3 found nothing and warned falsely.
BRANCH_COUNT=$(find "$EXPERIMENTS_ROOT/$BATCH_NAME/simulations" -maxdepth 4 \
    -path "*/run_*/output/run-out.branches" 2>/dev/null | wc -l | tr -d ' ')
if [ "$BRANCH_COUNT" -eq 0 ]; then
    echo "WARNING: no run-out.branches found under $EXPERIMENTS_ROOT/$BATCH_NAME" >&2
    echo "  The sweep will produce empty tables unless the simulations are present." >&2
else
    echo "Found $BRANCH_COUNT run(s) with a true genealogy."
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMIT_DIR="$REPO_ROOT/results/$BATCH_NAME/truetree_submission_$STAMP"
mkdir -p "$SUBMIT_DIR"
SBATCH_SCRIPT="$SUBMIT_DIR/submit_truetree_sweep.sbatch"

cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=truetree_${BATCH_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${SWEEP_TIME}
#SBATCH --cpus-per-task=${SWEEP_CPUS}
#SBATCH --mem=${SWEEP_MEM_GB}G
#SBATCH --output=${PROJECT_ROOT}/${LOG_DIR}/truetree_${BATCH_NAME}_%j.out
#SBATCH --error=${PROJECT_ROOT}/${LOG_DIR}/truetree_${BATCH_NAME}_%j.err

set -euo pipefail

# sbatch runs non-interactively, so neither mamba's shell function nor conda's
# base bin/ is on PATH. Source conda's profile explicitly from the configured
# base, then conda activate. NOTE: no backticks in this heredoc -- the delimiter
# is unquoted (variables must expand) so backticks would run as commands at
# staging. set -u must be off while activating: conda-forge activate.d hooks
# reference unset variables and would abort under set -e.
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}
set -u

cd "${PROJECT_ROOT}"

# Sweep parallelism is process-level (-j); keep each worker's BLAS
# single-threaded so N workers do not each spawn N threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

${PYTHON_BIN} scripts/sweep_truetree_homoplasy.py \\
    --batch "${BATCH_NAME}" \\
    --data-root data/ \\
    --experiments-root "${EXPERIMENTS_ROOT}" \\
    --ref-genbank "${REF_GENBANK}" \\
    --epitope-sites "${EPITOPE_SITES}" \\
    -j ${SWEEP_JOBS}
EOF
chmod +x "$SBATCH_SCRIPT"

mkdir -p "$REPO_ROOT/$LOG_DIR"

echo "Staged true-genealogy sweep job:"
echo "  script:      $SBATCH_SCRIPT"
echo "  batch:       $BATCH_NAME"
echo "  branches:    $EXPERIMENTS_ROOT/$BATCH_NAME/simulations/<config>/run_N/output/run-out.branches"
echo "  epitopes:    $EPITOPE_SITES"
echo "  resources:   ${SWEEP_CPUS} cpus, ${SWEEP_MEM_GB}G, ${SWEEP_TIME}, -j ${SWEEP_JOBS}"
echo "  outputs:     results/aggregated/$BATCH_NAME/truetree_recurrence_by_run.csv"
echo "               results/aggregated/$BATCH_NAME/truetree_origin_counts_by_run.csv"
echo "               results/aggregated/$BATCH_NAME/truetree_confusability_by_run.csv"

if [ -n "$SUBMIT" ]; then
    echo "Submitting..."
    sbatch "$SBATCH_SCRIPT"
else
    echo
    echo "Not submitted. To launch:"
    echo "  sbatch $SBATCH_SCRIPT"
    echo "Or re-run this command with --submit."
fi
