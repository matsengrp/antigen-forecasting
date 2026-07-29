#!/bin/bash
# Reproduce the manuscript figures. Run this where the paper repo
# (../antigen-tex) is checked out — the notebooks and scripts write their
# PDFs/PNGs into ../antigen-tex/figures/, so this is a LOCAL step.
#
# Reproduction graph (verified from each notebook's build/BATCH constant):
#
#   NEW supplement figures (this review) — rendered by default:
#     - figure3_candidates_panel          notebook manuscript-figure-3-variant-
#         assignment-compare-aggregated (BATCH=2026-07-04-reviewer-runs). This is
#         the ONLY figure that consumes the cluster sweep: it reads
#         results/aggregated/<batch>/*.csv, so refresh + pull those first
#         (scripts/reproduce_batch.sh on HPC, then sync results/aggregated/).
#     - figureS5_mutation_*               scripts/make_figureS5.sh (flu-final;
#         local, ~2 min, deterministic). The figS5 notebook is interactive-only
#         and does not save, so reproduction goes through the scripts.
#
#   ORIGINAL manuscript figures — pinned to fixed local builds; add --originals.
#   These do NOT use the reviewer-runs sweep. They render only where each build's
#   data is present locally (some builds are git-ignored / cluster-produced):
#     - manuscript-figure-2-simulation-summary               (flu-final)
#     - manuscript-figure-4-freq-errors                      (flu-simulated-150k-samples-seq)
#     - manuscript-figure-4-growth-rate-inference-plots      (flu-final)
#     - manuscript-figure-5-growth-rate-errors               (flu-final)
#     - manuscript-figure-6-growth-rate-zoom-ins             (flu-simulated-150k-samples-final)
#     - manuscript-figures-S1-and-S2-sim-summary-stats       (flu-final)
#
# Each item is attempted independently; the script reports per-item pass/fail and
# exits non-zero if any failed (a failure is usually a missing local build —
# render where that build's data exists, or sync it first).
#
# Usage:
#   scripts/render_figures.sh [<experiment>] [--originals] [--timeout SECONDS]
#
# Examples:
#   scripts/render_figures.sh 2026-07-04-reviewer-runs           # the two new supplement figures
#   scripts/render_figures.sh 2026-07-04-reviewer-runs --originals

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
NB_DIR="$REPO_ROOT/notebooks"

EXPERIMENT=""
WITH_ORIGINALS=""
TIMEOUT="1800"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --originals) WITH_ORIGINALS=1 ;;
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

# Original-manuscript figure notebooks (fixed local builds; --originals).
ORIGINAL_NOTEBOOKS=(
    manuscript-figure-2-simulation-summary
    manuscript-figure-4-freq-errors
    manuscript-figure-4-growth-rate-inference-plots
    manuscript-figure-5-growth-rate-errors
    manuscript-figure-6-growth-rate-zoom-ins
    manuscript-figures-S1-and-S2-sim-summary-stats
)

FAILED=()

run_notebook() {
    local nb="$1"
    local path="$NB_DIR/$nb.ipynb"
    if [ ! -f "$path" ]; then
        echo "  MISSING  $nb (no such notebook)"; FAILED+=("$nb"); return
    fi
    echo "  RUN      $nb"
    if ( cd "$NB_DIR" && jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout="$TIMEOUT" "$nb.ipynb" >/dev/null 2>&1 ); then
        echo "  OK       $nb"
    else
        echo "  FAIL     $nb (missing local build data? render where it exists)"
        FAILED+=("$nb")
    fi
}

echo "==> New supplement figures"
run_notebook manuscript-figure-3-variant-assignment-compare-aggregated

echo "  RUN      figureS5 (scripts/make_figureS5.sh)"
if "$SCRIPT_DIR/make_figureS5.sh" >/dev/null 2>&1; then
    echo "  OK       figureS5"
else
    echo "  FAIL     figureS5 (see scripts/make_figureS5.sh; needs flu-final + antigen-prime)"
    FAILED+=("figureS5")
fi

if [ -n "$WITH_ORIGINALS" ]; then
    echo "==> Original manuscript figures"
    for nb in "${ORIGINAL_NOTEBOOKS[@]}"; do
        run_notebook "$nb"
    done
else
    echo "(original manuscript figures skipped; pass --originals to render them)"
fi

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "${#FAILED[@]} item(s) failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi
echo "All figures rendered. Outputs under ../antigen-tex/figures/."
