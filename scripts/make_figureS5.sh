#!/bin/bash
# Reproduce the figure S5 mutation-homoplasy panels (supplement) from the
# committed flu-final build. This is a LOCAL step — it does not depend on the
# reviewer-runs cluster sweep. It runs the two homoplasy scripts the interactive
# notebook (manuscript-figureS5-mutation-homoplasy.ipynb) wraps, but which the
# notebook itself does not save from:
#
#   1. mutation_background_distances.py -> mutations / null / pairs / antigenic-null CSVs
#   2. plot_mutation_homoplasy.py       -> three figS5 PDFs+PNGs
#         figureS5_mutation_homoplasy_distance.{pdf,png}
#         figureS5_mutation_occurrence_counts.{pdf,png}
#         figureS5_mutation_antigenic_distance.{pdf,png}
#
# All numeric parameters are the script defaults, which match the notebook
# (anchor-tip 45b125db, min-origin-progeny 1, max-carriers 1500, null-samples
# 20000, seed 0) — so output is deterministic.
#
# Usage:
#   scripts/make_figureS5.sh [--build BUILD] [--output-dir DIR] [--work-dir DIR]
#
# Defaults: BUILD=flu-final, output-dir=../antigen-tex/figures (the paper repo).
# Examples:
#   scripts/make_figureS5.sh                             # writes into ../antigen-tex/figures
#   scripts/make_figureS5.sh --output-dir /tmp/figS5     # dry validation into a scratch dir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD="flu-final"
OUTPUT_DIR="$REPO_ROOT/../antigen-tex/figures"
WORK_DIR=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --build) shift; BUILD="${1:?--build needs a value}" ;;
        --output-dir) shift; OUTPUT_DIR="${1:?--output-dir needs a value}" ;;
        --work-dir) shift; WORK_DIR="${1:?--work-dir needs a value}" ;;
        *) echo "Unrecognized argument: $1" >&2; exit 1 ;;
    esac
    shift
done

cd "$REPO_ROOT"

DATA="data/$BUILD"
AUSPICE="$DATA/variant-assignment/phylogenetic/auspice.json"
FASTA="$DATA/antigen-outputs/unique_sequences.fasta"
REF_GB="$DATA/auspice/ref_HA.gb"
TIPS="$DATA/antigen-outputs/unique_tips.csv"
EPITOPES="../antigen-prime/src/main/resources/epitopeSites.txt"

for f in "$AUSPICE" "$FASTA" "$REF_GB" "$TIPS" "$EPITOPES"; do
    [ -f "$f" ] || { echo "ERROR: missing input: $f" >&2; exit 1; }
done

# Intermediate CSVs go to a work dir (temporary unless one is provided).
CLEANUP=""
if [ -z "$WORK_DIR" ]; then
    WORK_DIR="$(mktemp -d)"
    CLEANUP="$WORK_DIR"
fi
mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

echo "==> Step 1/2: compute homoplasy tables (build=$BUILD)"
python scripts/mutation_background_distances.py \
    --auspice-json "$AUSPICE" \
    --sequences-fasta "$FASTA" \
    --ref-genbank "$REF_GB" \
    --epitope-sites "$EPITOPES" \
    --tips-csv "$TIPS" \
    --output "$WORK_DIR/mutations.csv" \
    --null-output "$WORK_DIR/null.csv" \
    --pairs-output "$WORK_DIR/pairs.csv" \
    --antigenic-null-output "$WORK_DIR/antigenic_null.csv"

echo "==> Step 2/2: render figS5 panels -> $OUTPUT_DIR"
python scripts/plot_mutation_homoplasy.py \
    --mutations-csv "$WORK_DIR/mutations.csv" \
    --null-csv "$WORK_DIR/null.csv" \
    --pairs-csv "$WORK_DIR/pairs.csv" \
    --antigenic-null-csv "$WORK_DIR/antigenic_null.csv" \
    --output-dir "$OUTPUT_DIR"

[ -n "$CLEANUP" ] && rm -rf "$CLEANUP"
echo "Done. figS5 panels written under $OUTPUT_DIR"
