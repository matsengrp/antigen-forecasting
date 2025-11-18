#!/bin/bash
# assign-tree-based-variants.sh
#
# Create phylogenetic tree from sequences and assign variant clades based on
# antigenic epitope sites using augur pipeline and add_new_clades.py
#
# Usage:
#   scripts/assign-tree-based-variants.sh --sequences <path> --output-dir <path> [--metadata <path>]
#
# Required arguments:
#   --sequences    Path to input sequences (FASTA or alignment file)
#   --output-dir   Directory to store all intermediate and final outputs
#
# Optional arguments:
#   --metadata     Path to metadata TSV (default: data/flu-simulated-150k-samples-final/auspice/sequences_metadata.tsv)

set -euo pipefail

# Default paths
DEFAULT_METADATA="data/flu-simulated-150k-samples-final/auspice/sequences_metadata.tsv"
REF_GENBANK="data/flu-simulated-150k-samples-final/auspice/ref_HA.gb"
WEIGHTS_CONFIG="configs/weights_per_site_for_clades.json"
LINEAGE="h3n2"
SEGMENT="ha"

# Initialize variables
SEQUENCES=""
OUTPUT_DIR=""
METADATA="$DEFAULT_METADATA"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sequences)
            SEQUENCES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --metadata)
            METADATA="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --sequences <path> --output-dir <path> [--metadata <path>]"
            echo ""
            echo "Required arguments:"
            echo "  --sequences    Path to input sequences (FASTA or alignment)"
            echo "  --output-dir   Directory for outputs"
            echo ""
            echo "Optional arguments:"
            echo "  --metadata     Metadata TSV (default: $DEFAULT_METADATA)"
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SEQUENCES" ]]; then
    echo "Error: --sequences is required"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output-dir is required"
    exit 1
fi

# Validate input files exist
if [[ ! -f "$SEQUENCES" ]]; then
    echo "Error: Sequences file not found: $SEQUENCES"
    exit 1
fi

if [[ ! -f "$METADATA" ]]; then
    echo "Error: Metadata file not found: $METADATA"
    exit 1
fi

if [[ ! -f "$REF_GENBANK" ]]; then
    echo "Error: Reference GenBank file not found: $REF_GENBANK"
    exit 1
fi

if [[ ! -f "$WEIGHTS_CONFIG" ]]; then
    echo "Error: Weights config file not found: $WEIGHTS_CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Detect available CPU cores (cross-platform)
if command -v nproc &> /dev/null; then
    CPUS=$(nproc)
elif command -v sysctl &> /dev/null; then
    CPUS=$(sysctl -n hw.ncpu)
else
    CPUS=1
fi

echo "=========================================="
echo "Phylogenetic Variant Assignment Pipeline"
echo "=========================================="
echo "Sequences:   $SEQUENCES"
echo "Output dir:  $OUTPUT_DIR"
echo "Metadata:    $METADATA"
echo "CPUs:        $CPUS"
echo "=========================================="

# Detect if input is FASTA (needs alignment) or already aligned
ALIGNMENT=""
if [[ "$SEQUENCES" == *.fasta || "$SEQUENCES" == *.fa ]]; then
    echo "[1/9] Running alignment with augur align..."
    ALIGNMENT="$OUTPUT_DIR/sequences.alignment"
    augur align \
        --sequences "$SEQUENCES" \
        --reference-sequence "$REF_GENBANK" \
        --output "$ALIGNMENT" \
        --nthreads "$CPUS" \
        --fill-gaps
    echo "✓ Alignment complete"
else
    echo "[1/9] Using provided alignment..."
    ALIGNMENT="$SEQUENCES"
    echo "✓ Alignment ready"
fi

# Step 2: Build tree
echo "[2/9] Building phylogenetic tree..."
augur tree \
    --alignment "$ALIGNMENT" \
    --output "$OUTPUT_DIR/tree_raw.nwk" \
    --method iqtree \
    --nthreads "$CPUS"
echo "✓ Tree built"

# Step 3: Refine tree
echo "[3/9] Refining tree..."
augur refine \
    --tree "$OUTPUT_DIR/tree_raw.nwk" \
    --alignment "$ALIGNMENT" \
    --output-tree "$OUTPUT_DIR/tree.nwk" \
    --output-node-data "$OUTPUT_DIR/branch_lengths.json"
echo "✓ Tree refined"

# Step 4: Reconstruct ancestral sequences
echo "[4/9] Reconstructing ancestral sequences..."
augur ancestral \
    --tree "$OUTPUT_DIR/tree.nwk" \
    --alignment "$ALIGNMENT" \
    --output-node-data "$OUTPUT_DIR/nt_muts.json" \
    --inference joint
echo "✓ Ancestral sequences reconstructed"

# Step 5: Translate to amino acids
echo "[5/9] Translating to amino acids..."
augur translate \
    --tree "$OUTPUT_DIR/tree.nwk" \
    --ancestral-sequences "$OUTPUT_DIR/nt_muts.json" \
    --reference-sequence "$REF_GENBANK" \
    --output-node-data "$OUTPUT_DIR/aa_muts.json"
echo "✓ Translation complete"

# Step 6: Export to Auspice JSON
echo "[6/9] Exporting to Auspice JSON..."
augur export v2 \
    --tree "$OUTPUT_DIR/tree.nwk" \
    --metadata "$METADATA" \
    --node-data "$OUTPUT_DIR/branch_lengths.json" "$OUTPUT_DIR/nt_muts.json" "$OUTPUT_DIR/aa_muts.json" \
    --output "$OUTPUT_DIR/auspice_base.json"
echo "✓ Base Auspice JSON created"

# Step 7: Assign phylogenetic clades
echo "[7/9] Assigning phylogenetic clades..."
python scripts/add_new_clades.py \
    --input "$OUTPUT_DIR/auspice_base.json" \
    --lineage "$LINEAGE" \
    --segment "$SEGMENT" \
    --weights "$WEIGHTS_CONFIG" \
    --new-key clade \
    --output "$OUTPUT_DIR/auspice.json"
echo "✓ Clades assigned"

# Summary
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Final output: $OUTPUT_DIR/auspice.json"
echo ""
echo "Intermediate files:"
echo "  - Tree: $OUTPUT_DIR/tree.nwk"
echo "  - Mutations: $OUTPUT_DIR/nt_muts.json, $OUTPUT_DIR/aa_muts.json"
echo "  - Base JSON: $OUTPUT_DIR/auspice_base.json"
echo "=========================================="
