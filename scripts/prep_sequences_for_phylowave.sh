#!/bin/bash
#SBATCH --time=1-0
set -eu

# Script to prepare phylogenetic tree for PhyloWave analysis
# Based on methodology from Lefrancq et al. 2023 (bioRxiv)
#
# Usage: ./prep_sequences_for_phylowave.sh [BUILD_NAME] [CLOCK_RATE]
#
# Arguments:
#   BUILD_NAME: Name of the build directory (default: flu-simulated-150k-samples-final)
#   CLOCK_RATE: Molecular clock rate in subs/site/year (default: 0.0045 for H3N2 HA)
#
# Examples:
#   ./prep_sequences_for_phylowave.sh flu-simulated-150k-samples-final 0.0045
#   ./prep_sequences_for_phylowave.sh my-h3n2-data

# Get BUILD from command line argument or use default
BUILD="${1:-flu-simulated-150k-samples-final}"

# Get molecular clock rate from command line or use H3N2 HA default
# H3N2 HA: ~0.0045 subs/site/year
# H1N1 HA: ~0.0048 subs/site/year
# SARS-CoV-2: ~0.0008 subs/site/year
CLOCK_RATE="${2:-0.0045}"

# Fast mode option (skip bootstrap for faster testing)
FAST_MODE="${3:-true}"

# Dynamically detect number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    N_THREADS=$(sysctl -n hw.ncpu)
else
    # Linux
    N_THREADS=$(nproc)
fi

echo "=== PhyloWave Tree Preparation ==="
echo "Build: $BUILD"
echo "Molecular clock rate: $CLOCK_RATE subs/site/year"
echo "Fast mode: $FAST_MODE (skip bootstrap for faster tree building)"
echo "Detected $N_THREADS CPU cores"
echo ""

# Get the script's directory to determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Base data directory
PHYLOWAVE_DIR="$PROJECT_ROOT/data/$BUILD/phylowave"

# Input file paths (flexible: can use alignment or raw sequences)
REF_PATH="$PROJECT_ROOT/data/$BUILD/ref_HA.fasta"
RAW_SEQS_PATH="$PROJECT_ROOT/data/$BUILD/auspice/sequences.fasta"
METADATA_PATH="$PROJECT_ROOT/data/$BUILD/auspice/sequences_metadata.tsv"
ALIGNMENT_PATH="$PROJECT_ROOT/data/$BUILD/phylowave/sequences.aligned.fasta"

# Output file paths
PHYLOWAVE_ALIGNMENT="$PHYLOWAVE_DIR/sequences.aligned.fasta"
IQTREE_PREFIX="$PHYLOWAVE_DIR/iqtree"
RAW_TREE_PATH="$PHYLOWAVE_DIR/raw_tree.nwk"
REFINED_TREE_PATH="$PHYLOWAVE_DIR/time_tree.nwk"
NODE_DATA_PATH="$PHYLOWAVE_DIR/node_data.json"

# Validate required input files exist
echo "Validating input files..."

# Check for metadata
if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file not found at $METADATA_PATH"
    echo "Metadata must contain a 'date' column with sampling dates"
    exit 1
fi

# Check for sequences (either aligned or raw)
if [ ! -f "$ALIGNMENT_PATH" ] && [ ! -f "$RAW_SEQS_PATH" ]; then
    echo "Error: No sequence files found!"
    echo "  Looked for alignment: $ALIGNMENT_PATH"
    echo "  Looked for raw sequences: $RAW_SEQS_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$PHYLOWAVE_DIR"
export AUGUR_RECURSION_LIMIT=25000;

echo "Output directory: $PHYLOWAVE_DIR"
echo ""

# ==================== Step 0: Alignment (if needed) ====================
if [ -f "$ALIGNMENT_PATH" ]; then
    echo "Step 0: Using existing alignment..."
    echo "  Found: $ALIGNMENT_PATH"

    # Only copy if source and destination are different
    if [ "$ALIGNMENT_PATH" != "$PHYLOWAVE_ALIGNMENT" ]; then
        cp "$ALIGNMENT_PATH" "$PHYLOWAVE_ALIGNMENT"
    fi
    echo "  ✓ Alignment ready: $PHYLOWAVE_ALIGNMENT"
else
    echo "Step 0: Creating multiple sequence alignment from raw sequences..."
    echo "  Input: $RAW_SEQS_PATH"

    if [ ! -f "$REF_PATH" ]; then
        echo "Error: Reference sequence not found at $REF_PATH"
        echo "Reference is required for alignment"
        exit 1
    fi

    augur align \
        --sequences "$RAW_SEQS_PATH" \
        --reference-sequence "$REF_PATH" \
        --output "$PHYLOWAVE_ALIGNMENT" \
        --remove-reference \
        --fill-gaps \
        --nthreads $N_THREADS

    echo "  ✓ Alignment created: $PHYLOWAVE_ALIGNMENT"
fi

echo ""

# ==================== Step 1: Build ML tree with IQ-TREE ====================
echo "Step 1: Building maximum-likelihood phylogeny..."

# Skip if tree already exists
if [ -f "$RAW_TREE_PATH" ]; then
    echo "  ✓ ML tree already exists, skipping: $RAW_TREE_PATH"
    echo "  (Delete this file to re-run IQ-TREE)"
# Check if IQ-TREE is available
elif ! command -v iqtree2 &> /dev/null && ! command -v iqtree &> /dev/null; then
    echo "Warning: IQ-TREE not found, falling back to FastTree (via augur tree)"
    echo "  Note: IQ-TREE is recommended for PhyloWave (provides bootstrap support)"
    echo "  Install with: conda install -c bioconda iqtree"
    echo ""

    augur tree \
        --alignment "$PHYLOWAVE_ALIGNMENT" \
        --output "$RAW_TREE_PATH" \
        --nthreads $N_THREADS

    echo "  ✓ Tree built with FastTree: $RAW_TREE_PATH"
else
    # Use IQ-TREE with parameters from Lefrancq et al. 2023
    IQTREE_CMD=$(command -v iqtree2 || command -v iqtree)

    echo "Using IQ-TREE for maximum-likelihood inference..."
    echo "  Model: GTR+F+G (General Time Reversible + empirical base frequencies + Gamma)"

    if [ "$FAST_MODE" = "true" ]; then
        echo "  Bootstrap: SKIPPED (fast mode enabled)"
        echo "  Note: Lefrancq et al. 2023 did not use bootstrap for viral sequences"
        echo ""

        $IQTREE_CMD \
            -s "$PHYLOWAVE_ALIGNMENT" \
            -m GTR+F+G \
            -nt AUTO \
            --prefix "$IQTREE_PREFIX" \
            -redo
    else
        echo "  Bootstrap: 1000 replicates with BNNI optimization"
        echo "  WARNING: This may take 6-12 hours with 5000+ sequences"
        echo "  Use fast mode for quicker testing: ./prep_sequences_for_phylowave.sh BUILD CLOCK_RATE true"
        echo ""

        $IQTREE_CMD \
            -s "$PHYLOWAVE_ALIGNMENT" \
            -m GTR+F+G \
            -bb 1000 \
            -bnni \
            -nt AUTO \
            --prefix "$IQTREE_PREFIX" \
            -redo
    fi

    # Copy IQ-TREE output to standard location
    cp "${IQTREE_PREFIX}.treefile" "$RAW_TREE_PATH"

    echo "  ✓ IQ-TREE analysis complete"
    if [ "$FAST_MODE" = "true" ]; then
        echo "  ✓ Maximum-likelihood tree: $RAW_TREE_PATH"
    else
        echo "  ✓ Tree with bootstrap support: $RAW_TREE_PATH"
    fi
    echo "  ✓ Full IQ-TREE output: ${IQTREE_PREFIX}.*"
fi

echo ""

# ==================== Step 2: Time-calibrate tree ====================
echo "Step 2: Time-calibrating phylogeny with molecular clock..."

# Skip if time-calibrated tree already exists
if [ -f "$REFINED_TREE_PATH" ]; then
    echo "  ✓ Time-calibrated tree already exists, skipping: $REFINED_TREE_PATH"
    echo "  (Delete this file to re-run augur refine)"
else
    echo "  Clock rate: $CLOCK_RATE substitutions/site/year"
    echo "  Method: TreeTime (via augur refine)"
    echo ""

    augur refine \
        --tree "$RAW_TREE_PATH" \
        --alignment "$PHYLOWAVE_ALIGNMENT" \
        --metadata "$METADATA_PATH" \
        --output-tree "$REFINED_TREE_PATH" \
        --output-node-data "$NODE_DATA_PATH" \
        --timetree \
        --coalescent opt \
        --clock-rate "$CLOCK_RATE" \
        --clock-std-dev 0.0005

    echo "  ✓ Time-resolved tree: $REFINED_TREE_PATH"
    echo "  ✓ Branch lengths now in time units (years)"
fi
echo ""

# ==================== Step 3: Validation ====================
echo "=== PhyloWave Readiness Validation ==="
echo ""

VALIDATION_PASSED=true

# Check 1: Tree exists and has branch lengths
if [ -f "$REFINED_TREE_PATH" ]; then
    echo "✓ Time-resolved tree exists: $REFINED_TREE_PATH"

    if grep -q ":" "$REFINED_TREE_PATH"; then
        echo "✓ Tree has branch lengths (in time units)"
    else
        echo "✗ ERROR: Tree missing branch lengths!"
        VALIDATION_PASSED=false
    fi
else
    echo "✗ ERROR: Time-resolved tree not created!"
    VALIDATION_PASSED=false
fi

# Check 2: Alignment exists
if [ -f "$PHYLOWAVE_ALIGNMENT" ]; then
    echo "✓ Alignment exists: $PHYLOWAVE_ALIGNMENT"
else
    echo "✗ ERROR: Alignment file missing!"
    VALIDATION_PASSED=false
fi

# Check 3: Metadata has date column
if [ -f "$METADATA_PATH" ]; then
    echo "✓ Metadata exists: $METADATA_PATH"

    if head -1 "$METADATA_PATH" | grep -q "date"; then
        echo "✓ Metadata has 'date' column"
    else
        echo "✗ WARNING: Metadata missing 'date' column!"
        echo "  PhyloWave requires sampling dates for each sequence"
        VALIDATION_PASSED=false
    fi
else
    echo "✗ ERROR: Metadata file missing!"
    VALIDATION_PASSED=false
fi

# Check 4: Node data exists
if [ -f "$NODE_DATA_PATH" ]; then
    echo "✓ Node data exists: $NODE_DATA_PATH"
else
    echo "✗ WARNING: Node data file missing!"
fi

echo ""

# ==================== Step 4: Summary ====================
if [ "$VALIDATION_PASSED" = true ]; then
    echo "=== ✓ SUCCESS: Ready for PhyloWave Analysis ==="
else
    echo "=== ✗ VALIDATION FAILED: Check errors above ==="
    exit 1
fi

echo ""
echo "Output files:"
echo "  • Time-resolved tree: $REFINED_TREE_PATH"
echo "  • Alignment: $PHYLOWAVE_ALIGNMENT"
echo "  • Metadata: $METADATA_PATH"
echo "  • Node data: $NODE_DATA_PATH"

if [ -f "${IQTREE_PREFIX}.treefile" ]; then
    echo "  • IQ-TREE files: ${IQTREE_PREFIX}.*"
fi

echo ""
echo "Next steps for PhyloWave (Lefrancq et al. 2023 method):"
echo ""
echo "1. Compute fitness index on tree:"
echo "   • Input tree: $REFINED_TREE_PATH"
echo "   • For H3N2: timescale = 0.4 years, time window = 0.25 years"
echo ""
echo "2. Detect lineages using GAM on index dynamics:"
echo "   • Use mgcv package in R"
echo "   • Find optimal number of lineages via elbow plot"
echo ""
echo "3. Quantify lineage fitness with multinomial logistic regression:"
echo "   • Use Stan/cmdstanr"
echo "   • Estimate relative growth rates (β_i) for each lineage"
echo ""
echo "Reference: Lefrancq et al. 2023, 'Learning the fitness dynamics of"
echo "           pathogens from phylogenies' bioRxiv"
echo "           https://github.com/noemielefrancq/paper-index-fitness-dynamics-trees"
