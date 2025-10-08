#!/bin/bash
#SBATCH --time=1-0
set -eu

# Script to generate Auspice phylogenetic tree with clade/variant annotations
# Usage: ./make_auspice_tree.sh [BUILD_NAME]
# Default BUILD: flu-simulated-150k-samples

# Get BUILD from command line argument or use default
BUILD="${1:-flu-simulated-150k-samples-final}"

# Dynamically detect number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    N_THREADS=$(sysctl -n hw.ncpu)
else
    # Linux
    N_THREADS=$(nproc)
fi

echo "Detected $N_THREADS CPU cores"

# Get the script's directory to determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Base data directory
DATA_DIR="$PROJECT_ROOT/data/$BUILD/antigen-outputs"
AUSPICE_DIR="$PROJECT_ROOT/data/$BUILD/auspice"

# Input file paths
REF_PATH="$PROJECT_ROOT/data/$BUILD/ref_HA.fasta"
SEQS_PATH="$AUSPICE_DIR/sequences.fasta"
METADATA_PATH="$AUSPICE_DIR/sequences_metadata.tsv"
COLORMAP_PATH="$AUSPICE_DIR/variant_color_map.tsv"

# Output file paths
RAW_TREE_PATH="$AUSPICE_DIR/raw_tree.nwk"
REFINED_TREE_PATH="$AUSPICE_DIR/refined_tree.nwk"
NODE_DATA_PATH="$AUSPICE_DIR/node_data.json"
AUSPICE_TREE_PATH="$AUSPICE_DIR/final_tree.json"

# Validate required input files exist
echo "Validating input files for build: $BUILD"
if [ ! -f "$SEQS_PATH" ]; then
    echo "Error: Sequences file not found at $SEQS_PATH"
    exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file not found at $METADATA_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$AUSPICE_DIR"

export AUGUR_RECURSION_LIMIT=25000;

echo "Building auspice tree for $BUILD"
echo "Project root: $PROJECT_ROOT"
#echo "Data directory: $DATA_DIR"
echo "Output directory: $AUSPICE_DIR"
# Infer newick tree from fasta sequences
echo "Step 1: Building phylogenetic tree from sequences..."
augur tree --alignment "$SEQS_PATH" --output "$RAW_TREE_PATH" --nthreads $N_THREADS

# Refine tree
echo "Step 2: Refining tree with temporal information..."
augur refine --tree "$RAW_TREE_PATH" --metadata "$METADATA_PATH" --output-tree "$REFINED_TREE_PATH" --output-node-data "$NODE_DATA_PATH"

# Export tree for auspice
echo "Step 3: Exporting tree for Auspice visualization..."

# Check if colormap exists, otherwise export without it
if [ -f "$COLORMAP_PATH" ]; then
    echo "Using color map from $COLORMAP_PATH"
    augur export v2 --tree "$REFINED_TREE_PATH" --metadata "$METADATA_PATH" --node-data "$NODE_DATA_PATH" --color-by-metadata clade_membership --colors "$COLORMAP_PATH" --output "$AUSPICE_TREE_PATH"
else
    echo "Warning: Color map not found at $COLORMAP_PATH, exporting without custom colors"
    augur export v2 --tree "$REFINED_TREE_PATH" --metadata "$METADATA_PATH" --node-data "$NODE_DATA_PATH" --color-by-metadata clade_membership --output "$AUSPICE_TREE_PATH"
fi

echo "Done! Auspice tree saved to: $AUSPICE_TREE_PATH"