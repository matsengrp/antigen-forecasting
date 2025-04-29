#!/bin/bash
#SBATCH --time=1-0
set -eu

N_THREADS=16
BUILD="flu-simulated-150k-samples-antigenic-clusters"
## DEFINE PATHS
REF_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/ref_HA.fasta"
SEQS_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/sequences.fasta"
METADATA_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/sequences_metadata.tsv"
RAW_TREE_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/auspice/raw_tree.nwk"
REFINED_TREE_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/auspice/refined_tree.nwk"
NODE_DATA_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/auspice/node_data.json"
AUSPICE_TREE_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/auspice/final_tree.json"
COLORMAP_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/auspice/variant_color_map.tsv"


export AUGUR_RECURSION_LIMIT=10000;

echo "Building auspice tree for $BUILD"
# Infer newick tree from fasta sequences
augur tree --alignment $SEQS_PATH --output $RAW_TREE_PATH --nthreads $N_THREADS

# Refine tree
echo "Refining tree..."
augur refine --tree $RAW_TREE_PATH  --metadata $METADATA_PATH --output-tree $REFINED_TREE_PATH --output-node-data $NODE_DATA_PATH

# Export tree for auspice
echo "Exporting tree for auspice..."
augur export v2 --tree $REFINED_TREE_PATH --metadata $METADATA_PATH --node-data $NODE_DATA_PATH --color-by-metadata clade_membership --colors $COLORMAP_PATH --output $AUSPICE_TREE_PATH