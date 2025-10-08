#!/bin/bash
#SBATCH --time=1-0
set -eu

# Run this script from the antigen-forecasting repository root:
# bash scripts/make_sequence_embeddings.sh
#
# Activate conda environment before running this script with:
# mamba activate antigen

# Define build and paths
SEED=314159
N_THREADS=32
BUILD="flu-simulated-150k-samples-final"
REF_PATH="data/$BUILD/ref_HA.fasta"
SEQS_PATH="data/$BUILD/antigen-outputs/unique_sequences.fasta"
ALIGNMENT_PATH="data/$BUILD/pathogen-embed/sequences.alignment"
DISTANCE_MATRIX_PATH="data/$BUILD/pathogen-embed/distance-matrix.csv"
EMBEDDING_METHOD="mds"
N_COMPONENTS=10
EMBEDDINGS_PATH="data/$BUILD/pathogen-embed/$EMBEDDING_METHOD-embeddings.csv"
CLUSTERS_PATH="data/$BUILD/pathogen-embed/seq-clusters.csv"
FIGURE_PATH="data/$BUILD/pathogen-embed/clusters-$EMBEDDING_METHOD.pdf"

# Number of clusters for k-means
N_CLUSTERS=30

# Create output directory if it doesn't exist
mkdir -p "data/$BUILD/pathogen-embed"

# Align sequences from a fasta (make this a variable later)
echo "Aligning sequences for $BUILD..."
augur align --sequences $SEQS_PATH --reference-sequence $REF_PATH --output $ALIGNMENT_PATH --remove-reference --fill-gaps --nthreads $N_THREADS

# Create pairwise distance matrix from alignment
echo "Calculating pairwise distances for $BUILD..."
pathogen-distance --alignment $ALIGNMENT_PATH --output $DISTANCE_MATRIX_PATH

# Learn embedding from the distance matrix
echo "Learning $EMBEDDING_METHOD embeddings for $BUILD..."
pathogen-embed --alignment $ALIGNMENT_PATH --distance-matrix $DISTANCE_MATRIX_PATH --output-dataframe $EMBEDDINGS_PATH $EMBEDDING_METHOD --components $N_COMPONENTS

# Check if embeddings file was created
if [ ! -f "$EMBEDDINGS_PATH" ]; then
    echo "Error: Embeddings file not created at $EMBEDDINGS_PATH"
    exit 1
fi

# # Cluster the embeddings
# echo "Clustering embeddings with k-means (k=$N_CLUSTERS)..."
# python scripts/kmeans-cluster.py "$EMBEDDINGS_PATH" "$N_CLUSTERS" "$CLUSTERS_PATH" --column-prefix "$EMBEDDING_METHOD" --cluster-col-name "variant_seq" --random_seed "$SEED"

# # Check if clustering succeeded
# if [ -f "$CLUSTERS_PATH" ]; then
#     echo "Success! Sequence embeddings and clusters have been created:"
#     echo "  - Embeddings: $EMBEDDINGS_PATH"
#     echo "  - Clusters: $CLUSTERS_PATH"
# else
#     echo "Error: Clustering failed. Check the error messages above."
#     exit 1
# fi
