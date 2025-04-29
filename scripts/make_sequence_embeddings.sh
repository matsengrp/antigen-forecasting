#!/bin/bash
#SBATCH --time=1-0
set -eu

# Activate conda environment
mamba activate antigen

# Define build and paths
N_THREADS=32
BUILD="flu-simulated-150k-samples"
REF_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/ref_HA.fasta"
SEQS_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/sequences.fasta"
ALIGNMENT_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/pathogen-embed/sequences.alignment"
DISTANCE_MATRIX_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/pathogen-embed/distance-matrix.csv"
EMBEDDINGS_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/pathogen-embed/mds-embeddings.csv"
CLUSTERS_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/pathogen-embed/clusters-mds.csv"
FIGURE_PATH="$HOME/antigen-experiments/forecasting/data/$BUILD/pathogen-embed/clusters-mds.pdf"

# Align sequences from a fasta (make this a variable later)
echo "Aligning sequences for $BUILD..."
augur align --sequences $SEQS_PATH --reference-sequence $REF_PATH --output $ALIGNMENT_PATH --remove-reference --fill-gaps --nthreads $N_THREADS

# Create pairwise distance matrix from alignment
echo "Calculating pairwise distances for $BUILD..."
pathogen-distance --alignment $ALIGNMENT_PATH --output $DISTANCE_MATRIX_PATH

# Learn embedding from the distance matrix (make method a vairable name later, MDS for now)
echo "Learning MDS embeddings for $BUILD..."
pathogen-embed --alignment $ALIGNMENT_PATH --distance-matrix $DISTANCE_MATRIX_PATH --output-dataframe $EMBEDDINGS_PATH mds --components 3

# Cluster the embeddings
# pathogen-cluster --embedding $EMBEDDINGS_PATH --output-dataframe $CLUSTERS_PATH --output-figure $FIGURE_PATH --label-attribute variant --min-size 2 --min-samples 5 --distance-threshold 2.0