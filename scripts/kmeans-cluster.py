#!/usr/bin/env python3
"""
Perform k-means clustering on a DataFrame using specified columns.

Usage:
    python scripts/kmeans-cluster.py <input_df> <k> <output_path> [--column-prefix <prefix>] [--cluster-col-name <name>] [--random_seed <seed>] 

    Arguments:
    input_df: Path to the input DataFrame CSV file.
    k: Number of clusters (k) for k-means clustering.
    output_path: Path to write the DataFrame with assigned clusters.

    Options:
    --column-prefix <prefix>: Prefix of columns to use for clustering (e.g., 'ag', 'pc', 'mds'). Default is 'ag'.
    --cluster-col-name <name>: Name for the cluster assignment column. Default is 'cluster'.
    --random_seed <seed>: Random seed for reproducibility. Default is None.

"""
import pandas as pd
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def kmeans_clustering(input_df, k, output_path, column_prefix='ag', cluster_col_name='cluster', random_seed=None):
    """
    Perform k-means clustering on the input DataFrame.

    Args:
        input_df (DataFrame): Input DataFrame containing the data to be clustered.
        k (int): Number of clusters (k) for k-means clustering.
        output_path (str): Path to write the DataFrame with assigned clusters.
        column_prefix (str): Prefix of columns to use for clustering (e.g., 'ag', 'pc', 'mds'). Defaults to 'ag'.
        cluster_col_name (str): Name for the cluster assignment column. Defaults to 'cluster'.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    # Find all columns that start with the specified prefix
    feature_cols = [col for col in input_df.columns if col.startswith(column_prefix)]
    
    if not feature_cols:
        raise ValueError(f"No columns found with prefix '{column_prefix}'")
    
    print(f"Found {len(feature_cols)} columns with prefix '{column_prefix}': {feature_cols}")
    
    # Drop rows with NaN values in the feature columns
    input_df_clean = input_df.dropna(subset=feature_cols).copy()
    dropped_rows = len(input_df) - len(input_df_clean)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with NaN values")

    # Extract features
    X = input_df_clean[feature_cols]

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_seed)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Add cluster labels to DataFrame
    input_df_clean[cluster_col_name] = cluster_labels

    # Write DataFrame with assigned clusters to a file
    input_df_clean.to_csv(output_path, index=False)

    print(f"Clustering completed. Results saved to {output_path}")
    print(f"Cluster sizes: {pd.Series(cluster_labels).value_counts().sort_index().to_dict()}")

if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Perform k-means clustering on a DataFrame.")
    parser.add_argument("input_df", help="Path to the input DataFrame CSV file")
    parser.add_argument("k", type=int, help="Number of clusters (k)")
    parser.add_argument("output_path", help="Path to write the DataFrame with assigned clusters")
    parser.add_argument("--column-prefix", type=str, default='ag', 
                       help="Prefix of columns to use for clustering (e.g., 'ag', 'pc', 'mds'). Default: 'ag'")
    parser.add_argument("--cluster-col-name", type=str, default='cluster',
                       help="Name for the cluster assignment column. Default: 'cluster'")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility")

    # Parse command-line arguments
    args = parser.parse_args()

    # Load input DataFrame
    input_df = pd.read_csv(args.input_df)

    # Perform k-means clustering
    kmeans_clustering(input_df, args.k, args.output_path, 
                     column_prefix=args.column_prefix,
                     cluster_col_name=args.cluster_col_name,
                     random_seed=args.random_seed)