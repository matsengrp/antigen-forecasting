import pandas as pd
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def kmeans_clustering(input_df, k, output_path, random_seed=None):
    """
    Perform k-means clustering on the input DataFrame.

    Args:
        input_df (DataFrame): Input DataFrame containing the data to be clustered.
        k (int): Number of clusters (k) for k-means clustering.
        output_path (str): Path to write the DataFrame with assigned clusters.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
    # Drop any NaN values in the DataFrame
    input_df = input_df.dropna()

    # Extract features
    # Extracting features
    X = input_df[['ag1', 'ag2']]

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_seed)
    cluster_labels = kmeans.fit_predict(X)

    # Add cluster labels to DataFrame
    input_df['variant'] = cluster_labels

    # Write DataFrame with assigned clusters to a file
    input_df.to_csv(output_path, index=False)

    print(f"Clustering completed. Results saved to {output_path}")

if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Perform k-means clustering on a DataFrame.")
    parser.add_argument("input_df", help="Path to the input DataFrame CSV file")
    parser.add_argument("k", type=int, help="Number of clusters (k)")
    parser.add_argument("output_path", help="Path to write the DataFrame with assigned clusters")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility")

    # Parse command-line arguments
    args = parser.parse_args()

    # Load input DataFrame
    input_df = pd.read_csv(args.input_df)

    # Perform k-means clustering
    kmeans_clustering(input_df, args.k, args.output_path, args.random_seed)