"""
Title: Find Optimal Number of Neighbors for Clustering in scRNA-seq Data

Description:
------------
This script determines the optimal value of `n_neighbors` for constructing the neighborhood graph 
in single-cell RNA sequencing (scRNA-seq) datasets using the Scanpy framework.

The pipeline includes:
1. Preprocessing steps: filtering, normalization, log transformation, selection of highly variable genes,
   scaling, and PCA.
2. For each candidate value of `n_neighbors`, the script:
    - Constructs a neighborhood graph using PCA space.
    - Applies Leiden clustering.
    - Computes the Silhouette Score to assess clustering quality.
3. The optimal number of neighbors is chosen as the value that yields the highest Silhouette Score.

This score reflects the cohesion and separation of the resulting clusters in PCA space,
allowing for objective hyperparameter tuning prior to applying dimensionality reduction (e.g., UMAP, t-SNE)
or pseudotime inference methods.

Usage:
------
- Set the path to your `.h5ad` file(s) in the `DATA_DIR`.
- The function returns the optimal `n_neighbors` value for a given dataset.
- Optional: Uncomment the plotting section to visualize how Silhouette Score changes with neighborhood size.

"""


import scanpy as sc
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def find_optimal_n_neighbors(filepath, n_pcs=30, neighbor_range=range(5, 31, 5)):
    """
    Determine the optimal number of neighbors for clustering in scRNA-seq data.
    
    Parameters:
    -----------
    filepath : str
        Path to the input .h5ad single-cell dataset.
    n_pcs : int, optional (default=30)
        Number of principal components to use for neighborhood graph construction.
    neighbor_range : iterable, optional (default=range(5, 31, 5))
        Range of neighbor values to test for optimal clustering.
    
    Returns:
    --------
    best_k : int
        The number of neighbors that yields the highest Silhouette Score.
    """

    # Load dataset
    adata = sc.read_h5ad(filepath)

    # Preprocessing pipeline (similar to the methods in the manuscript)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)

    silhouette_scores = []
    tested_neighbors = []

    for k in neighbor_range:
        adata_tmp = adata.copy()
        # Compute neighborhood graph using k neighbors and n_pcs principal components
        sc.pp.neighbors(adata_tmp, n_neighbors=k, n_pcs=n_pcs)
        # Perform Leiden clustering on the neighborhood graph
        sc.tl.leiden(adata_tmp, key_added="leiden_tmp")

        # Calculate Silhouette Score based on PCA embedding and Leiden clusters
        score = silhouette_score(adata_tmp.obsm["X_pca"], adata_tmp.obs["leiden_tmp"])
        silhouette_scores.append(score)
        tested_neighbors.append(k)

    # Uncomment below lines to visualize the silhouette scores across neighbor values
    # plt.plot(tested_neighbors, silhouette_scores, marker='o')
    # plt.xlabel("Number of Neighbors (n_neighbors)")
    # plt.ylabel("Silhouette Score")
    # plt.title("Silhouette Scores vs. Number of Neighbors")
    # plt.grid(True)
    # plt.show()

    # Identify the neighbor number with the highest silhouette score
    best_k = tested_neighbors[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

    print(f"Best n_neighbors: {best_k} (Silhouette Score = {best_score:.3f})")
    return best_k

if __name__ == "__main__":
    # Define base data directory (adjust as needed or pass as an argument)
    DATA_DIR = "./data/"  # <- Set to your data folder path or relative path

    # Example usage for Pancreas dataset
    pancreas_path = DATA_DIR + "pancreas.h5ad"
    optimal_k_pancreas = find_optimal_n_neighbors(pancreas_path, n_pcs=30)

    # Uncomment and adjust paths to run for other datasets:
    # pbmc3k_path = DATA_DIR + "pbmc3k.h5ad"
    # optimal_k_pbmc = find_optimal_n_neighbors(pbmc3k_path, n_pcs=25)

    # bat_path = DATA_DIR + "tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad"
    # optimal_k_bat = find_optimal_n_neighbors(bat_path, n_pcs=25)
