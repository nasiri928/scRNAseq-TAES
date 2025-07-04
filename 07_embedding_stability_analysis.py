
"""
Title: Run-to-Run Stability Analysis of UMAP/t-SNE Embeddings in scRNA-seq Data

Description:
------------
This script evaluates the stability (robustness) of nonlinear embeddings such as UMAP or t-SNE 
when applied to single-cell RNA sequencing (scRNA-seq) data.

Due to the stochastic nature of these dimensionality reduction techniques, their outputs may vary 
significantly across different runs with different random seeds. This can affect downstream analysis 
such as trajectory inference or clustering.

To assess stability, the script does the following for each dataset:
1. Applies a preprocessing pipeline (filtering, normalization, PCA, neighbors).
2. Computes the chosen embedding (UMAP or t-SNE) multiple times (`n_runs`) with different random seeds.
3. For each pair of embedding runs, computes pairwise distances and calculates the Spearman correlation 
   between distance vectors.
4. Reports the **mean** and **standard deviation** of these Spearman correlations as a measure of stability.

Higher mean correlation and lower standard deviation indicate more stable (reproducible) embeddings.

Usage:
------
- Adjust `embedding_method` to `"umap"` or `"tsne"`.
- Adjust the dataset paths in the `datasets` dictionary.
- Run the script to print out stability scores for each dataset.
- Use this information to select more stable embedding techniques for trajectory inference or visualization.

"""


import scanpy as sc
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

# Configuration
embedding_method = "tsne"  # Options: "umap" or "tsne"
n_runs = 5

# Dataset file paths (update to your local or repo paths)
datasets = {
    "pbmc3k": "./data/pbmc3k.h5ad",
    "pancreas": "./data/pancreas.h5ad",
    "bat": "./data/tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad"
}

# Optimal parameters for each dataset
optimal_npcs = {"pbmc3k": 25, "pancreas": 30, "bat": 25}
optimal_k = {"pbmc3k": 15, "pancreas": 30, "bat": 30}

def compute_embedding_stability(file_path, dataset_name, method="umap", n_runs=5):
    """
    Compute run-to-run stability of embeddings by calculating
    the mean and standard deviation of pairwise Spearman correlations
    between pairwise distances of embeddings across multiple runs.

    Parameters:
    -----------
    file_path : str
        Path to the h5ad dataset file.
    dataset_name : str
        Dataset identifier (used to retrieve optimal params).
    method : str
        Embedding method, either "umap" or "tsne".
    n_runs : int
        Number of independent runs with different random seeds.

    Returns:
    --------
    mean_corr : float
        Mean Spearman correlation across runs.
    std_corr : float
        Standard deviation of Spearman correlations.
    """

    adata = sc.read(file_path)

    # Basic preprocessing steps
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=optimal_k[dataset_name], n_pcs=optimal_npcs[dataset_name])

    embeddings = []
    for seed in range(n_runs):
        adata_tmp = adata.copy()
        if method == "umap":
            sc.tl.umap(adata_tmp, random_state=seed)
            embeddings.append(adata_tmp.obsm["X_umap"])
        elif method == "tsne":
            sc.tl.tsne(adata_tmp, random_state=seed)
            embeddings.append(adata_tmp.obsm["X_tsne"])

    # Calculate Spearman correlation between pairwise distances of embeddings from different runs
    correlations = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            d1 = pdist(embeddings[i])
            d2 = pdist(embeddings[j])
            corr, _ = spearmanr(d1, d2)
            correlations.append(corr)

    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)

    return mean_corr, std_corr

# Run stability analysis for all datasets
for name, path in datasets.items():
    mean_corr, std_corr = compute_embedding_stability(path, name, method=embedding_method, n_runs=n_runs)
    print(f"{name} | mean Spearman correlation across runs = {mean_corr:.4f} ± {std_corr:.4f}")
