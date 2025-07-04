"""
Title: Sensitivity Analysis of UMAP and t-SNE Embeddings for scRNA-seq Data

Description:
------------
This script evaluates the sensitivity of UMAP and t-SNE embeddings to their key hyperparameters 
using single-cell RNA sequencing (scRNA-seq) datasets.

Specifically, the script performs the following:

1. Loads three benchmark scRNA-seq datasets (PBMC3k, Pancreas, BAT).
2. Applies standard preprocessing steps including normalization, log1p transformation, 
   selection of highly variable genes, scaling, and PCA.
3. For **UMAP**, tests multiple `n_neighbors` values (e.g., 5, 10, 15, ..., 100) and computes:
   - Clustering using Leiden algorithm
   - Silhouette Score as a measure of clustering quality
   - UMAP visualizations colored by clusters with silhouette scores shown
4. For **t-SNE**, tests different `perplexity` values (e.g., 5, 10, 30, 50) and repeats the same steps.
5. Saves all UMAP and t-SNE visualizations with hyperparameter values and silhouette scores to the `results/` folder.

This analysis provides insights into how embedding quality and cluster structure change with 
key hyperparameters, helping to select robust configurations for downstream analyses 
like trajectory inference or cell type annotation.

Usage:
------
- Make sure `.h5ad` files are available in `./data/`.
- Adjust optimal `n_pcs` and `n_neighbors` if needed.
- Run the script to generate all plots for UMAP and t-SNE sensitivity.
- Output files are saved as: `./results/umap_sensitivity_*.png` and `./results/tsne_sensitivity_*.png`

"""


import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# ---------- Optimal Parameters for each dataset ----------
optimal_npcs = {
    "pbmc3k": 25,
    "pancreas": 30,
    "bat": 25
}
optimal_neighbors = {
    "pbmc3k": 15,
    "pancreas": 30,
    "bat": 30
}

# ---------- Load datasets ----------
adata_pbmc = sc.read("./data/pbmc3k.h5ad")
adata_pancreas = sc.read("./data/pancreas.h5ad")
adata_bat = sc.read("./data/tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad")

# ---------- Preprocessing all datasets ----------
for adata in [adata_pbmc, adata_pancreas, adata_bat]:
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata._inplace_subset_var(adata.var.highly_variable)
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)

# ---------- UMAP Sensitivity Analysis ----------
def umap_sensitivity_scanpy(adata_raw, dataset_name):
    """
    Performs UMAP sensitivity analysis over different n_neighbors values.
    Calculates silhouette scores for each embedding.
    Saves the UMAP plots showing clustering consistency.
    """
    n_neighbors_list = [5, 10, 15, 30, 50, 100]
    silhouette_scores = []
    n_pcs = optimal_npcs[dataset_name]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, n_neighbors in enumerate(n_neighbors_list):
        adata = adata_raw.copy()
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(adata, random_state=42)
        sc.tl.leiden(adata, resolution=1.0)

        coords = adata.obsm["X_umap"]
        labels = adata.obs["leiden"]
        sil_score = silhouette_score(coords, labels)
        silhouette_scores.append(sil_score)

        ax = axes[i // 3, i % 3]
        sc.pl.umap(adata, color='leiden', show=False, ax=ax,
                   title=f"n_neighbors={n_neighbors}\nSil={sil_score:.2f}", s=10)

    plt.tight_layout()
    plt.savefig(f"./results/umap_sensitivity_{dataset_name}.png", dpi=300)
    plt.close()
    return n_neighbors_list, silhouette_scores

# ---------- t-SNE Sensitivity Analysis ----------
def tsne_sensitivity_scanpy(adata_raw, dataset_name):
    """
    Performs t-SNE sensitivity analysis over different perplexity values.
    Calculates silhouette scores for each embedding.
    Saves the t-SNE plots showing clustering consistency.
    """
    perplexities = [5, 10, 30, 50]
    silhouette_scores = []
    n_pcs = optimal_npcs[dataset_name]
    n_neighbors = optimal_neighbors[dataset_name]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, perplexity in enumerate(perplexities):
        adata = adata_raw.copy()
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.tsne(adata, perplexity=perplexity, random_state=42)
        sc.tl.leiden(adata, resolution=1.0)

        coords = adata.obsm["X_tsne"]
        labels = adata.obs["leiden"]
        sil_score = silhouette_score(coords, labels)
        silhouette_scores.append(sil_score)

        ax = axes[i // 2, i % 2]
        sc.pl.tsne(adata, color='leiden', show=False, ax=ax,
                   title=f"perplexity={perplexity}\nSil={sil_score:.2f}", s=10)

    plt.tight_layout()
    plt.savefig(f"./results/tsne_sensitivity_{dataset_name}.png", dpi=300)
    plt.close()
    return perplexities, silhouette_scores

# ---------- Run All Analyses ----------
print("Running UMAP sensitivity for PBMC...")
umap_sensitivity_scanpy(adata_pbmc, "pbmc3k")

print("Running t-SNE sensitivity for PBMC...")
tsne_sensitivity_scanpy(adata_pbmc, "pbmc3k")

print("Running UMAP sensitivity for Pancreas...")
umap_sensitivity_scanpy(adata_pancreas, "pancreas")

print("Running t-SNE sensitivity for Pancreas...")
tsne_sensitivity_scanpy(adata_pancreas, "pancreas")

print("Running UMAP sensitivity for BAT...")
umap_sensitivity_scanpy(adata_bat, "bat")

print("Running t-SNE sensitivity for BAT...")
tsne_sensitivity_scanpy(adata_bat, "bat")
