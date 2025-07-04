
"""
Title: Evaluate TAES Score for Dimensionality Reduction Methods in scRNA-seq Data

Description:
------------
This script evaluates and compares the performance of different dimensionality reduction 
methods (PCA, t-SNE, UMAP, Diffusion Maps) on single-cell RNA sequencing (scRNA-seq) data 
using the **TAES score**, a custom metric defined as the average of:
    - **Silhouette Score**: quantifies clustering quality
    - **Trajectory Score**: measures how well the embedding preserves pseudotemporal structure 
      using Diffusion Pseudotime (DPT) correlation

Main Steps:
-----------
1. Load and preprocess three benchmark scRNA-seq datasets:
    - Normalize, log-transform, select highly variable genes, scale, PCA
2. For each dataset and embedding method:
    - Compute the embedding
    - Perform Leiden clustering
    - Compute:
        a. Silhouette score based on clusters
        b. Trajectory score: average Spearman correlation between DPT pseudotime 
           and embedding dimensions
        c. TAES = (Silhouette + Trajectory) / 2
3. Save results to CSV file (`taes_with_n_neighbors.csv`)
4. Generate a barplot comparing TAES scores across methods and datasets

Output:
-------
- Tabular results of silhouette, trajectory, and TAES scores for each method/dataset
- A summary barplot saved as `taes_comparison_with_n_neighbors.png` under `./results/`

Usage:
------
- Ensure `.h5ad` dataset files are present in `./data/`.
- Adjust optimal PCA and neighbor values if needed.
- Run the script to generate and visualize TAES scores.
- Use this analysis to guide the selection of dimensionality reduction methods for 
  downstream scRNA-seq tasks such as clustering, pseudotime analysis, or visualization.

"""

import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress scanpy verbosity for cleaner output
sc.settings.verbosity = 0

# Dataset file paths (update to your relative paths)
datasets = {
    "pbmc3k": "./data/pbmc3k.h5ad",
    "pancreas": "./data/pancreas.h5ad",
    "bat": "./data/tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad"
}

# Optimal PCA components per dataset
optimal_npcs = {
    "pbmc3k": 25,
    "pancreas": 30,
    "bat": 25
}

# Optimal number of neighbors per dataset
optimal_k = {
    "pbmc3k": 15,
    "pancreas": 30,
    "bat": 30
}

methods = ["pca", "tsne", "umap", "diffmap"]
results = []

def compute_trajectory_score(adata, key="X_emb", n_neighbors=15, n_pcs=30):
    """
    Compute the trajectory correlation score using Diffusion Maps and DPT pseudotime.
    """
    # Compute diffusion map if not already present
    if "X_diffmap" not in adata.obsm:
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)  # Important for diffusion map
        sc.tl.diffmap(adata)

    # Compute neighborhood graph on diffusion map representation
    sc.pp.neighbors(adata, use_rep="X_diffmap", n_neighbors=n_neighbors)

    # Set root cell for pseudotime based on largest cluster if leiden is available
    if 'leiden' in adata.obs:
        root_cluster = adata.obs['leiden'].value_counts().idxmax()
        root_index = adata.obs[adata.obs['leiden'] == root_cluster].index[0]
        adata.uns['iroot'] = adata.obs_names.get_loc(root_index)
    else:
        adata.uns['iroot'] = 0

    # Calculate Diffusion Pseudotime
    sc.tl.dpt(adata)

    pt = adata.obs['dpt_pseudotime']
    emb = adata.obsm[key]
    dims = emb.shape[1]

    # Average absolute Spearman correlation between pseudotime and each embedding dimension
    traj_corr = np.mean([
        abs(spearmanr(pt, emb[:, i]).correlation)
        for i in range(dims)
    ])
    return traj_corr

for ds_name, ds_file in datasets.items():
    adata = sc.read(ds_file)
    n_pcs = optimal_npcs[ds_name]
    n_neighbors = optimal_k[ds_name]

    # Basic preprocessing
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    sc.tl.leiden(adata)

    for method in methods:
        adata_tmp = adata.copy()

        # Get or compute embeddings
        if method == "pca":
            emb = adata_tmp.obsm["X_pca"]
        elif method == "tsne":
            sc.tl.tsne(adata_tmp, n_pcs=n_pcs)
            emb = adata_tmp.obsm["X_tsne"]
        elif method == "umap":
            sc.pp.neighbors(adata_tmp, n_pcs=n_pcs, n_neighbors=n_neighbors)
            sc.tl.umap(adata_tmp)
            emb = adata_tmp.obsm["X_umap"]
        elif method == "diffmap":
            sc.pp.neighbors(adata_tmp, n_pcs=n_pcs, n_neighbors=n_neighbors)
            sc.tl.diffmap(adata_tmp)
            emb = adata_tmp.obsm["X_diffmap"]

        adata_tmp.obsm["X_emb"] = emb
        sil = silhouette_score(emb, adata_tmp.obs["leiden"])
        traj = compute_trajectory_score(adata_tmp, key="X_emb", n_neighbors=n_neighbors, n_pcs=n_pcs)
        taes = (sil + traj) / 2

        results.append({
            "dataset": ds_name,
            "method": method.upper(),
            "silhouette": sil,
            "trajectory": traj,
            "taes": taes
        })

# Save results to CSV (update path as needed)
df = pd.DataFrame(results)
df.to_csv("./results/taes_with_n_neighbors.csv", index=False)

# Plot TAES comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="method", y="taes", hue="dataset")
plt.title("TAES Score Comparison")
plt.ylabel("TAES Score")
plt.xlabel("Embedding Method")
plt.ylim(0, 1)
plt.legend(title="Dataset")
plt.tight_layout()
plt.savefig("./results/taes_comparison_with_n_neighbors.png", dpi=300)
plt.show()
