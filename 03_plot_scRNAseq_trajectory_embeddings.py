"""
Title: Visualize Diffusion Pseudotime Trajectories in Multiple Embedding Spaces

Description:
------------
This script processes single-cell RNA-seq datasets to generate pseudotime trajectory plots
projected onto different low-dimensional embedding spaces (PCA, t-SNE, UMAP, Diffusion Maps).

For each dataset, the following steps are performed:

1. Preprocessing:
   - Filter genes and cells
   - Normalize and log-transform expression data
   - Select highly variable genes
   - Scale and apply PCA

2. Neighborhood Graph Construction:
   - Construct k-nearest neighbor graph using optimal `n_pcs` and `n_neighbors`
   - Apply Leiden clustering to identify communities

3. Embedding Computation:
   - Compute PCA, t-SNE, UMAP, and Diffusion Maps embeddings

4. Pseudotime Inference:
   - Set the root cell as a representative of the largest Leiden cluster
   - Infer diffusion pseudotime (DPT)

5. Visualization:
   - For each embedding method, plot cells colored by pseudotime values
   - Save the pseudotime trajectory plots in the output directory

Usage:
------
- Make sure your `.h5ad` files are correctly placed in the specified `base_path`.
- The output will be a set of `.png` files saved under `figures2/` showing trajectories
  in different embeddings for each dataset.

"""



import scanpy as sc
import matplotlib.pyplot as plt
import os

# Base path for datasets and output directory
base_path = "./data"  # Change this to your data folder in GitHub repo or local machine
output_dir = os.path.join(base_path, "figures2")
os.makedirs(output_dir, exist_ok=True)  # Ensure output folder exists

# Dataset file paths
datasets = {
    "pbmc3k": os.path.join(base_path, "pbmc3k.h5ad"),
    "pancreas": os.path.join(base_path, "pancreas.h5ad"),
    "bat": os.path.join(base_path, "tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad")
}

# Optimal number of PCs and neighbors for each dataset
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

def process_and_plot(adata, dataset_name, n_pcs, n_neighbors):
    """
    Preprocess the dataset, compute embeddings, diffusion pseudotime,
    and save trajectory plots for PCA, t-SNE, UMAP, and Diffusion Maps.

    Parameters:
    -----------
    adata : AnnData
        Single-cell dataset
    dataset_name : str
        Name identifier for saving files
    n_pcs : int
        Number of principal components for neighbor graph
    n_neighbors : int
        Number of neighbors for graph construction
    """

    # Preprocessing
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)

    # Neighborhood graph construction and clustering
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    sc.tl.leiden(adata)

    # Compute embeddings
    sc.tl.pca(adata)
    sc.tl.tsne(adata, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.tl.diffmap(adata)

    # Set root cell for pseudotime (most abundant cluster)
    root_cluster = adata.obs['leiden'].value_counts().idxmax()
    root_cell = adata.obs[adata.obs['leiden'] == root_cluster].index[0]
    adata.uns['iroot'] = adata.obs_names.get_loc(root_cell)
    sc.tl.dpt(adata)

    # Save trajectory plots colored by diffusion pseudotime for each embedding
    for method in ['pca', 'tsne', 'umap', 'diffmap']:
        sc.pl.embedding(
            adata, basis=method, color='dpt_pseudotime',
            show=False, title=f"{method.upper()} - {dataset_name}"
        )
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_trajectory_{method}.png"), dpi=300)
        plt.close()

    return adata

# Run processing on all datasets
for name, path in datasets.items():
    print(f"Processing {name} ...")
    adata = sc.read_h5ad(path)
    process_and_plot(adata, name, optimal_npcs[name], optimal_neighbors[name])
