"""
Title: Generate Embedding and Pseudotime Visualizations for scRNA-seq Datasets

Description:
------------
This script preprocesses single-cell RNA sequencing (scRNA-seq) datasets and generates
visualizations for dimensionality reduction and diffusion pseudotime inference.

For each dataset (e.g., PBMC3K, pancreas, BAT), the pipeline performs the following steps:

1. Preprocessing:
   - Gene filtering
   - Normalization and log-transformation
   - Highly variable gene selection
   - Scaling
   - Principal Component Analysis (PCA)

2. Graph Construction and Clustering:
   - Constructs a neighborhood graph using optimal `n_pcs` and `n_neighbors`
   - Applies Leiden clustering

3. Dimensionality Reduction:
   - Computes PCA, t-SNE, UMAP, and Diffusion Maps embeddings

4. Pseudotime Inference:
   - Identifies the root cell based on the largest Leiden cluster
   - Computes Diffusion Pseudotime (DPT)

5. Visualization:
   - Saves a grid of PCA, t-SNE, UMAP, and Diffusion Map plots colored by Leiden clusters
   - Saves a Diffusion Map plot colored by pseudotime trajectory

Usage:
------
- Make sure to set the correct file paths for your `.h5ad` datasets.
- Outputs will be saved in the specified `figures1_2` folder under the dataset base directory.
- The script is useful for visual inspection of embedding structures and pseudotemporal progression.


"""




import scanpy as sc
import matplotlib.pyplot as plt
import os

# Base directory for datasets and output figures
base_path = "./data"  # Change this to your local data directory or repository data folder
output_dir = os.path.join(base_path, "figures1_2")
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Paths to H5AD dataset files
datasets = {
    "pbmc3k": os.path.join(base_path, "pbmc3k.h5ad"),
    "pancreas": os.path.join(base_path, "pancreas.h5ad"),
    "bat": os.path.join(base_path, "tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad")
}

# Optimal parameters for each dataset (number of PCs and neighbors)
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
    Preprocess the data, perform dimensionality reduction, clustering,
    compute diffusion pseudotime, and save embedding and trajectory plots.

    Parameters:
    -----------
    adata : AnnData object
        The loaded single-cell dataset.
    dataset_name : str
        Identifier for the dataset, used in output filenames.
    n_pcs : int
        Number of principal components to use for neighbors graph.
    n_neighbors : int
        Number of neighbors for graph construction.
    """

    # Preprocessing
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)

    # Compute neighborhood graph and clustering
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    sc.tl.leiden(adata)

    # Compute embeddings
    sc.tl.diffmap(adata)
    sc.tl.tsne(adata, n_pcs=n_pcs)
    sc.tl.umap(adata)

    # Set root cell for diffusion pseudotime (most abundant cluster)
    root_cluster = adata.obs['leiden'].value_counts().idxmax()
    root_cell = adata.obs[adata.obs['leiden'] == root_cluster].index[0]
    adata.uns['iroot'] = adata.obs_names.get_loc(root_cell)

    # Compute diffusion pseudotime
    sc.tl.dpt(adata)

    # Plot embeddings colored by Leiden clusters
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    sc.pl.pca(adata, color='leiden', show=False, ax=axs[0, 0], title='PCA')
    sc.pl.tsne(adata, color='leiden', show=False, ax=axs[0, 1], title='t-SNE')
    sc.pl.umap(adata, color='leiden', show=False, ax=axs[1, 0], title='UMAP')
    sc.pl.diffmap(adata, color='leiden', show=False, ax=axs[1, 1], title='Diffusion Maps')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_embeddings_leiden.png"), dpi=300)
    plt.close()

    # Plot diffusion pseudotime trajectory
    sc.pl.diffmap(adata, color='dpt_pseudotime', show=False)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_trajectory.png"), dpi=300)
    plt.close()

    return adata

# Process all datasets and generate figures
for name, path in datasets.items():
    print(f"Processing dataset: {name}")
    adata = sc.read_h5ad(path)
    process_and_plot(adata, name, n_pcs=optimal_npcs[name], n_neighbors=optimal_neighbors[name])
