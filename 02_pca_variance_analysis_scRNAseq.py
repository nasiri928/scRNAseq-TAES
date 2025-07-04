"""
Title: PCA Variance Analysis for scRNA-seq Datasets

Description:
------------
This script performs Principal Component Analysis (PCA) on a single-cell RNA sequencing (scRNA-seq)
dataset and prints the explained variance ratio for each principal component, along with the
cumulative explained variance.

This information helps determine how many principal components (PCs) are sufficient to capture
the majority of the variability in the data before applying dimensionality reduction or clustering methods.

The processing steps include:
1. Standard preprocessing: filtering, normalization, log transformation, selection of highly variable genes,
   and scaling.
2. PCA with up to `max_pcs` components.
3. Reporting of:
    - The individual explained variance ratio for each PC.
    - The cumulative variance explained up to each PC.

Usage:
------
- Set the correct path to your `.h5ad` dataset file.
- Run the script to print variance ratios and help determine a suitable number of PCs to retain.

Typical application includes deciding how many dimensions to retain for methods like UMAP, t-SNE,
neighborhood graph construction, or pseudotime inference.

"""



import scanpy as sc
import numpy as np

def print_pca_variance(file_path, max_pcs=50):
    """
    Load the dataset, perform PCA up to max_pcs components,
    and print the explained variance ratio of each component and cumulative variance.

    Parameters:
    -----------
    file_path : str
        Path to the input .h5ad dataset file.
    max_pcs : int, optional (default=50)
        Maximum number of principal components to compute.
    
    Returns:
    --------
    var_ratio : numpy.ndarray
        Explained variance ratio for each principal component.
    cum_var : numpy.ndarray
        Cumulative explained variance ratio.
    """

    print(f"🔎 Loading data from: {file_path}")
    adata = sc.read_h5ad(file_path)

    # Basic preprocessing (adjust as necessary for your dataset)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)

    # Perform PCA
    sc.tl.pca(adata, n_comps=max_pcs, svd_solver='arpack')

    var_ratio = adata.uns['pca']['variance_ratio']
    cum_var = np.cumsum(var_ratio)

    print("\nPC\tExplained Variance Ratio\tCumulative Explained Variance")
    for i in range(len(var_ratio)):
        print(f"{i+1}\t{var_ratio[i]:.6f}\t\t\t{cum_var[i]:.6f}")

    return var_ratio, cum_var

if __name__ == "__main__":
    # Base directory for dataset files (adjust as needed)
    DATA_DIR = "./data/"

    # Example usage with pancreas dataset
    pancreas_path = DATA_DIR + "pancreas.h5ad"

    # Uncomment to test other datasets:
    # bat_path = DATA_DIR + "tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad"
    # pbmc3k_path = DATA_DIR + "pbmc3k.h5ad"

    print_pca_variance(pancreas_path, max_pcs=50)
