# scRNAseq-TAES

This repository contains a collection of scripts for analyzing single-cell RNA sequencing (scRNA-seq) datasets, focusing on clustering, dimensionality reduction, pseudotime trajectory inference, and embedding quality evaluation using the TAES score.

---

## Project Structure

- `data/`  
  Contains raw scRNA-seq datasets in `.h5ad` format.

- `scripts/`  
  Contains 8 Python scripts for different stages of analysis:

  1. **01_find_optimal_neighbors_tune_clustering.py**  
     Determines the optimal number of neighbors (`n_neighbors`) for clustering.

  2. **02_pca_variance_analysis_scRNAseq.py**  
     Performs PCA variance ratio analysis to help decide the number of principal components to retain.

  3. **03_plot_scRNAseq_trajectory_embeddings.py**  
     Visualizes pseudotime trajectories on multiple embeddings (PCA, t-SNE, UMAP, Diffusion Maps).

  4. **04_calculate_taes_slingshot_monocle3.py**  
     Calculates the TAES score using pseudotime inferred by Slingshot and Monocle3.

  5. **05_generate_embeddings_and_pseudotime_figures.py**  
     Generates embeddings and pseudotime trajectory figures for the datasets.

  6. **06_sensitivity_analysis_umap_tsne.py**  
     Conducts sensitivity analysis of UMAP and t-SNE parameters on clustering results.

  7. **07_embedding_stability_analysis.py**  
     Assesses embedding stability across multiple runs by computing Spearman correlations.

  8. **08_calculate_taes_scores.py**  
     Computes final TAES scores (average of Silhouette score and pseudotime correlation) across embeddings.

---

## Requirements

- **Python 3.8+**  
- Python packages:  
  `scanpy`, `anndata`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `scikit-learn`, `rpy2`  
- **R and Bioconductor packages** for running Slingshot and Monocle3-related analyses:  
  - `SingleCellExperiment`  
  - `slingshot`  
  - `monocle3`

---

## Usage Guide

1. Clone the repository and switch to the `project-root` branch:  
   ```bash
   git clone -b project-root https://github.com/nasiri928/scRNAseq-TAES.git
   cd scRNAseq-TAES
