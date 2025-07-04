# scRNAseq-TAES

This repository contains a collection of scripts for analyzing single-cell RNA sequencing (scRNA-seq) datasets, focusing on clustering, dimensionality reduction, pseudotime trajectory inference, and embedding quality evaluation using the TAES score.

---

## 📁 Project Structure

- `data/`  
  Contains raw `.h5ad` scRNA-seq datasets used in the analysis.  

  Currently available:
  - `pbmc3k.h5ad`  
  - `tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad`  

  Additional datasets can be downloaded from the links below.

- `scripts/`  
  Contains Python scripts for each stage of analysis:

  1. `01_find_optimal_neighbors_tune_clustering.py`  
     → Optimize the number of neighbors (`n_neighbors`) for clustering.

  2. `02_pca_variance_analysis_scRNAseq.py`  
     → Analyze PCA variance ratio to select `n_pcs`.

  3. `03_plot_scRNAseq_trajectory_embeddings.py`  
     → Visualize pseudotime trajectories across embeddings (PCA, t-SNE, UMAP, Diffusion Maps).

  4. `04_calculate_taes_slingshot_monocle3.py`  
     → Calculate TAES score using pseudotime inferred from Slingshot and Monocle3.

  5. `05_generate_embeddings_and_pseudotime_figures.py`  
     → Generate trajectory plots for each embedding.

  6. `06_sensitivity_analysis_umap_tsne.py`  
     → Analyze sensitivity of UMAP and t-SNE to parameter changes.

  7. `07_embedding_stability_analysis.py`  
     → Assess stability of embeddings across runs (using Spearman correlation).

  8. `08_calculate_taes_scores.py`  
     → Final TAES computation: average of silhouette score and pseudotime correlation.

---

## 📦 Dataset Download

If you want to use the full set of datasets used in the experiments:

- **PBMC3K**  
  Included in the repository: `data/pbmc3k.h5ad`

- **BAT (Brown Adipose Tissue)**  
  - GitHub: `data/tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad`  
  - Also available from Figshare:  
    [Download BAT from Figshare](https://figshare.com/articles/dataset/Tabula_Muris_Senis_FACS_spleen/12654728?file=23872493)

- **Pancreas**  
  - Not included in the repository due to file size limit.  
  - Download from Figshare:  
    [Download pancreas.h5ad from Figshare](https://figshare.com/articles/dataset/pancreas_h5ad/21878426)

After downloading, place them manually into the `data/` folder to use the scripts.

---

## 🧰 Requirements

- **Python 3.8+**
- Required Python packages:  
  `scanpy`, `anndata`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `scikit-learn`, `rpy2`

- **R + Bioconductor packages** (for Slingshot and Monocle3):  
  - `SingleCellExperiment`  
  - `slingshot`  
  - `monocle3`

---

## 🚀 Usage Guide

1. Clone the repository:

```bash
git clone https://github.com/nasiri928/scRNAseq-TAES.git
cd scRNAseq-TAES
