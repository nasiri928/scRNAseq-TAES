"""
📌  Description (FA/EN)
==============================================================
This script calculates the **TAES score** (Trajectory-Aligned Embedding Score)
for single-cell RNA-seq datasets using multiple dimensionality reduction methods
(PCA, t-SNE, UMAP, DiffMap) and two pseudotime inference algorithms
(Slingshot and Monocle3).

TAES is computed as the average of:
1. Silhouette score of clustering (Leiden)
2. Spearman correlation between pseudotime and embedding coordinates
==============================================================

📦 Requirements (Python + R):
--------------------------------------------------------------
Python packages (install via pip):
    pip install scanpy anndata matplotlib seaborn pandas numpy scikit-learn rpy2

R packages (install via R/RStudio):
    install.packages("BiocManager")
    BiocManager::install(c("SingleCellExperiment", "slingshot", "monocle3"))

⚠️ Note:
- Monocle3 requires specific Bioconductor versions; check compatibility.
- Make sure R is installed and added to PATH so that `rpy2` can detect it.
- R packages must be installed in the same R version that rpy2 links to.

==============================================================

📁 Outputs:
--------------------------------------------------------------
- CSV file of results: `all-taes.csv`
- Barplot visualizations for TAES using Slingshot and Monocle3:
    - `taes_slingshot.png`
    - `taes_monocle3.png`

Ensure output directories exist or are created manually before running the script.

"""

# --------------------------------------------------------------
# 📦 Imports
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# 📚 Load R packages
SingleCellExperiment = importr("SingleCellExperiment")
slingshot = importr("slingshot")
monocle3 = importr("monocle3")
base = importr("base")
s4 = importr("S4Vectors")

# 📁 Dataset paths
datasets = {
    "pbmc3k": "F:/Python Code/pbmc3k.h5ad",
    "pancreas": "F:/Python Code/pancreas.h5ad",
    "bat": "F:/Python Code/tabula-muris-senis-facs-processed-official-annotations-BAT.h5ad"
}

# 🧠 PCA & neighborhood parameters
optimal_npcs = {"pbmc3k": 25, "pancreas": 30, "bat": 25}
optimal_k = {"pbmc3k": 15, "pancreas": 30, "bat": 30}

# 🔁 Compute trajectory scores
def compute_traj_scores(adata, embedding_key, max_pcs_used=30):
    emb = adata.obsm[embedding_key]
    if embedding_key == "X_pca" and emb.shape[1] > max_pcs_used:
        emb = emb[:, :max_pcs_used]
    leiden = adata.obs["leiden"].astype(str)

    def traj_corr(pt, emb):
        valid = ~np.isnan(pt)
        return np.mean([abs(spearmanr(pt[valid], emb[valid, i]).correlation) for i in range(emb.shape[1])])

    # --- Slingshot pseudotime
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_emb_df = pandas2ri.py2rpy(pd.DataFrame(emb, index=adata.obs_names))
        r_leiden = robjects.FactorVector(leiden)

    robjects.globalenv["r_df"] = r_emb_df
    robjects.globalenv["clusters"] = r_leiden
    robjects.r('''
        library(SingleCellExperiment)
        library(slingshot)
        n_cells <- nrow(r_df)
        sce <- SingleCellExperiment(assays = list(counts = matrix(0, nrow=1, ncol=n_cells)))
        reducedDims(sce) <- SimpleList(PCA = as.matrix(r_df))
        rownames(reducedDims(sce)[["PCA"]]) <- rownames(r_df)
        colnames(reducedDims(sce)[["PCA"]]) <- colnames(r_df)
        sce <- slingshot(sce, clusterLabels = clusters, reducedDim = 'PCA')
    ''')
    sl_pt = np.array(robjects.r('as.numeric(colData(sce)$slingPseudotime_1)'))
    score_sl = traj_corr(sl_pt, emb)

    # --- Monocle3 pseudotime
    expr_mat = adata.raw.X if adata.raw is not None else adata.X
    if hasattr(expr_mat, "toarray"):
        expr_mat = expr_mat.toarray()

    raw_genes = np.array(adata.raw.var_names)
    valid_cells = np.array((expr_mat > 0).sum(axis=1)).flatten() > 0
    valid_genes = np.array((expr_mat > 0).sum(axis=0)).flatten() > 0

    expr_mat_filtered = expr_mat[valid_cells][:, valid_genes]
    valid_cells_idx = np.array(adata.obs_names)[valid_cells]
    valid_genes_idx = raw_genes[valid_genes]

    gene_meta = pd.DataFrame(index=valid_genes_idx)
    gene_meta["gene_short_name"] = valid_genes_idx
    cell_meta = pd.DataFrame(index=valid_cells_idx)
    cell_meta["cluster"] = leiden.loc[valid_cells_idx].values

    expr_mat_T = expr_mat_filtered.T
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_expr = pandas2ri.py2rpy(pd.DataFrame(expr_mat_T, index=valid_genes_idx, columns=valid_cells_idx))
        r_gene_meta = pandas2ri.py2rpy(gene_meta)
        r_cell_meta = pandas2ri.py2rpy(cell_meta)

    robjects.globalenv["r_expr"] = r_expr
    robjects.globalenv["r_gene_meta"] = r_gene_meta
    robjects.globalenv["r_cell_meta"] = r_cell_meta

    robjects.r('''
        library(monocle3)
        cds <- new_cell_data_set(as.matrix(r_expr), cell_metadata = r_cell_meta, gene_metadata = r_gene_meta)
        cds <- preprocess_cds(cds, num_dim = min(10, ncol(r_expr)))
        cds <- reduce_dimension(cds)
        cds <- cluster_cells(cds)
        cds@colData$cluster <- r_cell_meta$cluster
        cds <- learn_graph(cds)
        root_cell <- rownames(cds@colData)[cds@colData$cluster == names(which.max(table(cds@colData$cluster)))][1]
        cds <- order_cells(cds, root_cells = root_cell)
    ''')
    try:
        mo_pt = np.array(robjects.r('as.numeric(pseudotime(cds))'))
        score_mo = traj_corr(mo_pt, emb)
    except Exception:
        score_mo = np.nan

    return {'Slingshot': score_sl, 'Monocle3': score_mo}

# 🔄 Main loop
results = []
for name, path in datasets.items():
    adata = sc.read_h5ad(path)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    adata.raw = adata.copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=optimal_npcs[name])
    sc.pp.neighbors(adata, n_neighbors=optimal_k[name], n_pcs=optimal_npcs[name])
    sc.tl.leiden(adata)
    sc.tl.umap(adata)
    sc.tl.diffmap(adata)
    sc.tl.tsne(adata, n_pcs=optimal_npcs[name])

    embed_keys = {
        'PCA': 'X_pca',
        'UMAP': 'X_umap',
        'DiffMap': 'X_diffmap',
        't-SNE': 'X_tsne'
    }

    for method, key in embed_keys.items():
        sil = silhouette_score(adata.obsm[key], adata.obs['leiden'])
        trajs = compute_traj_scores(adata.copy(), key)
        for pt_method, tc in trajs.items():
            taes = (sil + tc) / 2 if not np.isnan(tc) else np.nan
            results.append({
                'dataset': name,
                'embedding': method,
                'pseudotime_method': pt_method,
                'silhouette': sil,
                'trajectory_corr': tc,
                'TAES': taes
            })

# 💾 Save results
results_df = pd.DataFrame(results)
results_df.to_csv("F:/Python Code/Figure-Slingshot and Monocle3/all-taes.csv", index=False)

# 📊 Plot results
for pt_method in ['Slingshot', 'Monocle3']:
    plt.figure(figsize=(10, 6))
    data = results_df[results_df['pseudotime_method'] == pt_method]
    sns.barplot(data=data, x="embedding", y="TAES", hue="dataset", palette="Set2")
    plt.title(f"TAES Scores across Embeddings ({pt_method})")
    plt.ylabel("TAES Score")
    plt.xlabel("Embedding Method")
    plt.ylim(0, 1)
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig(f"F:/Python Code/Figure-Slingshot and Monocle3/taes_{pt_method.lower()}.png", dpi=300)
    plt.close()
