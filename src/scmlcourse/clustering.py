from pathlib import Path

import scanpy as sc
from sklearn.cluster import KMeans


def preprocess(adata, target_sum=1e6, n_neighbors=15, n_pcs=10, plots_dir=None):
    """Normalize, log-transform, and embed `adata` for clustering.

    Runs normalize_total + log1p, then PCA, then neighbors + UMAP on the
    resulting PCA embedding. Modifies `adata` in place and returns it.

    If `plots_dir` is given, the PCA variance ratio, PCA, and UMAP plots are
    saved there via scanpy's built-in `save` argument instead of being shown.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    sc.pp.pca(adata)
    if plots_dir is not None:
        sc.settings.figdir = Path(plots_dir)
    sc.pl.pca_variance_ratio(
        adata,
        save=".png" if plots_dir is not None else None,
        show=plots_dir is None,
    )
    sc.pl.pca(
        adata,
        color=["perturbation_2", "perturbation_2", "pct_counts_mt", "pct_counts_mt"],
        dimensions=[(0, 1), (2, 3), (0, 1), (2, 3)],
        ncols=2,
        size=2,
        save="_pca.png" if plots_dir is not None else None,
        show=plots_dir is None,
    )

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        color="perturbation_2",
        # Setting a smaller point size to get prevent overlap
        size=2,
        save="_umap.png" if plots_dir is not None else None,
        show=plots_dir is None,
    )
    return adata


def cluster_leiden(adata, resolutions=(0.2, 0.3, 0.4, 0.5, 1), plots_dir=None):
    """Run Leiden clustering on `adata` at each of `resolutions`.

    Requires `sc.pp.neighbors` to have been run already. Stores each
    clustering as `adata.obs[f"leiden_{res}"]` and returns `adata`.

    If `plots_dir` is given, the UMAP plot colored by each resolution's
    clustering is saved there via scanpy's built-in `save` argument instead
    of being shown.
    """
    for res in resolutions:
        sc.tl.leiden(adata, flavor="igraph", resolution=res, key_added=f"leiden_{res}")

    if plots_dir is not None:
        sc.settings.figdir = Path(plots_dir)
    sc.pl.umap(
        adata,
        color=[f"leiden_{res}" for res in resolutions] + ["perturbation_2"],
        save="_leiden.png" if plots_dir is not None else None,
        show=plots_dir is None,
    )
    return adata


def cluster_kmeans(adata, n_clusters_range=range(2, 5), plots_dir=None):
    """Run KMeans clustering on `adata.obsm["X_pca"]` for each k in `n_clusters_range`.

    Skips the first PC (`X_pca[:, 1:]`). Stores each clustering as
    `adata.obs[f"kmeans_{n_clusters}"]` and returns `adata`.

    If `plots_dir` is given, the UMAP plot colored by each k's clustering is
    saved there via scanpy's built-in `save` argument instead of being shown.
    """
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters)
        adata.obs[f"kmeans_{n_clusters}"] = kmeans.fit_predict(adata.obsm["X_pca"][:, 1:])

    if plots_dir is not None:
        sc.settings.figdir = Path(plots_dir)
    sc.pl.umap(
        adata,
        color=[f"kmeans_{n_clusters}" for n_clusters in n_clusters_range] + ["perturbation_2", "pct_counts_mt"],
        size=2,
        save="_kmeans.png" if plots_dir is not None else None,
        show=plots_dir is None,
    )
    return adata
