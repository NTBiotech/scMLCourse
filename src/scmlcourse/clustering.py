import scanpy as sc
from sklearn.cluster import KMeans


def preprocess(adata, target_sum=1e6, n_neighbors=15, n_pcs=10):
    """Normalize, log-transform, and embed `adata` for clustering.

    Runs normalize_total + log1p, then PCA, then neighbors + UMAP on the
    resulting PCA embedding. Modifies `adata` in place and returns it.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    sc.pp.pca(adata)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    return adata


def cluster_leiden(adata, resolutions=(0.2, 0.3, 0.4, 0.5, 1)):
    """Run Leiden clustering on `adata` at each of `resolutions`.

    Requires `sc.pp.neighbors` to have been run already. Stores each
    clustering as `adata.obs[f"leiden_{res}"]` and returns `adata`.
    """
    for res in resolutions:
        sc.tl.leiden(adata, flavor="igraph", resolution=res, key_added=f"leiden_{res}")
    return adata


def cluster_kmeans(adata, n_clusters_range=range(2, 5)):
    """Run KMeans clustering on `adata.obsm["X_pca"]` for each k in `n_clusters_range`.

    Skips the first PC (`X_pca[:, 1:]`). Stores each clustering as
    `adata.obs[f"kmeans_{n_clusters}"]` and returns `adata`.
    """
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters)
        adata.obs[f"kmeans_{n_clusters}"] = kmeans.fit_predict(adata.obsm["X_pca"][:, 1:])
    return adata
