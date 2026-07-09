import gc
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# Import of raw data written by Nicolas
PERTURBATION_COL_RENAME = {
    "sgRNA": "perturbation",
    "condition": "perturbation_2",
}


def chunk_expression_csv(csv_path, data_dir, chunksize=100):
    """Split a large genes-by-cells expression CSV into per-chunk .h5ad files.

    Each chunk covers `chunksize` genes (rows of the CSV) and all cells;
    chunks already written to `data_dir` are skipped so the function is
    resumable.
    """
    data_dir = Path(data_dir)
    reader = pd.read_csv(csv_path, index_col=0, chunksize=chunksize)

    for i, chunk in enumerate(reader):
        chunk_path = data_dir / f"chunk_{i}.h5ad"
        if chunk_path.exists():
            continue
        gene_names = chunk.index.to_numpy()
        cell_names = chunk.columns.to_numpy()
        X = sparse.csr_matrix(chunk.to_numpy(dtype=np.float32)).T.tocsr()
        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=cell_names),
            var=pd.DataFrame(index=gene_names),
        )
        adata.write_h5ad(chunk_path, compression="gzip")
        gc.collect()


def load_raw_adata(data_dir, metadata_path, n_chunks=None, out_path=None):
    """Concatenate the chunked .h5ad files and attach cell metadata.

    Loads the first `n_chunks` chunk files (all of them if `n_chunks` is
    None) from `data_dir`, concatenates them along the gene axis, and joins
    in per-cell metadata from `metadata_path`, adding convenience
    perturbation columns.
    """
    data_dir = Path(data_dir)
    chunk_paths = sorted(data_dir.glob("chunk_*.h5ad"))
    if n_chunks is not None:
        chunk_paths = chunk_paths[:n_chunks]

    adata = ad.concat([ad.read_h5ad(p) for p in chunk_paths], axis=1)

    meta = pd.read_csv(metadata_path, index_col=0, header=[0, 1]).droplevel(1, axis=1)
    adata.obs = meta.loc[adata.obs.index]
    adata.obs["NO_SITE"] = adata.obs["sgRNA"].map(lambda p: isinstance(p, str) and "NO_SITE" in p)
    adata.obs["ONE_NON-GENE_SITE"] = adata.obs["sgRNA"].map(
        lambda p: isinstance(p, str) and "ONE_NON-GENE_SITE" in p
    )
    for col, name in PERTURBATION_COL_RENAME.items():
        adata.obs[name] = adata.obs[col].copy()
    if not out_path is None:
        adata.write(out_path)
    return adata


# Quality control written by Atanas

INPUT_FILE = '/home/ubuntu/data/frangieh/rna.h5ad'


def load_qc_adata(input_file=INPUT_FILE):
    adata = ad.read_h5ad(input_file)
    print('Number of Cells:', adata.n_obs)
    print('Number of Genes:', adata.n_vars)
    return adata


def calculate_qc_metrics(adata):
    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
    return adata


## Outlier filtering
def outlier_func(adata_object, metric: str, nmads: int, print_tresholds=False):

    metric_median = np.median(adata_object.obs[metric])
    metric_mad = np.median(np.absolute(adata_object.obs[metric] - metric_median))
    upper_T = metric_median + (nmads*metric_mad)
    lower_T = metric_median - (nmads*metric_mad)

    if print_tresholds:
        print('upper: ', upper_T)
        print('lower: ', lower_T)

    return (adata_object.obs[metric] > upper_T) |(adata_object.obs[metric] < lower_T)


def _save_fig(fig, plots_dir, label, name):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f"{label}_{name}.png", bbox_inches="tight")


def qc_plots(adata, plots_dir=None, label="qc"):
    ax = sc.pl.violin(
    adata,
    ["total_counts", "log1p_total_counts"],
    jitter=0.4,
    multi_panel=True,
    show=plots_dir is None,
    )
    if plots_dir is not None:
        _save_fig(ax.figure, plots_dir, label, "counts")
    ax = sc.pl.violin(
    adata,
    ["n_genes_by_counts",  "log1p_n_genes_by_counts"],
    jitter=0.4,
    multi_panel=True,
    show=plots_dir is None,
    )
    if plots_dir is not None:
        _save_fig(ax.figure, plots_dir, label, "genes")
    ax = sc.pl.violin(
    adata,
    ['pct_counts_mt','pct_counts_in_top_50_genes'],
    jitter=0.4,
    multi_panel=True,
    show=plots_dir is None,
    )
    if plots_dir is not None:
        _save_fig(ax.figure, plots_dir, label, "pct_counts")


def qc_scatter(adata, plots_dir=None, label="qc"):
    ax = sc.pl.scatter(
        adata,
        "total_counts",
        "n_genes_by_counts",
        color="pct_counts_mt",
        show=plots_dir is None,
    )
    if plots_dir is not None:
        _save_fig(ax.figure, plots_dir, label, "scatter")


# **Notes:** Looking at the AnnData object, we have 218 331 Cells. Due to the high number of available cells, we could afford to be more strict with our filtering.
#
#
# * `pct_counts_mt`: we can see, that all cells over a thershold of 18 have already been filtered.
# * `tota_counts` and `n_genes_by_counts`: There are still some outliers noticable
# * `log1p_total_counts`, `log1p_n_genes_by_counts`, `pct_counts_in_top_50_genes`: There are still some outliers noticable


### Min Cells and Genes filters
def filter_min_genes_cells(adata, min_genes=200, min_cells=20):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


### QC Metrics
def flag_qc_outliers(adata, verbose=False):
    if verbose:
        print(outlier_func(adata, 'total_counts', 7).value_counts())
        print(outlier_func(adata, 'n_genes_by_counts', 4).value_counts())
        print(outlier_func(adata, 'log1p_total_counts', 4).value_counts())
        print(outlier_func(adata, 'log1p_n_genes_by_counts', 5).value_counts())
    return (outlier_func(adata, 'total_counts', 6) |
            outlier_func(adata, 'log1p_total_counts', 6) |
            outlier_func(adata, 'n_genes_by_counts', 6) |
            outlier_func(adata, 'log1p_n_genes_by_counts', 6) |
            outlier_func(adata, 'pct_counts_mt', 6) |
            outlier_func(adata, 'pct_counts_in_top_50_genes', 5))


def filter_qc_outliers(adata):
    adata.obs['outlier'] = flag_qc_outliers(adata)
    print(adata.obs['outlier'].value_counts())
    adata = adata[~adata.obs['outlier'], :].copy()
    return adata


## Doublet Detection
def detect_doublets(adata, batch_key="perturbation_2"):
    print(adata.obs.columns)
    #print(adata.obs['tissue_type'].value_counts())
    print(adata.obs[batch_key].value_counts())
    sc.pp.scrublet(adata, batch_key=batch_key)
    return adata


# Alternative doublet-detection approach considered (scvi/solo), kept for reference:
#scvi.model.SCVI.setup_anndata(test)
#vae = scvi.model.SCVI(test)
#vae.train()
#solo = scvi.external.SOLO.from_scvi_model(vae)
#solo.train()
#df = solo.predict()
#df['prediction'] = solo.predict(soft = False)
#
#df.index = df.index.map(lambda x: x[:-2])
#
#df


def run_qc(input_file=INPUT_FILE, adata=None, plots_dir=None, out_path=None, scrub_dublets=False,
    min_genes=200, min_cells=20,
    ):
    """Run the full QC notebook flow: load data, compute metrics, plot,
    explore min genes/cells filtering, flag+filter outliers, and detect
    doublets. Returns the outlier-filtered AnnData.

    If `plots_dir` is given, QC plots are saved there (via scanpy's `save`
    argument) instead of shown, annotated with a "raw"/"filtered" label so
    the before/after filtering plots don't overwrite each other.
    """
    if adata is None:
        adata = load_qc_adata(input_file)
    if not plots_dir is None:
        if not Path(plots_dir).exists():
            Path(plots_dir).mkdir()
    
    adata.raw = adata.copy()
    print(adata)
    adata = calculate_qc_metrics(adata)
    qc_plots(adata, plots_dir=plots_dir, label="raw")
    qc_scatter(adata, plots_dir=plots_dir, label="raw")

    adata = filter_min_genes_cells(adata, min_genes=min_genes, min_cells=min_cells)
    print(adata)
    adata = filter_qc_outliers(adata)
    print(adata)
    qc_plots(adata, plots_dir=plots_dir, label="filtered")
    qc_scatter(adata, plots_dir=plots_dir, label="filtered")
    if scrub_dublets:
        adata = detect_doublets(adata)
    
    # log and normalize
    sc.pp.log1p(adata)
    sc.pp.normalize_total(adata, target_sum=1e6)
    if not out_path is None:
        adata.write(out_path)
    return adata