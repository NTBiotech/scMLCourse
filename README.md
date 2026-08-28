# Practical machine learning for single cell multiomics - Seminar report

For this project, we analyzed a CRISPR perturbation scRNA-seq dataset containing the gene expression profiles of melanoma cells across three experimental cell culture conditions. The CRISPR perturbation targeted 248 genes and included an additional single-cell surface protein measurements. 

The dataset was provided as an `AnnData` H5ad file (`rna.h5ad`). In order to solve the given tasks, a perliminary inspection, exploration and quality control processing was done to ensure the high quality of data.

### Quality control processing

The quality control processing was done as shown in the JupyterLab `QC.ipynb` notebook. Briefly, the raw `AnnData` dataset contained the gene expression data for 218 331 cells across 23 712 genes. Quality control was performed based on the discussed criterai during the course and following the recommended Scanpy workflow, as outlined in Chapter 8 in the Single-cell best practices textbook (1). From the initial QC plots we can see:

* `pct_counts_mt`: we can see, that all cells over a thershold of 18 have already been filtered.
* `tota_counts` and `n_genes_by_counts`: There are still some outliers noticable
* `log1p_total_counts`, `log1p_n_genes_by_counts`, `pct_counts_in_top_50_genes`: There are still some outliers noticable

Due to the high number of cells, we can afford to be more strict with our filtering. Cells with less than 200 expressed genes and genes expressed in less than 20 cells were filtered out from the dataset. For the following QC metrics, a threshold of 3 (Initially 5) MADs was set for the following metrics to filter any possible outliers: `total_counts`, `log1p_total_counts`, `n_genes_by_counts`, `log1p_n_genes_by_counts`, `pct_counts_mt`, `pct_counts_in_top_50_genes`.

Note: Doublets would normally be called using the scrublet impementation in scanpy, however running the `sc.pp.scrublet(subset)` line always crahsed the VM for some reason, so the MAD treshold above was lowered to 3 from 5.

Lastly, the dataset was normalized, the top 200 highly variable genes were selected and the `AnnData` Object was exported as `rna.qc.hvg200.h5ad`. For Task 1 - Condition classification, an additional dataset with only the top 100 HVGs was exported.

## Reports

Our solutions to the given tasks are in the notebooks folder at:

1. [Classification](./notebooks/task_1/README.md)
2. [Clustering](./notebooks/task_2/README.md)
3. [Perturbation Effect Modeling](./notebooks/task_3/README.md)
