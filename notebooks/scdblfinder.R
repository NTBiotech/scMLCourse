library(scDblFinder)
library(Matrix)
data_dir = "/home/ubuntu/data/frangieh" # SET DATA DIRECTORY
data_mat <- Matrix::readMM(paste0(data_dir, "/RNA_counts.mtx"))
cellnames_df <- read.csv(paste0(data_dir, "/RNA_cellnames.csv"))
coldata <- DataFrame(cellnames=cellnames_df[,"cellnames"], day=cellnames_df[, "day"])
sce <- SingleCellExperiment(list(counts=t(data_mat)), colData=coldata)
scdbl <- scDblFinder(sce, samples="day")
doublet_score = scdbl$scDblFinder.score
doublet_class = scdbl$scDblFinder.class
df = data.frame(unlist(list(doublet_score)), unlist(list(doublet_class)))
names(df) = c("doublet_score", "doublet_class")
rownames(df) <- cellnames_df[,"cellnames"]
write.csv(df, paste0(data_dir, "/doublets_RNA_scdblfinder.csv"))