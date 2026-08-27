## Clustering 
### Task
Which genetic perturbations show similar effects, if any? Which clustering method(s) best
capture(s) the underlying biology?

Apply different clustering methods to the data, visualize and compare the results. Students
working alone only need to use data from the co-culture condition. Interpret the results
biologically. Do the clusters correspond to what you would expect based on the findings
described in the paper, e.g., the pathways that are discussed?

### Data Selection
Data from Frangieh et al. was preprocessed through the scmlcource.preprocessing.preprocess function, which wraps the workflow described in [qc](./qc.md). Additionally, for clustering, we filter out perturbation conditions that are represented by less than 100 cells.

### Clustering Methods
For clustering we choose two widespread and accessible methods:
1. #### Leiden Clustering
    Leiden Clustering is one of the most used clustering method in scRNA data analysis. We use scanpy's implementation in the scanpy.tl.leiden function.
    Leiden clustering was developed by [1], as an improvement on the louvain clustering algorithm by [2]. A graphical description of the louvain algorithm is depicted here:
    ![Louvain Algorithm](../plots/louvain_clustering_scheme.jpg)
    Leiden clustering has additional refinement steps between modularity optimization and agglomeration, in order to break up weakly connected groups.

    As input for leiden clustering, scanpy.tl.leiden uses a neighbour graph that is generated from principal component loadings.

2. #### K-Means clustering
    This relatively simple clustering method, that results in non-hierarchical separation. Observations are aggregated and separated, based on a distance metric, into a predetermined number of *k* clusters. This figure describes the algorithm:

    ![KMeans Algorithm](../plots/kmeans_clustering_scheme.jpg)

    We utilize scikit-learns implementation of k-means clustering in sklearn.cluster.KMeans. As input we use the PCA loadings from scanpy's scanpy.tl.pca function.

### Interpretation of Clusters
Frangieh et al. provide a classification of perturbed genes in their supplement tables (Supplementary Table 5), that sorts CRISPR targets into 10 biological categories. Unfortunately, this provided classification only supplies description for a small subset of perturbation targets. To still perform meaningfull analysis of the generated clusters, we perform ranked gene set enrichment analysis using the gseapy library with the "MSigDB_Hallmark_2020" gene set.

### Results
We calibrate leiden clusterings to a resolution of 0.2 and k-means clustering with k=5, resulting in ca. 5 clusters each. 
Here we show umaps and gene enrichment examples for well differentiated and undefined clusters.
#### Control Condition
##### UMAP
![control umap](../plots/task_2/Control/umap_clusters.png)
##### Leiden Cluster GEA
![cluster_1_leiden](../plots/task_2/Control/cluster_1_leiden.png)![cluster_4_leiden](../plots/task_2/Control/cluster_4_leiden.png)
##### K-Means Cluster GEA
![cluster_2_kmeans](../plots/task_2/Control/cluster_2_kmeans.png)![cluster_5_kmeans](../plots/task_2/Control/cluster_5_kmeans.png)

#### Co-Culture Condition
##### UMAP
![control umap](../plots/task_2/Co-culture/umap_clusters.png)
##### Leiden Cluster GEA
![cluster_1_leiden](../plots/task_2/Co-culture/cluster_1_leiden.png)![cluster_3_leiden](../plots/task_2/Co-culture/cluster_3_leiden.png)
##### K-Means Cluster GEA
![cluster_2_kmeans](../plots/task_2/Co-culture/cluster_2_kmeans.png)![cluster_4_kmeans](../plots/task_2/Co-culture/cluster_4_kmeans.png)

#### IFNγ Treated Condition
##### UMAP
![control umap](../plots/task_2/Co-culture/umap_clusters.png)
##### Leiden Cluster GEA
![cluster_1_leiden](../plots/task_2/Co-culture/cluster_1_leiden.png)![cluster_3_leiden](../plots/task_2/Co-culture/cluster_3_leiden.png)
##### K-Means Cluster GEA
![cluster_2_kmeans](../plots/task_2/Co-culture/cluster_2_kmeans.png)![cluster_4_kmeans](../plots/task_2/Co-culture/cluster_4_kmeans.png)

### References
- [1]: V. A. Traag, L. Waltman, and N. J. van Eck. From louvain to leiden: guaranteeing well-connected communities. Scientific Reports, mar 2019. URL: https://doi.org/10.1038/s41598-019-41695-z, doi:10.1038/s41598-019-41695-z.
- [2]: Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment, 2008(10):P10008, oct 2008. URL: https://doi.org/10.1088/1742-5468/2008/10/P10008, doi:10.1088/1742-5468/2008/10/p10008.




