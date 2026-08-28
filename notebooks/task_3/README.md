# Task 3 - Perturbation prediction
### Task

Can a model predict transcriptome changes for a gene that was not knocked out in the training data?

Before model training, select a subset of 50 perturbations for modeling and explain your choice. Of these, select a subset of perturbations that are held out, i.e., not used during training or hyperparameter selection, only at test time. If the number of selected genes is still too high for the model you want to train, you may subset it further and explain your choice. You may also use unperturbed cells.

The target variable to predict by your models is the mean log2 fold change (LFC) per condition and perturbation target. Train two (if one student), three (if two students), or four (if three students) types of models. Here, a type of model is a feature engineering strategy combined with a learning algorithm. At least one of the models should be deliberately simplistic. Evaluate the model performance and uncertainty based on suitable metrics and interpret the results biologically. You may also use the protein data, however, this is not required. Students working alone only need to use data from the co-culture condition.

## Classical Statistical Approaches
(by Atanas)

For this task, the main goal for the model is to predict changes in gene expression for genes not featured in the training data. The main target variable for this task is the mean log2 fold change per condition and perturbation target for a selected set of 50 perturbations. As such, this would be considered a regression problem. The dataset used for this task is the previously mentioned dataset with 200 HVGs.

### Experimental design

Due to the complexety of this task, as with the previous task, several consessions were made. Firstly, due to technical difficulties, kernel crashes and overall longer training times, the number of perturbations was reduced from 50 to 30, leaving 25 perturbation conditions for training and 5 for testing. In addition, the culture conditions were also limited to Control and Co-culture to further reduce the dataset. The perturbation conditions were chosen based on the number of cell per perturbation condition. The top 30 perturbations with the highest cell counts, excluding the control cells, were selected. The first 25 of them were used for training, while the final 5 were used for testing. 

The target mean log2 fold changes were claculated per condition per perturbatino target using the scanpy function `sc.get.rank_genes_groups`. Each observation (cell) was assigned a target variable value based on their condition and perturbation grouping, with cells within the same group getting the same value. For each observation, the target variable was a 30-dimensional array, for each selected perturbation target.

In order to compile the feature set, 2 approaches were implemented. All genes in the dataset were sorted by thei normalized dispersion column `dispersions_norm`, which results from the scanpy HVG selection and the top 2000 HVGs were selected.The authors of the publication _"...performed pooled Perturb-CITE-seq screens of ICR program genes in a patient-derived tumor-TIL co-culture model, targeting 248 genes of the ICR signature (744 targeting guides)..."_. As such, the `adata.obs` column `guide_id` to filter out all genes, that have been targeted and could carry information regarding the perturbation condition. In order to avoid leakage, all perturbed genes were removed from the feature set. 

A gene expression matrix containing all cells with the chosen training perturbations for all genes in the selected feature set was split into a Training and Validation in a 75%/25% split. For this regression task, 2 regression models from `scikit-learn` were selected: a ridge linear regression model and a random forest regressor model. The experimental set up was based on `scikit-learn`'s API and was the same for both models. A grid search parameter optimiazation with stratified k-fold cross-validation (k=3) was used to optimize hyperparameters for the models using the training set, on which the best performing model was fitted.

### Model evaluation

For regression tasks, the models were evaluated based on the :

* Mean Absolute Error: Average absolute difference between predictions and actuals.
* Mean Squared Error: Squares errors to penalize large deviations more.
* Root Mean Squared Error: Same units as target, penalizes large errors.
* R^2 Score: Proportion of variance explained by the model.
* Person score: The correlation between the true and predicted targets.

Based on the evaluation metrics, the Random forest regresor showed a slighlty better performance across all metrics in this multi-target regression task. As the target variable is multi-dimensional, looking at the mean values does provide much useful information, however the individual performance metrics per target indicate, the predicted log2FC for some pertrubation genes are better than for other perturbation genes. However, there are several main issues with these evaluation metrics. The main problem can be seen looking into the boxplots showing the wide scale of the log2FC values for the different genes, which could make all error-based performance metrics difficult to interpret.

```Python
# Model evaluation - validation set

# Ridge regression
#R-squared:  -0.14179773381220684
#Mean absolute error:  1.245088393456913
#Mean squared error:  12.93050721254891
#Root mean squared error: 1.6089768970212894
#Pearson:  0.7961969578295705

# Random forest regression
#R-squared:  0.0890854956818329
#Mean absolute error:  1.1075968113183619
#Mean squared error:  10.272217751606357
#Root mean squared error: 1.4362768737280658
#Pearson:  0.8324481961867805
```

While the performance metrics on the testing perturbations seem comparable to those on the validation set, there are several crucial factors to consider. Firstly, the training perturbations are still present in the target vector and may skew the model performance metrics, since they would contribute more towards the individual metrics. As mentioned, it would be useful to look at the performance metrics for the individual perturbation genes. 

```python
# Model evaluation - testing set

# Ridge regression
#R-squared:  -2.537997574055322
#Mean absolute error:  1.2168186090567799
#Mean squared error:  12.000184783181112
#Root mean squared error: 1.59421253310336
#Pearson:  0.8246698610910892

# Random forest regression
#R-squared:  -0.471253932722074
#Mean absolute error:  1.0875212658101172
#Mean squared error:  9.226183980509036
#Root mean squared error: 1.3534550274305877
#Pearson:  0.8587254365765324
```
Overall, looking at the plotted residuals between the true and predicted values, as well as the scatterplots of true vs. predicted log2FC values, both models perform quite badly. In this case with low training and test performance both models would be considered biased and underfitted.

### Outlook

As with Task 1, the dataset was reduced in favour of faster training times. While performing badly, both the Ridge regression and the Random forst showed some capacity to model the data and there are several alternative approaches that may show better results. An alternate feature engineering strategy, such as adding the top principle components could help the models distinguish between individual perturbation groupings. Another approach would require the use of more complex models, which would be better suited to fit the relation ship between a perturbed gene and a single-cell gene expression profile. 

## References

1. Heumos, L., Schaar, A.C., Lance, C. et al. Best practices for single-cell analysis across modalities. Nat Rev Genet (2023). https://doi.org/10.1038/s41576-023-00586-w
2. Marchetti, F., Legnaro, E., & Guastavino, S. (2025). Multiclass threshold-based classification. arXiv preprint arXiv:2505.11276.
3. https://stats.stackexchange.com/questions/611695/how-to-adjust-the-classification-thresholds-in-a-multiclass-classification-probl
4. https://www.genecards.org/
5. Laffin B and Petrash JM (2012) Expression of the Aldo-Ketoreductases AKR1B1 and AKR1B10 in Human Cancers. Front. Pharmacol. 3:104. doi: 10.3389/fphar.2012.00104
6. Waters JA, Urbano I, Robinson M and House CD (2022) Insulin-like growth factor binding protein 5: Diverse roles in cancer. Front. Oncol. 12:1052457. doi: 10.3389/fonc.2022.1052457


## Neural Network Approaches
(by Nicolas)

### Methods

#### Models

We attempt to fit the conditional space of perturbation effects using multi-layer perceptrons (MLPs), since they can approximate non-linear and additive effects of perturbations.

This has been implemented before in variational autoencoders (VAEs) such as trVAE[^1] or CPA[^2]. Other neural network architectures have also been used for this task, such as generative adversarial networks (GANs) (GRouNdGAN[^3]), graph neural networks (GNNs) (GEARS[^4]), and transformers (scGPT[^5]).

We chose the variational autoencoder for its relatively simple architecture, ease of implementation, and high interpretability. To condition our model, we inject the perturbation condition into the sampled latent embeddings, which serve as input for the decoder. Here, we encode a limited number of perturbations *p* as a one-hot encoding of size *p* × *n*, where *n* is the total number of dimensions/genes in the encoder's input. This allows prediction of perturbation effects for all possible conditions, including unseen perturbations.

As a comparative baseline, we train three MLPs on the same perturbation prediction task. The inputs to these models are:
1. Only expression data
2. Only perturbation condition
3. Both expression data and perturbation condition

#### Data Preparation

We preprocess the data as done in tasks 1 and 2. Additionally, we normalize, log-transform, and select the top 2000 most highly variable genes. As a subset of perturbation conditions, we choose the 50 most common perturbations called in the dataset. To allow for effective generalization, we use the union of highly variable genes and top perturbations as the input for the encoder. Samples with no perturbation are included in the processed dataset.

We split the data into training and test datasets, with 40 perturbations plus control in the training dataset and 10 perturbations in the test dataset. We then further split the training set into training and validation sets, while stratifying by perturbation condition.

### Discussion

We trained and tested three baseline MLP models on the task of perturbation effect prediction. Here, we observe rapid overfitting in the model trained only on expression profiles. Adding the perturbation condition to the input seemed to prevent overfitting but resulted in a decrease in correlation when predicting the test set.

Our cVAE model required careful calibration of the Kullback-Leibler (KL) weighting warmup to prevent collapse of the latent distribution. The model performed adequately in reconstructing expression values in both the test and validation sets, but lacked the ability to predict the LFC in the test set. This can be partly explained by the primitive mechanism of LFC prediction, which involves conditioning the decoder on control and perturbation and calculating the cell-specific LFC. Success in this task would suggest that the model parameterizes the underlying gene regulatory network (GRN), which is extremely challenging, even for state-of-the-art models. Further analysis of the model's predictions might uncover avenues for improvement, but we are limited in this task by time and resources. Moreover, recent efforts in benchmarking perturbation modeling techniques bring the use of deep neural networks into question. One example is Ahlmann-Eltze et al.[^6], who compared sophisticated deep learning approaches to deliberately simple linear baselines.

A more in-depth treatment of this problem would involve testing models through cross-validation, comparing them to linear baselines, and evaluating published models such as scVI and PCA, as suggested by Bendidi et al.[^7]

---

[^1]: Lotfollahi, M., Naghipourfar, M., Theis, F. J., & Wolf, F. A. (2020). Conditional out-of-distribution generation for unpaired data using transfer VAE. *Bioinformatics*.
[^2]: Lotfollahi, M., Klimovskaia Susmelj, A., De Donno, C., Hetzel, L., Ji, Y., Ibarra, I. L., ... Theis, F. J. (2023). Predicting cellular responses to complex perturbations in high-throughput screens. *Molecular Systems Biology*, e11517.
[^3]: Zinati, Y., Takiddeen, A., & Emad, A. (2024). GRouNdGAN: GRN-guided simulation of single-cell RNA-seq data using causal generative adversarial networks. *Nature Communications*, 15, 4055.
[^4]: Roohani, Y., Huang, K., & Leskovec, J. (2024). Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology*, 42, 927–935.
[^5]: Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., & Wang, B. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, 21, 1470–1480.
[^6]: Ahlmann-Eltze, C., Huber, W., & Anders, S. (2025). Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nature Methods*, 22, 1657–1661.
[^7]: Bendidi, I., Whitfield, S., Kenyon-Dean, K., Ben Yedder, H., El Mesbahi, Y., Noutahi, E., & Denton, A. K. (2024). Benchmarking transcriptomics foundation models for perturbation analysis: one PCA still rules them all. *arXiv preprint* arXiv:2410.13956.