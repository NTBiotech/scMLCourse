# Practical machine learning for single cell multiomics - Seminar report

For this project, we analyzed a CRISPR perturbation scRNA-seq dataset containing the gene expression profiles of melanoma cells across three experimental cell culture conditions. The CRISPR perturbation targeted 248 genes and included an additional single-cell surface protein measurements. 

The dataset was provided as an `AnnData` H5ad file (`rna.h5ad`). In order to solve the given tasks, a perliminary inspection, exploration and quality control processing was done to ensure the high quality of data.

## Quality control processing

The quality control processing was done as shown in the JupyterLab `QC.ipynb` notebook. Briefly, the raw `AnnData` dataset contained the gene expression data for 218 331 cells across 23 712 genes. Quality control was performed based on the discussed criterai during the course and following the recommended Scanpy workflow, as outlined in Chapter 8 in the Single-cell best practices textbook (1). From the initial QC plots we can see:

* `pct_counts_mt`: we can see, that all cells over a thershold of 18 have already been filtered.
* `tota_counts` and `n_genes_by_counts`: There are still some outliers noticable
* `log1p_total_counts`, `log1p_n_genes_by_counts`, `pct_counts_in_top_50_genes`: There are still some outliers noticable

Due to the high number of cells, we can afford to be more strict with our filtering. Cells with less than 200 expressed genes and genes expressed in less than 20 cells were filtered out from the dataset. For the following QC metrics, a threshold of 3 (Initially 5) MADs was set for the following metrics to filter any possible outliers: `total_counts`, `log1p_total_counts`, `n_genes_by_counts`, `log1p_n_genes_by_counts`, `pct_counts_mt`, `pct_counts_in_top_50_genes`.

Note: Doublets would normally be called using the scrublet impementation in scanpy, however running the `sc.pp.scrublet(subset)` line always crahsed the VM for some reason, so the MAD treshold above was lowered to 3 from 5.

Lastly, the dataset was normalized, the top 200 highly variable genes were selected and the `AnnData` Object was exported as `rna.qc.hvg200.h5ad`. For Task 1 - Condition classification, an additional dataset with only the top 100 HVGs was exported.

## Task 1 - Condition Classification

Before we begin with the classification, it is important to perform a brief initial exploratoration and look into the class fractions. Based on their distribution, we can see that the IFNy class (0.442871) is slightly overrepresented in the dataset in comparison to Co-culture (0.279471) and Control (0.277659). Using the QC processed dataset we generated, we can reduce the feature set to the selected 100 HVGs. Due to some issues with kernel crashes during training, I had to cap the dataset size to only 5% of its original size. Increasing the dataset could help improve the model.

### Experimental design 

The `scikit-learn` library provides a useful function (`train_test_split()`), which we can use to split or data into a Train/Validate/Test sets(60%/20%/20%), stratified by `y` to ensure equal class distribution across all sets, as described in the library documentation and user guide. For the classification task, 2 concentional classifiers from `scikit-learn` were selected: a logistic regression model and a decision tree model. The experimental set up was based on `scikit-learn`'s API and was the same for both models. A grid search parameter optimiazation with stratified k-fold cross-validation (k=3) was used to optimize hyperparameters for the models using the training set, on which the best performing model was fitted.

### Model evaluation

Model performance was evaluated on the validation set based on the following metrics:
* **Accuracy**: Correct predictions/ Total predictions. Suscepatble to class imbalances.
* **Precision**: TP/(TP+FP). The proportion of correctly predicted positives among all predicted positives.
* **Recall aka Sensitivity**: TP/(TP+FN); The proportion of actual positives correctly identified.
* **F1 Score**: Balancing mean of precision and recall. 2*((Precision*Recall)/(Precision+Recall)).
* **Area under the ROCurve** (One-vs-Rest) and ROC Plot: Average area under the ROC curve calculated for each class using a one-vs-rest approach. It measures how well the classifier distinguishes each class from all other classes, weighting every class equally.
* **Confusion Matrix**: Summarizes TP, TN, FP, FN counts and rates(nomralized).
The `scikit-learn` library also provised a `classification_report()` function, which calculates the evaluation metrics for each individual class.

```python
#LogisticRegression(C=np.float64(10000.0), l1_ratio=0, max_iter=2000,random_state=1)
#accuracy         : 0.711
#balanced acc.    : 0.708
#f1               : 0.705
#precision        : 0.702
#recall           : 0.708
#macro AUROC (OvR): 0.881
#macro AUCPR      : 0.786

#DecisionTreeClassifier(ccp_alpha=0.001, class_weight='balanced', max_depth=100,min_samples_split=10, random_state=1)
#accuracy         : 0.665
#balanced acc.    : 0.677
#f1               : 0.662
#precision        : 0.67
#recall           : 0.677
#macro AUROC (OvR): 0.816
#macro AUCPR      : 0.677
```
While both models show room for improvemnet, the `LogisticRegression` Classifier seemed to outperform the `DecisionTree` Classifier across all metrics. In a classical classification problem, one conventional stratedy to improve a classifier's performance is to adjust the treshold for prediction between the two classes with the goal to improve model performance (e.g. the `TunedThresholdClassifierCV()` by `scikit-learn`). There have methods published on how to approach this problem in a multi-class classification (2). As a general "proof-of-concept", a simple class-frequency correction was implemented (3). As we can see in the perfomance metrics comparison, while there seems to be a drop in model accurace for both models, the `Logisticregression` model showed slight improvemnts in precision, recall and the overall balanced accuracy. This could be attributed to two main reasons: The class imbalance favours slightly only one class, while the other two are comparable, as discussed in the beginning, and during training, we stratify our training by class to reduce the chances of our model only learning one class. 

```Python
# LogRegression

#adjT_evaluation(y_valid, valid_prob_logreg, valid_pred_logreg, freq, np.unique(y))
#Performance metrics:
#accuracy         : 0.711
#balanced acc.    : 0.708
#f1               : 0.705
#precision        : 0.702
#recall           : 0.708

#Performance metrics using class-frequency adjusted scores:
#accuracy         : 0.703 -
#balanced acc.    : 0.718 +
#f1               : 0.701 -
#precision        : 0.705 +
#recall           : 0.718 +

# Decision Tree
#Performance metrics:
#accuracy         : 0.665
#balanced acc.    : 0.677
#f1               : 0.662
#precision        : 0.67
#recall           : 0.677

#Performance metrics using class-frequency adjusted scores:
#accuracy         : 0.649 -
#balanced acc.    : 0.668 -
#f1               : 0.647 -
#precision        : 0.665 -
#recall           : 0.668 +
```

Evaluating the models on the testing data, we can see a slight imprevemnt for both models across all metrics. Bad performance during both training and testing could indicate a high-biased or underfitted model. Bad model performance during training would be a sign of an overfitted high-variance model. While we see some improvement during testing,  this variability in model performance is still small enough, that it could be attributed to other factors, such as the testing set being "easier" due to random sampling during the train/validation/test splitting. (Note: I also tested this with a larger feature set, i.e. more HVGs, and the models came out almost perfect each time. This I would consider an overfitted high-variability model.)

```python
#LogisticRegression(C=np.float64(10000.0), l1_ratio=0, max_iter=2000, random_state=1)
#accuracy:  0.725
#balanced accuracy:  0.723
#f1:  0.719
#precision:  0.716
#recall:  0.723
#AUROC-OvR:  0.891

#DecisionTreeClassifier(ccp_alpha=0.001, class_weight='balanced', max_depth=100,min_samples_split=10, random_state=1)
#accuracy:  0.685
#balanced accuracy:  0.697
#f1:  0.682
#precision:  0.693
#recall:  0.697
#AUROC-OvR:  0.841
```

### Feature improtance

Lastly, we can look into feature importance to determine which features were most informartive for each model. For both models, we looked into 2 differrnt approaches. 

#### Logistic regression

Starting with `LogisticRegression`, we can look into the coefficients for each feature to determine its contribution to the model. For the 3 conditions, the top 5 most informative features based on model coefficients were:

* Control: CXCL10, CXCL11, CXCL9, TRAC and CCL8
* Co-culture: CXCL10, CXCL11, GNLY, CCL8 and KRT17
* INFy: CXCL10, CXCL11, CCL8, CXCL9 and GNLY

As we can see, the the majority of the most informative features are consistent across all three classes. Genes CXCL10, CXCL11 and CXCL9 had positive coefficients for the conditions Co-culture and IFNy and negative for Control cells, indicating they were crucial for distinguishing between the classes. Looking into the individual entries on the GeneCards database, CXCL10 can play a role as a chemoattractant for T cells and can be expressed as a response to IFN-y. CXCL11 and CXCL9 can also act as chemoattractants for T cells and are induced by IFN-y. The CCL8, GNLY genes were the most informative features separating the Co-culture and IFNy classes. CCL8 is a known chemoattractant for immune cells. GNLY encodes a small saposin-like protein with a cytotoxic effect secreted by various cytotoxic cells, among which T cells and has a known anticancer role.

Investigating the permutation-based feature importance showed an additional informative feature, AKR1B10, a gene encoding aldo-ketoreductase family member and often overexpressed in various cancers among which melanoma (5).

#### Decision tree

Looking into the impurity-based feature importance used by the `Decision tree` classifier, there is some overlap with the most imporant features for the logistic regression model, however, some genes, such as IGFBP5, MSMP and CXCL3 were ranked higher. The insulin-like growth factor binding protein 5 (IGFBP5) is known for its varying effects with studies indicitacing its pro- and anti-tumor effects, but is known to be involved in the progression of multiple cancers, among which melanoma. MSMP has been previously linked to prostate cancer and CXCL3, as the other chemokines is a known chemoatractant molecule involved in the immune response.

### Outlook

While both models' perfomance was good, there are several ways this classification can be improved. Starting from the initial data, our feature selection strategy is very simple. An informed approach, where we incorporate features known to be highly expressed in or characteritic for cancer cells, such as AKR1B10 for instance could help improve our models significantly. Other types of feature engineering, such as incorporating PCs into the feature set might also be beneficial. Another method would be to use a machine learning model with feature restriction term, such as LASSO, to fit on our data in a One-vs-Rest classification, which could help us find the most infomrative features to identify each condition against all others. Other machine learning approaches, such as ensemble learning could also help us improve model performace. 

Lastly, it must be noted that several consessions were made because of technical performance, such as capping the dataset at 5%. The choice model hyperparameters was also made to keep training times managable. Expanding the grid search done form both models could also improve model performance.  


## Task 3 - Perturbation prediction

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

## Refrences

1. Heumos, L., Schaar, A.C., Lance, C. et al. Best practices for single-cell analysis across modalities. Nat Rev Genet (2023). https://doi.org/10.1038/s41576-023-00586-w
2. Marchetti, F., Legnaro, E., & Guastavino, S. (2025). Multiclass threshold-based classification. arXiv preprint arXiv:2505.11276.
3. https://stats.stackexchange.com/questions/611695/how-to-adjust-the-classification-thresholds-in-a-multiclass-classification-probl
4. https://www.genecards.org/
5. Laffin B and Petrash JM (2012) Expression of the Aldo-Ketoreductases AKR1B1 and AKR1B10 in Human Cancers. Front. Pharmacol. 3:104. doi: 10.3389/fphar.2012.00104
6. Waters JA, Urbano I, Robinson M and House CD (2022) Insulin-like growth factor binding protein 5: Diverse roles in cancer. Front. Oncol. 12:1052457. doi: 10.3389/fonc.2022.1052457