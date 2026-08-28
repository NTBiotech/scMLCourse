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

### Feature importance

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
