## Perturbation Effect Prediction

### Task

Can a model predict transcriptome changes for a gene that was not knocked out in the training data?

Before model training, select a subset of 50 perturbations for modeling and explain your choice. Of these, select a subset of perturbations that are held out, i.e., not used during training or hyperparameter selection, only at test time. If the number of selected genes is still too high for the model you want to train, you may subset it further and explain your choice. You may also use unperturbed cells.

The target variable to predict by your models is the mean log2 fold change (LFC) per condition and perturbation target. Train two (if one student), three (if two students), or four (if three students) types of models. Here, a type of model is a feature engineering strategy combined with a learning algorithm. At least one of the models should be deliberately simplistic. Evaluate the model performance and uncertainty based on suitable metrics and interpret the results biologically. You may also use the protein data, however, this is not required. Students working alone only need to use data from the co-culture condition.

## Neural Network Approach

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