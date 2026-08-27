## Perturbation Effect Prediction
### Task
Can a model predict transcriptome changes for a gene that was not knocked out in the
training data?

Before model training, select a subset of 50 perturbations for modeling and explain your
choice. Of these, select a subset of perturbations that are held out, i.e., not used during training or hyperparameter selection, only at test time. If the number of selected genes is still too high for the model you want to train, you may subset it further and explain your choice.
You may also use unperturbed cells.

The target variable to predict by your models is the mean log2 fold change per condition and perturbation target. Train two (if one student), three (if two students) or four (if three students) types of models. Here, a type of model is a feature engineering strategy combined with a learning algorithm. At least one of the models should be deliberately simplistic. Evaluate the model performance and uncertainty based on suitable metrics and interpret the results biologically.
You may also use the protein data, however, this is not required. Students working alone only need to use data from the co-culture condition.
## Neural Network Approach:
### Methods
#### Models
We attempt to fit the conditional space of perturbation effects using multi-layer-perceptrons (MLP), since they can approximate non-linear and additive effects of perturbations.

This has been implemented before in variational autoencoders (VAEs) such as, trVAE or CPA.
Other neural network architectures such as generative adversarial networks (GANs) (GRouNdGAN), graph-neural networks (GNNs) (GEARS) and transformers (scGPT). 
We choose the variational autoencoder for it's relatively simple architecture, ease of implementation and high interpretability. To condition our model, we inject the perturbation condition into the sampled latent embeddings, which serve as input for the decoder. Here we use encode a limited number of perturbations p in a on-hot encoding of size p x n, where n is the total number of dimensions/genes in the encoders input. This allows prediction of perturbation effect for all possible conditions, including unseen perturbations.

As a comparative baseline we train three MLPs on the same task of perturbation. The inputs to these models are:
1. Only expression data
2. Only perturbation condition
3. Both expression data and perturbation condition

#### Data Preparation
We preprocess data as done in task 1 and 2. Additionally, we normalize, log-transform and select the top 1000 most highly variable genes. As a subset of perturbation conditions, we choose the 50 most common perturbations called in the dataset. To allow for effective generalization, we use the union of highly variable genes and top perturbations as the input for the encoder. Samples with no perturbation are included in the processed dataset.

### Results
