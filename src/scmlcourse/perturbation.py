import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

class DataSet(Dataset):
    """Per-cell dataset yielding (expression, target log-fold-change signature) pairs.

    For a control cell (no perturbation), the target is an all-zero vector;
    otherwise it is the perturbed gene's column from `fold_change_mtx`.
    """

    def __init__(self, adata, fold_change_mtx):
        self.adata = adata
        self.fold_change_mtx = fold_change_mtx

    def __getitem__(self, index):
        """Return (expression, target log-fold-change signature) for the cell at `index`."""
        name = self.adata.obs_names[index]
        x = self.adata[name].X.toarray()
        counts = torch.as_tensor(x).squeeze(0).float()
        target = self.adata.obs.loc[name, "perturbed_gene"]
        if pd.isna(target) or target.strip().lower() in ["ctrl", "control"]:
            y = torch.zeros(self.adata.n_vars)
        else:
            y = torch.as_tensor(np.array(self.fold_change_mtx[target]))
        return counts, y

    def __len__(self):
        """Return the number of cells."""
        return self.adata.n_obs

class DataModule(pl.LightningDataModule):
    """Lightning DataModule for perturbation-effect prediction.

    Loads `adata_path`, restricts to the `n_pert` most frequent perturbed
    genes plus highly variable genes, computes per-gene log-fold-change
    signatures vs. control cells (via `rank_genes_groups`), holds out
    `n_test_pert` perturbations as an unseen test set, and splits the
    remaining perturbations into stratified train/val sets.
    """

    def __init__(self, adata_path:Path, n_pert=50, n_test_pert=10, val_size=0.1, batch_size=100, retain_adata=False, control="control", n_top_genes=1000, n_workers=4):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

        self.adata = sc.read_h5ad(adata_path)

        self.adata.obs["perturbed_gene"] = self.adata.obs["perturbation"].map(lambda x: x.split("_")[0] if isinstance(x, str) else x)
        self.adata.obs["perturbed_gene"] = self.adata.obs[["perturbed_gene","perturbation_2"]].apply(lambda x: str(x["perturbed_gene"]+s), axis=1)
        counts_per_perturbation = self.adata.obs["perturbed_gene"].value_counts()
        target_genes = counts_per_perturbation.sort_values(ascending=False).index[:n_pert+1] #assume control is in the most populated

        self.adata = self.adata[self.adata.obs["perturbed_gene"].map(lambda x: x in target_genes or pd.isna(x))]

        sc.pp.normalize_total(self.adata, target_sum=1e6)
        sc.pp.log1p(self.adata)
        sc.pp.highly_variable_genes(self.adata,n_top_genes=n_top_genes)
        self.adata = self.adata[:,(self.adata.var["highly_variable"] | self.adata.var_names.map(lambda x: x in target_genes))]

        # select test set genes
        test_genes = np.random.choice([x for x in target_genes if x != control], n_test_pert, replace=False)
        train_genes = target_genes[(test_genes[None, :] != target_genes.to_numpy()[:,None]).all(1)]
        
        self.adata.obs["perturbed_gene"] = self.adata.obs["perturbed_gene"].map(lambda x: "Ctrl" if pd.isna(x) else x)
        sc.tl.rank_genes_groups(self.adata, "perturbed_gene", use_raw=False, reference=control)

        df = sc.get.rank_genes_groups_df(self.adata, group=None)
        logfoldchanges = (df.pivot(index="names", columns="group", values="logfoldchanges")
                .reindex(self.adata.var_names))   # now rows == var order
        
        print(test_genes, train_genes)
        train_adata = self.adata[self.adata.obs["perturbed_gene"].map(lambda x: x in train_genes or pd.isna(x))]
        test_adata = self.adata[self.adata.obs["perturbed_gene"].map(lambda x: x in test_genes)]
        self.n_vars = self.adata.n_vars
        if not retain_adata:
            del self.adata
        # split val train
        perturbations = train_adata.obs["perturbed_gene"].to_numpy()
        perturbations[pd.isna(perturbations)] = "Ctrl"
        train_obs, val_obs = train_test_split(train_adata.obs_names,test_size=val_size, stratify=perturbations)

        self.test_dataset = DataSet(adata = test_adata, fold_change_mtx=logfoldchanges[test_genes])
        self.train_dataset = DataSet(adata=train_adata[train_obs], fold_change_mtx=logfoldchanges[[x for x in train_genes if x != control]])
        self.val_dataset = DataSet(adata=train_adata[val_obs], fold_change_mtx=logfoldchanges[[x for x in train_genes if x != control]])

    def train_dataloader(self):
        """Return the DataLoader for the training split."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers
        )
    def val_dataloader(self):
        """Return the DataLoader for the validation split."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
        )
    def test_dataloader(self):
        """Return the DataLoader for the held-out test-perturbation split."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers
        )

def train(
    model,
    data_module,
    max_epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-3,
    hidden_dims: tuple = (512, 512),
    log_dir: str = "logs",
    run_name: str = "pert_regressor",
    seed: int = 0,
):
    """Train `model` on `data_module` with early stopping and checkpointing.

    Sets up CSV logging, early stopping / model checkpointing / LR
    monitoring callbacks, fits the model, then evaluates the best
    checkpoint on the test set. Returns `(model, test_scores)`.
    """
    pl.seed_everything(seed, workers=True)
 
 
    loggers = [
        CSVLogger(save_dir=log_dir, name=run_name),
    ]
 
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            min_delta=1e-4,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=log_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="{epoch}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
 
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=True,
    )
 
    trainer.fit(model, datamodule=data_module)
    test_scores = trainer.test(model, datamodule=data_module, ckpt_path="best")
    return model, test_scores


class Baseline(pl.LightningModule):
    """MLP regressor predicting a cell's perturbation log-fold-change signature from its expression."""

    def __init__(self, mlp_kwargs:dict, loss=nn.MSELoss, lr=1e-5):
        super().__init__()
        self.module = MLP(**mlp_kwargs, )
        self.loss = loss()
        self.lr = lr

    def _step(self, batch, batch_idx, stage:str):
        """Compute loss and log loss/Pearson metrics for a train/val/test batch."""
        x, log_fold_change = batch
        y = self.module(x)
        loss = self.loss(log_fold_change, y)
        bs = x.size(0)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=bs,
                 on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_pearson", self._pearson(y, log_fold_change), prog_bar=(stage != "train"),
                 batch_size=bs, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Run one training step."""
        self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Run one validation step."""
        self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Run one test step."""
        self._step(batch, batch_idx, "test")

    @staticmethod
    def _pearson(pred, target):
        """Mean per-sample Pearson correlation between predicted and true signatures."""
        p = pred - pred.mean(dim=1, keepdim=True)
        t = target - target.mean(dim=1, keepdim=True)
        num = (p * t).sum(dim=1)
        den = p.norm(dim=1) * t.norm(dim=1) + 1e-8
        return (num / den).mean()

    def configure_optimizers(self,):
        """Return an Adam optimizer with a val_loss-monitored ReduceLROnPlateau scheduler."""
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


class MLP(nn.Module):
    """Feed-forward network of Linear-activation-Dropout blocks with a final Linear output layer."""

    def __init__(self, in_dim, out_dim, hidden_dims=(512, 512), dropout=0.1,
                 activation=nn.ReLU, out_activation=None) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), activation(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], out_dim))
        if out_activation is not None:
            layers.append(out_activation())
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        """Apply the network to input `x`."""
        return self.net(x)
 
