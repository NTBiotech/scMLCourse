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
        # raw (pre-normalization) counts are the model input
        self.X = torch.as_tensor(adata.layers["raw"].toarray()).float()
        self.pert = adata.obs["perturbed_condition"].to_numpy()
        n_cells, n_genes = adata.n_obs, adata.n_vars

        fc = fold_change_mtx.to_numpy()          # (n_genes, n_perturbations)
        col_idx = {g: i for i, g in enumerate(fold_change_mtx.columns)}

        # target: the perturbed condition's logFC signature, or all-zero for
        # control cells (no matching column in fold_change_mtx)
        y = np.zeros((n_cells, n_genes), dtype=np.float32)
        for i, target in enumerate(self.pert):
            if not (pd.isna(target) or ("ctrl" in str(target).strip().lower()) or  ("control" in str(target).strip().lower())):
                y[i] = fc[:, col_idx[target]]

        self.y = torch.as_tensor(y).float()
        # one hot encode perturbations
        self.pert = (adata.var_names.to_numpy()[None,:] == adata.obs["perturbed_gene"].to_numpy()[:,None])

    def __getitem__(self, index):
        """Return (raw expression, target logFC signature, one-hot perturbed gene) for the cell at `index`."""
        return self.X[index], self.y[index], self.pert[index]
        #name = self.adata.obs_names[index]
        #x = self.adata[name].X.toarray()
        #counts = torch.as_tensor(x).squeeze(0).float()
        #target = self.adata.obs.loc[name, "perturbed_gene"]
        #if pd.isna(target) or target.strip().lower() in ["ctrl", "control"]:
        #    y = torch.zeros(self.adata.n_vars)
        #else:
        #    y = torch.as_tensor(np.array(self.fold_change_mtx[target]))
        #return counts, y

    def __len__(self):
        """Return the number of cells."""
        return len(self.X)

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
        self.conditions = np.unique(self.adata.obs["perturbation_2"])
        # "perturbed_gene": the target gene only; "perturbed_condition": gene+experimental
        # condition, since a gene's effect can differ across conditions (e.g. co-culture vs. mono-culture)
        self.adata.obs["perturbed_gene"] = self.adata.obs["perturbation"].map(lambda x: x.split("_")[0] if isinstance(x, str) else x)
        self.adata.obs["perturbed_condition"] = self.adata.obs[["perturbed_gene","perturbation_2"]].apply(lambda x: str(x["perturbed_gene"]+"_"+str(x["perturbation_2"])), axis=1)

        counts_per_perturbation = self.adata.obs["perturbed_gene"].value_counts()
        ranking = counts_per_perturbation.sort_values(ascending=False).index
        # filter out perturbed genes not in the variables
        mask = ranking.map(lambda x: (x in self.adata.var_names) or (x==control))
        _target_genes = ranking[mask][:n_pert+1] #assume control is in the most populated

        # restrict to cells with one of the top n_pert perturbed genes (+ control)
        self.adata = self.adata[self.adata.obs["perturbed_gene"].map(lambda x: x in _target_genes)]
        self.adata.layers["raw"] = self.adata.X.copy()

        sc.pp.normalize_total(self.adata, target_sum=1e6)
        sc.pp.log1p(self.adata)
        sc.pp.highly_variable_genes(self.adata,n_top_genes=n_top_genes)

        # keep highly variable genes plus the target genes themselves (so
        # every perturbed gene appears in the model's input/output space)
        mask = (self.adata.var["highly_variable"] | self.adata.var_names.map(lambda x: x in _target_genes))
        self.adata = self.adata[:,mask]

        # hold out n_test_pert perturbed genes entirely (never seen in training)
        self._test_genes = np.random.choice([x for x in _target_genes if x.split("_")[0] != control], n_test_pert, replace=False)
        self._train_genes = _target_genes[(self._test_genes[None, :] != _target_genes.to_numpy()[:,None]).all(1)]
        # expand each gene into one perturbed_condition entry per experimental condition
        test_genes = []
        for g in self._test_genes:
            test_genes.extend([f"{g}_{c}" for c in self.conditions])
        train_genes = []
        for g in self._train_genes:
            train_genes.extend([f"{g}_{c}" for c in self.conditions])
        logfoldchanges = []
        # compute each perturbed_condition's logFC signature vs. its matching
        # control, separately per experimental condition (since the control
        # baseline differs between conditions)
        for c in self.conditions:
            subset = self.adata[self.adata.obs["perturbation_2"]==c].copy()
            print(c,subset.obs["perturbed_condition"])
            sc.tl.rank_genes_groups(subset, "perturbed_condition", reference=f"{control}_{c}", method="wilcoxon", use_raw=False)
            df = sc.get.rank_genes_groups_df(subset, group=None)
            print(df.head())
            logfoldchanges.append(df.pivot(index="names", columns="group", values="logfoldchanges")
                    .reindex(self.adata.var_names))   # now rows == var order
        logfoldchanges = pd.concat(logfoldchanges, axis=1)
        print(f"logfoldchanges has shape {logfoldchanges.shape}")
        print(test_genes, train_genes)
        train_adata = self.adata[self.adata.obs["perturbed_condition"].map(lambda x: x in train_genes or pd.isna(x))]
        test_adata = self.adata[self.adata.obs["perturbed_condition"].map(lambda x: x in test_genes)]
        self.n_vars = self.adata.n_vars
        self.test_genes = test_genes
        self.train_genes = train_genes
        if not retain_adata:
            del self.adata
        # split val train, stratified so each perturbed_condition is
        # represented proportionally in both splits
        perturbations = train_adata.obs["perturbed_condition"].values
        train_obs, val_obs = train_test_split(train_adata.obs_names,test_size=val_size, stratify=perturbations)

        self.test_dataset = DataSet(adata = test_adata, fold_change_mtx=logfoldchanges[test_genes])
        self.train_dataset = DataSet(adata=train_adata[train_obs], fold_change_mtx=logfoldchanges[[x for x in train_genes if control not in x]])
        self.val_dataset = DataSet(adata=train_adata[val_obs], fold_change_mtx=logfoldchanges[[x for x in train_genes if control not in x]])

    def train_dataloader(self):
        """Return the DataLoader for the training split."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            persistent_workers=True,
        )
    def val_dataloader(self):
        """Return the DataLoader for the validation split."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            persistent_workers=True,
        )
    def test_dataloader(self):
        """Return the DataLoader for the held-out test-perturbation split."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            persistent_workers=True,
        )

def train(
    model,
    data_module,
    max_epochs: int = 100,
    patience: int = 15,
    log_dir: str = "logs",
    run_name: str = "pert_regressor",
    seed: int = 0,
    test_on:str="best",
    **kwargs
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
            dirpath=loggers[0].log_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="{epoch}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    print(f"Training for max {max_epochs} epochs...")
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
        **kwargs
    )
 
    trainer.fit(model, datamodule=data_module)
    test_scores = trainer.test(model, datamodule=data_module, ckpt_path=test_on)

    return model, test_scores, loggers[0].log_dir


class Baseline(pl.LightningModule):
    """MLP regressor predicting a cell's perturbation log-fold-change signature from its expression."""

    def __init__(self, mlp_kwargs:dict, loss=nn.MSELoss, lr=0.001):
        super().__init__()
        self.module = MLP(**mlp_kwargs, )
        self.loss = loss()
        self.lr = lr

    def _step(self, batch, batch_idx, stage:str):
        """Compute loss and log loss/Pearson metrics for a train/val/test batch."""
        x, log_fold_change, pert = batch
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
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Run one validation step."""
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Run one test step."""
        return self._step(batch, batch_idx, "test")

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


class condMLP(Baseline):
    '''MLP regressor predicting log-fold-change from perturbation condition.'''
    def _step(self, batch, batch_idx, stage:str):
            """Compute loss and log loss/Pearson metrics for a train/val/test batch."""
            x, log_fold_change, pert = batch
            y = self.module(pert.to(torch.float32))
            loss = self.loss(log_fold_change, y)
            bs = x.size(0)
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=bs,
                    on_step=(stage == "train"), on_epoch=True)
            self.log(f"{stage}_pearson", self._pearson(y, log_fold_change), prog_bar=(stage != "train"),
                    batch_size=bs, on_step=False, on_epoch=True)
            return loss

class pertMLP(Baseline):
    '''MLP regressor predicting log-fold-change from perturbation condition and expression data'''
    def _step(self, batch, batch_idx, stage:str):
            """Compute loss and log loss/Pearson metrics for a train/val/test batch."""
            x, log_fold_change, pert = batch
            y = self.module(torch.concat((x, pert.to(torch.float32)), dim=1))
            loss = self.loss(log_fold_change, y)
            bs = x.size(0)
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=bs,
                    on_step=(stage == "train"), on_epoch=True)
            self.log(f"{stage}_pearson", self._pearson(y, log_fold_change), prog_bar=(stage != "train"),
                    batch_size=bs, on_step=False, on_epoch=True)
            return loss


class cVAE(Baseline, pl.LightningModule):
    '''Conditional VAE that reconstructs expression conditioned on the perturbed gene.

    Encodes raw expression `x` to a latent Gaussian, decodes
    `[latent, perturbation_one_hot]` back to expression, and derives the
    predicted logFC signature at test time as `log2(decoded / decoded_ctrl)`
    (see `predict_fc`), where `decoded_ctrl` decodes the same latent with an
    all-zero (control) perturbation vector.
    '''
    def __init__(self, in_dim, latent_dim, encoder_kwargs:dict, decoder_kwargs:dict, lr=1e-5, pretrain_epochs=0, kl_midpoint=20, kl_slope=1):
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        encoder_kwargs["in_dim"] = in_dim
        decoder_kwargs["out_dim"] = in_dim
        # decoder input is encoder output + one_hot of perturbation
        encoder_kwargs["out_dim"] = latent_dim
        decoder_kwargs["in_dim"] = latent_dim+in_dim
        decoder_kwargs["out_activation"] = nn.ReLU
        encoder_kwargs["out_activation"] = None
        super().__init__(decoder_kwargs, lr=lr)
        
        print(encoder_kwargs)
        self.mu_encoder = MLP(**encoder_kwargs)
        print(encoder_kwargs)
        self.sigma_encoder = MLP(**encoder_kwargs)
        print(decoder_kwargs)
        self.decoder = self.module

        self.kl_midpoint = kl_midpoint
        self.kl_slope = kl_slope
        self.kl = 0.00001

        self.pretrain_epochs=pretrain_epochs
        self.pretrain = self.pretrain_epochs>0

    def _step(self, batch, batch_idx, stage:str):
        """Encode/reparameterize/decode a batch and log the reconstruction + KL loss.

        Loss is `mse(x, x_hat) + kl * self.kl` (`self.kl` is annealed by
        `kl_schedule`/`on_train_epoch_start`). Also logs the encoder/decoder
        statistics (min/max/mean of mu, logvar, latent, x_hat) during training.
        """
        x, gt_fc, pert = batch
        mu = self.mu_encoder(x)
        logvar = self.sigma_encoder(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        latent = mu + std * torch.randn_like(mu)
        x_hat = self.decoder(torch.concat([latent, pert], axis=1))

        kl = (-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)).mean()
        mse_x = self.loss(x, x_hat)
        loss = mse_x+kl*self.kl
        pearson = self._pearson(x, x_hat)
        
        bs = x.size(0)
        if stage=="train":
            self.log(f"{stage}_mu_min", mu.min(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_mu_max", mu.max(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_mu_mean", mu.mean(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_logvar_min", logvar.min(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_logvar_max", logvar.max(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_logvar_mean", logvar.mean(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_latent_min", latent.min(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_latent_max", latent.max(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_latent_mean", latent.mean(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_x_hat_min", x_hat.min(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_x_hat_max", x_hat.max(), batch_size=bs,on_step=False, on_epoch=True)
            self.log(f"{stage}_x_hat_mean", x_hat.mean(), batch_size=bs,on_step=False, on_epoch=True)
        
        self.log(f"{stage}_pearson", pearson, prog_bar=True, batch_size=bs,
                on_step=False, on_epoch=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=bs,
                on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_loss_kl", kl, prog_bar=True, batch_size=bs,
                on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_loss_mse_x", mse_x, prog_bar=True, batch_size=bs,
                on_step=(stage == "train"), on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Run one test step, evaluating the predicted logFC signature via `predict_fc`."""
        bs = batch[0].size(0)
        fc, loss, pearson_fc, pearson_x = self.predict_fc(batch, batch_idx)
        self.log(f"test_loss", loss, prog_bar=True, batch_size=bs, on_epoch=True)
        self.log(f"pearson_fc", pearson_fc, prog_bar=True, batch_size=bs, on_epoch=True)
        self.log(f"pearson_x", pearson_x, prog_bar=True, batch_size=bs, on_epoch=True)
        return loss

    def on_train_epoch_start(self):
        """Update the KL weight for this epoch per `kl_schedule` and log it."""
        self.kl = self.kl_schedule(self.current_epoch)
        self.log("kl_factor", self.kl, prog_bar=True)
        self.log("lr", self.lr, prog_bar=True)

    def kl_schedule(self, time_step, ):
        """Sigmoid KL-annealing weight, ramping up around epoch `kl_midpoint`."""
        return float(1 / (1. + np.exp(self.kl_slope * (self.kl_midpoint - float(time_step)))))

    def predict_fc(self, batch, batch_idx):
        """Derive the predicted logFC signature from paired perturbed/control decodes.

        Decodes the same latent with the batch's perturbation vector and
        with an all-zero (control) vector, then takes
        `log2(decoded_pert / decoded_ctrl)` as the predicted signature.
        Returns `(predicted_fc, loss_vs_ground_truth_fc, pearson_fc, pearson_x)`.
        """
        stage = "fc_prediction"
        x, gt_fc, pert = batch
        mu = self.mu_encoder(x)
        logvar = self.sigma_encoder(x)
        std = torch.exp(0.5 * logvar)
        latent = mu + std * torch.randn_like(mu)
        x_hat = self.decoder(torch.concat([latent, pert], axis=1))
        x_ctrl_hat = self.decoder(torch.concat([latent, torch.zeros_like(pert)], axis=1))
        fc = self.log2foldchange(x_hat, x_ctrl_hat)
        loss = self.loss(fc, gt_fc)
        pearson_fc = self._pearson(fc, gt_fc)
        pearson_x = self._pearson(x_hat, x)
        return fc, loss, pearson_fc, pearson_x

    def log2foldchange(self, y, ctrl, eps=1e-6) -> torch.Tensor:
        """Elementwise log2(y / ctrl), clamping both to `eps` to avoid log(0)/div-by-0."""
        y = torch.clamp(y, min=eps)
        ctrl = torch.clamp(ctrl, min=eps)
        return torch.log2(y/ctrl)

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