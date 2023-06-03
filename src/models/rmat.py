import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim

from src.models.common import CrossAttentionType, CrossAttentionLayer, Head
from src.datamodules import DataBatch
from src.huggingmolecules import RMatConfig
from src.huggingmolecules.models import RMatModel
from src.huggingmolecules.models.models_common_utils import clones


class RmatModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        rmat_config: RMatConfig = RMatConfig.get_default(use_bonds=False),
        latent_size: int = None,
        targets=[],
        thresholds={},
        activity_importance=0.0,
        **kwargs,
    ):
        assert len(targets) > 0
        assert all((target in ["Ki", "IC50", "binding_score"]) for target in targets)
        super(RmatModel, self).__init__()
        self.save_hyperparameters()

        # Base RMats
        self.ligand_rmat = RMatModel(rmat_config)

        # Aggregator producing the final outputs from the RMats outputs
        if latent_size is None:
            latent_size = rmat_config.d_model
        self.aggregator = nn.Sequential(
            nn.Linear(
                in_features=rmat_config.d_model,
                out_features=rmat_config.d_model,
            ),
            # nn.BatchNorm1d(rmat_config.d_model),
            nn.ReLU(),
            nn.Linear(in_features=rmat_config.d_model, out_features=latent_size),
            # nn.BatchNorm1d(latent_size),
        )

        # train head only on values under the threshold.
        # threshold head is trained to discard outputs
        heads = {}
        threshold_heads = {}
        self.thresholds = {}
        self.targets = targets
        for target in targets:
            heads[target] = Head(latent_size, 1, 2)
            if target in thresholds.keys():
                threshold_heads[target] = {}
                threshold_heads[target] = Head(latent_size, 1, 2, "sigmoid")
                self.thresholds[target] = thresholds[target]
        self.heads = nn.ModuleDict(heads)
        self.threshold_heads = nn.ModuleDict(threshold_heads)
        self.class_weights = {
            "Ki": 0.15,
            "IC50": 9.0,
            "binding_score": 0.4,
        }

    def forward(self, batch: DataBatch):
        x, y = self._prepare_masked_batch(batch)

        ligand_batch = x["data"].ligand_features
        # PL logger requires it to be float
        self.log(
            "train_num_ligand_nodes",
            float(x["data"].ligand_features.node_features.size(1)),
        )

        ligand_batch_mask = (
            torch.sum(torch.abs(ligand_batch.node_features), dim=-1) != 0
        )

        ligand_latent = self.ligand_rmat.src_embed(ligand_batch.node_features)
        ligand_distances_matrix = self.ligand_rmat.dist_rbf(
            ligand_batch.distance_matrix
        )

        if self.hparams.rmat_config.use_bonds:
            ligand_edges_att = torch.cat(
                (
                    ligand_batch.bond_features,
                    ligand_batch.relative_matrix,
                    ligand_distances_matrix,
                ),
                dim=1,
            )
        else:
            ligand_edges_att = None

        ligand_encoded = self.ligand_rmat.encoder(
            ligand_latent, ligand_batch_mask, edges_att=ligand_edges_att
        )

        latent_code = self.aggregator(ligand_encoded)  # TODO - the shape could be wrong

        output = {}
        for target in self.targets:
            output[target] = {}

            if target in self.threshold_heads.keys():
                threshold_input = latent_code[x["mask"][target]["threshold"], None]
                out_threshold = self.threshold_heads[target](threshold_input)
                output[target]["threshold"] = out_threshold

            val_input = latent_code[x["mask"][target]["value"], None]
            out_value = self.heads[target](val_input)
            output[target]["value"] = out_value

        return output, y

    def training_step(self, batch: DataBatch, batch_idx):
        y_hat, y = self(batch)
        loss = 0
        for target in self.targets:
            # if whole target was discarded by mask then loss on empty tensors is NaN
            if target in self.threshold_heads and len(y[target]["threshold"]) > 0:
                loss_threshold = nn.functional.binary_cross_entropy(
                    y_hat[target]["threshold"],
                    y[target]["threshold"],
                    weight=self._get_class_weights(target, y),
                )

                loss += loss_threshold
                accuracy, predicted_labels = self._summarise_predictions(
                    target, y, y_hat
                )

                self._log_classification(
                    "train", accuracy, loss_threshold, predicted_labels, target, y
                )

            # if whole target was discarded by mask then loss on empty tensors is NaN
            if len(y[target]["value"]) > 0:
                loss_value = nn.functional.mse_loss(
                    y_hat[target]["value"], y[target]["value"]
                )

                loss_value = self._scale_loss(loss_value, target)
                loss += loss_value

                self.log("train/loss_" + target + "/value", loss_value)

        self.log("train/loss", loss)
        return loss

    def _log_classification(
        self, step_type: str, accuracy, loss_threshold, predicted_labels, target, y
    ):
        self.log(f"{step_type}/loss_{target}/threshold", loss_threshold)
        self.log(f"{step_type}/{target}_threshold_accuracy", accuracy)
        self.log(
            f"{step_type}/{target}_threshold_mean_predicted_label",
            predicted_labels.float().mean(),
        )
        self.log(
            f"{step_type}/{target}_threshold_mean_label",
            y[target]["threshold"].mean(),
        )

    def _summarise_predictions(self, target, y, y_hat):
        predicted_labels = y_hat[target]["threshold"] > 0.5
        accuracy = (predicted_labels == (y[target]["threshold"] > 0.5)).float().mean()
        return accuracy, predicted_labels

    def _scale_loss(self, loss_value, target):
        if target in {"IC50", "Ki"}:
            loss_value *= self.hparams.activity_importance
        return loss_value

    def _get_class_weights(self, target, y) -> torch.Tensor:
        weights = (1 - y[target]["threshold"]) + (
            y[target]["threshold"] * self.class_weights[target]
        )

        return weights.reshape(-1, 1)

    def _prepare_masked_batch(self, batch):
        # WARNING!!!
        # masking assumes y is of shape (batch_size,1)
        y = {}
        x = {"data": batch}
        x["mask"] = {}
        for target in self.targets:
            values = batch.get_regression_target(target)

            x["mask"][target] = {}
            mask = ~(torch.isnan(values))
            y[target] = {}
            if target in self.threshold_heads:
                # for threshold head mask only nans
                x["mask"][target]["threshold"] = mask
                threshold_mask = (self.thresholds[target][0] <= values) & (
                    values <= self.thresholds[target][1]
                )
                # ones * (value < threshold)
                threshold_target = (
                    torch.ones_like(values) * threshold_mask.int().float()
                )
                # filter out entries where value==NaN
                threshold_target = threshold_target[mask, None]
                y[target]["threshold"] = threshold_target
                mask = mask & threshold_mask

            x["mask"][target]["value"] = mask
            y[target]["value"] = values[x["mask"][target]["value"], None]
        return x, y

    def validation_step(self, batch, batch_idx):
        y_hat, y = self(batch)
        loss = 0
        for target in self.targets:
            # if whole target was discarded by mask then loss on empty tensors is NaN
            if target in self.threshold_heads and len(y[target]["threshold"]) > 0:
                loss_threshold = nn.functional.binary_cross_entropy(
                    y_hat[target]["threshold"],
                    y[target]["threshold"],
                    weight=self._get_class_weights(target, y),
                )

                loss += loss_threshold
                accuracy, predicted_labels = self._summarise_predictions(
                    target, y, y_hat
                )

                self._log_classification(
                    "val", accuracy, loss_threshold, predicted_labels, target, y
                )

            # if whole target was discarded by mask then loss on empty tensors is NaN
            if len(y[target]["value"]) > 0:
                loss_value = nn.functional.mse_loss(
                    y_hat[target]["value"], y[target]["value"]
                )

                loss_value = self._scale_loss(loss_value, target)
                loss += loss_value

                self.log("val/loss_" + target + "/value", loss_value)

        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self(batch)
        loss = 0
        for target in self.targets:
            # if whole target was discarded by mask then loss on empty tensors is NaN
            if target in self.threshold_heads and len(y[target]["threshold"]) > 0:
                loss_threshold = nn.functional.binary_cross_entropy(
                    y_hat[target]["threshold"],
                    y[target]["threshold"],
                    weight=self._get_class_weights(target, y),
                )

                loss += loss_threshold
                accuracy, predicted_labels = self._summarise_predictions(
                    target, y, y_hat
                )

                self._log_classification(
                    "test", accuracy, loss_threshold, predicted_labels, target, y
                )

            # if whole target was discarded by mask then loss on empty tensors is NaN
            if len(y[target]["value"]) > 0:
                loss_value = nn.functional.mse_loss(
                    y_hat[target]["value"], y[target]["value"]
                )

                loss_value = self._scale_loss(loss_value, target)
                loss += loss_value

                self.log("test/loss_" + target + "/value", loss_value)

        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
