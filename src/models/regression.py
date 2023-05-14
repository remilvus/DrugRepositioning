from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim

from src.datamodules import DataBatch
from src.huggingmolecules import RMatConfig
from src.huggingmolecules.models import RMatModel
from torch_geometric.nn import SchNet


class CatLayer(pl.LightningModule):
    def forward(self, x: List[torch.Tensor]):
        return torch.cat(x, dim=1)


class RegressionModel(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        cross_attention_head=False,
        ligand_schnet=True,
        target_schnet=True,
        **kwargs
    ):
        super(RegressionModel, self).__init__()
        self.save_hyperparameters()
        self.ligand_schnet = ligand_schnet
        self.target_schnet = target_schnet
        if ligand_schnet:
            self.ligand_encoder = SchNet(
                hidden_channels=64, num_filters=64, num_interactions=4
            )
        else:
            self.ligand_encoder = RMatModel(RMatConfig())

        if target_schnet:
            self.target_encoder = SchNet(
                hidden_channels=64, num_filters=64, num_interactions=4
            )
        else:
            self.target_encoder = RMatModel(RMatConfig())

        if cross_attention_head:
            raise NotImplementedError("")
        else:
            self.net = torch.nn.Sequential(CatLayer(), nn.Linear(2, 2))

    def forward(self, x: DataBatch):
        target_embedding = self.target_encoder(
            x.target_features.node_z,
            x.target_features.node_pos,
            x.target_features.batch,
        )
        ligand_embedding = self.forward_ligand(x)

        output = self.net([target_embedding, ligand_embedding])
        return output

    def forward_target(self, x: DataBatch):
        if self.target_schnet:
            return self.target_encoder(
                x.target_features.node_z,
                x.target_features.node_pos,
                x.target_features.batch,
            )
        else:
            return self.target_encoder(x.ligand_features)

    def forward_ligand(self, x: DataBatch):
        if self.ligand_schnet:
            return self.ligand_encoder(
                x.target_features.node_z,
                x.target_features.node_pos,
                x.target_features.batch,
            )
        else:
            return self.ligand_encoder(x.ligand_features)

    def training_step(self, batch, batch_idx):
        y = torch.cat(
            [batch.activity.unsqueeze(1), batch.binding_score.unsqueeze(1)], dim=1
        )
        y_hat = self(batch)
        loss = nn.functional.mse_loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     pass
    #
    # def test_step(self, batch, batch_idx):
    #     pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
