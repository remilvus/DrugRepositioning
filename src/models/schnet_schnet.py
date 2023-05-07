import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch_geometric.nn import SchNet


class SchnetSchnet(pl.LightningModule):
    def __init__(self,
                 lr=1e-3,
                 **kwargs):
        super(SchnetSchnet, self).__init__()
        self.save_hyperparams()
        self.schnet_ligand = SchNet(hidden_channels=64, num_filters=64, num_interactions=4)
        self.schnet_target = SchNet(hidden_channels=64, num_filters=64, num_interactions=4)
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        target_embedding = self.schnet_target(x.ligand_features)
        ligand_embedding = self.schnet_ligand(x.ligand_features)
        combined = torch.cat([target_embedding, ligand_embedding], dim=1)
        output = self.linear(combined)
        return output

    def training_step(self, batch, batch_idx):
        y = torch.cat([batch.activity, batch.binding_score], dim=1)
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
