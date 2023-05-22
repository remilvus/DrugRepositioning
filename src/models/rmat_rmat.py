import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim

from common import CrossAttentionType, CrossAttentionLayer
from src.datamodules import DataBatch
from src.huggingmolecules import RMatConfig
from src.huggingmolecules.models import RMatModel
from src.huggingmolecules.models.models_common_utils import clones


class RmatRmatModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        cross_attention_type: CrossAttentionType = "NONE",
        rmat_config: RMatConfig = RMatConfig(),
        **kwargs
    ):
        super(RmatRmatModel, self).__init__()
        self.save_hyperparameters()

        # Base RMats
        self.ligand_rmat = RMatModel(rmat_config)
        self.target_rmat = RMatModel(rmat_config)

        # Cross-attention layers
        ca_layer = CrossAttentionLayer(rmat_config)
        self.ligand_ca_layers = clones(ca_layer, rmat_config.encoder_n_layers)
        self.target_ca_layers = clones(ca_layer, rmat_config.encoder_n_layers)

        # Aggregator producing the final outputs from the RMats outputs
        # TODO: calc in features
        self.aggregator = nn.Sequential(nn.Linear(in_features=..., out_features=2))

    def forward(self, x: DataBatch):
        self.log("train_num_ligand_nodes", x.ligand_features.node_z.size(0))
        self.log("train_num_target_nodes", x.target_features.node_z.size(0))

        ligand_batch = x.ligand_features
        target_batch = x.target_features

        ligand_batch_mask = (
            torch.sum(torch.abs(ligand_batch.node_features), dim=-1) != 0
        )
        target_batch_mask = (
            torch.sum(torch.abs(target_batch.node_features), dim=-1) != 0
        )

        ligand_latent = self.ligand_rmat.src_embed(ligand_batch.node_features)
        target_latent = self.target_rmat.src_embed(target_batch.node_features)

        ligand_distances_matrix = self.ligand_rmat.dist_rbf(
            ligand_batch.distance_matrix
        )
        target_distances_matrix = self.target_rmat.dist_rbf(
            target_batch.distance_matrix
        )

        ligand_edges_att = torch.cat(
            (
                ligand_batch.bond_features,
                ligand_batch.relative_matrix,
                ligand_distances_matrix,
            ),
            dim=1,
        )
        target_edges_att = torch.cat(
            (
                target_batch.bond_features,
                target_batch.relative_matrix,
                target_distances_matrix,
            ),
            dim=1,
        )

        for (
            ligand_rmat_encoder_layer,
            target_rmat_encoder_layer,
            ligand_ca_layer,
            target_ca_layer,
        ) in zip(
            self.ligand_rmat.encoder.layers,
            self.target_rmat.encoder.layers,
            self.ligand_ca_layers,
            self.target_ca_layers,
        ):
            ligand_latent = ligand_rmat_encoder_layer(
                ligand_latent, ligand_batch_mask, edges_att=ligand_edges_att
            )
            target_latent = target_rmat_encoder_layer(
                target_latent, target_batch_mask, edges_att=target_edges_att
            )

            if self.hparams.cross_attention_type in [
                CrossAttentionType.LIGAND,
                CrossAttentionType.BOTH,
            ]:
                new_ligand_latent = ligand_ca_layer(
                    target_latent,
                    ligand_latent,
                    ligand_batch_mask,
                    edges_att=target_edges_att,
                    edges_att_v=ligand_edges_att,
                )
            else:
                new_ligand_latent = ligand_latent
            if self.hparams.cross_attention_type in [
                CrossAttentionType.TARGET,
                CrossAttentionType.BOTH,
            ]:
                new_target_latent = target_ca_layer(
                    ligand_latent,
                    target_latent,
                    target_batch_mask,
                    edges_att=ligand_edges_att,
                    edges_att_v=target_edges_att,
                )
            else:
                new_target_latent = target_latent
            ligand_latent, target_latent = new_ligand_latent, new_target_latent

        ligand_encoded = self.ligand_rmat.encoder.norm(ligand_latent)
        target_encoded = self.target_rmat.encoder.norm(target_latent)

        output = self.aggregator(ligand_encoded, target_encoded)

        return output

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
