import torch
from torch import nn

from src.models.FlexSchNet import Flexible_latent_SchNet

from src.models.common import CrossAttentionType, CrossAttentionLayer, Head
from src.datamodules import DataBatch
from src.huggingmolecules import RMatConfig
from src.huggingmolecules.models import RMatModel
from src.huggingmolecules.models.models_common_utils import clones
from src.models.rmat_rmat import RmatRmatModel


class SchnetRmatModel(RmatRmatModel):
    def __init__(self,
                 lr: float = 1e-3,
                 cross_attention_type: CrossAttentionType = CrossAttentionType.NONE,
                 rmat_config: RMatConfig = RMatConfig.get_default(use_bonds=False),
                 latent_size: int = None,
                 schnet_latent_size: int = 128,
                 targets=[],
                 thresholds={},
                 activity_importance=0.0,
                 **kwargs,
                 ):
        if latent_size is None:
            latent_size = (rmat_config.d_model+schnet_latent_size)//2
        super().__init__(self,
                         cross_attention_type=cross_attention_type,
                         rmat_config=rmat_config,
                         latent_size=latent_size,
                         targets=targets,
                         thresholds=thresholds,
                         activity_importance=activity_importance,
                         **kwargs)
        # TODO: config?
        self.target_net = Flexible_latent_SchNet(latent_size=schnet_latent_size,
                                                 num_interactions=rmat_config.encoder_n_layers
                                                 )
        in_size = rmat_config.d_model + schnet_latent_size
        self.aggregator = nn.Sequential(
            nn.Linear(
                in_features=in_size,
                out_features=in_size,
            ),
            # nn.BatchNorm1d(2 * rmat_config.d_model),
            nn.ReLU(),
            nn.Linear(in_features=in_size, out_features=latent_size),
            # nn.BatchNorm1d(latent_size),
        )

    def forward(self, x):
        ligand_batch = x["data"].ligand_features
        target_batch = x["data"].target_features
        # PL logger requires it to be float
        self.log(
            "train_num_ligand_nodes",
            float(x["data"].ligand_features.node_features.size(1)),
        )
        # self.log(
        #     "train_num_target_nodes",
        #     float(x["data"].target_features.node_features.size(1)),
        # )

        ligand_batch_mask = (
                torch.sum(torch.abs(ligand_batch.node_features), dim=-1) != 0
        )
        # target_batch_mask = (
        #         torch.sum(torch.abs(target_batch.node_features), dim=-1) != 0
        # )

        ligand_latent = self.ligand_net.src_embed(ligand_batch.node_features)
        target_latent = self.target_net(z=target_batch.node_z,
                                        pos=target_batch.node_pos,
                                        batch=target_batch.batch)

        ligand_distances_matrix = self.ligand_net.dist_rbf(
            ligand_batch.distance_matrix
        )
        # target_distances_matrix = self.target_net.dist_rbf(
        #     target_batch.distance_matrix
        # )

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
            ligand_edges_att = target_edges_att = None

        for (
                ligand_rmat_encoder_layer,
                #target_rmat_encoder_layer,
                ligand_ca_layer,
                #target_ca_layer,
        ) in zip(
            self.ligand_net.encoder.layers,
            #self.target_net.encoder.layers,
            self.ligand_ca_layers,
            #self.target_ca_layers,
        ):
            ligand_latent = ligand_rmat_encoder_layer(
                ligand_latent, ligand_batch_mask, edges_att=ligand_edges_att
            )
            # target_latent = target_rmat_encoder_layer(
            #     target_latent, target_batch_mask, edges_att=target_edges_att
            # )

            if False and self.hparams.cross_attention_type in [
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
            if False and self.hparams.cross_attention_type in [
                CrossAttentionType.TARGET,
                CrossAttentionType.BOTH,
            ]:
                pass
                # new_target_latent = target_ca_layer(
                #     ligand_latent,
                #     target_latent,
                #     target_batch_mask,
                #     edges_att=ligand_edges_att,
                #     edges_att_v=target_edges_att,
                # )
            else:
                new_target_latent = target_latent
            ligand_latent, target_latent = new_ligand_latent, new_target_latent

        ligand_encoded = self.ligand_net.encoder.norm(ligand_latent)
        # target_encoded = self.target_net.encoder.norm(target_latent)
        target_encoded = target_latent
        # print(torch.mean(target_latent))
        # print(target_encoded.shape,ligand_encoded.shape)
        # Aggregating from dummy node
        latent_code = self.aggregator(
            torch.cat([ligand_encoded[:, 0, :], target_encoded], dim=1)
        )
        output = self.get_head_outputs(latent_code, x)

        return output
