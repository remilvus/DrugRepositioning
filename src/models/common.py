from enum import Enum

import torch
from torch import nn

from src.huggingmolecules import RMatConfig
from src.huggingmolecules.models.models_common_utils import MultiHeadedAttention

from src.huggingmolecules.models.models_rmat import RMatAttention


class CrossAttentionType(Enum):
    NONE = 0
    TARGET = 1
    LIGAND = 2
    BOTH = 3


class CrossAttentionLayer(nn.Module):
    def __init__(self, rmat_config: RMatConfig):
        super(CrossAttentionLayer, self).__init__()

        self.cross_attn = MultiHeadedAttention(
            h=rmat_config.encoder_n_attn_heads,
            d_model=rmat_config.d_model,
            dropout=rmat_config.dropout,
            attention=RMatAttention(rmat_config),
        )
        self.vk_norm = nn.LayerNorm(rmat_config.d_model)
        self.q_norm = nn.LayerNorm(rmat_config.d_model)
        self.dropout = nn.Dropout(rmat_config.dropout)

        self.size = rmat_config.d_model

    def forward(self, vk: torch.Tensor, q: torch.Tensor, mask: torch.Tensor, **kwargs):
        x = vk
        vk = self.vk_norm(vk)
        q = self.q_norm(q)
        a = self.cross_attn(q, vk, vk, mask=mask, **kwargs)
        return x + self.dropout(a)
