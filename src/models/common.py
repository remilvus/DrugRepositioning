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
        self.qk_norm = nn.LayerNorm(rmat_config.d_model)
        self.v_norm = nn.LayerNorm(rmat_config.d_model)
        self.dropout = nn.Dropout(rmat_config.dropout)

        self.size = rmat_config.d_model

    def forward(self, qk: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, **kwargs):
        qk = self.qk_norm(qk)
        v = self.v_norm(v)
        a = self.cross_attn(qk, qk, v, mask=mask, **kwargs)
        return v + self.dropout(a)
