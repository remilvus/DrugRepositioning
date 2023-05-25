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


class Head(nn.Module):

    def __init__(self, in_size: int, out_size: int, layers: int = 2, activation='identity',
                 narrowing: bool = False):
        super().__init__()
        activations_list = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'identity': nn.Identity(),
        }
        layers_ = []
        if narrowing:
            sizes = torch.linspace(in_size, out_size, layers + 1, dtype=torch.int16)
        else:
            sizes = [in_size for i in range(layers)] + [out_size]
        for layer_in_size, layer_out_size in zip(sizes[:-2], sizes[1:-1]):
            layers_.append(nn.Linear(layer_in_size, layer_out_size, bias=True))
            # layers_.append(nn.BatchNorm1d(layer_out_size)),
            layers_.append(nn.LeakyReLU())
        layers_.append(nn.Linear(sizes[-2], sizes[-1]))
        layers_.append(activations_list[activation])
        self.layers = nn.Sequential(*layers_)

    def forward(self, x):
        return self.layers(x)
