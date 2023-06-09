# Heavily based on featurization from huggingmolecules

from dataclasses import dataclass
from typing import *

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

from src.huggingmolecules.featurization.featurization_api import RecursiveToDeviceMixin
from src.huggingmolecules.featurization.featurization_mat_utils import (
    add_dummy_node,
    build_position_matrix,
    build_atom_features_matrix,
    build_adjacency_matrix,
    pad_sequence,
)
from src.huggingmolecules.featurization.featurization_rmat_utils import (
    build_bond_features_matrix,
    build_relative_matrix,
    add_mask_feature,
)
from .featurizer import Featurizer


@dataclass
class RMatMoleculeEncoding:
    node_features: np.ndarray
    bond_features: np.ndarray
    distance_matrix: np.ndarray
    relative_matrix: np.ndarray
    generated_features: Optional[List[float]]
    y: Optional[np.ndarray]


@dataclass
class RMatBatchEncoding(RecursiveToDeviceMixin):
    node_features: torch.FloatTensor
    bond_features: torch.FloatTensor
    relative_matrix: torch.FloatTensor
    distance_matrix: torch.FloatTensor
    generated_features: Optional[torch.FloatTensor]
    y: Optional[torch.FloatTensor]
    batch_size: int

    def __len__(self):
        return self.batch_size


class RMatFeaturizer(Featurizer):
    def get_features(self):
        mol = self.mol

        node_features = build_atom_features_matrix(mol, self.use_bonds)

        if self.use_bonds:
            bond_features = build_bond_features_matrix(mol)
            adj_matrix = build_adjacency_matrix(mol)
        else:
            bond_features = None
            adj_matrix = None

        pos_matrix = build_position_matrix(mol)
        dist_matrix = pairwise_distances(pos_matrix)

        node_features, adj_matrix, dist_matrix, bond_features = add_dummy_node(
            node_features=node_features,
            adj_matrix=adj_matrix,
            dist_matrix=dist_matrix,
            bond_features=bond_features,
        )

        if self.use_bonds:
            relative_matrix = build_relative_matrix(adj_matrix)
        else:
            relative_matrix = None
        bond_features, node_features = add_mask_feature(bond_features, node_features)

        return RMatMoleculeEncoding(
            node_features=node_features,
            bond_features=bond_features,
            distance_matrix=dist_matrix,
            relative_matrix=relative_matrix,
            generated_features=None,
            y=None,
        )

    def collate_fn(self, encodings):
        node_features = pad_sequence(
            [torch.tensor(e.node_features).float() for e in encodings]
        )
        dist_matrix = pad_sequence(
            [torch.tensor(e.distance_matrix).float() for e in encodings]
        )
        if self.use_bonds:
            bond_features = pad_sequence(
                [torch.tensor(e.bond_features).float() for e in encodings]
            )
            relative_matrix = pad_sequence(
                [torch.tensor(e.relative_matrix).float() for e in encodings]
            )
        else:
            bond_features = None
            relative_matrix = None

        return RMatBatchEncoding(
            node_features=node_features,
            bond_features=bond_features,
            relative_matrix=relative_matrix,
            distance_matrix=dist_matrix,
            generated_features=None,
            y=None,
            batch_size=len(encodings),
        )
