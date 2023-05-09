from dataclasses import dataclass

import numpy as np
import torch

from .featurizer import Featurizer
from src.huggingmolecules.featurization.featurization_api import RecursiveToDeviceMixin


@dataclass
class SchnetMoleculeEncoding:
    node_z: np.ndarray
    node_pos: np.ndarray


@dataclass
class SchnetBatchEncoding(RecursiveToDeviceMixin):
    node_z: torch.LongTensor
    node_pos: torch.FloatTensor
    batch: torch.LongTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class SchnetFeaturizer(Featurizer):
    def get_features(self):
        mol = self.mol

        node_z = []
        node_pos = []
        for i, atom in enumerate(mol.GetAtoms()):
            node_z.append(atom.GetAtomicNum())
            node_pos.append(mol.GetConformer().GetAtomPosition(i))

        node_z = np.array(node_z, dtype=np.int64)
        node_pos = np.array(node_pos, dtype=np.float32)

        return SchnetMoleculeEncoding(node_z=node_z,
                                      node_pos=node_pos)

    @staticmethod
    def collate_fn(encodings):
        node_z = torch.from_numpy(np.concatenate([encoding.node_z for encoding in encodings]))
        node_pos = torch.from_numpy(np.concatenate([encoding.node_pos for encoding in encodings]))

        repeat_counts = [encoding.node_z.size for encoding in encodings]
        batch = torch.from_numpy(np.repeat(np.arange(len(encodings), dtype=np.int64), repeat_counts))

        return SchnetBatchEncoding(node_z=node_z,
                                   node_pos=node_pos,
                                   batch=batch,
                                   batch_size=len(encodings))
