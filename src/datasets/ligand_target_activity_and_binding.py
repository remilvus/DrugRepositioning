from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from src.featurizers.featurizer import Featurizer


@dataclass(eq=True, frozen=True)
class Ligand:
    chembl_id: str
    smiles: str


@dataclass(eq=True, frozen=True)
class Target:
    name: str
    pdb_file: Path


@dataclass
class DataPoint:
    ligand: Ligand
    ligand_features: Any
    target: Target
    target_features: Any
    activity: float
    binding_score: float


class LigandTargetActivityAndBinding(Dataset):
    def __init__(
        self,
        ligands: set,
        targets: set,
        ligand_featurizer: Featurizer,
        target_featurizer: Featurizer,
    ):
        self.data = []
        self.ligands = {}
        self.targets = {}
        self.ligand_featurizer = ligand_featurizer
        self.target_featurizer = target_featurizer

        for target in targets:
            df = pd.read_csv(target.pdb_file.with_suffix(".csv"))
            for i in range(len(df)):
                ligand = Ligand(df["ChEMBL_ID"].iloc[i], df["smiles"].iloc[i])
                if ligand in ligands:
                    activity = df["activity"].iloc[i]
                    binding_score = df["binding_score"].iloc[i]
                    self.data.append(
                        DataPoint(ligand, None, target, None, activity, binding_score)
                    )

    def __getitem__(self, index):
        data_point = deepcopy(
            self.data[index]
        )  # We do not want to keep features in self.data
        data_point.ligand_features = self.ligand_featurizer.from_smiles(
            data_point.ligand.smiles
        ).get_features()
        # data_point.target_features = self.ligand_featurizer.from_smiles(
        #     data_point.ligand.smiles
        # ).get_features()
        data_point.target_features = self.target_featurizer.from_pdb(
            data_point.target.pdb_file
        ).get_features()
        return data_point

    def __len__(self):
        return len(self.data)

    @staticmethod
    def available_ligands(path):
        ligands = set()
        for csv_file in path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            ligands.update(
                {
                    Ligand(chembl_id, smiles)
                    for chembl_id, smiles in df[["ChEMBL_ID", "smiles"]].itertuples(
                        index=False, name=None
                    )
                }
            )
        return ligands

    @staticmethod
    def available_targets(path):
        targets = {
            Target(f.stem, f)
            for f in path.glob("*.pdb")
            if (f.with_suffix(".csv")).is_file()
        }
        return targets
