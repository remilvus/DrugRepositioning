from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
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
    pocket_center: tuple[float, float, float]


@dataclass
class DataPoint:
    ligand: Ligand
    ligand_features: Any
    target: Target
    target_features: Any
    activity_Ki: float
    activity_IC50: float
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
                    activity_Ki, activity_IC50, binding_score = np.nan, np.nan, np.nan
                    if "activity_Ki" in df.columns:
                        activity_Ki = df["activity_Ki"].iloc[i]
                    if "activity_IC50" in df.columns:
                        activity_IC50 = df["activity_IC50"].iloc[i]
                    if "binding_score" in df.columns:
                        binding_score = df["binding_score"].iloc[i]
                    self.data.append(
                        DataPoint(
                            ligand,
                            None,
                            target,
                            None,
                            activity_Ki=activity_Ki,
                            activity_IC50=activity_IC50,
                            binding_score=binding_score,
                        )
                    )

    def __getitem__(self, index):
        data_point = deepcopy(
            self.data[index]
        )  # We do not want to keep features in self.data
        data_point.ligand_features = self.ligand_featurizer.load_from_smiles(
            data_point.ligand.smiles
        ).get_features()
        data_point.target_features = self.target_featurizer.load_from_pdb(
            data_point.target.pdb_file, data_point.target.pocket_center
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
        pockets_df = pd.read_csv(path / "pockets" / "pockets.csv")
        pockets = {
            target: (x, y, z)
            for target, x, y, z in pockets_df[["target", "x", "y", "z"]].itertuples(
                index=False, name=None
            )
        }

        targets = {
            Target(f.stem, f, pockets[f.stem])
            for f in path.glob("*.pdb")
            if f.with_suffix(".csv").is_file() and f.stem in pockets.keys()
        }
        return targets
