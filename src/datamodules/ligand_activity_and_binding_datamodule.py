from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

from src.datamodules import DataBatch
from src.datasets import LigandTargetActivityAndBinding, DataPoint
from src.featurizers import Featurizer


def wrap_collate_fn(ligand_collate_fn):
    def collate_fn(datapoints: list[DataPoint]):
        ligands = [datapoint.ligand for datapoint in datapoints]

        ligands_features = ligand_collate_fn(
            [datapoint.ligand_features for datapoint in datapoints]
        )

        activities_Ki = torch.stack(
            [torch.tensor(datapoint.activity_Ki).float() for datapoint in datapoints]
        )
        activities_IC50 = torch.stack(
            [torch.tensor(datapoint.activity_IC50).float() for datapoint in datapoints]
        )
        binding_scores = torch.stack(
            [torch.tensor(datapoint.binding_score).float() for datapoint in datapoints]
        )

        return DataBatch(
            ligands,
            ligands_features,
            None,
            None,
            activities_Ki.float(),
            activities_IC50.float(),
            binding_scores.float(),
        )

    return collate_fn


class LigandActivityAndBindingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ligand_featurizer: Featurizer,
        target: str,
        path: Path = Path("data/"),
        batch_size: int = 32,
        test_size: float = 0.1,
        val_size: float = 0.1,
        num_workers: int = 0,
    ):
        super().__init__()
        self.ligand_featurizer = ligand_featurizer
        self.path = path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.collate_fn = wrap_collate_fn(ligand_featurizer.collate_fn)
        self.train_dataset = (
            self.val_dataset
        ) = self.test_dataset = self.predict_dataset = None
        all_targets = list(LigandTargetActivityAndBinding.available_targets(self.path))
        all_targets = {t.name: t for t in all_targets}
        if target not in all_targets:
            raise ValueError(
                f"{target} is not available! Available targets: {all_targets}"
            )
        self.target = all_targets[target]

    def setup(self, stage: str):
        all_ligands = list(
            LigandTargetActivityAndBinding.available_ligands(
                self.path, self.target.name
            )
        )

        train_ligands, test_ligands = train_test_split(
            all_ligands, test_size=self.test_size
        )
        train_ligands, val_ligands = train_test_split(
            train_ligands, test_size=self.val_size
        )

        train_ligands, val_ligands, test_ligands = (
            set(train_ligands),
            set(val_ligands),
            set(test_ligands),
        )

        if stage == "fit":
            self.train_dataset = LigandTargetActivityAndBinding(
                train_ligands,
                {self.target},
                self.ligand_featurizer,
            )
            self.val_dataset = LigandTargetActivityAndBinding(
                val_ligands, {self.target}, self.ligand_featurizer
            )
        if stage == "test":
            self.test_dataset = LigandTargetActivityAndBinding(
                test_ligands,
                {self.target},
                self.ligand_featurizer,
            )
        if stage == "predict":
            self.predict_dataset = LigandTargetActivityAndBinding(
                test_ligands,
                {self.target},
                self.ligand_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
