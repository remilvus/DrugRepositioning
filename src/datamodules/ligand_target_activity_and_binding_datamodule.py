from dataclasses import dataclass
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.datasets import LigandTargetActivityAndBinding, DataPoint, Ligand, Target
from src.featurizers import Featurizer
from src.huggingmolecules import RecursiveToDeviceMixin


@dataclass
class DataBatch(RecursiveToDeviceMixin):
    ligand: list[Ligand]
    ligand_features: torch.FloatTensor
    target: list[Target]
    target_features: torch.FloatTensor
    activity: torch.FloatTensor
    binding_score: torch.FloatTensor


def merge_collate_fns(ligand_collate_fn, target_collate_fn):
    def collate_fn(datapoints: list[DataPoint]):
        targets = [datapoint.target for datapoint in datapoints]
        ligands = [datapoint.ligand for datapoint in datapoints]

        targets_features = target_collate_fn(
            [datapoint.target_features for datapoint in datapoints]
        )
        ligands_features = ligand_collate_fn(
            [datapoint.ligand_features for datapoint in datapoints]
        )

        activities = torch.stack(
            [torch.tensor(datapoint.activity).float() for datapoint in datapoints]
        )
        binding_scores = torch.stack(
            [torch.tensor(datapoint.binding_score).float() for datapoint in datapoints]
        )

        return DataBatch(
            ligands,
            ligands_features,
            targets,
            targets_features,
            activities,
            binding_scores,
        )

    return collate_fn


class LigandTargetActivityAndBindingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ligand_featurizer: Featurizer,
        target_featurizer: Featurizer,
        path: Path = Path("data/"),
        batch_size: int = 32,
        test_size: float = 0.2,
        val_size: float = 0.00001,
        num_workers: int = 0,
    ):
        super().__init__()
        self.ligand_featurizer = ligand_featurizer
        self.target_featurizer = target_featurizer
        self.path = path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.collate_fn = merge_collate_fns(
            ligand_featurizer.collate_fn, target_featurizer.collate_fn
        )
        self.train_dataset = (
            self.val_dataset
        ) = self.test_dataset = self.predict_dataset = None

    def setup(self, stage: str):
        all_ligands = list(LigandTargetActivityAndBinding.available_ligands(self.path))
        all_targets = list(LigandTargetActivityAndBinding.available_targets(self.path))

        train_ligands, test_ligands = train_test_split(
            all_ligands, test_size=self.test_size
        )
        train_ligands, val_ligands = train_test_split(
            train_ligands, test_size=self.val_size
        )
        train_targets, test_targets = train_test_split(
            all_targets, test_size=self.test_size
        )
        train_targets, val_targets = train_test_split(
            train_targets, test_size=self.val_size
        )

        train_ligands, val_ligands, test_ligands = (
            set(train_ligands),
            set(val_ligands),
            set(test_ligands),
        )
        train_targets, val_targets, test_targets = (
            set(train_targets),
            set(val_targets),
            set(test_targets),
        )

        if stage == "fit":
            self.train_dataset = LigandTargetActivityAndBinding(
                train_ligands,
                train_targets,
                self.ligand_featurizer,
                self.target_featurizer,
            )
            self.val_dataset = LigandTargetActivityAndBinding(
                val_ligands, val_targets, self.ligand_featurizer, self.target_featurizer
            )
        if stage == "test":
            self.test_dataset = LigandTargetActivityAndBinding(
                test_ligands,
                test_targets,
                self.ligand_featurizer,
                self.target_featurizer,
            )
        if stage == "predict":
            self.predict_dataset = LigandTargetActivityAndBinding(
                test_ligands,
                test_targets,
                self.ligand_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
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
