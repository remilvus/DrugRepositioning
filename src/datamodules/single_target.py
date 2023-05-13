from sklearn import model_selection
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.utils import load_datasets, SmilesDataset


class SingleTargetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        target: str,
        data_dir: str = "../data",
        batch_size: int = 32,
        test_size: float = 0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_name = target
        self.df = load_datasets()[target]
        self.df_train, self.df_val = model_selection.train_test_split(
            self.df, test_size=test_size
        )
        print(f"Training set size: {len(self.df_train)}")
        print(f"Validation set size: {len(self.df_val)}")


class SingleTargetSmilesDataModule(SingleTargetDataModule):
    def __init__(self, target: str, data_dir: str = "../data", batch_size: int = 32):
        super().__init__(target, data_dir, batch_size)

        smiles = "".join(list(self.df["smiles"]))
        self.tokens = sorted(tuple(set(smiles)))
        self.vocab_size = len(self.tokens) + 1  # adding one accounts for padding
        self.sentence_length = max(len(s) for s in self.df["smiles"])
        print(f"Max smiles length: {self.sentence_length}")
        print(f"Total number of tokens: {self.vocab_size}")

    def train_dataloader(self):
        dataset = SmilesDataset(
            self.df_train, tokens=self.tokens, max_len=self.sentence_length
        )
        return DataLoader(dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        dataset = SmilesDataset(
            self.df_val, tokens=self.tokens, max_len=self.sentence_length
        )
        return DataLoader(dataset, batch_size=self.batch_size)
