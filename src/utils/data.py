from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_datasets(data_root="../data") -> Dict[str, pd.DataFrame]:
    data_root = Path(data_root)
    frames = {
        f.name: pd.read_csv(f, header=None, names=["smiles", "chembl", "target"])
        for f in data_root.iterdir()
    }
    for name, df in frames.items():
        all_rows = len(df)
        df.dropna(axis=0, inplace=True)
        rows = len(df)
        if all_rows > rows:
            print(f"Dropped {all_rows - rows} for {name} (reason: NaN)")
    return frames


def get_target_names(data_root="../data") -> List[str]:
    names = [f.name for f in Path(data_root).iterdir()]
    return sorted(names)


class SmilesDataset(Dataset):
    def __init__(self, df, tokens, max_len):
        self.df = df
        self.padding_token = "pad"
        self.tokens = {self.padding_token: 0}
        for i, token in enumerate(tokens):
            self.tokens[token] = i + 1
        self.max_len = max_len
        self.prepare_items()

    def prepare_items(self):
        smiles = list(self.df["smiles"])
        tokenized_smiles = [[self.tokens[x] for x in smile] for smile in smiles]

        for i, smile in enumerate(tokenized_smiles):
            padding = [self.tokens[self.padding_token]] * (self.max_len - len(smile))
            tokenized_smiles[i] += padding
        self.tokenized_smiles = torch.IntTensor(tokenized_smiles)

        target = list(self.df["target"])
        self.target = np.clip(0, 100, target)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.tokenized_smiles[idx], self.target[idx]


class FeaturesDataset(Dataset):
    def __init__(self, df, featurizer):
        self.df = df
        self.featurizer = featurizer
        self.prepare_items()

    def prepare_items(self):
        smiles = list(self.df["smiles"])
        self.featurized_smiles = featurizer(smiles)

        target = list(self.df["target"])
        self.target = np.clip(0, 100, target)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.featurized_smiles[idx], self.target[idx]


if __name__ == "__main__":
    frames = load_datasets()
    for name, df in frames.items():
        print(name)
        print(df.head())
        print("-" * 20 + "\n")
