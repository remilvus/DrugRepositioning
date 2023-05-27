from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data import load_datasets


def prepare_datasets(results_root="../../results"):
    results = Path(results_root).joinpath("target")
    results.mkdir(exist_ok=True, parents=True)

    frames = load_datasets(data_root="../../data/raw_data", dropna=False)

    for name, df in frames.items():
        activity_name = [c for c in df.columns if c.startswith('activity')][0]
        if 'binding_score' not in df.columns:
            continue
        df = df[df[activity_name] > 0]
        duplicates = df[df['smiles'].duplicated(keep=False)]
        mean_values = duplicates.groupby('smiles')[[activity_name, 'binding_score']].mean().reset_index()

        df_without_duplicates = df.drop_duplicates(subset='smiles')
        df_merged = pd.concat([df_without_duplicates, mean_values])
        df_merged[activity_name] = np.log10(df_merged[activity_name])

        quantile = df_merged[activity_name].quantile(0.05)
        print(name, activity_name, quantile)
        # df_merged['most_active'] = df_merged['activity'] < quantile
        df_merged.to_csv(f'../../data/{name}')


prepare_datasets()
