from itertools import product

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from datamodules.single_target import SingleTargetSmilesDataModule
from utils.data import get_target_names
from model import SmilesTransformer


if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)

    configs = {
        'lr': [1e-3, 3e-3],
        'num_layers': [4],
        'num_heads': [2, 16, 64],
        'hidden_size': [256, 1024],
        'dropout': [0.1],
        'name': get_target_names(),
    }
    configs = product(*[zip([name] * len(values), values)
                       for name, values in configs.items()])

    for hyperparams in configs:
        hyperparams = dict(hyperparams)
        name = hyperparams['name']
        print(f"Training with config:\n{hyperparams}")
        datamodule = SingleTargetSmilesDataModule(name)

        wandb_logger = WandbLogger(
                project="Drug Repositioning",
                save_dir="../logs",
                tags=["baseline", name],
                reinit=True
            )

        model = SmilesTransformer(
                vocab_size=datamodule.vocab_size,
                **hyperparams
            )

        trainer = pl.Trainer(
                max_epochs=50,
                log_every_n_steps=5,
                devices=1,
                accelerator="gpu",
                precision=32,
                logger=wandb_logger,
                fast_dev_run=False,
            )
        trainer.fit(model=model, datamodule=datamodule)

        wandb_logger.experiment.finish()
        wandb_logger.finalize('success')
