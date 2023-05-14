from itertools import product

import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from src.models import RegressionModel
from src.datamodules import LigandTargetActivityAndBindingDataModule
from src.featurizers import RMatFeaturizer, SchnetFeaturizer

if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)

    configs = {
        "lr": [1e-3],
        "ligand_schnet": [False, True],
        "target_schnet": [True],
    }
    configs = product(
        *[zip([name] * len(values), values) for name, values in configs.items()]
    )

    for hyperparams in configs:
        hyperparams = dict(hyperparams)

        if hyperparams["ligand_schnet"]:
            ligand_featurizer = SchnetFeaturizer
        else:
            ligand_featurizer = RMatFeaturizer
        if hyperparams["target_schnet"]:
            target_featurizer = SchnetFeaturizer
        else:
            target_featurizer = RMatFeaturizer

        wandb_logger = WandbLogger(
            project="Drug Repositioning",
            entity="drug_repositioning",
            save_dir="logs",
            tags=[],
            reinit=True,
        )

        datamodule = LigandTargetActivityAndBindingDataModule(
            ligand_featurizer,
            target_featurizer,
            num_workers=0,
            batch_size=1,
        )

        model = RegressionModel(**hyperparams)

        # trainer = pl.Trainer(
        #     max_epochs=50,
        #     log_every_n_steps=5,
        #     devices=1,
        #     accelerator='auto',
        #     precision=32,
        #     logger=wandb_logger,
        #     fast_dev_run=False,
        # )
        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="gpu",
            # limit_train_batches=1,
            profiler="advanced",
            log_every_n_steps=100,
        )
        trainer.fit(model=model, datamodule=datamodule)

        wandb_logger.experiment.finish()
        wandb_logger.finalize("success")
