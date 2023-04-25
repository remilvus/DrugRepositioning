import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from datamodules.single_target import SingleTargetSmilesDataModule
from utils.data import get_target_names
from model import SmilesTransformer


if __name__ == "__main__":
    for num_layers in [2, 4]:
        np.random.seed(0)
        torch.random.manual_seed(0)

        datamodule = SingleTargetSmilesDataModule(get_target_names()[0])

        wandb_logger = WandbLogger(
            project="Drug Repositioning", save_dir="../logs", tags=["baseline"]
        )

        model = SmilesTransformer(
            vocab_size=datamodule.vocab_size, num_layers=num_layers
        )

        trainer = pl.Trainer(
            limit_train_batches=100,
            limit_val_batches=10,
            max_epochs=50,
            devices=4,
            accelerator="cpu",
            precision=32,
            logger=wandb_logger,
            fast_dev_run=False,
        )
        trainer.fit(model=model, datamodule=datamodule)
