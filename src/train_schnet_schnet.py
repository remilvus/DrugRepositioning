import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from datamodules import LigandTargetActivityAndBindingDataModule
from models import SchnetSchnet
from featurizers import RMatFeaturizer, SchnetFeaturizer

if __name__ == '__main__':
    np.random.seed(0)
    torch.random.manual_seed(0)

    wandb_logger = WandbLogger(
        project='Drug Repositioning',
        entity='drug_repositioning',
        save_dir='../logs',
        tags=['schnet_schnet'],
        reinit=True
    )

    ligand_featurizer = RMatFeaturizer
    target_featurizer = SchnetFeaturizer
    datamodule = LigandTargetActivityAndBindingDataModule(ligand_featurizer, target_featurizer)

    model = SchnetSchnet()

    trainer = pl.Trainer(
        max_epochs=50,
        log_every_n_steps=5,
        devices=1,
        accelerator='auto',
        precision=32,
        logger=wandb_logger,
        fast_dev_run=False,
    )
    trainer.fit(model=model, datamodule=datamodule)

    wandb_logger.experiment.finish()
    wandb_logger.finalize('success')
