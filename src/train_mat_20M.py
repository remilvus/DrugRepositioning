import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics import MeanSquaredError
import pytorch_lightning as pl

from src.datamodules.single_target import SingleTargetFeaturesDataModule
from src.utils.data import get_target_names
from huggingmolecules.src.huggingmolecules.models.models_mat import MatModel, MatFeaturizer
from huggingmolecules.experiments.src import TrainingModule


if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)

    featurizer = MatFeaturizer.from_pretrained('mat_masking_20M')
    datamodule = SingleTargetFeaturesDataModule(get_target_names()[0], featurizer=featurizer)

    wandb_logger = WandbLogger(
        project="Drug Repositioning", save_dir="../logs", tags=["mat_20M"]
    )

    model = MatModel.from_pretrained('mat_masking_20M')

    pl_module = TrainingModule(model,
                               loss_fn=torch.nn.MSELoss(),
                               metric_cls=MeanSquaredError,
                               optimizer=torch.optim.Adam(model.parameters()))

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
    trainer.fit(pl_module, datamodule=datamodule)
