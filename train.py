from itertools import product

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

# from src.datamodules.ligand_activity_and_binding_datamodule import (
#     LigandActivityAndBindingDataModule,
# )
from src.datamodules import LigandTargetActivityAndBindingDataModule
from src.featurizers import RMatFeaturizer, SchnetFeaturizer
from src.huggingmolecules import RMatConfig
from src.models.common import CrossAttentionType
from src.models.rmat_rmat import RmatRmatModel
from src.models.rmat import RmatModel
from src.models.schnet_rmat import RMatSchNetModel

if __name__ == "__main__":
    np.random.seed(0)
    torch.random.manual_seed(0)
    # batch[].target[].name
    configs = {
        "lr": [1e-5],
        "batch_size": [8],
        "model": ["RMatSchnet"],# ["RMatRMat"],#
        # "model": ["RMatRMat", "RMatSchnet", "RMat"],
        # "target": ["5HT1A"],  # "CYP2C8", "5HT1A", "D2"
        "cross_attention": [CrossAttentionType.NONE],
        "targets": [
            ["binding_score", "IC50", "Ki"]
        ],  # in ['Ki','IC50','binding_score']
        "activity_importance": [1.0],
        "thresholds": [
            {
                "binding_score": (-torch.inf, -9.0),
                "IC50": (-torch.inf, np.log10(900.0)),
                "Ki": (-torch.inf, np.log10(3000.0)),
            }
        ],  # in ['Ki','IC50','binding_score']
        # "cross_attention": [CrossAttentionType.NONE, CrossAttentionType.LIGAND, CrossAttentionType.TARGET, CrossAttentionType.BOTH],
    }
    # docking score that is related to the free energy of binding of a ligand to a receptor.
    # For this type of docking score, the more negative the score, the better.

    # Generalnie przyjmuje się, że Ki poniżej 1000 (jednostką są nM) determinuje aktywność

    # consider IC50 of <100 nM to be active, 101 nM to 300 nM to be moderately active, and >300 nM to be inactive
    configs = product(
        *[zip([name] * len(values), values) for name, values in configs.items()]
    )

    for hyperparams in configs:
        hyperparams = dict(hyperparams)
        batch_size = hyperparams["batch_size"]

        ligand_featurizer = RMatFeaturizer(use_bonds=False, cutout=False)

        if hyperparams["model"] in ["RMatSchnet"]:
            target_featurizer = SchnetFeaturizer(
                use_bonds=False, cutout=True, cutout_radius=10.0
            )
        elif hyperparams["model"] in ["RMatRMat"]:
            target_featurizer = RMatFeaturizer(
                use_bonds=False, cutout=True, cutout_radius=10.0
            )

        # TODO: report avg, min, max number of atoms after cutouts

        wandb_logger = WandbLogger(
            project="Drug Repositioning",
            entity="drug_repositioning",
            save_dir="logs",
            tags=[],
            reinit=True,
        )

        if hyperparams["model"] == "RMat":
            pass
            # datamodule = LigandActivityAndBindingDataModule(
            #     ligand_featurizer,
            #     target=hyperparams["target"],
            #     num_workers=0,
            #     batch_size=batch_size,
            # )
        else:
            datamodule = LigandTargetActivityAndBindingDataModule(
                ligand_featurizer,
                target_featurizer,
                num_workers=0,
                batch_size=batch_size,
            )

        if hyperparams["model"] == "RMatRMat":
            model = RmatRmatModel(
                rmat_config=RMatConfig.get_default(use_bonds=False), **hyperparams
            )
        elif hyperparams["model"] == "RMatSchnet":
            # TODO: implement RMatSchnet
            model = RMatSchNetModel(
                rmat_config=RMatConfig.get_default(use_bonds=False), **hyperparams
            )
        elif hyperparams["model"] == "RMat":
            model = RmatModel(
                rmat_config=RMatConfig.get_default(use_bonds=False), **hyperparams
            )

        trainer = pl.Trainer(
            max_epochs=100,
            log_every_n_steps=1,
            devices=1,
            accelerator="auto",
            precision=32,
            logger=wandb_logger,
            fast_dev_run=False,
            # limit_train_batches=1,
            # profiler="advanced",
        )

        trainer.fit(model=model, datamodule=datamodule)

        wandb_logger.experiment.finish()
        wandb_logger.finalize("success")
