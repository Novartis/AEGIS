#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""cd4.py

This script validates our model against CD4 datasets.
"""

import logging
import warnings
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
from experiments.inference import setup_model
from mhciipresentation.callbacks import GPUUsageLogger, VectorLoggingCallback
from mhciipresentation.constants import AA_TO_INT
from mhciipresentation.loaders import fasta_parser
from mhciipresentation.metrics import (
    build_scalar_metrics,
    build_vector_metrics,
    compute_performance_metrics,
    save_performance_metrics,
)
from mhciipresentation.paths import RAW_DATA
from mhciipresentation.utils import (
    assign_pseudosequences,
    attach_pseudosequence,
    encode_aa_sequences,
    flatten_lists,
    get_accelerator,
    get_hydra_logging_directory,
    make_dir,
    render_precision_recall_curve,
    render_roc_curve,
    sample_from_human_uniprot,
)
from omegaconf import DictConfig
from pyprojroot import here
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

cfg: DictConfig

logger = logging.getLogger(__name__)


def make_inference_unlabeled(X, cfg, input_dim, dest_dir):
    X = torch.from_numpy(
        np.stack(
            [
                np.pad(
                    seq,
                    pad_width=(0, input_dim - len(seq)),
                    constant_values=AA_TO_INT["X"],
                )
                for seq in X
            ]
        ).astype(int)
    ).int()

    dataset = TensorDataset(
        torch.from_numpy(
            np.stack(
                [
                    np.pad(
                        seq,
                        pad_width=(0, input_dim - len(seq)),
                        constant_values=AA_TO_INT["X"],
                    )
                    for seq in X
                ]
            )
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.compute.n_cpu_loader,
    )

    device = get_accelerator(
        cfg.debug.debug, cfg.compute.n_gpu, cfg.compute.mps, cfg.compute.cuda
    )
    model = setup_model(
        input_dim=input_dim,
        n_tokens=len(list(AA_TO_INT.values())),
        cfg=cfg,
    )
    save_name = "aegis_inference"
    trainer = pl.Trainer(
        default_root_dir=get_hydra_logging_directory() / "predictions_logs",
        accelerator=device.type,
        devices=cfg.compute.n_gpu,
        num_nodes=cfg.compute.num_nodes,
        max_epochs=cfg.training.epochs,
        callbacks=[
            RichProgressBar(leave=True),
            VectorLoggingCallback(
                root=Path(get_hydra_logging_directory()) / "vector_logs"
            ),
            GPUUsageLogger(
                log_dir=get_hydra_logging_directory()
                / "tensorboard"
                / "gpu_usage"
            ),
        ],
        log_every_n_steps=1,
        benchmark=cfg.debug.benchmark,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        logger=[
            pl_loggers.TensorBoardLogger(
                save_dir=get_hydra_logging_directory()
                / "predict"
                / "tensorboard",
                name="aegis_inference",
            ),
            pl_loggers.CSVLogger(
                save_dir=get_hydra_logging_directory() / "predict" / "csv",
                name=save_name,
            ),
        ],
    )
    # This is specific to our model because of what we return in the predict_step method.
    # y_hat = trainer.predict(model, loader)[0]["y_hat"].reshape(-1)
    y_hat = trainer.predict(model, loader)

    y_hat_post = []
    for batch in y_hat:
        y_hat_post.append(batch["y_hat"])
    y_hat = torch.tensor(np.concatenate(y_hat_post).reshape(-1))
    print("Predictions")
    print(y_hat)


@hydra.main(
    version_base="1.3", config_path=str(here() / "conf"), config_name="config"
)
def main(cfg: DictConfig) -> None:
    logger.info("Example script to make inferences using AEGIS")
    logger.info(
        "Training with the following"
        f" config:\n{omegaconf.omegaconf.OmegaConf.to_yaml(cfg)}"
    )
    epitope_df = fasta_parser(here() / "example/input_fasta.fsa")
    epitope_df = attach_pseudosequence(epitope_df)
    epitope_df["peptides_and_pseudosequence"] = epitope_df["peptide"].astype(
        str
    ) + epitope_df["Pseudosequence"].astype(str)
    epitope_df.columns = [
        "Sequence",
        "MHC_molecule",
        "protein_id",
        "source_protein",
        "Name",
        "Pseudosequence",
        "peptides_and_pseudosequence",
    ]
    epitope_df["Sequence Length"] = epitope_df["Sequence"].str.len()

    epitope_df["Sequence Length"] = epitope_df["Sequence Length"].loc[
        epitope_df["Sequence Length"] <= 25
    ]

    data = pd.concat(
        [
            epitope_df[["peptides_and_pseudosequence", "Sequence"]],
        ]
    )

    if cfg.model.feature_set == "seq_mhc":
        input_dim = 33 + 2 + 34
        X = encode_aa_sequences(
            data.peptides_and_pseudosequence,
            AA_TO_INT,
        )
    elif cfg.model.feature_set == "seq_only":
        input_dim = 33 + 2
        X = encode_aa_sequences(
            data.Sequence,
            AA_TO_INT,
        )
    else:
        raise ValueError(
            f"Unknown feature set {cfg.model.feature_set}. "
            "Please choose from seq_only or seq_and_mhc"
        )

    make_inference_unlabeled(
        X,
        cfg,
        input_dim,
        get_hydra_logging_directory() / "performance_metrics",
    )


if __name__ == "__main__":
    main()
