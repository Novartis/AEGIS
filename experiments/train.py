# -*- coding: utf-8 -*-

"""run.py

Script to train the transformer encoder-based model and perceptron to classify
peptides as presented or not presented according to the peptide, peptide context
and pseudosequence of the MHCII with which the peptide was eluted.

"""
import argparse
import json
import logging
import os
import pprint
import time
import warnings
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from Bio.SeqIO.FastaIO import SimpleFastaParser
from mhciipresentation.callbacks import VectorLoggingCallback
from mhciipresentation.constants import (
    AA_TO_INT,
    LEN_BOUNDS_HUMAN,
    N_MOTIF_FOLDS,
    USE_CONTEXT,
    USE_GPU,
    USE_SUBSET,
)
from mhciipresentation.loaders import (
    load_iedb_data,
    load_iedb_idx,
    load_nod_data,
    load_nod_idx,
    load_pseudosequences,
    load_sa_random_idx,
)
from mhciipresentation.models import TransformerModel
from mhciipresentation.paths import LOGS_DIR
from mhciipresentation.utils import (
    add_peptide_context,
    check_cache,
    compute_performance_measures,
    count_parameters,
    encode_aa_sequences,
    flatten_lists,
    get_n_trainable_params,
    join_peptide_with_pseudosequence,
    make_dir,
    prepare_batch,
    save_model,
    save_training_params,
    set_seeds,
    setup_training_env,
    shuffle_features_and_labels,
)
from omegaconf import DictConfig
from pyprojroot import here
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm

warnings.filterwarnings("ignore")

cfg: DictConfig

logger = logging.getLogger(__name__)


def build_scalar_metrics():
    return {
        "accuracy": BinaryAccuracy(threshold=0.5),
        "precision": torchmetrics.Precision(task="binary"),
        "recall": torchmetrics.Recall(task="binary"),
        "f1": torchmetrics.F1Score(task="binary"),
        "matthews": torchmetrics.MatthewsCorrCoef(task="binary"),
        "cohen": torchmetrics.CohenKappa(task="binary"),
        "auroc": torchmetrics.AUROC(task="binary"),
    }


def build_vector_metrics():
    return {
        "roc": torchmetrics.ROC(task="binary"),
        "precision_recall_curve": torchmetrics.PrecisionRecallCurve(
            task="binary"
        ),
        "confusion_matrix": torchmetrics.ConfusionMatrix(task="binary"),
    }


def get_hydra_logging_directory() -> Path:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
    return Path(hydra_cfg["runtime"]["output_dir"])  # type: ignore


def pad_sequences(
    features: np.ndarray,
    pad_width: int,
) -> torch.Tensor:
    # Right padding
    seq_padded = [
        np.pad(
            seq,
            pad_width=(0, pad_width - len(seq)),
            constant_values=AA_TO_INT["X"],
        )
        for seq in tqdm(features)
    ]
    return torch.tensor(np.array(seq_padded, dtype=int))


def prepare_iedb_data() -> (
    Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    cache_file = Path(here() / "data/.cache/sa_data_ready_for_modelling.csv")
    if not cache_file.is_file():
        logger.info("Loading Data")
        sa_data = load_iedb_data()

        logger.info("Adding peptide context")
        sa_data["peptide_with_context"] = add_peptide_context(
            sa_data["peptide"], sa_data["peptide_context"]
        )

        logger.info("Adding pseudosequence to peptide with context")
        sa_data[
            "peptide_with_context_and_mhcii_pseudosequence"
        ] = join_peptide_with_pseudosequence(
            sa_data["peptide_with_context"],
            sa_data["Pseudosequence"],
        )

        sa_data[
            "peptide_with_mhcii_pseudosequence"
        ] = join_peptide_with_pseudosequence(
            sa_data["peptide"],
            sa_data["Pseudosequence"],
        )
        sa_data.to_csv(cache_file)
    else:
        logger.info("Loading cached data.")
        sa_data = pd.read_csv(cache_file, index_col=0)

    X_train_idx_iedb, X_val_idx_iedb, X_test_idx_iedb = load_iedb_idx()

    X_train_iedb = sa_data.iloc[X_train_idx_iedb["index"]]
    X_val_iedb = sa_data.iloc[X_val_idx_iedb["index"]]
    X_test_iedb = sa_data.iloc[X_test_idx_iedb["index"]]
    y_train_iedb, y_val_iedb, y_test_iedb = (
        X_train_iedb.target_value.values,
        X_val_iedb.target_value.values,
        X_test_iedb.target_value.values,
    )
    return (
        X_train_iedb,
        X_val_iedb,
        X_test_iedb,
        y_train_iedb,
        y_val_iedb,
        y_test_iedb,
    )


def prepare_nod_data():
    nod_data = load_nod_data()
    X_train_idx_nod, X_val_idx_nod, X_test_idx_nod = load_nod_idx()
    X_train_nod = nod_data.iloc[X_train_idx_nod["index"]].rename(
        columns={"Peptide Sequence": "peptide"}
    )
    X_val_nod = nod_data.iloc[X_val_idx_nod["index"]].rename(
        columns={"Peptide Sequence": "peptide"}
    )
    X_test_nod = nod_data.iloc[X_test_idx_nod["index"]].rename(
        columns={"Peptide Sequence": "peptide"}
    )

    pseudosequences = load_pseudosequences()
    iag7_pseudosequence = pseudosequences.loc[
        pseudosequences.Name == "H-2-IAg7"
    ].Pseudosequence.values[0]
    (
        X_train_nod["Pseudosequence"],
        X_val_nod["Pseudosequence"],
        X_test_nod["Pseudosequence"],
    ) = (
        iag7_pseudosequence,
        iag7_pseudosequence,
        iag7_pseudosequence,
    )
    X_train_nod[
        "peptide_with_mhcii_pseudosequence"
    ] = join_peptide_with_pseudosequence(
        X_train_nod["peptide"], X_train_nod["Pseudosequence"]
    )
    X_val_nod[
        "peptide_with_mhcii_pseudosequence"
    ] = join_peptide_with_pseudosequence(
        X_val_nod["peptide"], X_val_nod["Pseudosequence"]
    )
    X_test_nod[
        "peptide_with_mhcii_pseudosequence"
    ] = join_peptide_with_pseudosequence(
        X_test_nod["peptide"], X_test_nod["Pseudosequence"]
    )

    y_train_nod, y_val_nod, y_test_nod = (
        X_train_nod.label.values,
        X_val_nod.label.values,
        X_test_nod.label.values,
    )
    return (
        X_train_nod,
        X_val_nod,
        X_test_nod,
        y_train_nod,
        y_val_nod,
        y_test_nod,
    )


def prepare_data() -> (
    Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    if "iedb" in cfg.dataset.data_source:
        (
            X_train_iedb,
            X_val_iedb,
            X_test_iedb,
            y_train_iedb,
            y_val_iedb,
            y_test_iedb,
        ) = prepare_iedb_data()
        if cfg.dataset.data_source == "iedb":
            return (
                X_train_iedb,
                X_val_iedb,
                X_test_iedb,
                y_train_iedb,
                y_val_iedb,
                y_test_iedb,
            )
    if "nod" in cfg.dataset.data_source:
        (
            X_train_nod,
            X_val_nod,
            X_test_nod,
            y_train_nod,
            y_val_nod,
            y_test_nod,
        ) = prepare_nod_data()
        if cfg.dataset.data_source == "nod":
            return (
                X_train_nod,
                X_val_nod,
                X_test_nod,
                y_train_nod,
                y_val_nod,
                y_test_nod,
            )
    if cfg.dataset.data_source == "iedb_nod":
        X_train = pd.concat([X_train_iedb, X_train_nod])
        X_val = pd.concat([X_val_iedb, X_val_nod])
        X_test = pd.concat([X_test_iedb, X_test_nod])
        y_train = np.hstack([y_train_iedb, y_train_nod])
        y_val = np.hstack([y_val_iedb, y_val_nod])
        y_test = np.hstack([y_test_iedb, y_test_nod])
        return (X_train, X_val, X_test, y_train, y_val, y_test)


def select_features(X_train, X_val, X_test):
    if cfg.model.feature_set == "seq_only":
        relevant_col = "peptide"
    else:
        relevant_col = "peptide_with_mhcii_pseudosequence"

    return (
        X_train[relevant_col],
        X_val[relevant_col],
        X_test[relevant_col],
    )


def train_model(
    model, device, train_loader, val_loader, test_loader, save_name="aegis"
):
    save_path = Path(os.getcwd()) / cfg.paths.checkpoints / save_name
    logger.info(
        "Training with the following"
        f" config:\n{omegaconf.omegaconf.OmegaConf.to_yaml(cfg)}"
    )
    logger.info(f"Saving model in {save_path}")
    logger.info(f"Creating TensorBoard Logger")
    make_dir(get_hydra_logging_directory() / "tensorboard" / save_name)
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=get_hydra_logging_directory() / "tensorboard", name=save_name
    )
    csv_logger = pl_loggers.CSVLogger(
        save_dir=get_hydra_logging_directory() / "csv", name=save_name
    )
    if device.type == "mps":
        accelerator = "mps"
    elif device.type == "cuda":
        accelerator = "gpu"
    elif device.type == "cpu":
        accelerator = "cpu"
    else:
        raise ValueError("Unknown Pytorch Lightning Accelerator")
    logger.info(f"Set pytorch lightning accelerator as {accelerator}")
    logger.info(f"Instantiating Trainer")

    trainer = pl.Trainer(
        default_root_dir=save_path,
        accelerator=device.type,
        devices=cfg.compute.n_gpu,
        num_nodes=cfg.compute.num_nodes,
        max_epochs=cfg.training.epochs,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=False, mode="min", monitor="val_loss"
            ),
            LearningRateMonitor("epoch", log_momentum=cfg.debug.verbose),
            RichProgressBar(leave=True),
            VectorLoggingCallback(
                root=Path(get_hydra_logging_directory()) / "vector_logs"
            ),
        ],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
        benchmark=cfg.debug.benchmark,
    )

    trainer.fit(model, train_loader, val_loader)

    logger.info(f"Total number of parameters: {count_parameters(model)}")
    logger.info(f"Testing model on validation and test set.")

    val_result = trainer.validate(
        model, dataloaders=val_loader, verbose=cfg.debug.verbose
    )
    test_result = trainer.test(
        model, dataloaders=test_loader, verbose=cfg.debug.verbose
    )
    result = {
        "test": test_result,
        "val": val_result,
    }
    return model, result


@hydra.main(
    version_base="1.3", config_path=str(here() / "conf"), config_name="config"
)
def main(aegiscfg: DictConfig):
    """Main function to train the transformer on the iedb data and sa dataset"""
    global cfg
    cfg = aegiscfg

    device = setup_training_env(
        cfg.debug.debug,
        cfg.seed.seed,
        cfg.compute.n_gpu,
        cfg.compute.mps,
        cfg.compute.cuda,
    )

    torch.set_num_threads(cfg.compute.n_cpu)

    set_seeds(cfg.seed.seed)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    X_train, X_val, X_test = select_features(X_train, X_val, X_test)

    X_train = encode_aa_sequences(
        X_train,
        AA_TO_INT,
    )
    X_val = encode_aa_sequences(
        X_val,
        AA_TO_INT,
    )
    X_test = encode_aa_sequences(
        X_test,
        AA_TO_INT,
    )

    if cfg.model.feature_set == "seq_only":
        input_dim = 33 + 2
    elif cfg.model.feature_set == "seq_mhc":
        input_dim = 33 + 2 + 34
    else:
        raise ValueError(
            f"Unknown feature set {cfg.model.feature_set}. "
            "Please choose from seq_only or seq_and_mhc"
        )

    if cfg.debug.debug:
        X_train = X_train[: cfg.debug.n_samples_debug]
        X_val = X_val[: cfg.debug.n_samples_debug]
        X_test = X_test[: cfg.debug.n_samples_debug]
        y_train = y_train[: cfg.debug.n_samples_debug]
        y_val = y_val[: cfg.debug.n_samples_debug]
        y_test = y_test[: cfg.debug.n_samples_debug]

    X_train = pad_sequences(X_train, input_dim)
    X_val = pad_sequences(X_val, input_dim)
    X_test = pad_sequences(X_test, input_dim)

    X_train, y_train = shuffle_features_and_labels(X_train, y_train)

    # Transform labels to tensors
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    TrainingDataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        TrainingDataset,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.compute.n_cpu,
    )
    ValidationDataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        ValidationDataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.compute.n_cpu,
    )
    TestDataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        TestDataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.compute.n_cpu,
    )

    n_tokens = len(list(AA_TO_INT.values()))
    device = torch.device("cuda" if USE_GPU else "cpu")  # training device
    logger.info("Instantiating model")
    model = TransformerModel(
        seq_len=input_dim,
        n_tokens=n_tokens,
        embedding_size=cfg.model.aegis.embedding_size,
        n_attn_heads=cfg.model.aegis.n_attn_heads,
        enc_ff_hidden=cfg.model.aegis.enc_ff_hidden,
        ff_hidden=cfg.model.aegis.ff_hidden,
        n_layers=cfg.model.aegis.n_layers,
        dropout=cfg.model.aegis.dropout,
        pad_num=AA_TO_INT["X"],
        batch_size=cfg.training.batch_size,
        warmup_steps=cfg.training.learning_rate.warmup_steps,
        epochs=cfg.training.epochs,
        start_learning_rate=cfg.training.learning_rate.start_learning_rate,
        peak_learning_rate=cfg.training.learning_rate.peak_learning_rate,
        weight_decay=cfg.training.optimizer.weight_decay,
        loss_fn=nn.BCELoss(),
        scalar_metrics=build_scalar_metrics(),
        vector_metrics=build_vector_metrics(),
        n_gpu=cfg.compute.n_gpu,
        n_cpu=cfg.compute.n_cpu,
        steps_per_epoch=len(train_loader),
    )
    tic = timer()
    logger.info(f"Training start time is {tic}")

    model, result = train_model(
        model,
        device,
        train_loader,
        val_loader,
        test_loader,
        "transformer",
    )
    toc = timer()
    logger.info(f"Training end time is {toc}")
    total_time = toc - tic
    logger.info(f"Total training time is {total_time}")


if __name__ == "__main__":
    main()
