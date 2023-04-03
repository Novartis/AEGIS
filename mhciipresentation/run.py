#!/usr/bin/env python
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
from typing import Dict, Tuple

import hydra
import numpy as np
import OmegaConf
import pandas as pd
import pytorch_lightning as pl
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
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
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

cfg: DictConfig

logger = logging.getLogger(__name__)


def build_metrics(device):
    return {
        "mae": torchmetrics.MeanAbsoluteError().to(device),
        "mse": torchmetrics.MeanSquaredError().to(device),
    }


def get_hydra_logging_directory() -> Path:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
    return Path(hydra_cfg["runtime"]["output_dir"])  # type: ignore


def evaluate_transformer(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    model: torch.nn.Module,
    dataset_type: str,
    pad_width: int,
    pad_num: int,
    criterion: torch.nn.BCELoss,
) -> Dict:
    """Evaluates a given dataset

    Args:
        X (np.ndarray): input data of shape (n_samples, n_features)
        y (np.ndarray): known labels for samples of shape (n_samples, )
        batch_size (int): batch size to compute predictions
        device (torch.device): device on which the training should take place.
            Can be "cpu", "cuda"...
        model (torch.nn.Module): model used for evaluation
        dataset_type (str): string used for logging purposes.
        pad_width (int): width of the padding used to make sure all input
            sequences are of the same length
        pad_num [int]: number to be masked due to padding
        criterion (torch.nn.BCELoss): loss function

    Returns:
        Dict: dictionary containing the dataset performance measures.
    """

    bin_idx = np.where((y == 0.0) | (y == 1.0))[0]
    y = y[bin_idx]
    X = X[bin_idx]

    steps = list(range(0, X.shape[0], batch_size))
    y_pred_batches = list()

    for step in tqdm(steps):
        X_batch, _ = prepare_batch(step, batch_size, pad_width, pad_num, X, y)
        X_batch = (
            X_batch.cuda(device=device, non_blocking=True)
            if USE_GPU
            else X_batch
        )
        src_padding_mask = X_batch == pad_num
        src_padding_mask = (
            src_padding_mask.to(device) if USE_GPU else src_padding_mask
        )
        if batch_size != X_batch.size(0):  # only on last batch
            src_padding_mask[:, : X_batch.size(0)]

        y_pred_batches.append(
            model(X_batch, src_padding_mask).cpu().detach().numpy()
        )
    y_pred = np.vstack(y_pred_batches)
    loss = criterion(
        torch.Tensor(y_pred).view(-1, 1), torch.Tensor(y).view(-1, 1)
    )
    metrics = compute_performance_measures(y_pred, y)
    metrics["loss"] = loss.item()
    print(f"Metrics for {dataset_type}")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(metrics)
    return metrics  # type: ignore


def train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    criterion: torch.nn.BCELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_width: int,
) -> float:
    r"""Training function implementing backprop algorithm

    Args:
        model (nn.Module): model to be trained
        X_train (np.ndarray): training data of shape (n_samples, n_features)
        y_train (np.ndarray): training labels of shape (n_samples, )
        batch_size (int): batch size used for training
        criterion (torch.nn.BCELoss): loss function
        optimizer (torch.optim.Optimizer): optimizer algorithm used to update
            the model parameters
        device (torch.device): device on which the training should take place.
            Can be "cpu", "cuda"...
        pad_width (int): width of the padding used to make sure all input
            sequences are of the same length

    Returns:
        float: epoch loss
    """
    model_cuda = model.cuda(device) if USE_GPU else model
    model_cuda.train()  # turn on train mode
    total_loss = 0.0
    steps = list(range(0, X_train.shape[0], batch_size))
    epoch_loss = list()

    for step in tqdm(steps):
        data, targets = prepare_batch(
            step, batch_size, pad_width, AA_TO_INT["X"], X_train, y_train
        )
        data_cuda = (
            data.cuda(device=device, non_blocking=True) if USE_GPU else data
        )
        targets_cuda = (
            targets.cuda(device=device, non_blocking=True).double()
            if USE_GPU
            else targets.double()
        )

        src_padding_mask = data == AA_TO_INT["X"]

        src_padding_mask = (
            src_padding_mask.to(device) if USE_GPU else src_padding_mask
        )
        if batch_size != data_cuda.size(0):  # only on last batch
            src_padding_mask = src_padding_mask[:, : data_cuda.size(0)]

        optimizer.zero_grad()
        output = model_cuda(data_cuda, src_padding_mask)
        loss = criterion(output, targets_cuda)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        epoch_loss.append(loss.item())
    return sum(epoch_loss) / len(epoch_loss)


def prepare_iedb_data() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    cache_file = Path("./.cache/sa_data_ready_for_modelling.csv")
    if not cache_file.is_file():
        print("Loading Data")
        sa_data = load_iedb_data()

        print("Adding peptide context")
        sa_data["peptide_with_context"] = add_peptide_context(
            sa_data["peptide"], sa_data["peptide_context"]
        )

        print("Adding pseudosequence to peptide with context")
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
        print("Loading cached data.")
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

    if True:
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


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        y_test = np.hstack([y_test_nod, y_test_nod])
        return (X_train, X_val, X_test, y_train, y_val, y_test)


def select_features(X_train, X_val, X_test):
    if cfg.dataset.feature_set == "seq_only":
        relevant_col = "peptide"
    else:
        relevant_col = "peptide_with_mhcii_pseudosequence"

    return (
        X_train[relevant_col],
        X_val[relevant_col],
        X_test[relevant_col],
    )


def train_model(
    device, model, train_loader, val_loader, test_loader, save_name="aegis"
):
    save_path = Path(os.getcwd()) / cfg.paths.checkpoints / save_name
    logger.info(
        f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}"
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trainer = pl.Trainer(
            default_root_dir=save_path,
            accelerator=device.type,
            devices=cfg.compute.n_gpu,
            num_nodes=cfg.compute.num_nodes,
            max_epochs=cfg.training.epochs,
            callbacks=[
                ModelCheckpoint(
                    save_weights_only=True, mode="max", monitor="val_loss"
                ),
                LearningRateMonitor("epoch", log_momentum=cfg.debug.verbose),
                RichProgressBar(leave=True),
                RichModelSummary(),
            ],
            logger=[tb_logger, csv_logger],
            log_every_n_steps=1,
        )
    logger.info(f"Total number of parameters: {count_parameters(model)}")
    logger.info(f"Testing model on validation and test set.")
    val_result = trainer.test(
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
    global aegiscfg
    cfg = aegiscfg

    torch.manual_seed(cfg.seed.seed)
    device = setup_training_env(
        cfg.debug.debug,
        cfg.seed.seed,
        cfg.compute.n_gpu,
        cfg.compute.mps,
        cfg.compute.cuda,
    )

    set_seeds()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    X_train, X_val, X_test = select_features(X_train, X_val, X_test)
    longest_input = max(
        [X_train.map(len).max(), X_val.map(len).max(), X_test.map(len).max()]
    )
    X_train = encode_aa_sequences(
        X_train,
        AA_TO_INT,
    )
    X_val = encode_aa_sequences(
        X_val,
        AA_TO_INT,
    )

    X_train, y_train = shuffle_features_and_labels(X_train, y_train)

    TrainingDataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(TrainingDataset)
    ValidationDataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(ValidationDataset)
    TestDataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(TestDataset)

    max_len = 5000
    batch_size = max_len * torch.cuda.device_count()
    epochs = 500
    criterion = nn.BCELoss()

    lr = float(1e-5)
    patience = 10
    input_dim = (
        33 + 2 if cfg.dataset.feature_set == "seq_only" else 33 + 2 + 34
    )  # size of longest sequence (33, from NOD mice + start/stop)
    n_tokens = len(list(AA_TO_INT.values()))
    embedding_size = 128  # embedding dimension
    enc_ff_hidden = 64  # dimension of the feedforward nn model in the encoder
    ff_hidden = 1024  # dimension of the last feedforward network
    nlayers = 4  # number of nn.TransformerEncoderLayer
    n_attn_heads = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.3  # dropout probability
    device = torch.device("cuda" if USE_GPU else "cpu")  # training device

    print("Instantiating model")
    model = TransformerModel(
        input_dim,
        n_tokens,
        embedding_size,
        n_attn_heads,
        enc_ff_hidden,
        ff_hidden,
        nlayers,
        dropout,
        max_len,
        cfg.training.learning_rate.start_learning_rate,
        cfg.training.optimizer.weight_decay,
        max_len,
    )

    # # try:
    # model = model.to(device) if USE_GPU else model
    # optimizer = Adam(model.parameters(), lr=lr)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    training_params = {
        "start_lr": str(lr),
        "patience": str(patience),
        "input_dim": str(input_dim),
        "n_tokens": str(n_tokens),
        "embedding_size": str(embedding_size),
        "n_attn_heads": str(n_attn_heads),
        "enc_ff_hidden": str(enc_ff_hidden),
        "ff_hidden": str(ff_hidden),
        "nlayers": str(nlayers),
        "dropout": str(dropout),
        "max_len": str(max_len),
    }

    tic = timer()
    logger.info(f"Training start time is {tic}")

    model, result = train_model(
        model,
        device,
        train_loader,
        val_loader,
        test_loader,
        "graph_transformer",
    )
    toc = timer()
    logger.info(f"Training end time is {toc}")
    total_time = toc - tic
    logger.info(f"Total training time is {total_time}")

    #######

    # if USE_SUBSET:
    #     X_train = X_train[:10000]
    #     y_train = y_train[:10000]

    # best_loss = 10000
    # best_matthews = -1
    # best_recall = -1
    # best_precision = -1
    # no_progress = 0
    # print(f"Features: {cfg.dataset.feature_set}")
    # print(f"Data Sources: {cfg.dataset.data_source}")
    # print("batch size: %s" % batch_size)
    # print("# training data points: %i" % len(X_train))
    # print("# labels == 0: %i" % sum(y_train == 0))
    # print("# labels != 0: %i" % sum(y_train != 0))
    # print("Starting training")
    # for epoch in range(1, epochs + 1):
    #     epoch_start_time = time.time()
    #     avg_epoch_loss = train(
    #         model,
    #         X_train,
    #         y_train,
    #         batch_size,
    #         criterion,
    #         optimizer,
    #         device,
    #         input_dim,
    #     )
    #     elapsed = time.time() - epoch_start_time
    #     print(
    #         f"EPOCH NO: {epoch} took {elapsed} seconds with average loss "
    #         f"{avg_epoch_loss}"
    #     )
    #     model.eval()
    #     print("Evaluating model on training and validation set")
    #     with torch.no_grad():

    #         train_metrics = evaluate_transformer(
    #             X_train,
    #             y_train,
    #             batch_size,
    #             device,
    #             model,
    #             "train",
    #             input_dim,
    #             AA_TO_INT["X"],
    #             criterion,
    #         )
    #         if USE_SUBSET:
    #             val_metrics = evaluate_transformer(
    #                 X_val[:5000],
    #                 y_val[:5000],
    #                 batch_size,
    #                 device,
    #                 model,
    #                 "val",
    #                 input_dim,
    #                 AA_TO_INT["X"],
    #                 criterion,
    #             )
    #         else:
    #             val_metrics = evaluate_transformer(
    #                 X_val,
    #                 y_val,
    #                 batch_size,
    #                 device,
    #                 model,
    #                 "val",
    #                 input_dim,
    #                 AA_TO_INT["X"],
    #                 criterion,
    #             )
    #         epoch_metrics = {"train": train_metrics, "val": val_metrics}

    #     if epoch % 10 == 0:
    #         scheduler.step()

    #     print("Performance data handling")
    #     current_matthews = epoch_metrics["val"]["matthews_corrcoef"]
    #     current_recall = epoch_metrics["val"]["recall"]
    #     current_precision = epoch_metrics["val"]["precision"]
    #     current_loss = epoch_metrics["val"]["loss"]

    #     checkpoint_dir = log_dir / "checkpoints"
    #     checkpoint_basename = f"checkpoint_epoch_{epoch}"
    #     checkpoint_fname = checkpoint_dir / checkpoint_basename
    #     if current_matthews > best_matthews:
    #         best_matthews = current_matthews
    #         checkpoint_fname = checkpoint_dir / (
    #             checkpoint_basename + "_best_matthews"
    #         )
    #         print("Saving best MCC model")

    #     if current_recall > best_recall:
    #         best_recall = current_recall
    #         checkpoint_fname = checkpoint_dir / (
    #             checkpoint_basename + "_best_recall"
    #         )
    #         print("Saving best recall model")

    #     if current_precision > best_precision:
    #         best_precision = current_precision
    #         checkpoint_fname = checkpoint_dir / (
    #             checkpoint_basename + "_best_precision"
    #         )
    #         print("Saving best precision model")

    #     if current_loss < best_loss:
    #         best_loss = current_loss
    #         print("Saving best loss model")

    #         no_progress = 0
    #     else:
    #         no_progress += 1

    #     if "best" in str(checkpoint_fname):
    #         save_model(model, str(checkpoint_fname.with_suffix(".pth")))

    #     print("Saving epoch metrics")
    #     with open(log_dir / f"metrics/epoch_{epoch}.json", "w") as outfile:
    #         json.dump(epoch_metrics, outfile)

    #     if no_progress == patience:
    #         # Early stopping
    #         break

    # print("Training terminated.")


if __name__ == "__main__":
    main()
