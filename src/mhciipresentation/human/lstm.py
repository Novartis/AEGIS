#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""lstm.py

LSTM-based neural network to predict peptide presentation by MHCII on the
public data.

"""


import json
import pprint
import time
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from mhciipresentation.constants import (
    AA_TO_INT,
    N_MOTIF_FOLDS,
    USE_GPU,
    USE_SUBSET,
)
from mhciipresentation.loaders import load_motif_exclusion_idx, load_sa_el_data
from mhciipresentation.models import LSTMModel
from mhciipresentation.paths import CACHE_DIR, DATA_DIR
from mhciipresentation.utils import (
    add_peptide_context,
    check_cache,
    compute_performance_measures,
    encode_aa_sequences,
    join_peptide_with_pseudosequence,
    make_dir,
    prepare_batch,
    set_seeds,
    shuffle_features_and_labels,
)

torch.autograd.set_detect_anomaly(True)


def evaluate_lstm(
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
    steps_eval = list(range(0, X.shape[0], batch_size))
    y_pred_batches = list()

    for step in tqdm(steps_eval):
        X_batch, _ = prepare_batch(step, batch_size, pad_width, pad_num, X, y)

        X_batch = (
            X_batch.cuda(device=device, non_blocking=True)
            if USE_GPU
            else X_batch
        )
        y_pred_raw = model(X_batch)
        y_pred = y_pred_raw.cpu().detach().numpy()
        y_pred_batches.append(y_pred)

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
    dataset_size = X_train.shape[0]
    steps = list(range(0, dataset_size, batch_size))

    epoch_loss = list()
    for step in tqdm(steps):
        data, targets = prepare_batch(
            step, batch_size, pad_width, AA_TO_INT["X"], X_train, y_train
        )
        data_cuda = (
            data.cuda(device=device, non_blocking=True) if USE_GPU else data
        )
        targets_cuda = (
            targets.cuda(device=device, non_blocking=True)
            if USE_GPU
            else targets
        )
        output = model_cuda(data_cuda)
        # print(f"max(output), {max(output)}")
        loss = criterion(output, targets_cuda)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        epoch_loss.append(loss.item())

    return sum(epoch_loss) / len(epoch_loss)


def main():
    set_seeds()

    use_cache = check_cache("preprocessed_debug_sa_el_data.csv")
    if use_cache:
        print("Loading Data")
        sa_el_data = pd.read_csv(
            CACHE_DIR + "preprocessed_debug_sa_el_data.csv", index_col=0
        )
    else:
        print("Loading Data")
        sa_el_data = load_sa_el_data()

        print("Adding peptide context")
        sa_el_data["peptide_with_context"] = add_peptide_context(
            sa_el_data["peptide"], sa_el_data["peptide_context"]
        )

        print("Adding pseudosequence to peptide with context")
        sa_el_data[
            "peptide_with_context_and_mhcii_pseudosequence"
        ] = join_peptide_with_pseudosequence(
            sa_el_data["peptide_with_context"], sa_el_data["Pseudosequence"],
        )

        sa_el_data[
            [
                "peptide",
                "target_value",
                "peptide_with_context",
                "Pseudosequence",
                "peptide_with_context_and_mhcii_pseudosequence",
            ]
        ].to_csv(CACHE_DIR + "preprocessed_debug_sa_el_data.csv")

    print("Building directories to save checkpoints and logging metrics")
    now = datetime.now()
    training_start_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    base_log_dir = (
        DATA_DIR + "/modelling/lstm_human/" + training_start_time + "/"
    )
    make_dir(base_log_dir)

    for i in range(N_MOTIF_FOLDS):
        log_dir = base_log_dir + f"/split_{i + 1}/"
        make_dir(log_dir)
        print("Load indices and split data according to those")
        X_train_idx, X_val_idx = load_motif_exclusion_idx(i + 1)
        X_train = encode_aa_sequences(
            sa_el_data.iloc[
                X_train_idx["index"].tolist()
            ].peptide_with_context_and_mhcii_pseudosequence,
            AA_TO_INT,
        )
        X_val = encode_aa_sequences(
            sa_el_data.iloc[
                X_val_idx["index"].tolist()
            ].peptide_with_context_and_mhcii_pseudosequence,
            AA_TO_INT,
        )

        y_train = sa_el_data.iloc[
            X_train_idx["index"].tolist()
        ].target_value.values
        # y_train = np.random.choice([0, 1], size=y_train.shape[0], p=[0.5, 0.5])

        y_val = sa_el_data.iloc[
            X_val_idx["index"].tolist()
        ].target_value.values
        # y_val = np.random.choice([0, 1], size=y_val.shape[0], p=[0.5, 0.5])

        X_train, y_train = shuffle_features_and_labels(X_train, y_train)

        if USE_SUBSET:
            X_train = X_train[:512]
            y_train = y_train[:512]

        batch_size = 16384
        epochs = 200
        criterion = nn.BCELoss()
        lr = 0.01
        dropout = 0.1  # dropout probability
        device = torch.device("cuda") if USE_GPU else None  # training device

        print("Instantiating model")
        input_dim = 63
        output_dim = 1
        embedding_dim = 64
        hidden_size = 64
        n_layers = 2

        model = LSTMModel(
            input_dim,
            output_dim,
            embedding_dim,
            hidden_size,
            n_layers,
            dropout,
            device,
        )

        # model = model.apply(initialize_weights)
        model = model.to(device) if USE_GPU else model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)

        make_dir(log_dir + "/metrics/")
        make_dir(log_dir + "/checkpoints/")

        best_matthews = 0
        best_recall = 0
        best_precision = 0
        print("Starting training")
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            avg_epoch_loss = train(
                model,
                X_train,
                y_train,
                batch_size,
                criterion,
                optimizer,
                device,
                input_dim,
            )
            elapsed = time.time() - epoch_start_time
            print(
                f"EPOCH NO: {epoch} took {elapsed} seconds with average loss "
                f"{avg_epoch_loss}"
            )
            # scheduler.step()
            model.eval()
            print("Evaluating model on training and validation set")
            with torch.no_grad():
                train_metrics = evaluate_lstm(
                    X_train,
                    y_train,
                    batch_size,
                    device,
                    model,
                    "train",
                    input_dim,
                    AA_TO_INT["X"],
                    criterion,
                )
                val_metrics = evaluate_lstm(
                    X_val,
                    y_val,
                    batch_size,
                    device,
                    model,
                    "val",
                    input_dim,
                    AA_TO_INT["X"],
                    criterion,
                )
                epoch_metrics = {"train": train_metrics, "val": val_metrics}

            # print("Performance data handling")
            current_matthews = epoch_metrics["val"]["matthews_corrcoef"]
            current_recall = epoch_metrics["val"]["recall"]
            current_precision = epoch_metrics["val"]["precision"]

            if current_matthews > best_matthews:
                best_matthews = current_matthews
                print("Saving best MCC model")
                torch.save(
                    model,
                    log_dir
                    + "/checkpoints/"
                    + f"checkpoint_epoch_{epoch}_best_matthews.pth",
                )
            if current_recall > best_recall:
                best_recall = current_recall
                print("Saving best recall model")
                torch.save(
                    model,
                    log_dir
                    + "/checkpoints/"
                    + f"checkpoint_epoch_{epoch}_best_recall.pth",
                )
            if current_precision > best_precision:
                best_precision = current_precision
                print("Saving best precision model")
                torch.save(
                    model,
                    log_dir
                    + "/checkpoints/"
                    + f"checkpoint_epoch_{epoch}_best_precision.pth",
                )
            with open(f"{log_dir}/metrics/epoch_{epoch}.json", "w") as outfile:
                json.dump(epoch_metrics, outfile)

        print("Training terminated.")


if __name__ == "__main__":
    main()
