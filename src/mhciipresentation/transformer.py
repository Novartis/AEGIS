#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""transformer.py

Script to train the transformer encoder-based model and perceptron to classify
peptides as presented or not presented according to the peptide, peptide context
and pseudosequence of the MHCII with which the peptide was eluted.

"""
import json
import os
import pprint
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from mhciipresentation.constants import (
    AA_TO_INT,
    LEN_BOUNDS_HUMAN,
    N_MOTIF_FOLDS,
    USE_CONTEXT,
    USE_GPU,
    USE_SUBSET,
)
from mhciipresentation.loaders import (
    load_motif_exclusion_idx,
    load_mouse_random_idx,
    load_pseudosequences,
    load_public_mouse_data,
    load_sa_data,
    load_sa_el_data,
    load_sa_el_random_idx,
    load_sa_random_idx,
)
from mhciipresentation.models import TransformerModel
from mhciipresentation.paths import (
    CACHE_DIR,
    DATA_DIR,
    EPITOPES_DIR,
    RAW_DATA,
    SPLITS_DIR,
)
from mhciipresentation.utils import (
    add_peptide_context,
    check_cache,
    compute_performance_measures,
    encode_aa_sequences,
    flatten_lists,
    get_n_trainable_params,
    join_peptide_with_pseudosequence,
    make_dir,
    prepare_batch,
    save_model,
    save_training_params,
    set_seeds,
    shuffle_features_and_labels,
)

set_seeds()


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


def main():
    """Main function to train the transformer on the sa el public data"""
    set_seeds()

    use_cache = check_cache("preprocessed_cached_sa_data.csv")
    if use_cache:
        print("Loading Data")
        sa_data = pd.read_csv(
            CACHE_DIR + "preprocessed_cached_sa_data.csv", index_col=0
        )
    else:
        print("Loading Data")
        sa_data = load_sa_data()

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

        sa_data[
            [
                "peptide",
                "target_value",
                "peptide_with_context",
                "Pseudosequence",
                "peptide_with_mhcii_pseudosequence",
                "peptide_with_context_and_mhcii_pseudosequence",
            ]
        ].to_csv(CACHE_DIR + "preprocessed_cached_sa_data.csv")

    # print("Preparing epitope dataset.")
    # epitope_df = epitope_file_parser(EPITOPES_DIR + "CD4_epitopes.fsa")
    # epitope_df["Pseudosequence"] = get_pseudosequence(epitope_df)

    # non_presented_peptides_bounds = (
    #     (LEN_BOUNDS_HUMAN[0], LEN_BOUNDS_HUMAN[1])
    #     if USE_CONTEXT
    #     else LEN_BOUNDS_HUMAN
    # )

    # epitope_data = generate_epitope_data(
    #     epitope_df, non_presented_peptides_bounds
    # )
    # epitope_df = prepare_val_set(epitope_data)

    mouse_data = load_public_mouse_data()
    out_dir = SPLITS_DIR + "/mouse/" + "/random/"
    train_data = pd.read_csv(out_dir + "X_train.csv")
    train_data = train_data.loc[train_data["Peptide Sequence"].str.len() <= 25]

    val_data = pd.read_csv(out_dir + "X_val.csv")
    val_data = val_data.loc[val_data["Peptide Sequence"].str.len() <= 25]

    X_train_mouse = train_data["Peptide Sequence"]
    y_train_mouse = train_data["label"].values

    X_val_mouse = val_data["Peptide Sequence"]
    y_val_mouse = val_data["label"].values

    print("Building directories to save checkpoints and logging metrics")
    now = datetime.now()
    training_start_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    base_log_dir = (
        DATA_DIR + "/modelling/transformer_human/" + training_start_time + "/"
    )
    make_dir(base_log_dir)
    log_dir = base_log_dir + "sa_data_with_nod/"
    make_dir(log_dir)
    print("Load indices and split data according to those")
    X_train_idx, X_val_idx = load_sa_random_idx()

    X_train = sa_data.iloc[list(X_train_idx["index"].values)].peptide
    X_val = sa_data.iloc[list(X_val_idx["index"].values)].peptide

    X_train = pd.concat([X_train, X_train_mouse])
    X_val = pd.concat([X_val, X_val_mouse])

    X_train = encode_aa_sequences(
        X_train,
        AA_TO_INT,
    )
    X_val = encode_aa_sequences(
        X_val,
        AA_TO_INT,
    )

    # X_val = encode_aa_sequences(
    #     epitope_df.peptide_with_mhcii_pseudosequence, AA_TO_INT,
    # )

    y_train = sa_data.iloc[X_train_idx["index"].tolist()].target_value.values
    y_val = sa_data.iloc[X_val_idx["index"].tolist()].target_value.values

    y_train = np.hstack([y_train, y_train_mouse])
    y_val = np.hstack([y_val, y_val_mouse])
    # y_val = epitope_df.label.values

    X_train, y_train = shuffle_features_and_labels(X_train, y_train)

    if USE_SUBSET:
        X_train = X_train[:5000]
        y_train = y_train[:5000]

    batch_size = 4098
    epochs = 300
    criterion = nn.BCELoss()

    lr = 1e-5
    patience = 10
    input_dim = 25 + 2  # size of longest sequence (25 + start/stop)
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
        device,
    )
    get_n_trainable_params(model)
    model = nn.DataParallel(model, device_ids=[0])  # type: ignore
    try:
        model = model.to(device) if USE_GPU else model
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        training_params = {
            "debugging round": "no",
            "data": "sa data random split peptide with mhcii pseudosequence. input sequence dictated by longest sequence in test set.",
            "splitting": "random splitting with no test and val size of 0.01",
            "start_lr": lr,
            "scheduler": "exponential, updates every 10 epoch.",
            "patience": patience,
            "input_dim": input_dim,
            "n_tokens": n_tokens,
            "embedding_size": embedding_size,
            "n_attn_heads": n_attn_heads,
            "enc_ff_hidden": enc_ff_hidden,
            "ff_hidden": ff_hidden,
            "nlayers": nlayers,
            "dropout": dropout,
            "optimizer": "Adam",
        }

        save_training_params(training_params, log_dir)

        print("Building directories to save checkpoints and logging metrics")
        now = datetime.now()
        training_start_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        make_dir(log_dir)
        make_dir(log_dir + "/metrics/")
        make_dir(log_dir + "/checkpoints/")

        best_loss = 10000
        best_matthews = 0
        best_recall = 0
        best_precision = 0
        no_progress = 0
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
            model.eval()
            print("Evaluating model on training and validation set")
            with torch.no_grad():
                idx = np.random.choice(X_train.shape[0], 100000, replace=False)

                X_train_sampled = X_train[idx]
                y_train_sampled = y_train[idx]

                train_metrics = evaluate_transformer(
                    X_train_sampled,
                    y_train_sampled,
                    batch_size,
                    device,
                    model,
                    "train",
                    input_dim,
                    AA_TO_INT["X"],
                    criterion,
                )
                val_metrics = evaluate_transformer(
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

            if epoch % 10 == 0:
                scheduler.step()

            print("Performance data handling")
            current_matthews = epoch_metrics["val"]["matthews_corrcoef"]
            current_recall = epoch_metrics["val"]["recall"]
            current_precision = epoch_metrics["val"]["precision"]
            current_loss = epoch_metrics["val"]["loss"]

            checkpoint_fname = (
                log_dir + "/checkpoints/" + f"checkpoint_epoch_{epoch}"
            )
            if current_matthews > best_matthews:
                best_matthews = current_matthews
                checkpoint_fname = checkpoint_fname + "_best_matthews"
                print("Saving best MCC model")

            if current_recall > best_recall:
                best_recall = current_recall
                checkpoint_fname = checkpoint_fname + "_best_recall"
                print("Saving best recall model")

            if current_precision > best_precision:
                best_precision = current_precision
                checkpoint_fname = checkpoint_fname + "_best_precision"
                print("Saving best precision model")

            if current_loss < best_loss:
                best_loss = current_loss
                print("Saving best loss model")

                no_progress = 0
            else:
                no_progress += 1

            save_model(model, checkpoint_fname + ".pth")
            print("Saving epoch metrics")
            with open(f"{log_dir}/metrics/epoch_{epoch}.json", "w") as outfile:
                json.dump(epoch_metrics, outfile)

            if no_progress == patience:
                # Early stopping
                break

    except Exception as error:
        raise error
    finally:
        torch.cuda.empty_cache()
    print("Training terminated.")


if __name__ == "__main__":
    main()
