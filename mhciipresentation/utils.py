#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""utils.py

This file contains a number of utility functions used throughout the project.

"""
import copy
import functools
import json
import logging
import math
import os
import pickle
import random
from collections import Counter
from itertools import count, tee
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from mhciipresentation.constants import (
    AA_TO_INT,
    AMINO_ACIDS,
    FORCE_PREPROCESSING,
    USE_GPU,
)
from mhciipresentation.loaders import load_pseudosequences, load_uniprot
from mhciipresentation.paths import CACHE_DIR
from scipy import sparse
from sklearn.metrics import (
    PrecisionRecallDisplay,
    auc,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import Binarizer
from torch import backends, nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def set_seeds(seed=42, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)

    if reproducibility:
        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = True


def get_accelerator(debug=False, n_devices=1, use_mps=False, use_cuda=False):

    if use_mps and use_cuda:
        raise ValueError("Cannot use both MPS and CUDA.")
    if use_mps and n_devices > 1:
        raise ValueError("Cannot use MPS with multiple devices (yet).")

    if use_mps and not torch.backends.mps.is_available() and not debug:
        if not torch.backends.mps.is_built():
            logger.warning(
                "MPS not available because the current PyTorch install was not"
                " built with MPS enabled."
            )
        else:
            logger.warning(
                "MPS not available because the current MacOS version is not"
                " 12.3+ and/or you do not have an MPS-enabled device on this"
                " machine."
            )
        if debug:
            logger.info(
                "Using CPU instead of MPS because debug mode is enabled."
            )
            device = torch.device("cpu")
        else:
            logging.info(
                "Using MPS because debug mode is disabled and MPS is"
                " available."
            )
            device = torch.device("mps")
    elif (
        use_cuda
        and not use_mps
        and torch.cuda.is_available()
        and n_devices == 1
    ):
        logging.info(
            "Using CUDA (single GPU) because debug mode is disabled and CUDA"
            " is available."
        )
        device = torch.device("cuda:0")
    elif (
        use_cuda
        and not use_mps
        and torch.cuda.is_available()
        and n_devices > 1
    ):
        logging.info(
            "Using CUDA (multiple GPU) because debug mode is disabled and CUDA"
            " is available."
        )
        device = torch.device("cuda")
    else:
        logging.info(
            "Using CPU because debug mode is enabled or CUDA is not available"
            " or MPS is not available."
        )
        device = torch.device("cpu")

    # Accelerate training with lower precision
    torch.set_float32_matmul_precision("medium")

    logger.info(f"Using device {device}")
    return device


def setup_training_env(
    debug, seed, n_devices=1, use_mps=False, use_cuda=False
):
    set_seeds(seed=seed)
    device = get_accelerator(
        debug, n_devices, use_mps=use_mps, use_cuda=use_cuda
    )
    return device


def check_cache(input_file: str):
    if (
        not Path(CACHE_DIR).exists()
        or FORCE_PREPROCESSING
        or not Path(CACHE_DIR / input_file).exists()
    ):
        make_dir(CACHE_DIR)
        return False
    else:
        return True


def flatten_lists(lists: list) -> list:
    """Removes nested lists"""
    result = list()
    for _list in lists:
        _list = list(_list)
        if _list != []:
            result += _list
        else:
            continue
    return result


def shuffle_features_and_labels(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffles labels and features in the same order.

    Args:
        X (np.ndarray): features of shape (n_samples * n_features)
        y (np.ndarray): labels of shape (n_samples * 1)

    Returns:
        Tuple[np.ndarray, np.ndarray]: shuffles labels and features
    """
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def pandas2fasta(
    df: pd.DataFrame,
    fasta_file_name: str,
    sequence_col: str,
    description_col: str = None,
    description_generic: str = "peptide bound to MHCII",
):
    """Exports a pandas dataframe to a fasta file."""
    if Path(fasta_file_name).exists():
        os.remove(fasta_file_name)

    with open(fasta_file_name, "a") as f:
        for i in df[sequence_col].iteritems():
            if description_col is not None:
                record = SeqRecord(
                    Seq(i[1]),
                    id=str(i[0]),
                    description=df.iloc[i[0]][description_col],
                )

            else:
                record = SeqRecord(
                    Seq(i[1]), id=str(i[0]), description=description_generic
                )

            SeqIO.write(record, f, "fasta")


def uniquify(input_seq: list, suffs: Iterator[int] = count(1)):
    """Make all the items unique by adding a suffix (_1, _2, etc).

    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    seq = input_seq.copy()
    not_unique = [
        k for k, v in Counter(seq).items() if v > 1
    ]  # so we have: ['name', 'zip']
    # suffix generator dict - e.g., {'name': <my_gen>, 'zip': <my_gen>}
    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            # s was unique
            continue
        else:
            seq[idx] = seq[idx] + "_" + suffix
    return seq


def make_dir(directory: Path) -> None:
    """Makes directory and does not stop if it is already created.

    TODO: since converting to pathlib, this function is not needed anymore.

    Args:
        directory (Path): directory to be created
    """

    Path(directory).mkdir(parents=True, exist_ok=True)


def aa_seq_to_int(s: str, aa_to_int: dict) -> List[int]:
    """Maps an amino acid sequence to a list of strings

    Args:
        s (str): input sequence

    Raises:
        ValueError: when the sequence does not contain characters that can be
        mapped to integers

    Returns:
        List[int]: list of mapped integers
    """
    # Make sure only valid aa's are passed
    if not set(s).issubset(set(aa_to_int.keys())):
        raise ValueError(
            "Unsupported character(s) in sequence found:"
            f" {set(s).difference(set(aa_to_int.keys()))}"
        )
    return (
        [aa_to_int["start"]] + [aa_to_int[a] for a in s] + [aa_to_int["stop"]]
    )


def encode_aa_sequences(
    aa_sequences: pd.Series, aa_to_int: dict
) -> np.ndarray:
    """Encodes a given series of amino acids

    Args:
        aa_sequences (pd.Series): amino acids to encode

    Returns:
        np.ndarray: encoded array of integers
    """
    return np.array(
        [
            arr.tolist()
            for arr in aa_sequences.apply(
                lambda x: np.array(aa_seq_to_int(x, aa_to_int))
            ).values
        ],
        dtype=object,
    )


def add_peptide_context(
    peptide: pd.Series, peptide_context: pd.Series
) -> pd.Series:
    """Adds the peptide flanking regions to the peptide. If a peptide has
        a sequence ABCDEFGH, and the peptide context consists of UVWABCFGHXYZ,
        then this function returns UVWABCDEFGHXYZ.

    Args:
        peptide (pd.Series): amino acids eluted and sequenced using MS
        peptide_context (pd.Series): peptide context including PFRs

    Returns:
        pd.Series: peptide containing PFRs
    """
    before = peptide_context.str[:3].to_frame()
    before.columns = ["before"]
    after = peptide_context.str[-3:].to_frame()
    after.columns = ["after"]
    return before.join(peptide.to_frame()).join(after).agg("".join, axis=1)


def join_peptide_with_pseudosequence(
    peptide_with_context: pd.Series,
    pseudosequence: pd.Series,
) -> pd.Series:
    """Adds the peptide, pads it and appends the associated pseudosequence.
        Given a peptide ABCDEF and and a pseudosequence HIJK, the function
        returns a series of strings ABCDEFXXXXXHIJK, where the number of Xs
        depends on the padding width.

    Args:
        peptide_with_context (pd.Series): peptide containing the PFR
        pseudosequence (pd.Series): pseudosequence of the MHC molecule

    Returns:
        pd.Series: series used as input for the LSTM
    """
    return (
        peptide_with_context.to_frame()
        .join(pseudosequence.to_frame())
        .agg("".join, axis=1)
    )


def compute_performance_measures(
    y_pred: np.ndarray, y_true: np.ndarray
) -> Dict:
    """Computes necessary measures to assess model performance

    Args:
        y_pred (np.ndarray): predicted output by the model
        y_true (np.ndarray): ground truth label

    Returns:
        Dict: set of performance measures associated to predictions
    """
    y_pred_bin = Binarizer(threshold=0.5).transform(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_bin, average="binary"
    )
    auc = roc_auc_score(y_true, y_pred)
    matthews = matthews_corrcoef(y_true, y_pred_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "matthews_corrcoef": matthews,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prepare_batch(
    step: int,
    batch_size: int,
    pad_width: int,
    pad_num: int,
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares a given batch for processing by the model the following way:
        1. Pad the seq of ints to the longest seq length.
        2. Concat to a tensor
    TODO: make labels optional for evaluation
    Args:
        step (int): index of the starting sample of the batch
        features (np.ndarray): (n_samples, n_features) feature set to fetch batch
            samples from.
        labels (np.ndarray): (n_samples, 1 targets)
        batch_size (int): batch size desized
        pad_width (int): width of the padding used to make sure all input
            sequences are of the same length
        pad_num [int]: number to be masked due to padding

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: padded batch sequences of shape
            (batch_size, pad_width) and reshaped labels
    """
    dataset_size = len(features)
    if labels is None:
        labels = np.zeros(len(features))

    batch_in = []
    batch_target = []
    for sample_idx in range(step, min(step + batch_size, dataset_size), 1):

        in_seq = features[sample_idx]
        target = labels[sample_idx]

        batch_in.append(in_seq)
        batch_target.append(target)

    # Right padding
    seq_padded = [
        np.pad(
            seq, pad_width=(0, pad_width - len(seq)), constant_values=pad_num
        )
        for seq in batch_in
    ]
    return (
        torch.tensor(np.array(seq_padded, dtype=int)),
        torch.tensor(batch_target).reshape(-1, 1),
    )


def set_pandas_options() -> None:
    """Sets appropriate values for debugging."""
    pd.options.display.max_colwidth = 150
    pd.options.display.max_columns = 150
    pd.options.display.max_rows = 150


# One hot encoding functions
def onehot_encode_amino_acid_sequence(aa_seq: str) -> list:
    """Computes the one hot encoding of an amino acid sequence

    Args:
        aa_seq (str): input amino acid sequence

    Returns:
        list: list of the one-hot encoded amino acid sequence.
    """
    return [
        [0 if char != letter else 1 for char in AMINO_ACIDS]
        for letter in aa_seq
    ]


def encode_mhcii(
    fcontent: pd.DataFrame, column: str = "Pseudosequence"
) -> np.ndarray:
    """One hot encodes the MHCII pseudosequence


    Args:
        fcontent (pd.DataFrame): dataframe containing the pseudosequence to
            encode
        column (str): column name for the MHCII pseudosequence

    Returns:
        np.ndarray: one hot encoded MHCII
    """
    return np.matrix(
        fcontent[column]
        .str.pad(side="both", fillchar="X", width=12)
        .apply(
            lambda x: np.array(onehot_encode_amino_acid_sequence(x)).flatten()
        )
        .tolist()
    )


def encode_peptide(
    fcontent: pd.DataFrame, column: str = "peptide"
) -> np.ndarray:
    """One hot encodes the peptides.

    Args:
        fcontent (pd.DataFrame): dataframe containing the peptide sequences to
            encode
        column (str): column name for the peptide sequences

    Returns:
        np.ndarray: One hot encoded peptide sequence
    """
    # Returns matrix of shape (n_samples * 441 (21 AA * 21 (padded) peptides))
    return np.matrix(
        fcontent[column]
        .str.pad(side="both", fillchar="X", width=21)
        .apply(
            lambda x: np.array(onehot_encode_amino_acid_sequence(x)).flatten()
        )
        .tolist()
    )


def encode_context(
    fcontent: pd.DataFrame, column: str = "peptide_context"
) -> np.ndarray:
    """One hot encodes the peptide context

    Args:
        fcontent (pd.DataFrame): dataframe containing the peptide sequences to
            encode
        column (str): column used to encode the peptide context

    Returns:
        np.ndarray: encoded peptide context
    """
    # Encode context
    # Yields matrix of dimension (n_samples * 252 (21 AA * 12 (padded) peptides))
    return np.stack(  # type: ignore
        np.array(
            fcontent[column]
            .str.pad(side="both", fillchar="X", width=12)
            .apply(
                lambda x: np.array(
                    onehot_encode_amino_acid_sequence(x)
                ).flatten()
            )
        )
    )


def oh_encode(
    fcontent: pd.DataFrame,
) -> Tuple[sparse.csr.csr_matrix, pd.Series]:
    """Fully encodes the input dataframe

    Args:
        fcontent (pd.DataFrame): input dataframe containing the peptide
            sequences, mhcii pseudosequence and peptide context

    Returns:
        Tuple[sparse.csr.csr_matrix, pd.Series]: encoded features and target
            values
    """
    encoded_mhcii = encode_mhcii(fcontent)
    logger.info(f"Encoded MHCII is of shape {encoded_mhcii.shape}")
    encoded_peptide = encode_peptide(fcontent)
    logger.info(f"Encoded peptide is of shape {encoded_peptide.shape}")
    encoded_context = encode_context(fcontent)
    logger.info(f"Encoded context is of shape {encoded_context.shape}")
    encoded_fcontent = sparse.csr_matrix(
        np.hstack([encoded_mhcii, encoded_peptide, encoded_context])
    )
    logger.info(f"Final encoded object is of shape {encoded_fcontent.shape}")
    return encoded_fcontent, fcontent.target_value


def get_peptide_context(
    peptides: pd.Series, source_proteins: pd.Series
) -> pd.Series:
    """Gets the peptide context for each peptides (flanking regions ±3 aa before
    and after the peptides). Pads with X if a peptide is on a terminus of a
    protein.

    Args:
        peptides (pd.Series): series containing the sequence of the peptides
        source_proteins (pd.Series): source proteins from which a context can be
            extracted

    Returns:
        pd.Series: peptide with the context added to either end.
    """
    peptide_flanking_regions = list()
    for peptide in tqdm(peptides.str.strip().tolist()):
        if source_proteins.str.contains(peptide, case=False).any():
            after = (
                source_proteins.loc[
                    source_proteins.str.contains(peptide, case=False)
                ]
                .iloc[0]
                .split(peptide.upper())[1][:3]
                .ljust(3, "X")
            )
            before = (
                source_proteins.loc[
                    source_proteins.str.contains(peptide, case=False)
                ]
                .iloc[0]
                .split(peptide.upper())[0][-3:]
                .rjust(3, "X")
            )
            peptide_flanking_regions.append(
                {"peptide": peptide, "before": before, "after": after}
            )
    peptide_flanking_regions_df = pd.DataFrame(peptide_flanking_regions)
    peptide_flanking_regions_df = peptide_flanking_regions_df.dropna()
    return (
        peptide_flanking_regions_df["before"]
        .to_frame()
        .join(peptide_flanking_regions_df["peptide"].to_frame())
        .join(peptide_flanking_regions_df["after"].to_frame())
        .agg("".join, axis=1)
    )


def generate_negative_peptides(white_space: List, bounds: Tuple) -> pd.Series:
    """Generates negative peptides from the white space (protein space without
        any presented peptides)

    Args:
        white_space (List): non presented regions of proteins from which
            peptides are presented
        bounds (Tuple): bounds to use when generating negative peptides

    Returns:
        pd.Series: negative peptides
    """
    neg_peptides = list()
    # We loop through the length of ranges + 3 amino acids because of the PFRs
    logger.info("Generating negative peptides.")
    # We have to extend the range to bounds[1] + 4 due to range up to n-1.
    for length in tqdm(range(bounds[0], bounds[1] + 1)):
        for peptide in white_space:
            for i in range(len(peptide) - length + 1):
                neg_peptides.append(peptide[i : i + length])

    neg_peptides_series = pd.Series(neg_peptides)
    return neg_peptides_series.astype("str")


def get_white_space(peptides: pd.Series, proteins: pd.Series) -> List:
    """Get regions of the proteins which are not presented

    Args:
        peptides (pd.Series): presented peptides
        proteins (pd.Series): source proteins

    Returns:
        List: regions of proteins which are not presented but those proteins
            contain regions that are presented
    """
    white_space = list()
    logger.info("Generating white space")
    for peptide in tqdm(peptides.tolist()):
        proteins_containing_peptide = proteins.loc[
            proteins.str.contains(peptide)
        ].tolist()
        for protein in proteins_containing_peptide:
            white_space.append(protein.split(peptide))
    return flatten_lists(white_space)  # type: ignore


def take(n: int, to_take_from: Dict) -> Dict:
    """Returns n first elements from a dictionary

    Args:
        n (int): numbers of elements to take
        to_take_from (Dict): dictionary from which to take `n` elements

    Returns:
        List: list of elements
    """
    return {k: to_take_from[k] for k in list(to_take_from.keys())[:n]}


def save_model(model: nn.Module, out_dir: str) -> None:
    """Saves a PyTorch model in such a way that it can be loaded back into
        memory for inference

    Args:
        model (nn.Module): model to save
        out_dir (str): where to save the model
    """
    torch.save(model.state_dict(), out_dir)


def save_training_params(training_params: Dict, out_dir: Path) -> None:
    """Saves a .json file with the training hyperparameters of the model

    Args:
        training_params (Dict): contains all the hyperparameters used for
            training
        out_dir (str): where to save training_params
    """
    with open(out_dir / "training_params.json", "w") as outfile:
        json.dump(training_params, outfile)


def get_n_trainable_params(model: torch.nn.Module):
    """Prins the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): model from which to extract parameters
    """
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters in Millions: ", pytorch_total_params / 1e6)


def load_model_weights(
    model: nn.Module, model_dir: str, device: torch.device
) -> nn.Module:
    """Loads model weights

    Args:
        model (nn.Module): Loads PyTorch model weights
        model_dir (str): model directory
        device (torch.device): device on which to put the model

    Returns:
        nn.Module: model with weights
    """
    if USE_GPU:
        model.load_state_dict(copy.deepcopy(torch.load(model_dir)))
        model = model.to(device)
    else:
        model.load_state_dict(
            copy.deepcopy(
                torch.load(model_dir, map_location=torch.device("cpu"))
            )
        )
    return model


def make_predictions_with_transformer(
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    model: torch.nn.Module,
    pad_width: int,
    pad_num: int,
) -> np.ndarray:
    """Makes a prediction using a transformer model by building a masking
    tensor.

    Args:
        X (np.ndarray): input features
        y (np.ndarray): labels
        batch_size (int): size of each batch passed to the model.
        device (torch.device): device used for execution
        model (torch.nn.Module): model used to make predictions
        pad_width (int): padding width used to make sure each input sequence
            has the same length
        pad_num (int): number representing padding.

    Returns:
        np.ndarray: vstacked predictions for each input vector.
    """
    steps_eval = list(range(0, X.shape[0], batch_size))
    y_pred_batches = list()
    for step in tqdm(steps_eval, position=2, leave=False):
        X_batch, _ = prepare_batch(
            step, batch_size, pad_width, pad_num, X, None
        )
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
            src_padding_mask = src_padding_mask[
                : X_batch.size(0), : X_batch.size(1)
            ]

        y_pred_batches.append(
            model(X_batch, src_padding_mask).cpu().detach().numpy()
        )
    return np.vstack(y_pred_batches)  # type: ignore


def render_roc_curve(y_pred, y_true, dest_dir, title, fname):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(dest_dir, fname + ".png"))
    plt.close()


def render_precision_recall_curve(y_pred, y_true, dest_dir, title, fname):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # displ=PrecisionRecallDisplay(precision=precision, recall=recall)
    auprc = auc(recall, precision)
    points = np.array([recall, precision]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, axs = plt.subplots(1, 1)
    norm = plt.Normalize(thresholds.min(), thresholds.max())
    lc = LineCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(thresholds)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.5, 1.0])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.text(
        0.05,
        0.52,
        "AUPRC curve (area = %0.2f)" % auprc,
    )
    # displ.plot()
    plt.title(title)
    plt.savefig(os.path.join(dest_dir, fname + "prec_rec_curve" + ".png"))


def sample_peptides(hs_uniprot_str: str, peptide_length: int, n: int) -> List:
    """Samples peptide uniformly from uniprot.

    Args:
        hs_uniprot_str (str): homo sapiens uniprot proteins separated by |.
        peptide_length (int): desired peptide length
        n (int): number of peptide length

    Returns:
        List: sampled decoy peptides of the desired length.
    """

    protein_space_len = len(hs_uniprot_str)
    peptides: List = list()
    while len(peptides) < n:
        start = random.sample(range(protein_space_len), 1)[0]
        peptide_candidate = hs_uniprot_str[start : start + peptide_length]
        if "|" not in peptide_candidate:
            peptides.append(peptide_candidate)
        logger.info(
            "Sampling peptides from SwissProt space"
            f" {(len(peptides)/n)*100}% complete.",
            end="\r",
        )
    logger.info("")
    return peptides


def sample_from_human_uniprot(n_len: Dict) -> List:
    """Sample from human uniprot

    Args:
        n_len (Dict): length of peptides.

    Returns:
        List: list of sampled decoy. Nested list for each peptide length, needs
            to be flattened for use.
    """
    uniprot = load_uniprot()
    peptides = list()
    hs_uniprot = uniprot.loc[uniprot["Species ID"] == 9606]
    # We want to create one giant string to sample from to avoid sampling from
    # short proteins more often (row-wise sampling bias).
    hs_uniprot_str = "|".join(hs_uniprot.Sequence.tolist())
    for peptide_length in list(n_len.keys()):
        n_samples = n_len[peptide_length]
        peptides.append(
            sample_peptides(hs_uniprot_str, peptide_length, n_samples)
        )
    return peptides


def attach_pseudosequence(epitope_df: pd.DataFrame) -> pd.Series:
    """Returns a series of corresponding pseudosequences

    Args:
        fcontent (pd.DataFrame): peptide and associated MHC source information

    Returns:
        pd.Series: allelelist and mhcii molecules
    """
    mhcii_pseudosequences = load_pseudosequences()
    return epitope_df.merge(
        mhcii_pseudosequences,
        how="outer",
        left_on="MHC_molecule",
        right_on="Name",
    ).dropna()


def assign_pseudosequences(
    ligands: pd.DataFrame, decoys: pd.DataFrame
) -> pd.DataFrame:
    """Assigns pseudosequences to ligands appropriately.

    Args:
        ligands (pd.DataFrame): ligands
        decoys (pd.DataFrame): decoys (negative data)

    Returns:
        pd.DataFrame: decoys with assigned pseudosequences
    """
    decoys["Pseudosequence"] = "Placeholder"
    for index, value in (
        ligands[["Sequence Length", "Pseudosequence"]]
        .value_counts()
        .iteritems()
    ):
        peptide_length = index[0]
        pseudosequence = index[1]

        assigned = (
            decoys.loc[
                (decoys.Sequence.str.len() == peptide_length)
                & (decoys["Pseudosequence"] == "Placeholder")
            ]
            .sample(n=value)
            .assign(Pseudosequence=pseudosequence)
        )
        decoys.loc[
            decoys.Sequence.isin(assigned.Sequence), "Pseudosequence"
        ] = assigned.Pseudosequence

    return decoys


def save_obj(obj, path: Path) -> None:
    make_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_obj(path: Path) -> Any:
    """Loads pickle object from path.

    Args:
        path (Path): path to pickle file
        use_logging (bool, optional): Whether or not to use logging. Defaults to True.
        temporary (bool, optional): Whether or not the print statement should be temporary. Defaults to False.

    Raises:
        FileNotFoundError: if file does not exist
        NotADirectoryError: if file is not a file
        ValueError: if file is not a pickle file

    Returns:
        Any: loaded object
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj
