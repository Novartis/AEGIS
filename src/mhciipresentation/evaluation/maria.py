#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""maria.py

This script validates our model against datasets from against which MARIA was
evaluated.
"""

import argparse
import copy
import json
import pprint
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from torch import nn
from tqdm import tqdm

from mhciipresentation.constants import (
    AA_TO_INT,
    LEN_BOUNDS_HUMAN,
    USE_CONTEXT,
    USE_GPU,
    USE_SUBSET,
)
from mhciipresentation.inference import setup_model
from mhciipresentation.loaders import (
    load_K562_dataset,
    load_melanoma_dataset,
    load_pseudosequences,
    load_uniprot,
)
from mhciipresentation.paths import DATA_DIR, EPITOPES_DIR, RAW_DATA
from mhciipresentation.transformer import (
    TransformerModel,
    evaluate_transformer,
)
from mhciipresentation.utils import (
    assign_pseudosequences,
    compute_performance_measures,
    encode_aa_sequences,
    flatten_lists,
    make_predictions_with_transformer,
    render_roc_curve,
    sample_from_human_uniprot,
    set_pandas_options,
)

set_pandas_options()


def load_DRB1_0101_DRB1_0404() -> List[str]:
    """Loads DRB1_0101 and DRB1_0404

    Returns:
        List[str]: the two pseudosequences
    """
    pseudosequences = load_pseudosequences()
    return pseudosequences.loc[
        pseudosequences.Name.isin(["DRB1_0101", "DRB1_0404"])
    ].Pseudosequence.to_list()


def handle_K562_dataset(ligands: pd.DataFrame, title: str, fname: str) -> None:
    """Make predictions on K562 ligand dataset

    Args:
        ligands (pd.DataFrame): ligands to assign pseudosequences to.
        title (str): title of the resulting ROC curve.
        fname (str): filename of resulting plot.
    """
    # ligands = ligands.loc[ligands.Sequence.str.len() <= 21]
    n_len = ligands.Sequence.str.len().value_counts().sort_index().to_dict()
    decoys = sample_from_human_uniprot(n_len)
    ligands["label"] = 1
    decoys = pd.DataFrame(flatten_lists(decoys), columns=["Sequence"])
    decoys["label"] = 0
    decoys = assign_pseudosequences(ligands, decoys)
    dataset = pd.concat(
        [
            ligands[["Sequence", "label", "Pseudosequence"]],
            decoys[["Sequence", "label", "Pseudosequence"]],
        ]
    )
    # dataset["peptides_and_pseudosequence"] = dataset["Sequence"].astype(
    #     str
    # ) + dataset["Pseudosequence"].astype(str)

    device = torch.device("cuda" if USE_GPU else "cpu")  # training device
    model, input_dim = setup_model(device, FLAGS.model_wo_pseudo_path)

    X = encode_aa_sequences(dataset.Sequence, AA_TO_INT,)
    y = dataset.label.values
    batch_size = 5000
    predictions = make_predictions_with_transformer(
        X, y, batch_size, device, model, input_dim, AA_TO_INT["X"]
    )
    performance = compute_performance_measures(predictions, y)
    print(performance)
    render_roc_curve(predictions, y, FLAGS.results, title, fname)


def handle_melanoma_dataset(
    ligands: pd.DataFrame, title: str, fname: str
) -> None:
    """Makes predictions on the melanoma dataset.

    Args:
        ligands (pd.DataFrame): ligands eluted from melanoma tissues
        title (str): title of the resulting ROC curve.
        fname (str): filename of resulting plot.
    """
    ligands = ligands.loc[ligands.Sequence.str.len() <= 25]
    n_len = ligands.Sequence.str.len().value_counts().sort_index().to_dict()
    decoys = sample_from_human_uniprot(n_len)
    ligands["label"] = 1
    decoys = pd.DataFrame(flatten_lists(decoys), columns=["Sequence"])
    decoys["label"] = 0
    dataset = pd.concat(
        [ligands[["Sequence", "label"]], decoys[["Sequence", "label"]],]
    )
    device = torch.device("cuda" if USE_GPU else "cpu")  # training device
    model, input_dim = setup_model(device, FLAGS.model_wo_pseudo_path)
    X = encode_aa_sequences(dataset.Sequence, AA_TO_INT,)
    y = dataset.label.values
    batch_size = 5000
    predictions = make_predictions_with_transformer(
        X, y, batch_size, device, model, input_dim, AA_TO_INT["X"]
    )
    performance = compute_performance_measures(predictions, y)
    print(performance)
    render_roc_curve(predictions, y, FLAGS.results, title, fname)


def main():
    print("Handle K562 datasets")
    DRB1_0101_ligands, DRB1_0404_ligands = load_K562_dataset()
    # To exclude shorter peptides in the test set
    DRB1_0101_ligands = DRB1_0101_ligands.loc[
        DRB1_0101_ligands.Sequence.str.len() >= 15
    ]
    # To exclude peptides shorter than the binding pocket
    DRB1_0404_ligands = DRB1_0404_ligands.loc[
        (DRB1_0404_ligands.Sequence.str.len() >= 9)
    ]
    DRB1_0101, DRB1_0404 = load_DRB1_0101_DRB1_0404()
    DRB1_0101_ligands["Pseudosequence"] = DRB1_0101
    DRB1_0404_ligands["Pseudosequence"] = DRB1_0404
    handle_K562_dataset(
        DRB1_0101_ligands,
        "Differentiating K562 DRB1*01:01 ligands from decoys",
        "DRB1_0101_ligands",
    )
    handle_K562_dataset(
        DRB1_0404_ligands,
        "Differentiating K562 DRB1*04:04 ligands from decoys",
        "DRB1_0404_ligands",
    )

    melanoma_dataset = load_melanoma_dataset()
    print("Handle melanoma datasets")
    print(melanoma_dataset.shape)
    handle_melanoma_dataset(
        melanoma_dataset, "Differentiating melanoma dataset", "Melanoma"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model_with_pseudo_path",
    #     "-modp",
    #     type=str,
    #     help="Path to the checkpoint of the model to evaluate.",
    # )
    parser.add_argument(
        "--model_wo_pseudo_path",
        "-modwop",
        type=str,
        help="Path to the checkpoint of the model to evaluate.",
    )
    parser.add_argument(
        "--results",
        "-ress",
        type=str,
        help="Path storing the results should be stored.",
    )

    FLAGS = parser.parse_args()
    main()
