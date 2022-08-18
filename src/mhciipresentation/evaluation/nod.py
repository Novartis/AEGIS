#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""maria.py

This script validates our model against datasets from NOD mice
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

from mhciipresentation.constants import AA_TO_INT, USE_GPU
from mhciipresentation.inference import setup_model
from mhciipresentation.loaders import (
    load_internal_mouse_data,
    load_K562_dataset,
    load_melanoma_dataset,
    load_pseudosequences,
    load_uniprot,
)
from mhciipresentation.paths import (
    DATA_DIR,
    EPITOPES_DIR,
    RAW_DATA,
    SPLITS_DIR,
)
from mhciipresentation.transformer import (
    TransformerModel,
    evaluate_transformer,
)
from mhciipresentation.utils import (
    compute_performance_measures,
    encode_aa_sequences,
    flatten_lists,
    make_predictions_with_transformer,
    render_roc_curve,
    set_pandas_options,
)

set_pandas_options()


def handle_public_NOD(
    model: nn.Module, input_dim: int, device: torch.device, batch_size: int
):
    """Handles validation against part of the public NOD dataset

    Args:
        model (nn.Module): model used to make predictions
        input_dim (int): input dimension of the model
        device (torch.device): device to run the model on.
        batch_size (int): batch size to use for inference
    """
    in_dir = SPLITS_DIR + "/mouse/" + "/random/"
    test_data = pd.read_csv(in_dir + "X_test.csv")
    test_data = test_data.loc[test_data["Peptide Sequence"].str.len() <= 25]

    X = encode_aa_sequences(test_data["Peptide Sequence"], AA_TO_INT,)
    y = test_data.label.values

    predictions = make_predictions_with_transformer(
        X, y, batch_size, device, model, input_dim, AA_TO_INT["X"]
    )
    performance = compute_performance_measures(predictions, y)
    print(performance)
    render_roc_curve(
        predictions,
        y,
        FLAGS.results,
        "NOD mouse public dataset, different proteins",
        "nod_test",
    )


def main():
    device = torch.device("cuda" if USE_GPU else "cpu")  # training device
    model, input_dim = setup_model(device, FLAGS.model_wo_pseudo_path)
    batch_size = 5000
    handle_public_NOD(model, input_dim, device, batch_size)


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
