#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""nod.py

This script validates our model against datasets from NOD mice
"""

import argparse

import torch
from torch import nn

from mhciipresentation.constants import AA_TO_INT, USE_GPU
from mhciipresentation.transformer import prepare_nod_data
from mhciipresentation.inference import setup_model
from mhciipresentation.loaders import load_nod_data, load_nod_idx
from mhciipresentation.utils import (
    compute_performance_measures,
    encode_aa_sequences,
    make_dir,
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
    # nod_data = load_nod_data()
    # _, _, X_test_idx = load_nod_idx()
    # test_data = nod_data.iloc[X_test_idx["index"]]
    _,_,test_data,_,_,y_test = prepare_nod_data()

    if FLAGS.model_with_pseudo_path is not None:
        X = encode_aa_sequences(
            test_data.peptide_with_mhcii_pseudosequence,
            AA_TO_INT,
        )
    else:
        X = encode_aa_sequences(
            test_data["peptide"],
            AA_TO_INT,
        )

    #y = test_data.label.values

    predictions = make_predictions_with_transformer(
        X, batch_size, device, model, input_dim, AA_TO_INT["X"]
    )
    performance = compute_performance_measures(predictions, y_test)
    print(performance)
    make_dir(FLAGS.results)
    render_roc_curve(
        predictions,
        y_test,
        FLAGS.results,
        "NOD mouse public dataset, different proteins",
        "nod_test",
    )


def main():
    device = torch.device("cuda" if USE_GPU else "cpu")  # training device
    if FLAGS.model_wo_pseudo_path is not None:
        model, input_dim = setup_model(device, FLAGS.model_wo_pseudo_path)
    else:
        model, input_dim = setup_model(device, FLAGS.model_with_pseudo_path)

    batch_size = 5000
    handle_public_NOD(model, input_dim, device, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_with_pseudo_path",
        "-modp",
        type=str,
        help="Path to the checkpoint of the model to evaluate.",
    )
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
    parser.add_argument(
        "--features",
        required=False,
        help="Type of features to use",
        choices=[
            "seq_only",
            "seq_mhc",
        ],
        default="seq_mhc",
    )
    global FLAGS
    FLAGS = parser.parse_args()
    main()
