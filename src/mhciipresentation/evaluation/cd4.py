#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""cd4.py

This script validates our model against CD4 datasets.
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
    epitope_file_parser,
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
    attach_pseudosequence,
    compute_performance_measures,
    encode_aa_sequences,
    flatten_lists,
    make_predictions_with_transformer,
    render_roc_curve,
    sample_from_human_uniprot,
    set_pandas_options,
)

set_pandas_options()


def main():
    print("CD4")
    epitope_df = epitope_file_parser(EPITOPES_DIR + "CD4_epitopes.fsa")
    epitope_df["label"] = 1
    epitope_df = attach_pseudosequence(epitope_df)
    epitope_df["peptides_and_pseudosequence"] = epitope_df["peptide"].astype(
        str
    ) + epitope_df["Pseudosequence"].astype(str)
    epitope_df.columns = [
        "Sequence",
        "MHC_molecule",
        "protein_id",
        "source_protein",
        "label",
        "Name",
        "Pseudosequence",
        "peptides_and_pseudosequence",
    ]
    epitope_df["Sequence Length"] = epitope_df["Sequence"].str.len()

    epitope_df["Sequence Length"] = epitope_df["Sequence Length"].loc[
        epitope_df["Sequence Length"] <= 25
    ]

    n_len = epitope_df.Sequence.str.len().value_counts().sort_index().to_dict()
    decoys = sample_from_human_uniprot(n_len)
    decoys = pd.DataFrame(flatten_lists(decoys), columns=["Sequence"])
    decoys["label"] = 0
    decoys = assign_pseudosequences(epitope_df, decoys)
    decoys["peptides_and_pseudosequence"] = decoys["Sequence"].astype(
        str
    ) + decoys["Pseudosequence"].astype(str)
    data = pd.concat(
        [
            decoys[["peptides_and_pseudosequence", "label"]],
            epitope_df[["peptides_and_pseudosequence", "label"]],
        ]
    )

    X = encode_aa_sequences(data.peptides_and_pseudosequence, AA_TO_INT,)
    y = data.label.values
    batch_size = 5000

    device = torch.device("cuda" if USE_GPU else "cpu")  # training device
    model, input_dim = setup_model(device, FLAGS.model_with_pseudo_path)

    predictions = make_predictions_with_transformer(
        X, batch_size, device, model, input_dim, AA_TO_INT["X"]
    )
    performance = compute_performance_measures(predictions, y)
    print(performance)
    render_roc_curve(
        predictions,
        y,
        FLAGS.results,
        "CD4 validation set with pseudosequence",
        "CD4_test_with_pseudo",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_with_pseudo_path",
        "-modp",
        type=str,
        help="Path to the checkpoint of the model to evaluate.",
    )
    # parser.add_argument(
    #     "--model_wo_pseudo_path",
    #     "-modwop",
    #     type=str,
    #     help="Path to the checkpoint of the model to evaluate.",
    # )
    parser.add_argument(
        "--results",
        "-ress",
        type=str,
        help="Path storing the results should be stored.",
    )

    FLAGS = parser.parse_args()
    main()
