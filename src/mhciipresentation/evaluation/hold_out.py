#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""nod.py

This script validates our model against datasets from NOD mice
"""

import argparse

import torch
from torch import nn

from mhciipresentation.constants import AA_TO_INT, USE_GPU
from mhciipresentation.inference import setup_model
from mhciipresentation.loaders import load_iedb_data, load_iedb_idx
from mhciipresentation.transformer import prepare_iedb_data
from mhciipresentation.utils import (
    compute_performance_measures,
    encode_aa_sequences,
    make_dir,
    make_predictions_with_transformer,
    render_roc_curve,
    set_pandas_options,
)

set_pandas_options()


def handle_iedb_data_subset(
    model: nn.Module, input_dim: int, device: torch.device, batch_size: int
):
    """Handles validation against part of the public NOD dataset

    Args:
        model (nn.Module): model used to make predictions
        input_dim (int): input dimension of the model
        device (torch.device): device to run the model on.
        batch_size (int): batch size to use for inference
    """
    _, _, all_test_data, _, _, all_y_test = prepare_iedb_data()

    DRB1_alleles = ["DRB1_0101", "DRB1_0301", "DRB1_0401", "DRB1_0405",
                    "DRB1_0701", "DRB1_0802", "DRB1_0901", "DRB1_1101",
                    "DRB1_1201", "DRB1_1601","DRB1_1501"]

    HLA_DR_alleles = ["DRB3_0101", "DRB3_0202", "DRB5_0101"]

    HLA_DQ_alleles = ["HLA-DQA10501-DQB10201", "HLA-DQA10501-DQB10301", "HLA-DQA10301-DQB10302",
                      "HLA-DQA10401-DQB10402","HLA-DQA10101-DQB10501", "HLA-DQA10102-DQB10602"]

    HLA_DP_alleles = ['HLA-DPA10103-DPB10201', 'HLA-DPA10103-DPB10401', 'HLA-DPA10103-DPB10601',
                      'HLA-DPA10201-DPB10501', 'HLA-DPA10201-DPB11401', 'HLA-DPA10301-DPB10402']

    for allele_list in [DRB1_alleles, HLA_DR_alleles, HLA_DQ_alleles, HLA_DP_alleles]:
        for allele in allele_list:
            test_data = all_test_data[all_test_data.Alleles.str.contains(allele)].reset_index(drop=True)
            y_test = all_y_test[all_test_data.Alleles.str.contains(allele)]
            if len(y_test)<2:
                print("to few values for allele: %s" %allele)
                continue

            for i, j in enumerate(y_test):
                if j>=0.5:
                    y_test[i] = 1
        
                else:
                    y_test[i] = 0

            if sum(y_test)==len(y_test) or sum(y_test)==0:
                print("all values for allele: %s are the same" % allele)
                continue

    
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

    
            predictions = make_predictions_with_transformer(
                X, batch_size, device, model, input_dim, AA_TO_INT["X"]
            )
            performance = compute_performance_measures(predictions, y_test)
            print("Allele Name: %s" % allele)
            print("number datapoints: %i" %len(y_test)) 
            print("AUC: %(auc)s, F1: %(f1)s, MCC; %(matthews_corrcoef)s" % performance)
            print("")
#            print(performance)

#    make_dir(FLAGS.results)
#    render_roc_curve(
#        predictions,
#        y_test,
#        FLAGS.results,
#        "Allele: %s",
#        "nod_test",
#    )
#

def main():
    device = torch.device("cuda" if USE_GPU else "cpu")  # training device

    if FLAGS.model_with_pseudo_path is not None:
        model, input_dim, max_len = setup_model(device, FLAGS.model_with_pseudo_path)
    else:
        model, input_dim, max_len = setup_model(device, FLAGS.model_wo_pseudo_path)

    batch_size = max_len
    handle_iedb_data_subset(model, input_dim, device, batch_size)


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
        help="Path where the results should be stored.",
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
