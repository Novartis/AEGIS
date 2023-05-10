#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""nod.py

This script validates our model against datasets from NOD mice
"""

import argparse
import logging

import hydra
import torch
from experiments.train import prepare_iedb_data
from mhciipresentation.constants import AA_TO_INT, USE_GPU
from mhciipresentation.inference import make_inference, setup_model
from mhciipresentation.loaders import load_iedb_data, load_iedb_idx
from mhciipresentation.utils import (
    encode_aa_sequences,
    make_dir,
    make_predictions_with_transformer,
    render_roc_curve,
    set_pandas_options,
)
from omegaconf import DictConfig
from pyprojroot import here
from sklearn.preprocessing import Binarizer
from torch import nn

set_pandas_options()
logger = logging.getLogger(__name__)

cfg: DictConfig


@hydra.main(
    version_base="1.3", config_path=str(here() / "conf"), config_name="config"
)
def main(holdoutconfig: DictConfig):
    global cfg
    cfg = holdoutconfig
    _, _, data, _, _, y = prepare_iedb_data()

    DRB1_alleles = [
        "DRB1_0101",
        "DRB1_0301",
        "DRB1_0401",
        "DRB1_0405",
        "DRB1_0701",
        "DRB1_0802",
        "DRB1_0901",
        "DRB1_1101",
        "DRB1_1201",
        "DRB1_1601",
        "DRB1_1501",
    ]

    HLA_DR_alleles = ["DRB3_0101", "DRB3_0202", "DRB5_0101"]

    HLA_DQ_alleles = [
        "HLA-DQA10501-DQB10201",
        "HLA-DQA10501-DQB10301",
        "HLA-DQA10301-DQB10302",
        "HLA-DQA10401-DQB10402",
        "HLA-DQA10101-DQB10501",
        "HLA-DQA10102-DQB10602",
    ]

    HLA_DP_alleles = [
        "HLA-DPA10103-DPB10201",
        "HLA-DPA10103-DPB10401",
        "HLA-DPA10103-DPB10601",
        "HLA-DPA10201-DPB10501",
        "HLA-DPA10201-DPB11401",
        "HLA-DPA10301-DPB10402",
    ]

    for allele_list in [
        DRB1_alleles,
        HLA_DR_alleles,
        HLA_DQ_alleles,
        HLA_DP_alleles,
    ]:
        for allele in allele_list:
            allele_data = data[data.Alleles.str.contains(allele)].reset_index(
                drop=True
            )
            if len(allele_data) < 4:
                logger.info(
                    f"Too few alleles for {allele} ({len(allele_data)})"
                )
                continue
            y_alleles = data[
                data.Alleles.str.contains(allele)
            ].target_value.values
            # if len(y_test) < 2:
            #     print("to few values for allele: %s" % allele)
            #     continue
            # Binarize y_test with Binarizer
            y_alleles = Binarizer(threshold=0.5).transform(
                y_alleles.reshape(-1, 1)
            )

            if cfg.model.feature_set == "seq_mhc":
                input_dim = 33 + 2 + 34
                X = encode_aa_sequences(
                    allele_data.peptides_and_pseudosequence,
                    AA_TO_INT,
                )
            elif cfg.model.feature_set == "seq_only":
                input_dim = 33 + 2
                X = encode_aa_sequences(
                    allele_data.peptide,
                    AA_TO_INT,
                )
            else:
                raise ValueError(
                    f"Unknown feature set {cfg.model.feature_set}. "
                    "Please choose from seq_only or seq_and_mhc"
                )
            make_inference(
                X,
                y_alleles,
                cfg,
                input_dim,
                here() / "outputs" / "hold_out" / allele,
            )


if __name__ == "__main__":
    main()
