#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""xu.py

Validating against Xu dataset
"""

import argparse
import logging
from pathlib import Path

import hydra
import torch
from experiments.train import prepare_nod_data
from mhciipresentation.constants import AA_TO_INT, USE_GPU
from experiments.inference import make_inference
from mhciipresentation.loaders import (
    load_nod_data,
    load_nod_idx,
    load_you_dataset,
    load_xu_dataset,
)
from mhciipresentation.utils import (
    encode_aa_sequences,
    make_dir,
    make_predictions_with_transformer,
    render_precision_recall_curve,
    render_roc_curve,
    set_pandas_options,
    get_hydra_logging_directory,
)
from mhciipresentation.human.human import get_pseudosequences
from omegaconf import DictConfig
from pyprojroot import here
from torch import nn
import pandas as pd

set_pandas_options()
logger = logging.getLogger(__name__)

cfg: DictConfig


def prepare_xu_dataset():
    """Prepares the XU dataset for evaluation"""

    xu_dataset = load_xu_dataset()
    allelelist, mhcii_molecules = get_pseudosequences(
        xu_dataset,
    )
    mhcii_molecules = mhcii_molecules.reset_index()
    mhcii_molecules["Name"] = (
        "DRB" + mhcii_molecules.Name.str.split("DRB").str[-1]
    )
    xu_dataset = pd.merge(
        xu_dataset, mhcii_molecules, left_on="HLA", right_on="Name"
    )
    xu_dataset = xu_dataset.drop("Name", axis=1)
    xu_dataset["peptides_and_pseudosequence"] = (
        xu_dataset["peptide"] + xu_dataset["Pseudosequence"]
    )
    return xu_dataset


@hydra.main(
    version_base="1.3", config_path=str(here() / "conf"), config_name="config"
)
def main(xuconfig: DictConfig):
    global cfg
    cfg = xuconfig
    xu_dataset = prepare_xu_dataset()
    if cfg.model.feature_set == "seq_mhc":
        input_dim = 33 + 2 + 34
        X = encode_aa_sequences(
            xu_dataset.peptides_and_pseudosequence,
            AA_TO_INT,
        )
    elif cfg.model.feature_set == "seq_only":
        input_dim = 33 + 2
        X = encode_aa_sequences(
            xu_dataset.peptide,
            AA_TO_INT,
        )
    else:
        raise ValueError(
            f"Unknown feature set {cfg.model.feature_set}. "
            "Please choose from seq_only or seq_and_mhc"
        )
    make_inference(
        X,
        xu_dataset.result.values,
        cfg,
        input_dim,
        get_hydra_logging_directory() / "xu",
    )


if __name__ == "__main__":
    main()
