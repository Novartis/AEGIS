#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""nod.py

This script validates our model against datasets from NOD mice
"""

import argparse
import logging
from pathlib import Path

import hydra
import torch
from experiments.train import prepare_nod_data
from mhciipresentation.constants import AA_TO_INT, USE_GPU
from experiments.inference import make_inference
from mhciipresentation.loaders import load_nod_data, load_nod_idx
from mhciipresentation.utils import (
    encode_aa_sequences,
    make_dir,
    make_predictions_with_transformer,
    render_precision_recall_curve,
    render_roc_curve,
    set_pandas_options,
    get_hydra_logging_directory,
)
from omegaconf import DictConfig
from pyprojroot import here
from torch import nn

set_pandas_options()
logger = logging.getLogger(__name__)

cfg: DictConfig


@hydra.main(
    version_base="1.3", config_path=str(here() / "conf"), config_name="config"
)
def main(nodconfig: DictConfig):
    global cfg
    cfg = nodconfig
    _, _, data, _, _, y = prepare_nod_data()

    if cfg.model.feature_set == "seq_mhc":
        input_dim = 33 + 2 + 34
        X = encode_aa_sequences(
            data.peptide_with_mhcii_pseudosequence,
            AA_TO_INT,
        )
    elif cfg.model.feature_set == "seq_only":
        input_dim = 33 + 2
        X = encode_aa_sequences(
            data.peptide,
            AA_TO_INT,
        )
    else:
        raise ValueError(
            f"Unknown feature set {cfg.model.feature_set}. "
            "Please choose from seq_only or seq_and_mhc"
        )

    make_inference(
        X,
        y,
        cfg,
        input_dim,
        get_hydra_logging_directory() / "nod",
    )


if __name__ == "__main__":
    main()
