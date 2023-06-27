#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""maria.py

This script validates our model against datasets from against which MARIA was
evaluated.
"""

import argparse
import copy
import json
import logging
import pprint
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from mhciipresentation.constants import (
    AA_TO_INT,
    LEN_BOUNDS_HUMAN,
    USE_CONTEXT,
    USE_GPU,
    USE_SUBSET,
)
from experiments.inference import make_inference, setup_model
from mhciipresentation.loaders import (
    load_K562_dataset,
    load_melanoma_dataset,
    load_pseudosequences,
    load_uniprot,
)
from mhciipresentation.metrics import (
    build_scalar_metrics,
    build_vector_metrics,
    compute_performance_metrics,
    save_performance_metrics,
)
from mhciipresentation.paths import DATA_DIR, EPITOPES_DIR, RAW_DATA
from mhciipresentation.utils import (
    assign_pseudosequences,
    encode_aa_sequences,
    flatten_lists,
    get_accelerator,
    get_hydra_logging_directory,
    make_dir,
    make_predictions_with_transformer,
    render_precision_recall_curve,
    render_roc_curve,
    sample_from_human_uniprot,
    set_pandas_options,
)
from omegaconf import DictConfig
from pyprojroot import here
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

set_pandas_options()
logger = logging.getLogger(__name__)

cfg: DictConfig


def load_DRB1_0101_DRB1_0404() -> List[str]:
    """Loads DRB1_0101 and DRB1_0404

    Returns:
        List[str]: the two pseudosequences
    """
    pseudosequences = load_pseudosequences()
    return pseudosequences.loc[
        pseudosequences.Name.isin(["DRB1_0101", "DRB1_0404"])
    ].Pseudosequence.to_list()


def handle_K562_dataset(ligands: pd.DataFrame, fname: str) -> None:
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
    data = pd.concat(
        [
            ligands[["Sequence", "label", "Pseudosequence"]],
            decoys[["Sequence", "label", "Pseudosequence"]],
        ]
    )
    data["peptides_and_pseudosequence"] = data["Sequence"].astype(str) + data[
        "Pseudosequence"
    ].astype(str)

    device = torch.device("cuda" if USE_GPU else "cpu")  # training device

    if cfg.model.feature_set == "seq_mhc":
        input_dim = 33 + 2 + 34
        X = encode_aa_sequences(
            data.peptides_and_pseudosequence,
            AA_TO_INT,
        )
    elif cfg.model.feature_set == "seq_only":
        input_dim = 33 + 2
        X = encode_aa_sequences(
            data.Sequence,
            AA_TO_INT,
        )
    else:
        raise ValueError(
            f"Unknown feature set {cfg.model.feature_set}. "
            "Please choose from seq_only or seq_and_mhc"
        )

    make_inference(
        X,
        data.label.values,
        cfg,
        input_dim,
        get_hydra_logging_directory() / "K562" / fname,
    )


def handle_melanoma_dataset(ligands: pd.DataFrame, fname: str) -> None:
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
    data = pd.concat(
        [
            ligands[["Sequence", "label"]],
            decoys[["Sequence", "label"]],
        ]
    )
    device = torch.device("cuda" if USE_GPU else "cpu")  # training device

    if cfg.model.feature_set == "seq_only":
        input_dim = 33 + 2
        X = encode_aa_sequences(
            data.Sequence,
            AA_TO_INT,
        )
        make_inference(
            X,
            data.label.values,
            cfg,
            input_dim,
            get_hydra_logging_directory() / "melanoma",
        )
    else:
        logger.info("Not possible")


@hydra.main(
    version_base="1.3", config_path=str(here() / "conf"), config_name="config"
)
def main(mariaconfig: DictConfig) -> None:
    global cfg
    cfg = mariaconfig
    make_dir(Path("./data/evaluation/"))
    logger.info("Handle K562 datasets")
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
    logger.info("DRB1_0101")
    handle_K562_dataset(
        DRB1_0101_ligands,
        "DRB1_0101_ligands",
    )
    logger.info("DRB1_0404")
    handle_K562_dataset(
        DRB1_0404_ligands,
        "DRB1_0404_ligands",
    )
    logger.info("DRB1_0404")
    melanoma_dataset = load_melanoma_dataset()
    logger.info("Handle melanoma datasets")
    logger.info(melanoma_dataset.shape)
    handle_melanoma_dataset(melanoma_dataset, "Melanoma")


if __name__ == "__main__":
    main()
