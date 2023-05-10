#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""stratmann.py

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
from mhciipresentation.inference import make_inference, setup_model
from mhciipresentation.loaders import (
    load_K562_dataset,
    load_melanoma_dataset,
    load_nod_data,
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


def load_stratmann():
    stratmann = pd.read_csv(
        here() / "data" / "raw" / "stratmann.csv", header=None
    )
    stratmann.columns = ["peptide"]
    pseudosequences = load_pseudosequences()
    iag7_pseudosequence = pseudosequences.loc[
        pseudosequences.Name == "H-2-IAg7"
    ].Pseudosequence.values[0]
    stratmann["pseudosequence"] = iag7_pseudosequence
    stratmann["peptides_and_pseudosequence"] = stratmann.apply(
        lambda x: x.peptide + x.pseudosequence, axis=1
    )
    stratmann["label"] = 1
    # Add decoys to the dataset
    nod_data = load_nod_data()
    nod_data_decoy = nod_data.loc[nod_data.label == 0]
    # Sample the same number of decoys as there are targets
    stratmann_decoy = nod_data_decoy.sample(n=len(stratmann), random_state=42)
    # Rename the peptide column to match the stratmann data
    stratmann_decoy = stratmann_decoy.rename(
        columns={"Peptide Sequence": "peptide"}
    )
    stratmann_decoy["pseudosequence"] = iag7_pseudosequence
    stratmann_decoy["peptides_and_pseudosequence"] = stratmann_decoy.apply(
        lambda x: x.peptide + x.pseudosequence, axis=1
    )
    stratmann = pd.concat([stratmann, stratmann_decoy])
    return stratmann


@hydra.main(
    version_base="1.3", config_path=str(here() / "conf"), config_name="config"
)
def main(stratmanconfig: DictConfig) -> None:
    global cfg
    cfg = stratmanconfig
    df = load_stratmann()
    if cfg.model.feature_set == "seq_mhc":
        input_dim = 33 + 2 + 34
        X = encode_aa_sequences(
            df.peptides_and_pseudosequence,
            AA_TO_INT,
        )
    elif cfg.model.feature_set == "seq_only":
        input_dim = 33 + 2
        X = encode_aa_sequences(
            df.peptide,
            AA_TO_INT,
        )
    else:
        raise ValueError(
            f"Unknown feature set {cfg.model.feature_set}. "
            "Please choose from seq_only or seq_and_mhc"
        )
    make_inference(
        X,
        df.label.values,
        cfg,
        input_dim,
        here() / "outputs" / "stratmann",
    )


if __name__ == "__main__":
    main()
