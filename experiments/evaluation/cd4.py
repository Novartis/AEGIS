#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""cd4.py

This script validates our model against CD4 datasets.
"""

import logging
import warnings
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
from mhciipresentation.callbacks import GPUUsageLogger, VectorLoggingCallback
from mhciipresentation.constants import AA_TO_INT
from experiments.inference import make_inference
from mhciipresentation.loaders import epitope_file_parser
from mhciipresentation.metrics import (
    build_scalar_metrics,
    build_vector_metrics,
    compute_performance_metrics,
    save_performance_metrics,
)
from mhciipresentation.paths import RAW_DATA
from mhciipresentation.utils import (
    assign_pseudosequences,
    attach_pseudosequence,
    encode_aa_sequences,
    flatten_lists,
    get_accelerator,
    get_hydra_logging_directory,
    make_dir,
    render_precision_recall_curve,
    render_roc_curve,
    sample_from_human_uniprot,
)
from omegaconf import DictConfig
from pyprojroot import here
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

cfg: DictConfig

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3", config_path=str(here() / "conf"), config_name="config"
)
def main(cd4config: DictConfig) -> None:
    global cfg
    cfg = cd4config
    logger.info("CD4")
    logger.info(
        "Training with the following"
        f" config:\n{omegaconf.omegaconf.OmegaConf.to_yaml(cfg)}"
    )
    epitope_df = epitope_file_parser(RAW_DATA / "CD4_epitopes.fsa")
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
            decoys[["peptides_and_pseudosequence", "Sequence", "label"]],
            epitope_df[["peptides_and_pseudosequence", "Sequence", "label"]],
        ]
    )

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
        get_hydra_logging_directory() / "performance_metrics",
    )


if __name__ == "__main__":
    main()
