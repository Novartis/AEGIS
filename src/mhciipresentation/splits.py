#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""splits.py

This script generates different train, test and validation splits for the
various datasets used in the mhciipresentation project containing data coming
from single allele, eluted peptides as well as randomly sampled peptides as
achieved by Nielsen and colleagues in:
https://academic.oup.com/nar/article/48/W1/W449/5837056
"""

import os
import random
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mhciipresentation.constants import N_MOTIF_FOLDS
from mhciipresentation.human.human import load_raw_files, select_data_files
from mhciipresentation.loaders import load_nod_data, load_sa_data
from mhciipresentation.paths import LEVENSTEIN_DIR, RAW_DATA, SPLITS_DIR
from mhciipresentation.utils import make_dir


def remove_overlapping_peptides(
    peptides_1: Set,
    peptides_2: Set,
) -> pd.Series:
    """Removes peptides occuring in peptides_2 from peptides_1

    Args:
        peptides_1 (np.ndarray): peptides to remove from
        peptides_2 (np.ndarray): peptides to check against

    Returns:
        pd.Series: peptides_1, which does not contain elements present in
            peptides_2
    """
    # Creates a difference between two sets to get the duplicate peptides as
    # fast as possible
    peptides_1_reduced = peptides_1.difference(peptides_2)

    # Remove features and labels that correspond to duplicate peptides
    return peptides_1_reduced, peptides_2


def validate_split(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray = None
) -> None:
    """Validates the splits by checking absence of overlap of peptides among
        splits

    Args:
        X_train (np.ndarray): training features
        X_val (np.ndarray): validation features
        X_test (np.ndarray): testing features
    """

    print(
        f"Overlap between train and val"
        f" {len(set(X_train).intersection(set(X_val)))}"
    )
    if X_test is not None:
        print(
            f"Overlap between train and test"
            f" {len(set(X_train).intersection(set(X_test)))}"
        )

        print(
            f"Overlap between val and test"
            f" {len(set(X_val).intersection(set(X_test)))}"
        )


def save_idx(
    out_dir: Path,
    X_train_data: pd.DataFrame,
    X_val_data: pd.DataFrame,
    X_test_data: pd.DataFrame = None,
) -> None:
    X_train_data.index.to_frame(name="index").to_csv(
        out_dir / "X_train_idx.csv", index=False
    )
    X_val_data.index.to_frame(name="index").to_csv(
        out_dir / "X_val_idx.csv", index=False
    )
    if X_test_data is not None:
        X_test_data.index.to_frame(name="index").to_csv(
            out_dir / "X_test_idx.csv", index=False
        )


def label_dist_summary(
    data: pd.DataFrame, target_col: str, dataset_name: str
) -> None:
    """Summarizes the distribution of the two most common labels (ie 0 and 1)

    Note: labels for BA data are not accounted for here, therefore do not use this for counting purposes.

    Args:
        data (pd.DataFrame): data to summarize
        target_col (str): target column to summarize
        dataset_name (str): name of the dataset to summarize
    """
    value_cts = data[target_col].value_counts()
    print(
        f"Label distribution in {dataset_name}: negative samples = "
        f" {value_cts[0]}; positive samples = {value_cts[1]}"
    )


def random_splitting(
    data: pd.DataFrame,
    out_dir: str = SPLITS_DIR / "random/",
    eval_frac: float = 0.2,
    val_frac: float = 0.5,
) -> None:
    """Randomly splits the data and saves the resulting indices under SPLITS_DIR
        + "/random/".

    Args:
        data (pd.DataFrame): input dataset
        out_dir (str): output directory
        eval_frac (float): fraction of the total dataset used for evaluation
            (i.e. validation and testing)
        val_frac (float): fraction of the evaluation set used for validation
    """
    # *_tmp contains the data from all evaludation sets (validation + test sets)
    X_train, X_eval, y_train, y_eval = train_test_split(
        data.peptide.values,
        data.target_value.values,
        test_size=eval_frac,
        random_state=42,
    )

    # Remove peptides occuring in the test set from training set
    X_train, X_eval = remove_overlapping_peptides(set(X_train), set(X_eval))
    X_train_data = data[data["peptide"].isin(X_train)]
    X_eval_data = data[data["peptide"].isin(X_eval)]

    # Generate dev and validation set from test set
    X_val = X_eval_data.sample(frac=val_frac, random_state=42).peptide
    X_test = X_eval_data.drop(X_val.index).peptide

    # Remove peptides occuring in validation set from test set
    if X_test.shape[0] != 0:
        X_test, X_val = remove_overlapping_peptides(set(X_val), set(X_test))
        X_test_data = data[data["peptide"].isin(X_test)]
    else:
        X_test_data = None

    X_val_data = data[data["peptide"].isin(X_val)]

    if X_test_data is not None:
        validate_split(
            X_train_data.peptide, X_val_data.peptide, X_test_data.peptide
        )
    else:
        validate_split(
            X_train_data.peptide,
            X_val_data.peptide,
        )

    # Summary of samples sizes
    label_dist_summary(X_train_data, "target_value", "training")
    label_dist_summary(X_val_data, "target_value", "validation")
    if X_test_data is not None:
        label_dist_summary(X_test_data, "target_value", "testing")

    # Writing the data
    make_dir(out_dir)

    save_idx(out_dir, X_train_data, X_val_data, X_test_data)
    print("Written random splits successfully")


def random_splitting_nod(data: pd.DataFrame) -> None:
    """We stratify by protein name.

    Args:
        data (pd.DataFrame): dataset to stratify
    """
    unique_proteins = set(
        data.loc[data.label == 1]["Uniprot Accession"].to_list()
    )
    val_proteins = set(
        random.sample(unique_proteins, int(len(unique_proteins) * 0.1))
    )
    test_proteins = set(
        random.sample(
            unique_proteins - val_proteins, int(len(unique_proteins) * 0.1)
        )
    )
    train_proteins = unique_proteins - test_proteins.union(val_proteins)

    X_train_data = data.loc[data["Uniprot Accession"].isin(train_proteins)]
    X_val_data = data.loc[data["Uniprot Accession"].isin(val_proteins)]
    X_test_data = data.loc[data["Uniprot Accession"].isin(test_proteins)]

    X_train_data_neg = data.loc[data.label == 0].sample(len(X_train_data) * 5)
    data = data.loc[data.label == 0].loc[
        ~data["Peptide Sequence"].isin(
            set(X_train_data_neg["Peptide Sequence"].to_list())
        )
    ]
    X_val_data_neg = data.loc[data.label == 0].sample(len(X_val_data) * 5)
    data = data.loc[data.label == 0].loc[
        ~data["Peptide Sequence"].isin(
            set(X_val_data_neg["Peptide Sequence"].to_list()).union(
                set(X_train_data_neg["Peptide Sequence"].to_list())
            )
        )
    ]
    X_test_data_neg = data.loc[data.label == 0].sample(len(X_test_data) * 5)

    X_train_data = X_train_data.append(X_train_data_neg)
    X_val_data = X_val_data.append(X_val_data_neg)
    X_test_data = X_test_data.append(X_test_data_neg)

    # Summary of samples sizes
    label_dist_summary(X_train_data, "label", "training")
    label_dist_summary(X_val_data, "label", "validation")
    label_dist_summary(X_test_data, "label", "testing")

    # Writing the data
    out_dir = SPLITS_DIR / "random_nod"
    make_dir(out_dir)
    save_idx(out_dir, X_train_data, X_val_data, X_test_data)
    print("Written random splits successfully")


def main():
    print("Splitting SA EL + BA data randomly")
    sa_data = load_sa_data()
    make_dir(SPLITS_DIR)
    random_splitting(
        sa_data,
        out_dir=SPLITS_DIR / "random_iedb/",
        val_frac=0.5,
        eval_frac=0.05,
    )

    print("Random splitting of mouse data")
    mouse_data = load_nod_data()
    random_splitting_nod(mouse_data)


if __name__ == "__main__":
    main()
