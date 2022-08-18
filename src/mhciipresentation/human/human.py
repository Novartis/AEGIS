#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""human.py

This script extracts the single allele elution data from the raw data.

"""

import os
from typing import List, Tuple

import pandas as pd

from mhciipresentation.loaders import load_raw_files
from mhciipresentation.paths import PROCESSED_DATA, RAW_DATA


def select_data_files(list_of_files: List) -> List:
    """Selects files containing elution data either from training or testing
        partitions

    Args:
        list_of_files (List): list of files in a given directory

    Returns:
        List: filtered list of relevant files to process
    """
    return [
        fname
        for fname in list_of_files
        if "EL" in fname and "train" in fname or "test" in fname
    ]


def get_pseudosequences(
    fcontent: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the allele list and

    Args:
        fcontent (pd.DataFrame): peptide and associated MHC source information

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: allelelist and mhcii molecules
    """
    # Load necessary files
    allelelist = pd.read_csv(
        RAW_DATA + "allelelist.txt",
        sep=" ",
        index_col=0,
        names=["Name", "Alleles"],
    )
    mhcii_molecules = pd.read_csv(
        RAW_DATA + "pseudosequence_mapping.dat",
        sep="\t",
        names=["Name", "Pseudosequence"],
        index_col=0,
    )
    return (allelelist, mhcii_molecules)


def filter_el_samples(fcontent: pd.DataFrame) -> pd.DataFrame:
    """Finds elution data samples. This is done by looking at the peptide
        context, because ba data does not contain peptides with a context in the
        public data sets and adds Xs for padding.

    Args:
        fcontent (pd.DataFrame): list of samples containing eluted and binbing
        affinity data

    Returns:
        pd.DataFrame: samples containing only elution data samples
    """
    fcontent = fcontent[~fcontent["peptide_context"].str.contains("X")]
    # There's one string that seems to be 6 so we remove it to avoid cetain errors
    fcontent = fcontent.loc[fcontent["peptide_context"].str.len() == 12]
    return fcontent


def filter_sa_samples(fcontent: pd.DataFrame) -> pd.DataFrame:
    """Finds single-allele samples

    Args:
        fcontent (pd.DataFrame): samples containing both multi allele and single allele data

    Returns:
        pd.DataFrame: samples containing single alleles only
    """
    allelelist, mhcii_molecules = get_pseudosequences(fcontent)
    # First we need to be sure the publication/dataset where the MHC peptide
    # comes from contains one MHC molecule
    allelelist["number_of_alleles"] = allelelist.Alleles.str.split(",").apply(
        lambda x: len(x)
    )
    allelelist_subset = allelelist[allelelist["number_of_alleles"] == 1]

    # Get pseudosequences and return

    return fcontent.merge(
        allelelist_subset, left_on="MHC_molecule", right_on="Name", how="inner"
    ).merge(mhcii_molecules, left_on="Alleles", right_on="Name", how="inner")


def main():
    """Main function extracting single allele and elution data from the list of combinations"""
    list_of_peptide_files = select_data_files(os.listdir(RAW_DATA))
    list_of_peptide_files.sort()
    raw_files = load_raw_files(list_of_peptide_files)
    unique_samples = raw_files.drop_duplicates()
    sa_data = filter_sa_samples(unique_samples)
    sa_el_data = filter_el_samples(sa_data)
    # The df needs to be re-indexed from 0 to len(sa_el_data) - 1 otherwise we
    # have troubles down the road.
    sa_el_data.reset_index(drop=True).to_csv(PROCESSED_DATA + "sa_el_data.csv")
    sa_data.reset_index(drop=True).to_csv(PROCESSED_DATA + "sa_data.csv")


if __name__ == "__main__":
    main()
