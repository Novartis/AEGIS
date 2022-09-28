#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""loaders.py

This is the file where all data loader convenience functions are placed.

"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser
from scipy import sparse
from tqdm import tqdm

from mhciipresentation.constants import RAW_PUBLIC_FILE_COL_NAMES
from mhciipresentation.errors import FormatError
from mhciipresentation.paths import (
    CACHE_DIR,
    DATA_DIR,
    ENCODED_DATA,
    PROCESSED_DATA,
    RAW_DATA,
    SPLITS_DIR,
)


def load_pseudosequences() -> pd.DataFrame:
    """Loads pseudosequence file generated by Nielsen et al.

    Returns:
        pd.DataFrame: pseudosequences for most known human/mouse alleles.
    """
    mhcii_pseudosequences = pd.read_csv(
        os.path.join(RAW_DATA / "pseudosequence_mapping.dat"),
        sep="\t",
        names=["Name", "Pseudosequence"],
        index_col=0,
    ).reset_index()
    return mhcii_pseudosequences


def load_sa_el_data() -> pd.DataFrame:
    """Loads SA EL data only

    Returns:
        pd.DataFrame: SA EL data
    """
    return pd.read_csv(PROCESSED_DATA / "iedb_sa_data.csv", index_col=0)


def load_sa_data() -> pd.DataFrame:
    """Loads all SA data, including BA + EL data.

    Returns:
        pd.DataFrame: BA + EL data.
    """
    return pd.read_csv(PROCESSED_DATA / "sa_data.csv", index_col=0)


def load_iedb_idx() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads SA EL radom indexes with peptide exclusion.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val and test
            indices.
    """
    in_dir = SPLITS_DIR / "random_iedb/"
    X_train_idx = pd.read_csv(in_dir / "X_train_idx.csv")
    X_val_idx = pd.read_csv(in_dir / "X_val_idx.csv")
    X_test_idx = pd.read_csv(in_dir / "X_test_idx.csv")
    return X_train_idx, X_val_idx, X_test_idx


def load_nod_idx() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads SA EL radom indexes with peptide exclusion.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val and test
            indices.
    """
    in_dir = SPLITS_DIR / "random_nod/"
    X_train_idx = pd.read_csv(in_dir / "X_train_idx.csv")
    X_val_idx = pd.read_csv(in_dir / "X_val_idx.csv")
    X_test_idx = pd.read_csv(in_dir / "X_test_idx.csv")
    return X_train_idx, X_val_idx, X_test_idx


def load_sa_random_idx() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load random SA index.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: indices of train and validation sets
            to index sa_data.
    """
    in_dir = SPLITS_DIR / "/random_sa/"
    X_train_idx = pd.read_csv(in_dir + "X_train_idx.csv")
    X_val_idx = pd.read_csv(in_dir + "X_val_idx.csv")
    return X_train_idx, X_val_idx


def load_mouse_random_idx() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load randomized mouse data index. Data does not contain overlapping
        peptides.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: data with randomized
        index
    """
    in_dir = SPLITS_DIR / "/mouse/random/"
    X_train_idx = pd.read_csv(in_dir / "X_train_idx.csv")
    X_val_idx = pd.read_csv(in_dir / "X_val_idx.csv")
    X_test_idx = pd.read_csv(in_dir / "X_test_idx.csv")
    return X_train_idx, X_val_idx, X_test_idx


def load_motif_exclusion_idx(fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load motif exclusion dataset

    Args:
        fold (int): folder index

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: loaded train/val split using index.
    """
    in_dir = SPLITS_DIR + "/motifs/"
    X_train_idx = pd.read_csv(in_dir / f"split_{fold}/" / "X_train_idx.csv")
    X_val_idx = pd.read_csv(in_dir / f"split_{fold}/" / "X_val_idx.csv")
    return X_train_idx, X_val_idx


def load_sa_el_levenstein_idx() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Loads Levenstein stratification of the datasets.

    Returns:
        Tuple[ pd.DataFrame, pd.DataFrame, pd.DataFrame ]: Levenstein
    """
    in_dir = SPLITS_DIR / "/levenstein/"
    X_train_idx = pd.read_csv(in_dir / "X_train_idx.csv")
    X_val_idx = pd.read_csv(in_dir / "X_val_idx.csv")
    X_test_idx = pd.read_csv(in_dir / "X_test_idx.csv")
    return X_train_idx, X_val_idx, X_test_idx


def load_training_data() -> Tuple[sparse.csr.csr_matrix, np.ndarray]:
    """Loads training data for sa el features as sparse matrix.

    Returns:
        Tuple[sparse.csr.csr_matrix, np.ndarray]: [description]
    """
    # Load features
    features = sparse.load_npz(ENCODED_DATA / "encoded_sa_el_features.npz")

    # Load labels
    labels = pd.read_csv(
        ENCODED_DATA / "encoded_sa_el_labels.csv", index_col=0
    )

    return features, labels


def load_raw_file(fname: str) -> pd.DataFrame:
    """Load individual raw file from netMHCIIpan v4 dataset.

    Args:
        fname (str): file name

    Returns:
        pd.DataFrame: loaded files
    """
    df = pd.read_csv(
        RAW_DATA / fname, names=RAW_PUBLIC_FILE_COL_NAMES, sep="\t"
    )
    return df


def load_raw_files(list_of_peptide_files: List) -> pd.DataFrame:
    """Load raw files from netMHCIIpan v4 dataset.

    Args:
        list_of_peptide_files (List): [description]

    Returns:
        pd.DataFrame: [description]
    """
    loaded_data = pd.DataFrame()
    for fname in tqdm(list_of_peptide_files):
        fcontent = load_raw_file(fname)
        fcontent["file_name"] = fname
        loaded_data = pd.concat([loaded_data, fcontent])
    return loaded_data


def load_nod_data() -> pd.DataFrame:
    """Load peprocessed mouse data
    Source: https://www.nature.com/articles/s41590-020-0623-7.pdf?proof=t

    Returns:
        pd.DataFrame: preprocessed mouse data.
    """
    return pd.read_csv(
        PROCESSED_DATA / "preprocessed_public_mouse_data.csv", index_col=0
    )


def load_iedb_data() -> pd.DataFrame:
    """Loads preprocessed iedb data.

    Returns:
        pd.DataFrame: IEDB data.
    """
    return pd.read_csv(PROCESSED_DATA / "sa_data.csv", index_col=0)


def load_public_mouse_train_data() -> pd.DataFrame:
    """Load peprocessed mouse data
    Source: https://www.nature.com/articles/s41590-020-0623-7.pdf?proof=t

    Returns:
        pd.DataFrame: preprocessed mouse data.
    """
    return pd.read_csv(RAW_DATA / "train_set.csv", index_col=0)


def load_K562_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load K562 antigens bound to DRB1_0101 and DRB1_0404.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: [description]
    """
    supp_tables = pd.ExcelFile(RAW_DATA / "supplementary_tables.xlsx")
    DRB1_0101_ligands = supp_tables.parse("TableS5", header=1)
    DRB1_0404_ligands = supp_tables.parse("TableS6", header=1)
    return DRB1_0101_ligands, DRB1_0404_ligands


def load_uniprot() -> pd.DataFrame:
    """Load SwissProt database (human reviewed UniProt)

    Raises:
        FormatError: Formatting error if introduced in the UniProt header.

    Returns:
        pd.DataFrame: Formatted UniProt data
    """
    print("Loading SwissProt Dataset.")
    uniprot_columns = [
        "Database",
        "Protein ID",
        "Description",
        "Species",
        "Species ID",
        "Gene Name",
        "Protein Existence",
        "Sequence Version",
    ]
    uniprot = pd.DataFrame(columns=uniprot_columns)
    if not Path(CACHE_DIR / "uniprot_df_shard_1.csv").is_file():
        # Caching results
        with open(RAW_DATA / "uniprot_sprot.fasta") as handle:
            seq_counter = 0
            file_counter = 0
            for values in tqdm(SimpleFastaParser(handle)):
                values_parsed = list()
                parsed = (
                    values[0]
                    .replace(" OS=", "|")
                    .replace(" OX=", "|")
                    .replace(" GN=", "|")
                    .replace(" PE=", "|")
                    .replace(" SV=", "|")
                    .split("|")
                )
                sequence = values[1]
                database = parsed[0]
                protein_id = parsed[1]
                description = parsed[2]
                species = parsed[3]
                species_id = parsed[4]

                if len(parsed) == 8:
                    gene_name = parsed[5]
                    protein_existence = parsed[6]
                    sequence_version = parsed[7]
                elif len(parsed) == 7:
                    gene_name = "X"
                    protein_existence = parsed[5]
                    sequence_version = parsed[6]
                else:
                    raise FormatError(
                        "There is a problem with the source file used to store "
                        "UNIPROT data"
                    )
                uniprot = pd.concat(
                    [
                        uniprot,
                        pd.DataFrame(
                            [
                                database,
                                protein_id,
                                description,
                                species,
                                species_id,
                                gene_name,
                                protein_existence,
                                sequence_version,
                                sequence,
                            ],
                            index=[
                                "Database",
                                "Protein ID",
                                "Description",
                                "Species",
                                "Species ID",
                                "Gene Name",
                                "Protein Existence",
                                "Sequence Version",
                                "Sequence",
                            ],
                        ).T,
                    ]
                )
                if seq_counter % 10000 == 0:
                    uniprot.to_csv(
                        CACHE_DIR / f"uniprot_df_shard_{file_counter}.csv"
                    )
                    file_counter += 1
                    uniprot = pd.DataFrame(columns=uniprot_columns)
                seq_counter += 1
    else:
        uniprot_shards = [
            uniprot_shard
            for uniprot_shard in os.listdir(CACHE_DIR)
            if "uniprot_df" in uniprot_shard
        ]
        uniprot = pd.DataFrame(columns=uniprot_columns)
        for shard in uniprot_shards:
            uniprot_shard_df = pd.read_csv(CACHE_DIR / shard, index_col=0)
            uniprot = uniprot.append(uniprot_shard_df)
    return uniprot


def load_melanoma_dataset() -> pd.DataFrame:
    """Loads melanoma dataset from the MARIA paper.
    Source: https://www.nature.com/articles/ncomms13404#ref-CR57

    Returns:
        pd.DataFrame: melanoma dataset with cleaned columns.
    """
    melanoma_excel_file = pd.ExcelFile(RAW_DATA / "melanoma_untyped.xlsx")
    melanoma_table = melanoma_excel_file.parse(
        "Supplementary data 2", header=1
    )

    # Remove HLA-I peptides
    melanoma_table = melanoma_table.dropna(
        subset=[col for col in melanoma_table.columns if "HLA-II" in col],
        how="all",
    )

    # Remove columns with clutter (HLA-I removes HLA-I and HLA-II)
    melanoma_table = melanoma_table[
        [col for col in melanoma_table.columns if "HLA-I" not in col]
    ]

    return melanoma_table


def epitope_file_parser(fname):
    epitope_df = pd.DataFrame()
    with open(fname) as handle:
        for values in SimpleFastaParser(handle):
            values_parsed = list()
            values_parsed = values[0].split(" ")
            values_parsed.append(values[1])
            epitope_df = pd.concat([epitope_df, pd.DataFrame(values_parsed).T])
    epitope_df.columns = [
        "peptide",
        "MHC_molecule",
        "protein_id",
        "source_protein",
    ]
    epitope_df.index = range(len(epitope_df))
    return epitope_df
