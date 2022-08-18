#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mouse.py

File preprocessing mouse data. The script:
- Cleans up positive peptides
- Generates negative peptides
- Decorates negative peptides with appropriate amounts of PTMs

"""

import os
import random
import xml.etree.ElementTree as ET
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from mhciipresentation.constants import LEN_BOUNDS_MOUSE, PTM_LIST
from mhciipresentation.errors import FormatError, SamplingSizeError
from mhciipresentation.paths import MOUSE_PUBLIC
from mhciipresentation.utils import (
    generate_negative_peptides,
    get_peptide_context,
    get_white_space,
)

# Set seed for script
random.seed(42)


def parse_protein_sequences_xml_file(file: str) -> pd.DataFrame:
    """Parse protein XML file

    Args:
        file (str): file to protein xml file from Wan et al.

    Returns:
        pd.DataFrame: source proteins relevant for the data
    """
    if not os.path.isfile(file):
        raise FileNotFoundError("The file was not found")
    try:
        tree = ET.parse(file)
    except ET.ParseError:
        print("Error parsing file.")
    try:
        root = tree.getroot()
        protein_sequences = list()
        for child in root:
            for subchild in child:
                if "DBSequence" in subchild.tag:
                    entry = dict()
                    for subsubchild in subchild:
                        if "Seq" in subsubchild.tag:
                            entry["sequence"] = subsubchild.text
                        if "cvParam" in subsubchild.tag:
                            entry["source protein"] = subsubchild.attrib[
                                "value"
                            ]
                    protein_sequences.append(entry)
        return pd.DataFrame(protein_sequences)
    except ET.ParseError:
        print(
            "The file was probably not formatted according to the expected "
            "schema."
        )


def parse_protein_data() -> pd.Series:
    """Parse source protein data

    Returns:
        pd.DataFrame: parse protein data
    """
    protein_sequences_1 = parse_protein_sequences_xml_file(
        MOUSE_PUBLIC + "Unanue_7733_180712_05.mzid"
    )
    protein_sequences_2 = parse_protein_sequences_xml_file(
        MOUSE_PUBLIC + "Unanue_7872_190321_06.mzid"
    )
    protein_sequences_3 = parse_protein_sequences_xml_file(
        MOUSE_PUBLIC + "Unanue_7715_180601_08.mzid"
    )
    protein_sequences_4 = parse_protein_sequences_xml_file(
        MOUSE_PUBLIC + "Unanue_7797-7873.mzid"
    )
    protein_sequences_5 = parse_protein_sequences_xml_file(
        MOUSE_PUBLIC + "Unanue_7681-7683-7707-7772.mzid"
    )
    return (
        pd.concat(
            [
                protein_sequences_1,
                protein_sequences_2,
                protein_sequences_3,
                protein_sequences_4,
                protein_sequences_5,
            ]
        )
        .dropna()
        .drop_duplicates()
        .sequence
    )


def parse_peptide_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse mouse peptide data

    Raises:
        FileNotFoundError: if the file is not found

    Returns:
        Tuple[pd.DataFrame,pd.DataFrame]: peptide without and with accession numbers
    """
    xl_file = MOUSE_PUBLIC + "NIHMS1556366-supplement-SuppTables1-5-5.xlsx"
    if not os.path.isfile(xl_file):
        raise FileNotFoundError("The Excel file was not found.")

    wan_data = pd.ExcelFile(xl_file)
    islet = wan_data.parse("Islet All peptides")
    pancreatic_lymph_nodes = wan_data.parse("pLN All peptides")
    whole_spleen_peptides = wan_data.parse("Whole spleen peptides")
    b_cell_deficient = wan_data.parse("uMT spleen peptides")
    b_cell_peptides = wan_data.parse("B cell peptides")
    dense_core_granules = wan_data.parse("DCGs")
    crinosomes = wan_data.parse("crinosomes")
    secretome = wan_data.parse("Secretome")
    hybrid_insulin_peptide = wan_data.parse("Free HIPs")

    peptides = pd.concat(
        [
            islet,
            pancreatic_lymph_nodes,
            whole_spleen_peptides,
            b_cell_deficient,
            b_cell_peptides,
        ]
    )[["Sequence Coverage", "Uniprot Accession"]].rename(
        columns={"Sequence Coverage": "Peptide Sequence"}
    )

    peptides = pd.concat(
        [
            peptides,
            crinosomes[["Peptide Sequence", "Uniprot Accession"]],
            secretome[["Peptide Sequence", "Uniprot Accession"]],
            dense_core_granules[["Peptide Sequence", "Uniprot Accession"]],
        ]
    )

    peptides["Peptide Sequence"] = peptides["Peptide Sequence"].str.upper()

    return peptides["Peptide Sequence"].reset_index(drop=True), peptides


def deduplicate_unresolved_peptides(peptides: pd.Series) -> pd.Series:
    """Sorts out peptides which are in the form v1/v2. That means they could not
        be resolved via MS.

    Args:
        peptides (pd.Series): peptides presented by the MHCII

    Returns:
        pd.Series: returns peptides together with all versions of unresolved
            peptides
    """
    # There is probably uncertainty about which is the correct amino acid, but we remove it.
    # To err on the side of caution we want to make sure that both peptides are in there so we can construct a conservative view of the "white space"
    uncertain_elements = list(
        set(peptides.loc[peptides.str.contains("/")].tolist())
    )
    peptides = peptides.loc[~peptides.isin(uncertain_elements)]
    possible_peptides = list()
    for element in uncertain_elements:
        for peptide in element.split("/"):
            possible_peptides.append(peptide)
    peptides = peptides.astype("str")
    peptides = peptides.append(pd.Series(possible_peptides))
    return peptides.reset_index(drop=True).drop_duplicates()


def clean_positive_peptides(peptides: pd.Series) -> pd.Series:
    """Clean peptides

    Args:
        peptides (pd.Series): peptides that are presented

    Returns:
        pd.Series: clean peptides
    """
    return (
        peptides.loc[
            ~peptides.str.contains(":")
        ]  # Remove wrongly formatted peptide
        .loc[~peptides.str.contains("/")]  # Removes ambiguous peptide
        .loc[
            peptides.str.len().isin(
                [i for i in range(LEN_BOUNDS_MOUSE[0], LEN_BOUNDS_MOUSE[1])]
            )
        ]  # Filters length of peptides
        .str.strip()  # Removes spaces
        .drop_duplicates()
    )


def clean_negative_peptides(peptides: pd.Series) -> pd.Series:
    """Clean peptides (remove PTMs, among others)

    Args:
        peptides (pd.Series): peptides that are presented

    Returns:
        pd.Series: clean peptides
    """
    return (
        peptides.str.replace("m", "M")  # Removes PTMs
        .str.replace("c", "C")
        .str.replace("n", "N")
        .str.replace("q", "Q")
        .str.strip()  # Removes spaces
        .loc[~peptides.str.contains(":")]  # Remove wrongly formatted peptide
        .loc[~peptides.str.contains("/")]  # Removes ambiguous peptide
        .drop_duplicates()
    )


def filter_length(series: pd.Series, bounds: tuple,) -> pd.Series:
    """Filters a dataframe `column` by string length as given by `bounds`.

    Args:
        series (pd.Series): input series
        bounds (tuple): bounds used to filter the series

    Returns:
        pd.DataFrame: series containing peptides with the given `bounds`.
    """
    return series.loc[series.str.len().between(bounds[0], bounds[1])]


def compute_str_len(series: pd.Series) -> pd.Series:
    """Computes the string length of a series.

    Args:
        series (pd.Series): input series

    Returns:
        pd.Series: the series containing the values of the length of each row's
            string.
    """
    # Series needs to be of type string
    return series.str.len()


def compute_ptm_aa_per_peptide(
    df: pd.DataFrame, column="peptides"
) -> pd.DataFrame:
    """Counts the amount of decorated amino acid among the amino acids in the
        `PTM_LIST`.

    Args:
        df (pd.DataFrame): input dataframe containing the peptide sequences.
        column (str, optional): the column containing the peptides.
            Defaults to "peptides".

    Returns:
        pd.DataFrame: the dataframe with additional columns in the form
            `PTM_{decorated_aa}`.
    """
    for aa in PTM_LIST:
        df[f"PTM_{aa.lower()}"] = (
            df[column].str.findall(f"{aa.lower()}").str.len()
        )
    return df


def compute_ptm_able_aa_per_peptide(
    df: pd.DataFrame, column="peptides"
) -> pd.DataFrame:
    """Computes the amount of amino acids which can be decorated per peptide.

     Args:
         df (pd.DataFrame): the input dataframe
         column (str, optional): The columns used. Defaults to "peptides".

     Returns:
         pd.DataFrame: contains the columns for each amino acid that can contain
    PTMs
    """
    for aa in PTM_LIST:
        df[f"PTM_able_{aa}"] = df[column].str.findall(f"{aa}").str.len()
    return df


def compute_ptm_per_peptide_length(
    df: pd.DataFrame,
    columns_to_count=["peptide_length", "PTM_c", "PTM_m", "PTM_q", "PTM_n",],
) -> pd.DataFrame:
    """Computes the amount of samples per combination of PTM and peptide lengths

    Args:
        df (pd.DataFrame): input dataframe containing the peptide sequences
        columns_to_count (list, optional): Columns to use for the `value_counts`.
            Defaults to [ "peptide_length", "PTM_c", "PTM_m", "PTM_q", "PTM_n", ].

    Raises:
        FormatError: if the columns used to count do not correspond to the ones available

    Returns:
        pd.DataFrame: dataframe with the `value_counts` of the columns_to_count, sorted by index.
    """
    if any(item not in df.columns for item in columns_to_count):
        raise FormatError(
            "Check the input df column names and set columns_to_counts appropriately."
        )
    return df[columns_to_count].value_counts().sort_index()


def select_by_peptide_length(
    df: pd.DataFrame, peptide_length: int
) -> pd.DataFrame:
    """Select peptides corresponding to `peptide_length`.

    Args:
        df (pd.DataFrame): input dataframe
        peptide_length (int): peptide lengths used to filter the dataframe

    Returns:
        pd.DataFrame: dataframe containing only peptides with the given
            `peptide_length`.
    """
    return df.loc[df.peptide_length == peptide_length]


def select_ptm_able_peptides(df: pd.DataFrame, n_ptms: dict) -> pd.DataFrame:
    """Select peptides which contain at least `n_ptms[aa]` amino acids which can be modified.

    Args:
        df (pd.DataFrame): input dataframe
        n_ptms (dict): containins all the ptms that need to be selected

    Returns:
        pd.DataFrame: returned dataframe
    """
    for aa in PTM_LIST:
        df = df.loc[df[f"PTM_able_{aa}"] >= n_ptms[f"n_{aa.lower()}"]]

    return df


def sample_from_df(df: pd.DataFrame, n_to_sample: int) -> pd.DataFrame:
    """Samples `n_to_sample` samples from `df` without
        replacement. Raises an error if it's not possible.

    Args:
        df_to_select_from (pd.DataFrame): dataframe to sample from
        n_to_sample (int): number of samples

    Raises:
        SamplingSizeError: error raised when sampling without replacement is not possible

    Returns:
        pd.DataFrame: sampled dataframe
    """
    if len(df) < n_to_sample:
        raise SamplingSizeError(
            "There are not enough eligible peptides to sample from without replacement"
        )
    # random_state is added for reproducibility
    return df.sample(n=n_to_sample, replace=False, random_state=42)


def filter_negative_data(
    neg_data: pd.DataFrame,
    peptide_length: int,
    n_ptms: dict,
    n_samples: int,
    factor: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filters out the relevant negative data according to peptide length,
        peptides that can be decorated, and samples the appropriate number of
        samples.

    Args:
        neg_data (pd.DataFrame): negative input data
        peptide_length (int): the peptide length used to select the data
        n_ptms (dict): containins all the ptms that need to be selected
        n_samples (int): contains the number of samples that are required
        factor (int): the factor by which we want to sample negative data from
        compared to the positive data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: sampled peptides and amended negative data not containing
            the sampled peptides.
    """

    # Select peptides which contain the required length
    neg_data_selection = select_by_peptide_length(neg_data, peptide_length)
    neg_data_selection = select_ptm_able_peptides(neg_data_selection, n_ptms)

    # Once the appropriate data has been selected, we sample the appropriate number of samples
    n_to_sample = n_samples * factor
    sampled_peptides = sample_from_df(neg_data_selection, n_to_sample)

    # We have to remove the sampled peptides to make sure they are not sampled again
    neg_data = neg_data.drop(sampled_peptides.index)
    return sampled_peptides, neg_data


def insert_ptms_in_peptides(
    sampled_peptides: pd.DataFrame, n_ptms: dict
) -> pd.DataFrame:
    """Decorates the peptides with the required amount of PTMs at random.

    Args:
        sampled_peptides (pd.DataFrame): sampled peptides to decorate
        n_ptms (dict): containins all the ptms that need to be selected

    Returns:
        pd.DataFrame: decorated peptides
    """
    for index_neg, row in sampled_peptides.iterrows():
        peptide = row.peptides
        modified_peptide = peptide
        for idx_aa, aa in enumerate(PTM_LIST):
            if row[f"PTM_able_{aa}"] == n_ptms[f"n_{aa.lower()}"]:
                modified_peptide = modified_peptide.replace(aa, aa.lower())
            elif n_ptms[f"n_{aa.lower()}"] != 0:
                idx_of_ptm_able_aa_to_replace = random.sample(
                    range(row[f"PTM_able_{aa}"]), n_ptms[f"n_{aa.lower()}"],
                )
                idx_of_ptm_able_aa = [
                    i
                    for i, letter in enumerate(modified_peptide)
                    if letter == aa
                ]
                for i in idx_of_ptm_able_aa_to_replace:
                    idx_of_ptm = idx_of_ptm_able_aa[i]
                    modified_peptide = (
                        modified_peptide[:idx_of_ptm]
                        + aa.lower()
                        + modified_peptide[idx_of_ptm + 1 :]
                    )
            else:
                continue
        sampled_peptides.at[index_neg, "peptides"] = modified_peptide
    return sampled_peptides


def decorate_negative_data(
    neg_data: pd.DataFrame,
    freq_PTM_per_peptide_length: pd.DataFrame,
    factor: int,
) -> pd.DataFrame:
    """Decorates the negative data with identical proportions of PTMs

    Args:
        neg_data (pd.DataFrame): negative input data
        freq_PTM_per_peptide_length (pd.DataFrame): frequency of PTM per
            peptide length
        factor (int): the factor by which we want to sample negative data from
        compared to the positive data

    Returns:
        pd.DataFrame: appropriately decorated negative data
    """
    # Filter and decorate negative data
    # We make deep copies to avoid modifying the original data without explicitely wanting to do so.
    neg_data = neg_data.copy()
    neg_data_decorated = pd.DataFrame()

    # Loop through all possible lengths and PTM # combination present in the positive data to sample and decorate accordingly
    for index, n_samples in tqdm(
        freq_PTM_per_peptide_length.iteritems(),
        total=len(freq_PTM_per_peptide_length),
    ):
        peptide_length = index[0]
        n_ptms = {
            "n_c": index[1],
            "n_m": index[2],
            "n_q": index[3],
            "n_n": index[4],
        }
        sampled_peptides, neg_data = filter_negative_data(
            neg_data, peptide_length, n_ptms, n_samples, factor
        )
        sampled_peptides = insert_ptms_in_peptides(sampled_peptides, n_ptms)
        neg_data_decorated = neg_data_decorated.append(sampled_peptides)
    return neg_data_decorated


def validate_results(negative_data: pd.DataFrame, positive_data: pd.DataFrame):
    """Validates the generated results by checking the proportion of
        combinations of PTMs in the positive and negative data

    Args:
        negative_data (pd.DataFrame): negative input data
        positive_data (pd.DataFrame): positive input data
    """
    freq_PTMs_per_peptide_positive = positive_data.peptides.str.findall(
        r"[cmqn]"
    ).str.len().value_counts().sort_index() / len(positive_data)

    freq_PTMs_per_peptide_negative = negative_data.peptides.str.findall(
        r"[cmqn]"
    ).str.len().value_counts().sort_index() / len(negative_data)

    print(
        f"Are the PTM frequencies the same overall? "
        f"{freq_PTMs_per_peptide_positive == freq_PTMs_per_peptide_negative}"
    )

    freq_PTM_per_peptide_length_pos = compute_ptm_per_peptide_length(
        positive_data
    ) / len(positive_data)

    freq_PTM_per_peptide_length_neg = compute_ptm_per_peptide_length(
        compute_ptm_aa_per_peptide(negative_data)
    ) / len(negative_data)

    equal_all_props = all(
        freq_PTM_per_peptide_length_pos == freq_PTM_per_peptide_length_neg
    )
    print(
        f"Are the specific combinations of peptide lengths and PTM decorations "
        f"the same overall? {equal_all_props}"
    )


def cleanup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Removes helper columns containing the peptide_length info and starting
        with `PTM_`

    Args:
        df (pd.DataFrame): input dataframe containing helper columns like
            PTM_able_c

    Returns:
        pd.DataFrame: dataframe, without helper columns
    """
    return df.drop(columns=[i for i in df.columns if "PTM_" in i]).drop(
        columns=["peptide_length"]
    )


def main():
    # Parse input data
    print("Parsing input data")
    proteins = parse_protein_data()
    peptides, peptides_with_protein_info = parse_peptide_data()

    # Handle positive peptides
    print("Handling positive peptides")
    positive_data = clean_positive_peptides(peptides)

    # Generate raw negative peptides
    print("Generate raw negative peptides")
    peptides = deduplicate_unresolved_peptides(peptides)
    peptides = clean_negative_peptides(peptides)
    white_space = get_white_space(peptides, proteins)
    negative_data = generate_negative_peptides(white_space, LEN_BOUNDS_MOUSE)

    # Filter and decorate negative peptides
    print("Handle negative peptide decorations")
    negative_data = negative_data.to_frame()
    positive_data = positive_data.to_frame()

    print("Remove helper columns")
    positive_data = positive_data.merge(
        peptides_with_protein_info,
        left_on="Peptide Sequence",
        right_on="Peptide Sequence",
        how="left",
    )
    print("Writing data")
    positive_data["label"] = 1
    negative_data["label"] = 0
    negative_data["Uniprot Accession"] = 0
    negative_data.columns = ["Peptide Sequence", "label", "Uniprot Accession"]
    data = positive_data.append(negative_data)
    data = data.sample(frac=1)
    data = data.reset_index(drop=True)
    data.to_csv(MOUSE_PUBLIC + "preprocessed_public_mouse_data.csv")


if __name__ == "__main__":
    main()
