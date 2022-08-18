#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_human.py

Test functions for mhciipresentation/human/human.py

"""

import pandas as pd
import pytest

from mhciipresentation.human.human import (
    filter_el_samples,
    filter_sa_samples,
    select_data_files,
)

test_file_selection = [
    (
        [
            "train_EL2.txt",
            "test_EL3.txt",
            "train_BA5.txt",
            "train_BA2.txt",
            "pseudosequence_mapping.dat",
        ],
        ["train_EL2.txt", "test_EL3.txt",],
    )
]


@pytest.mark.parametrize("list_of_files, expected", test_file_selection)
def test_select_data_files(list_of_files, expected):
    """test select data files"""
    assert select_data_files(list_of_files) == expected


raw_df = pd.DataFrame(
    data=[
        # MA sample
        ["DTQFVRFDSDAASPRGEPR", 1.0, "Bergseng__9051_PITOUT", "YVDDTQEPRAPW",],
        # SA BA sample (has no context)
        [
            "VLQAGFFLLTRILTIPQSLD",
            1.0,
            "HLA-DPA10103-DPB10201",
            "XXXVLQSLDXXX",
        ],
        # SA EL sample (has context + target in {1, 0})
        [
            "YWIWSPASLNEKTPKGH",
            0.0,
            "Abelin__MAPTAC_DRB1_1303",
            "TYFYWIKGHSVF",
        ],
        # SA BA sample (no context + target in [0, 1])
        [
            "AAASVPAADKFKTFE",
            0.203668,
            "HLA-DPA10103-DPB10201",
            "XXXAAATFEXXX",
        ],
    ],
    columns=["peptide", "target_value", "MHC_molecule", "peptide_context",],
)

df_test_filter_sa_samples = [
    (
        raw_df,
        # Only sa samples are left
        pd.DataFrame(
            data=[
                [
                    "VLQAGFFLLTRILTIPQSLD",
                    1.000000,
                    "HLA-DPA10103-DPB10201",
                    "XXXVLQSLDXXX",
                    "HLA-DPA10103-DPB10201",
                    1,
                    "YAFFMFSGGAILNTLFGQFEYFDIEEVRMHLGMT",
                ],
                [
                    "AAASVPAADKFKTFE",
                    0.203668,
                    "HLA-DPA10103-DPB10201",
                    "XXXAAATFEXXX",
                    "HLA-DPA10103-DPB10201",
                    1,
                    "YAFFMFSGGAILNTLFGQFEYFDIEEVRMHLGMT",
                ],
                [
                    "YWIWSPASLNEKTPKGH",
                    0.000000,
                    "Abelin__MAPTAC_DRB1_1303",
                    "TYFYWIKGHSVF",
                    "DRB1_1303",
                    1,
                    "QEFFIASGAAVDAIMESSFDYYSIDKATYHVGFT",
                ],
            ],
            columns=[
                "peptide",
                "target_value",
                "MHC_molecule",
                "peptide_context",
                "Alleles",
                "number_of_alleles",
                "Pseudosequence",
            ],
        ),
    )
]


@pytest.mark.parametrize("input_df, expected", df_test_filter_sa_samples)
def test_filter_sa_samples(input_df, expected):
    """test select data files"""
    pd.testing.assert_frame_equal(filter_sa_samples(input_df), expected)


df_test_filter_el_samples = [
    (
        raw_df,
        # Only el samples are left
        pd.DataFrame(
            data=[
                [
                    "DTQFVRFDSDAASPRGEPR",
                    1.0,
                    "Bergseng__9051_PITOUT",
                    "YVDDTQEPRAPW",
                ],
                [
                    "YWIWSPASLNEKTPKGH",
                    0.0,
                    "Abelin__MAPTAC_DRB1_1303",
                    "TYFYWIKGHSVF",
                ],
            ],
            columns=[
                "peptide",
                "target_value",
                "MHC_molecule",
                "peptide_context",
            ],
        ),
    )
]


@pytest.mark.parametrize("input_df, expected", df_test_filter_el_samples)
def test_filter_el_samples(input_df, expected):
    """test select data files"""
    pd.testing.assert_frame_equal(
        filter_el_samples(input_df).reset_index(drop=True), expected
    )
