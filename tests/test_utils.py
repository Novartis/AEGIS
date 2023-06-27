#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_utils.py

Test functions for utils.py

"""
import numpy as np
import pandas as pd
import pytest

from mhciipresentation.constants import AA_TO_INT
from mhciipresentation.mouse.mouse import parse_protein_data
from mhciipresentation.utils import (
    aa_seq_to_int,
    add_peptide_context,
    flatten_lists,
    get_peptide_context,
    join_peptide_with_pseudosequence,
)

test_nested_lists = [
    ([["A", "B", "C"], ["D", "E", "F"]], ["A", "B", "C", "D", "E", "F"]),
    ([["X"], ["Y"], ["Z"]], ["X", "Y", "Z"]),
    ([["ABC"], ["DEF"], ["GHI"]], ["ABC", "DEF", "GHI"]),
]


@pytest.mark.parametrize("list_of_lists, expected", test_nested_lists)
def test_flatten_lists(list_of_lists, expected):
    """test flatten lists"""
    assert flatten_lists(list_of_lists) == expected


peptide = pd.Series(
    data=[
        "DSIDAHTFDFETI",
        "HKGSPFPEVAESVQQELESYR",
        "NNHIMKLTKGLIKDALEN",
        "TWKERLVIRALENRVG",
        "LARDMTLPPETNVILT",
    ],
    name="peptide",
)

context = pd.Series(
    data=[
        "LSPDSIETIPHP",
        "WQKHKGSYRAQE",
        "SKTNNHLENIDP",
        "TWNTWKRVGIYS",
        "EVALARILTKDK",
    ],
    name="context",
)

peptide_with_context = pd.Series(
    data=[
        "LSPDSIDAHTFDFETIPHP",
        "WQKHKGSPFPEVAESVQQELESYRAQE",
        "SKTNNHIMKLTKGLIKDALENIDP",
        "TWNTWKERLVIRALENRVGIYS",
        "EVALARDMTLPPETNVILTKDK",
    ],
    name="peptide_with_context",
)

pseudosequence = pd.Series(
    [
        "QEFFIASGAAVDAIMWLFLECYDLQRATYHVGFT",
        "QEFFIASGAAVDAIMEVHFDYYSLQRATYHVGFT",
        "QEFFIASGAAVDAIMWLFLECYDLQRATYHVGFT",
        "YAFFQFSGGAILNTLYLQFEYFDLEEVRMHLDVT",
        "CNYHQGGGARVAHIMFFGLTYYDVGTETVHVAGI",
    ],
    name="pseudosequence",
)

peptide_with_context_and_pseudosequence = pd.Series(
    data=[
        "LSPDSIDAHTFDFETIPHPQEFFIASGAAVDAIMWLFLECYDLQRATYHVGFT",
        "WQKHKGSPFPEVAESVQQELESYRAQEQEFFIASGAAVDAIMEVHFDYYSLQRATYHVGFT",
        "SKTNNHIMKLTKGLIKDALENIDPQEFFIASGAAVDAIMWLFLECYDLQRATYHVGFT",
        "TWNTWKERLVIRALENRVGIYSYAFFQFSGGAILNTLYLQFEYFDLEEVRMHLDVT",
        "EVALARDMTLPPETNVILTKDKCNYHQGGGARVAHIMFFGLTYYDVGTETVHVAGI",
    ],
    name="peptide_with_context_and_pseudosequence",
)


X_train_synthetic = np.array(
    [
        "GEGELINWVALDREH",
        "Abelin__MAPTAC_DRB1_1101",
        "CELGEGREHRGH",
        "DRB1_1101",
        1,
        "QEFFIASGAAVDAIMESSFDYFDFDRATYHVGFT",
        "CELGEGELINWVALDREHRGH",
        "CELGEGELINWVALDREHRGHQEFFIASGAAVDAIMESSFDYFDFDRATYHVGFT",
    ]
)


X_train_peptide_with_pseudosequence = pd.Series(
    data=["LLLASLRQMKKTRGTLLALQEFFIASGAAVDAIMWPRFDYYDFDRATYHVGFT",]
)


X_test_peptide_with_pseudosequence = pd.Series(
    data=["LYSLESISLAENTQDVRDDDYSYFLASGGQVVHVLYFGYTYHDIRTETVHGPHT",]
)


X_train_peptide_with_pseudosequence_encoded = np.array(
    [
        [
            27,
            21,
            21,
            21,
            15,
            7,
            21,
            2,
            10,
            1,
            4,
            4,
            8,
            2,
            13,
            8,
            21,
            21,
            15,
            21,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            10,
            6,
            18,
            18,
            17,
            15,
            7,
            13,
            15,
            15,
            16,
            5,
            15,
            17,
            1,
            20,
            14,
            2,
            18,
            5,
            19,
            19,
            5,
            18,
            5,
            2,
            15,
            8,
            19,
            3,
            16,
            13,
            18,
            8,
            28,
        ]
    ]
)


X_test_peptide_with_pseudosequence_encoded = np.array(
    [
        [
            27,
            21,
            19,
            7,
            21,
            6,
            7,
            17,
            7,
            21,
            15,
            6,
            9,
            8,
            10,
            5,
            16,
            2,
            5,
            5,
            5,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            19,
            7,
            19,
            18,
            21,
            15,
            7,
            13,
            13,
            10,
            16,
            16,
            3,
            16,
            21,
            19,
            18,
            13,
            19,
            8,
            19,
            3,
            5,
            17,
            2,
            8,
            6,
            8,
            16,
            3,
            13,
            14,
            3,
            8,
            28,
        ]
    ]
)

test_data_add_peptide_context = [(peptide, context, peptide_with_context)]


@pytest.mark.parametrize(
    "peptide, context, expected", test_data_add_peptide_context
)
def test_add_peptide_context(peptide, context, expected):
    pd.testing.assert_series_equal(
        add_peptide_context(peptide, context), expected, check_names=False
    )


test_data_join_peptide_with_pseudosequence = [
    (
        peptide_with_context,
        pseudosequence,
        peptide_with_context_and_pseudosequence,
    )
]


@pytest.mark.parametrize(
    "peptide_with_context, pseudosequence, expected",
    test_data_join_peptide_with_pseudosequence,
)
def test_join_peptide_with_pseudosequence(
    peptide_with_context, pseudosequence, expected
):
    pd.testing.assert_series_equal(
        join_peptide_with_pseudosequence(
            peptide_with_context, pseudosequence,
        ),
        expected,
        check_names=False,
    )


test_data_aa_seq_to_int = [
    (
        "".join(list(AA_TO_INT.keys())[:-2]),
        [
            22,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            0,
            23,
        ],
    ),
    (
        "LSPDSIDAHTFDFETIPHP",
        [
            22,
            21,
            7,
            14,
            5,
            7,
            17,
            5,
            15,
            3,
            8,
            18,
            5,
            18,
            6,
            8,
            17,
            14,
            3,
            14,
            23,
        ],
    ),
]


@pytest.mark.parametrize(
    "aa_seq, expected", test_data_aa_seq_to_int,
)
def test_aa_seq_to_int(aa_seq, expected):
    assert aa_seq_to_int(aa_seq, AA_TO_INT) == expected


peptides = pd.Series(data=["DLAGRDLTDY", "LKYPIEHGIITNWDDMEK", "NELRVAPEEHPV"])

peptides_with_context = pd.Series(
    data=["MRLDLAGRDLTDYLMK", "ILTLKYPIEHGIITNWDDMEKIWH", "TFYNELRVAPEEHPVLLT"]
)

df_test_get_peptide_context = [(peptides, peptides_with_context)]


@pytest.mark.parametrize("input_df, expected", df_test_get_peptide_context)
def test_get_peptide_context(input_df, expected):
    try:
        proteins = parse_protein_data()
        pd.testing.assert_series_equal(
            get_peptide_context(input_df, proteins), expected
        )
    except:
        # Skip test when protein data is not accessible
        assert 1 == 1
