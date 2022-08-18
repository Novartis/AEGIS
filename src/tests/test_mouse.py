#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_mouse.py

Test functions for mhciipresentation/mouse/mouse.py

"""

import pandas as pd
import pytest

from mhciipresentation.mouse.mouse import (
    clean_negative_peptides,
    clean_positive_peptides,
    compute_ptm_aa_per_peptide,
    compute_ptm_able_aa_per_peptide,
    compute_ptm_per_peptide_length,
    compute_str_len,
    deduplicate_unresolved_peptides,
    generate_negative_peptides,
    get_black_space,
)

raw_df = pd.Series(
    data=[
        "496:508AGQAVSAYSEERD",  # contains anomaly
        "NVEIDPEIQ/NVELDPEIQ",  # unresolved peptide
        "   KYPIEHGIITNWDDmEKI",  # Contains spaces
        "NELRVAPEEHPV",  # Duplicate
        "NELRVAPEEHPV",
    ]
)

cleaned_df = pd.Series(
    data=["KYPIEHGIITNWDDmEKI", "NELRVAPEEHPV"], index=[2, 3]
)

df_test_clean_positive_samples = [(raw_df, cleaned_df)]


@pytest.mark.parametrize("input_df, expected", df_test_clean_positive_samples)
def test_clean_positive_peptides(input_df, expected):
    pd.testing.assert_series_equal(clean_positive_peptides(input_df), expected)


peptides_test = pd.Series(
    data=["NVEIDPEIQ/NVELDPEIQ", "DLAGRDLTDY", "KYPIEHGIITNWDDmEKI"]
)
deduplicated_peptides = pd.Series(
    data=["DLAGRDLTDY", "KYPIEHGIITNWDDmEKI", "NVEIDPEIQ", "NVELDPEIQ"]
)

df_test_deduplicate_unresolved_peptides = [
    (peptides_test, deduplicated_peptides)
]


@pytest.mark.parametrize(
    "input_df, expected", df_test_deduplicate_unresolved_peptides
)
def test_deduplicate_unresolved_peptidest(input_df, expected):
    pd.testing.assert_series_equal(
        deduplicate_unresolved_peptides(input_df), expected,
    )


peptides_dirty = pd.Series(
    data=["KYPIEHGIITNWDDmEKI", "KYPIEHGqInNWDDmEKI", "KYPIEHGII/TNWDDmEKI"]
)
negative_peptides_clean = pd.Series(
    data=["KYPIEHGIITNWDDMEKI", "KYPIEHGQINNWDDMEKI"]
)

df_test_clean_negative_data = [(peptides_dirty, negative_peptides_clean)]


@pytest.mark.parametrize("input_df, expected", df_test_clean_negative_data)
def test_clean_negative_peptides(input_df, expected):
    pd.testing.assert_series_equal(
        clean_negative_peptides(input_df), expected,
    )


sample_peptide = pd.Series(data="LKYPIEHGIITNWDDMEK")

source_protein = pd.Series(
    data=[
        "MCDEDETTALVCDNGSGLVKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLK"
        "YPIEHGIITNWDDMEKIWHHTFYNELRVAPEEHPTLLTEAPLNPKANREKMTQIMFETFNVPAMYVAIQA"
        "VLSLYASGRTTGIVLDSGDGVTHNVPIYEGYALPHAIMRLDLAGRDLTDYLMKILTERGYSFVTTAEREI"
        "VRDIKEKLCYVALDFENEMATAASSSSLEKSYELPDGQVITIGNERFRCPETLFQPSFIGMESAGIHETT"
        "YNSIMKCDIDIRKDLYANNVMSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILAS"
        "LSTFQQMWITKQEYDEAGPSIVHRKCF"
    ]
)

black_space = [
    "MCDEDETTALVCDNGSGLVKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILT",
    "IWHHTFYNELRVAPEEHPTLLTEAPLNPKANREKMTQIMFETFNVPAMYVAIQAVLSLYASGRTTGIVLDSGDG"
    "VTHNVPIYEGYALPHAIMRLDLAGRDLTDYLMKILTERGYSFVTTAEREIVRDIKEKLCYVALDFENEMATAAS"
    "SSSLEKSYELPDGQVITIGNERFRCPETLFQPSFIGMESAGIHETTYNSIMKCDIDIRKDLYANNVMSGGTTMY"
    "PGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWITKQEYDEAGPSIVHRKCF",
]

df_test_get_black_space = [(sample_peptide, source_protein, black_space)]


@pytest.mark.parametrize(
    "input_peptide, input_proteins, expected", df_test_get_black_space
)
def test_get_black_space(input_peptide, input_proteins, expected):
    assert get_black_space(input_peptide, input_proteins) == expected


df_generate_negative_peptides = [(black_space)]


@pytest.mark.parametrize("black_space", df_generate_negative_peptides)
def test_df_generate_negative_peptides(black_space):
    """We can only assert shape correctness due to the sheer amount of data."""
    generated_negative_peptides = generate_negative_peptides(black_space)
    assert generated_negative_peptides.shape == (7825,)


peptides_with_ptm = pd.Series(
    data=["MRLDLAGRDLTDYLMKcmq", "ILTLKYPIEHGIITnWDDMEKIWH"], name="peptides",
)

positive_with_str_len = pd.Series(data=[19, 24], name="peptides",)


df_compute_str_len = [(peptides_with_ptm, positive_with_str_len)]


@pytest.mark.parametrize("peptides, expected", df_compute_str_len)
def test_df_compute_str_len(peptides, expected):
    """We can only assert shape correctness due to the sheer amount of data."""
    pd.testing.assert_series_equal(compute_str_len(peptides), expected)


peptides_with_ptm = pd.DataFrame(
    data=[["MRLDLAGRDLTDYLMKcmq", 19], ["ILTLKYPIEHGIITnWDDMEKIWH", 24]],
    columns=["peptides", "peptide_length"],
)
positive_with_str_and_ptmed_aa = pd.DataFrame(
    data=[
        ["MRLDLAGRDLTDYLMKcmq", 19, 1, 1, 1, 0],
        ["ILTLKYPIEHGIITnWDDMEKIWH", 24, 0, 0, 0, 1],
    ],
    columns=["peptides", "peptide_length", "PTM_c", "PTM_m", "PTM_q", "PTM_n"],
)
df_ptm_per_aa = [(peptides_with_ptm, positive_with_str_and_ptmed_aa)]


@pytest.mark.parametrize("peptides, expected", df_ptm_per_aa)
def test_df_compute_ptm_aa_per_peptide(peptides, expected):
    """We can only assert shape correctness due to the sheer amount of data."""
    pd.testing.assert_frame_equal(
        compute_ptm_aa_per_peptide(peptides), expected
    )


ptm_able_aa_input = pd.DataFrame(
    data=[["MRLDLAGRDLTDYLMKCMQ", 19], ["ILTLKYPIEHGIITNWDDMEKIWH", 24],],
    columns=["peptides", "peptide_length"],
)

ptm_able_aa_output = pd.DataFrame(
    data=[
        ["MRLDLAGRDLTDYLMKCMQ", 19, 1, 3, 1, 0],
        ["ILTLKYPIEHGIITNWDDMEKIWH", 24, 0, 1, 0, 1],
    ],
    columns=[
        "peptides",
        "peptide_length",
        "PTM_able_C",
        "PTM_able_M",
        "PTM_able_Q",
        "PTM_able_N",
    ],
)


@pytest.mark.parametrize("peptides, expected", df_ptm_per_aa)
def test_df_compute_ptm_able_per_peptide(peptides, expected):
    """We can only assert shape correctness due to the sheer amount of data."""
    pd.testing.assert_frame_equal(
        compute_ptm_able_aa_per_peptide(peptides), expected
    )


freq_ptm = (
    pd.DataFrame(
        [(19, 1, 1, 1, 0, 1), (24, 0, 0, 0, 1, 1)],
        columns=[
            "peptide_length",
            "PTM_c",
            "PTM_m",
            "PTM_q",
            "PTM_n",
            "counts",
        ],
    )
    .set_index(["peptide_length", "PTM_c", "PTM_m", "PTM_q", "PTM_n"])
    .squeeze()
)

df_test_compute_ptm_able_per_peptide = [
    (positive_with_str_and_ptmed_aa, freq_ptm)
]


@pytest.mark.parametrize(
    "peptides, expected", df_test_compute_ptm_able_per_peptide
)
def test_df_compute_ptm_able_per_peptide(peptides, expected):
    """We can only assert shape correctness due to the sheer amount of data."""
    freq = compute_ptm_per_peptide_length(peptides)
    freq.name = "counts"
    pd.testing.assert_series_equal(freq, expected)
