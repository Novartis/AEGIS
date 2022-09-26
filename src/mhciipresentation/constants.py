#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""constants.py

This file provides constants used throughout this project.

"""

# Compute-related options
N_JOBS = 10
USE_GPU = True

N_MOTIF_FOLDS = 5

# Whether or not to accelerate some script by using the cahced precomputed
# dataset as input (used for debugging)

FORCE_PREPROCESSING = False
USE_SUBSET = False

# amino acids used in this project (i.e. natural AAs + X, a placeholder AA)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"

# Columns in the raw the netMHCIIpanv4 files
RAW_PUBLIC_FILE_COL_NAMES = [
    "peptide",
    "target_value",
    "MHC_molecule",
    "peptide_context",
]

# Mapping of amino acid to intermediate int representation
AA_TO_INT = {
    "M": 1,
    "R": 2,
    "H": 3,
    "K": 4,
    "D": 5,
    "E": 6,
    "S": 7,
    "T": 8,
    "N": 9,
    "Q": 10,
    "C": 11,
    "U": 12,
    "G": 13,
    "P": 14,
    "A": 15,
    "V": 16,
    "I": 17,
    "F": 18,
    "Y": 19,
    "W": 20,
    "L": 21,
    "X": 0,  # Unknown
    "start": 22,
    "stop": 23,
}

# Mapping of amino acid with PTMs to intermediate int representation
AA_TO_INT_PTM = {
    "M": 1,
    "m": 2,  # the lowercase letters are for PTMs
    "R": 3,
    "H": 4,
    "K": 5,
    "D": 6,
    "E": 7,
    "S": 8,
    "T": 9,
    "N": 10,
    "n": 11,
    "Q": 12,
    "q": 13,
    "C": 14,
    "c": 15,
    "U": 16,
    "G": 17,
    "P": 18,
    "A": 19,
    "V": 20,
    "I": 21,
    "F": 22,
    "Y": 23,
    "W": 24,
    "L": 25,
    "X": 0,  # Unknown
    "start": 26,
    "stop": 27,
}

# Length bounds for the positive peptides for the mouse dataset including PFRs
LEN_BOUNDS_MOUSE = (9, 33)
LEN_BOUNDS_HUMAN = (13, 17)
LEN_BOUNDS_HUMAN_EXPANDED = (9, 25)

# The list of possible PTMs which can be decorated
PTM_LIST = ["C", "M", "Q", "N"]

# Whether or not to use the context
USE_CONTEXT = False
