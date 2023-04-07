# -*- coding: utf-8 -*-
"""generate_deepmhcii_data.py

The objective of this script is to format the data we have to the format DeepMHCII needs.
"""

import pandas as pd
from mhciipresentation.loaders import load_pseudosequences
from mhciipresentation.paths import RAW_DATA
from pyprojroot import here


def main():
    print("Generating DeepMHCII data")

    # First we handle the training data. Labels are not binarized and contain the affinity score.
    # We also add a column for the allele.
    iedb_data = pd.read_csv(here() / "data/processed/sa_data.csv", index_col=0)
    train_idx = pd.read_csv(here() / "data/splits/random_iedb/X_train_idx.csv")

    train_data = iedb_data.iloc[train_idx["index"]]
    train_data = train_data[["peptide", "target_value", "Alleles"]]
    train_data.to_csv(
        here() / "baselines/DeepMHCII/data/aegis_data/train_data.csv",
        index=False,
        header=False,
        sep="\t",
    )

    # Same for validation data
    val_idx = pd.read_csv(here() / "data/splits/random_iedb/X_val_idx.csv")
    val_data = iedb_data.iloc[val_idx["index"]]
    val_data = val_data[["peptide", "target_value", "Alleles"]]
    val_data.to_csv(
        here() / "baselines/DeepMHCII/data/aegis_data/val_data.csv",
        index=False,
        header=False,
        sep="\t",
    )

    # Then we handle the test data. Labels are binarized.
    test_idx = pd.read_csv(here() / "data/splits/random_iedb/X_test_idx.csv")
    test_data = iedb_data.iloc[test_idx["index"]]
    # Binarize labels
    test_data["target_value"] = test_data["target_value"].apply(
        lambda x: 1 if x > 0.5 else 0
    )
    test_data = test_data[["peptide", "target_value", "Alleles"]]
    test_data.to_csv(
        here() / "baselines/DeepMHCII/data/aegis_data/test_data.csv",
        index=False,
        header=False,
        sep="\t",
    )


if __name__ == "__main__":
    main()
