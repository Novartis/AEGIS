# -*- coding: utf-8 -*-
"""select_best_variant.py
Select the best model variant.

Reads from outputs/variants and selects the best model based on the test loss.
"""

import itertools

import pandas as pd
from pyprojroot import here

feature_set = ["seq_only", "seq_mhc"]
data_source = ["iedb", "iedb_nod", "nod"]
layers = [2, 4, 8]
seeds = [0, 1, 2, 3]


def main():
    combinations = list(
        itertools.product(feature_set, data_source, layers, seeds)
    )
    best_performance = []
    for comb in combinations:
        model_path = f"outputs/variants/{comb[0]}-{comb[1]}-{comb[2]}-{comb[3]}/csv/transformer/version_0/metrics.csv"
        df = pd.read_csv(here() / model_path)
        epoch_metrics = (
            df.dropna(subset="epoch").groupby("epoch").mean().reset_index()
        )
        best_epoch = epoch_metrics["val_auroc/dataloader_idx_0"].idxmax()
        best_epoch_metrics = epoch_metrics.loc[best_epoch]
        cols = [
            idx
            for idx in best_epoch_metrics.index
            if "test" not in idx and "lr-" not in idx
        ]
        best_epoch_metrics = best_epoch_metrics.loc[cols].dropna().to_dict()
        best_epoch_metrics["feature_set"] = comb[0]
        best_epoch_metrics["data_source"] = comb[1]
        best_epoch_metrics["layers"] = comb[2]
        best_epoch_metrics["seed"] = comb[3]
        best_performance.append(best_epoch_metrics)
    best_performance = pd.DataFrame(best_performance)

    best_performance_mean = best_performance.groupby(
        ["feature_set", "data_source", "layers"]
    ).mean()
    best_performance_std = best_performance.groupby(
        ["feature_set", "data_source", "layers"]
    ).std()
    best_performance = pd.concat(
        [best_performance_mean, best_performance_std], axis=1
    )
    best_performance.drop(columns=["seed"]).to_latex(
        here() / "outputs/best_performance.tex"
    )

    # bst_perf = best_performance.iloc[
    #     best_performance.loc[
    #         best_performance.data_source.str.contains("iedb")
    #     ]["val_auroc/dataloader_idx_0"].idxmax()
    # ]


if __name__ == "__main__":
    main()
