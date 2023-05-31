#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""performance_variants

The aim of this script is to visualize the performance of all model variants

"""

import os
import pandas as pd
import re

# from mhciipresentation.performance_utils import load_scalars
import itertools
from pyprojroot import here
from functools import partial


def highlight_best(s, best_models):
    return [
        "\\textbf{" + cell + "}" if cell == best_models[col] else cell
        for col, cell in s.iteritems()
    ]


def make_table(df_stats):
    non_metric_cols = ["seed", "layers", "feature_set", "data_source", "path"]

    metrics = set(
        col[:-5]
        for col in df_stats.columns
        if (col.endswith("_mean") or col.endswith("_stad"))
        and col[:-5] not in non_metric_cols
    )

    for metric in metrics:
        df_stats[metric] = (
            "$"
            + df_stats[metric + "_mean"].map(lambda x: f"{x:.3f}")
            + " \pm "
            + df_stats[metric + "_stad"].map(lambda x: f"{x:.3f}")
            + "$"
        )

    best_models = df_stats.loc[
        df_stats[
            [metric for metric in df_stats.columns if "_mean" in metric]
        ].idxmax()
    ]

    df_stats = df_stats.drop(
        columns=[
            metric + suffix
            for metric in metrics
            for suffix in ["_mean", "_stad"]
        ]
    )

    df_stats.set_index(["data_source", "feature_set", "layers"], inplace=True)
    df_stats.style.set_caption("Model variants ranked by performance")
    # df_stats.style.highlight_max(
    #     subset=[col for col in df_stats.columns if col not in non_metric_cols]
    # )
    df_stats.columns = [
        re.sub(r"(val_.*)/dataloader_idx_0", r"val \1", col)
        if "val_" in col
        else col
        for col in df_stats.columns
    ]

    df_stats.columns = [
        re.sub(r"(val_.*)/dataloader_idx_1", r"test \1", col)
        if "val_" in col
        else col
        for col in df_stats.columns
    ]

    df_stats.columns = [
        col.replace("val val", "val").replace("test val", "test")
        for col in df_stats.columns
    ]

    train_cols = sorted([col for col in df_stats.columns if "train" in col])
    val_cols = sorted([col for col in df_stats.columns if "val" in col])
    test_cols = sorted([col for col in df_stats.columns if "test" in col])

    df_stats = df_stats[train_cols + val_cols + test_cols]

    latex = df_stats.to_latex(
        index=True,
        escape=False,
        sparsify=True,
        multirow=True,
        multicolumn=True,
        multicolumn_format="c",
        bold_rows=True,
    ).replace("_", " ")

    with open(here() / "scripts/figures/variants_table.tex", "w") as f:
        f.write(latex)


def get_curve_data(row):
    path = (
        "outputs/variants/"
        + row["feature_set"]
        + "-"
        + row["data_source"]
        + "-"
        + row["layers"]
        + "-"
        + str(row["seed"])
        + "/vector_logs/"
        + str(row["epoch"])
    )
    row


def make_plot_curves(df_raw, df_stats):
    df_raw.apply(get_curve_data, axis=1)


def main():
    # Load data
    df_raw = pd.read_csv(
        here() / "outputs/variants/raw_ranking_performance.csv"
    )
    df_stats = pd.read_csv(here() / "outputs/variants/ranked_models.csv")

    # First, make a nice table with the variants.
    make_table(df_stats)

    # Plot curves for best epoch with errors over seeds
    df_raw["epoch"] = df_raw["epoch"].astype(int)
    make_plot_curves(df_raw, df_stats)


if __name__ == "__main__":
    main()
