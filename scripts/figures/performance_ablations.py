#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""performance_variants.py

The aim of this script is to visualize the performance of all model variants

"""

import os
import pandas as pd
import re
import torch

from mhciipresentation.utils import load_obj
import itertools
from pyprojroot import here
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from plotting_utils import (
    build_confusion_matrix,
    build_precision_recall_curve,
    build_roc_curve,
    build_loss_curve,
)


def build_dest_dir(ext):
    dest_dir = here() / f"scripts/figures/generated/variants/{ext}"
    dest_dir.mkdir(exist_ok=True, parents=True)
    return dest_dir


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

    df_stats.set_index(["data_source", "feature_set"], inplace=True)
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
    metrics_to_show_in_table = [
        c
        for c in df_stats.columns
        if "matthews" in c or "auroc" in c or "f1" in c
    ]
    df_stats = df_stats[metrics_to_show_in_table]
    latex = df_stats.to_latex(
        index=True,
        escape=False,
        sparsify=True,
        multirow=True,
        multicolumn=True,
        multicolumn_format="c",
        bold_rows=True,
    ).replace("_", " ")

    with open(here() / "scripts/figures/ablation_table.tex", "w") as f:
        f.write(latex)


def get_vectors_logs_path(row):
    path = (
        str(here())
        + "/outputs/variants/"
        + str(row["feature_set"])
        + "-"
        + str(row["data_source"])
        + "-"
        + str(row["layers"])
        + "-"
        + str(row["seed"])
        + "/vector_logs/"
        + str(row["epoch"])
    )
    row["path"] = path
    return row


def get_metrics_logs_path(row):
    path = (
        str(here())
        + "/outputs/variants/"
        + str(row["feature_set"])
        + "-"
        + str(row["data_source"])
        + "-"
        + str(row["layers"])
        + "-"
        + str(row["seed"])
        + "/csv/transformer/version_0/metrics.csv"
    )
    row["metrics"] = path
    return row


def make_plot_curves(df_raw, df_stats):
    df_raw = df_raw.apply(get_vectors_logs_path, axis=1)
    df_raw = df_raw.apply(get_metrics_logs_path, axis=1)
    feature_set = ["seq_mhc"]
    data_source = ["iedb"]
    embedding = [
        "true",
    ]
    all_ones = ["true", "false"]
    seeds = [0, 1, 2, 3]
    for combination in itertools.product(
        feature_set, data_source, embedding, all_ones, seeds
    ):
        df_raw_subset = df_raw.loc[
            (df_raw["feature_set"] == combination[0])
            & (df_raw["data_source"] == combination[1])
            & (df_raw["layers"] == combination[2])
        ]
        curve_types = [
            "confusion_matrix",
            "precision_recall_curve",
            "roc",
            "loss",
        ]
        splits = ["train", "val", "test"]
        annot = f"Feature set: {combination[0]}, Data source: {combination[1]}, All ones: {combination[3]}"
        ext = f"{combination[0]}_{combination[1]}_{combination[2]}"
        dest_dir = build_dest_dir(ext)
        for comb in itertools.product(curve_types, splits):
            curve_type, split = comb
            if curve_type == "confusion_matrix":
                build_confusion_matrix(df_raw_subset, split, dest_dir, annot)
            elif curve_type == "precision_recall_curve":
                build_precision_recall_curve(
                    df_raw_subset, split, dest_dir, annot
                )
            elif curve_type == "roc":
                build_roc_curve(df_raw_subset, split, dest_dir, annot)
            elif curve_type == "loss":
                build_loss_curve(df_raw_subset, split, dest_dir, annot)
            else:
                raise ValueError("Curve type not recognized")


def main():
    # Load data
    df_raw = pd.read_csv(
        here() / "outputs/ablations/raw_ranking_performance.csv"
    )
    df_stats = pd.read_csv(here() / "outputs/ablations/ranked_models.csv")

    # First, make a nice table with the variants.
    make_table(df_stats)

    # Plot curves for best epoch with errors over seeds
    # df_raw["epoch"] = df_raw["epoch"].astype(int)
    # make_plot_curves(df_raw, df_stats)


if __name__ == "__main__":
    main()
