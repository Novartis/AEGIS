#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""performance.py

Performance assessment of a model using the training and validation set.

python src/mhciipresentation/performance.py
    --metrics_path /path/to/metrics/
    --plotting_path /path/to/plotting/dir/
    --name "Plot name"

"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mhciipresentation.paths import DATA_DIR

# Original colnames
ORIGINAL_COLNAMES = [
    "f1",
    "precision",
    "recall",
    "auc",
    "matthews_corrcoef",
    "loss",
    "tn",
    "tp",
    "fn",
    "fp",
]
ORIGINAL_COLNAMES_SORTED = [
    "f1",
    "precision",
    "recall",
    "auc",
    "matthews_corrcoef",
    "loss",
    "tn",
    "tp",
    "fn",
    "fp",
]
PERF_COLUMNS = [
    "F1",
    "Precision",
    "Recall",
    "AUC",
    "MCC",
    "Loss",
    "TN",
    "TP",
    "FN",
    "FP",
]


def main():
    epochs = os.listdir(FLAGS.metrics_path + "/metrics/")
    sorted_idx = [int(i[6:-5]) for i in epochs]
    epochs_run = max(sorted_idx)

    metrics = list()
    for i in range(2, epochs_run + 1):
        with open(f"{FLAGS.metrics_path}/metrics/epoch_{i}.json") as outfile:
            metrics_epoch = json.load(outfile)
        metrics.append(metrics_epoch)

    df_perf_train = pd.DataFrame(columns=ORIGINAL_COLNAMES)
    df_perf_test = pd.DataFrame(columns=ORIGINAL_COLNAMES)
    for epoch in metrics:
        df_perf_train = df_perf_train.append(epoch["train"], ignore_index=True)
        df_perf_test = df_perf_test.append(epoch["val"], ignore_index=True)

    df_perf_test = df_perf_test[ORIGINAL_COLNAMES_SORTED]
    df_perf_train = df_perf_train[ORIGINAL_COLNAMES_SORTED]

    df_perf_test.columns = PERF_COLUMNS
    df_perf_train.columns = PERF_COLUMNS

    sns.set_palette("Set2")

    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.suptitle(f"Learning Curves of the {FLAGS.name}", y=0.9)
    pos_idx = 1
    xlab = "Epoch"
    for i in PERF_COLUMNS[:6]:
        ax = fig.add_subplot(3, 2, pos_idx)
        sns.lineplot(
            data=pd.concat(
                [df_perf_train[i], df_perf_test[i]],
                axis=1,
                keys=["train", "val"],
            ),
            palette=["red", "green"],
            ax=ax,
        )
        ax.set_ylabel(i)
        if i != "Loss":
            plt.ylim(0, 1)

        if pos_idx == 5 or pos_idx == 6:
            ax.set_xlabel(xlab)
        pos_idx += 1
    plt.savefig(
        FLAGS.plotting_path + FLAGS.name + "_learning_curves_perf_metrics.png"
    )

    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.suptitle(f"Confusion matrix of the {FLAGS.name}", y=0.9)
    pos_idx = 1
    for i in PERF_COLUMNS[6:]:
        ax = fig.add_subplot(3, 2, pos_idx)
        plot = sns.lineplot(
            data=pd.concat(
                [df_perf_train[i], df_perf_test[i]],
                axis=1,
                keys=["train", "val"],
            ),
            palette=["red", "green"],
            ax=ax,
        )
        ax.set_ylabel(i)

        if pos_idx == 4 or pos_idx == 3:
            ax.set_xlabel(xlab)
        pos_idx += 1
    plt.savefig(
        FLAGS.plotting_path + FLAGS.name + "_confusion_matrix_evolution.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_path", "-perfp", type=str, help="path to the training logs",
    )
    parser.add_argument(
        "--plotting_path",
        "-plotp",
        type=str,
        help="path to the plotting directory",
    )
    parser.add_argument(
        "--name", "-n", type=str, help="Name of the model",
    )
    FLAGS = parser.parse_args()
    main()
