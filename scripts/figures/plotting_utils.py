# -*- coding: utf-8 -*-
"""plotting_utils.py

Plotting utilities
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def build_confusion_matrix(df_raw_subset, split):
    confusion_matrix_data = []
    for path in df_raw_subset.path.tolist():
        confusion_matrix_data.append(
            load_obj(Path(path) / f"{split}_confusion_matrix.pkl")
        )
    mean_cm = torch.stack(confusion_matrix_data).float().mean(dim=0)
    std_cm = torch.stack(confusion_matrix_data).float().std(dim=0)
    fig, ax = plt.subplots()
    im = ax.imshow(mean_cm, cmap="cool")
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    # Add text annotations
    for i in range(mean_cm.shape[0]):
        for j in range(mean_cm.shape[1]):
            text = ax.text(
                j,
                i,
                f"{mean_cm[i, j]:.2f} $\pm$ {std_cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    # Add axis labels and title
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")

    # Show the plot
    plt.savefig(
        here() / f"scripts/figures/outputs/confusion_matrix_{split}.pdf",
    )
    plt.close()


def build_precision_recall_curve(df_raw_subset, split):
    precision_recall_curve_data = []
    for path in df_raw_subset.path.tolist():
        precision_recall_curve_data.append(
            load_obj(Path(path) / f"{split}_precision_recall_curve.pkl")
        )
    for idx, curve in enumerate(precision_recall_curve_data):
        precision, recall, thresholds = curve
        sorted_indices = np.argsort(recall)
        precision = precision[sorted_indices]
        recall = recall[sorted_indices]
        recall_levels = np.linspace(0, 1, 100)
        interp_func = interp1d(
            recall,
            precision,
            bounds_error=False,
            fill_value=(precision[0], precision[-1]),
        )
        # Store the interpolated curve
        if idx == 0:
            curves = interp_func(recall_levels)
        else:
            curves = np.vstack([curves, interp_func(recall_levels)])

    # Calculate mean and standard deviation along that new dimension
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)

    # Plot mean curve with std deviation
    plt.plot(recall_levels, mean_curve, label="Mean curve")
    plt.fill_between(
        recall_levels,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2,
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve for {split} split")
    plt.savefig(
        here() / f"scripts/figures/outputs/precision_recall_curve_{split}.pdf",
    )
    plt.close()


def build_roc_curve(df_raw_subset, split):
    roc_curve_data = []
    for path in df_raw_subset.path.tolist():
        roc_curve_data.append(load_obj(Path(path) / f"{split}_roc.pkl"))
    for idx, curve in enumerate(roc_curve_data):
        fpr, tpr, thresholds = curve
        sorted_indices = np.argsort(fpr)
        tpr = tpr[sorted_indices]
        fpr = fpr[sorted_indices]
        fpr_levels = np.linspace(0, 1, 100)
        interp_func = interp1d(
            fpr,
            tpr,
            bounds_error=False,
            fill_value=(tpr[0], tpr[-1]),
        )
        # Store the interpolated curve
        if idx == 0:
            curves = interp_func(fpr_levels)
        else:
            curves = np.vstack([curves, interp_func(fpr_levels)])

    # Calculate mean and standard deviation along that new dimension
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)

    # Plot mean curve with std deviation
    plt.plot(fpr_levels, mean_curve, label="Mean curve")
    plt.fill_between(
        fpr_levels,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2,
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve for {split} split")
    plt.savefig(
        here() / f"scripts/figures/outputs/roc_curve_{split}.pdf",
    )
    plt.close()
