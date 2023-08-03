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
import pandas as pd


def add_annot(annot, pos="top"):
    if pos == "top":
        plt.text(
            0.5,
            0.05,
            annot,
            transform=plt.gca().transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )
    elif pos == "bottom":
        plt.text(
            0.5,
            0.05,
            annot,
            transform=plt.gca().transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )
    else:
        raise ValueError(f"pos must be either 'top' or 'bottom', got {pos}")


def build_confusion_matrix(
    df_raw_subset, split, dest_dir, annot, load_data=True
):
    if load_data:
        confusion_matrix_data = []
        for path in df_raw_subset.path.tolist():
            confusion_matrix_data.append(
                load_obj(Path(path) / f"{split}_confusion_matrix.pkl")
            )
    else:
        confusion_matrix_data = df_raw_subset
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
    add_annot(annot, pos="top")
    plt.savefig(
        dest_dir / f"confusion_matrix_{split}.pdf",
    )
    plt.close()


def build_precision_recall_curve(
    df_raw_subset, split, dest_dir, annot, load_data=True
):
    if load_data:
        precision_recall_curve_data = []
        for path in df_raw_subset.path.tolist():
            precision_recall_curve_data.append(
                load_obj(Path(path) / f"{split}_precision_recall_curve.pkl")
            )
    else:
        precision_recall_curve_data = df_raw_subset

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
    add_annot(annot, pos="bottom")
    plt.savefig(
        dest_dir / f"precision_recall_curve_{split}.pdf",
    )
    plt.close()


def build_roc_curve(df_raw_subset, split, dest_dir, annot, load_data=True):
    if load_data:
        roc_curve_data = []
        for path in df_raw_subset.path.tolist():
            roc_curve_data.append(load_obj(Path(path) / f"{split}_roc.pkl"))
    else:
        roc_curve_data = df_raw_subset

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
    plt.plot(fpr_levels, mean_curve, label="Mean curve", color="orange")
    plt.fill_between(
        fpr_levels,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2,
        color="lightcoral",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="blue")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if split == "melanoma":
        title = f"Differentiating Melanoma Dataset"
    elif split == "DRB1_0101_ligands":
        title = "Differentiating K562 DRB1*01:01 ligands from decoys"
    elif split == "DRB1_0404_ligands":
        title = "Differentiating K562 DRB1*04:04 ligands from decoys"
    else:
        title = f"ROC curve for {split} split"

    plt.title(title)
    add_annot(annot, pos="bottom")
    plt.savefig(dest_dir / f"roc_curve_{split}.pdf")
    plt.close()


def build_loss_curve(df_raw_subset, split, dest_dir, annot, load_data=True):
    if load_data:
        roc_curve_data = []
        for path in df_raw_subset.metrics.tolist():
            roc_curve_data.append(pd.read_csv(path))
    else:
        roc_curve_data = df_raw_subset

    loss_iters = []
    for idx, logs in enumerate(roc_curve_data):
        if split == "train":
            loss_col = "train_loss"
        elif split == "val":
            loss_col = "val_loss/dataloader_idx_0"
        elif split == "test":
            loss_col = "val_loss/dataloader_idx_1"
        else:
            raise ValueError(f"Split {split} not recognized")
        loss_iters.append(logs[loss_col].dropna())
    loss_iters = np.array(loss_iters)

    # Make plot of all the curves
    for idx, loss in enumerate(loss_iters):
        plt.plot(loss)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Loss curve for {split} split")
    add_annot(annot, pos="top")
    plt.savefig(
        dest_dir / f"loss_curve_{split}.pdf",
    )
    plt.close()
