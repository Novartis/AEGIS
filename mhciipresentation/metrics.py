# -*- coding: utf-8 -*-
"""metrics.py

Some metric utilities

"""
from pathlib import Path
from typing import Dict

import torch
import torchmetrics
from mhciipresentation.utils import save_obj
from sklearn.preprocessing import Binarizer


def build_scalar_metrics():
    return {
        "accuracy": torchmetrics.Accuracy(threshold=0.5, task="binary"),
        "precision": torchmetrics.Precision(task="binary"),
        "recall": torchmetrics.Recall(task="binary"),
        "f1": torchmetrics.F1Score(task="binary"),
        "matthews": torchmetrics.MatthewsCorrCoef(task="binary"),
        "cohen": torchmetrics.CohenKappa(task="binary"),
        "auroc": torchmetrics.AUROC(task="binary"),
    }


def build_vector_metrics():
    return {
        "roc": torchmetrics.ROC(task="binary"),
        "precision_recall_curve": torchmetrics.PrecisionRecallCurve(
            task="binary"
        ),
        "confusion_matrix": torchmetrics.ConfusionMatrix(task="binary"),
    }


def compute_performance_metrics(scalar_metrics, vector_metrics, y, y_hat):
    y_true = (
        torch.Tensor(
            Binarizer(threshold=0.5).transform(y.view(-1, 1).double().cpu())
        )
        .reshape(-1)
        .int()
    )
    scalar_metric_values = {}
    for metric in scalar_metrics.keys():
        scalar_metric_values[metric] = scalar_metrics[metric](
            y_hat.cpu(), y
        ).float()

    vector_metric_values = {}
    for metric in vector_metrics.keys():
        vector_metric_values[metric] = vector_metrics[metric](
            y_hat.cpu(), y_true
        )
    return scalar_metric_values, vector_metric_values


def save_performance_metrics(
    dest_path: Path, scalar_metric_values: Dict, vector_metric_values: Dict
):
    save_obj(scalar_metric_values, dest_path / "scalar_metrics.pkl")
    save_obj(vector_metric_values, dest_path / "vector_metrics.pkl")
