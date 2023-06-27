#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""performance_variants

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


def tensor_to_float(tensor):
    return tensor.item()


def process_combination(combination, dset, subset):
    skip = False
    if subset is None:
        dest_dir = (
            here()
            / f"scripts/figures/generated/inference/{dset}/{combination[0]}-{combination[1]}-{combination[2]}"
        )
    else:
        dest_dir = (
            here()
            / f"scripts/figures/generated/inference/{dset}/{subset}/{combination[0]}-{combination[1]}-{combination[2]}"
        )
        if "maria" in dset:
            if "melanoma" not in dset:
                dest_dir = (
                    here()
                    / f"outputs/inference/{dset.split('/')[0]}/{combination[0]}-{combination[1]}-{combination[2]}/{dset.split('/')[1]}/{dset.split('/')[2]}/"
                )
            else:
                dest_dir = (
                    here()
                    / f"outputs/inference/{dset.split('/')[0]}/{combination[0]}-{combination[1]}-{combination[2]}/melanoma/"
                )
    dest_dir.mkdir(exist_ok=True, parents=True)

    comb = f"{combination[0]}-{combination[1]}"
    paths = []
    if dset == "cd4":
        print("cd4")
    for seed in range(4):
        if subset is None:
            inference_path = (
                here()
                / f"outputs/inference/{dset}/{combination[0]}-{combination[1]}-{combination[2]}-{seed}/performance_metrics"
            )
            if "maria" in dset:
                if "melanoma" not in dset:
                    inference_path = (
                        here()
                        / f"outputs/inference/{dset.split('/')[0]}/{combination[0]}-{combination[1]}-{combination[2]}-{seed}/{dset.split('/')[1]}/{dset.split('/')[2]}/"
                    )
                else:
                    inference_path = (
                        here()
                        / f"outputs/inference/{dset.split('/')[0]}/{combination[0]}-{combination[1]}-{combination[2]}-{seed}/melanoma/"
                    )
            if "xu" in dset or "you" in dset:
                inference_path = (
                    here()
                    / f"outputs/inference/{dset}/{combination[0]}-{combination[1]}-{combination[2]}-{seed}/{dset}"
                )
        else:
            inference_path = (
                here()
                / f"outputs/inference/{dset}/{combination[0]}-{combination[1]}-{combination[2]}-{seed}/{subset}/performance_metrics"
            )
        if not inference_path.exists():
            print("-------------------")
            print(f"Path {inference_path} does not exist")
            print("-------------------")
            skip = True
        paths.append(inference_path)
    if skip:
        return None

    scalars = []
    vectors = []
    # Load data
    for path in paths:
        scalars.append(load_obj(path / "scalar_metrics.pkl"))
        vectors.append(load_obj(path / "vector_metrics.pkl"))
    scalars = pd.DataFrame(scalars)
    scalars = scalars.applymap(tensor_to_float)
    scalars = scalars[
        [
            c
            for c in scalars.columns
            if "matthews" in c or "auroc" in c or "f1" in c
        ]
    ]
    mean_scalars = scalars.mean()
    std_scalars = scalars.std()

    scalars = pd.concat([mean_scalars, std_scalars], axis=1).rename(
        columns={0: "mean", 1: "std"}
    )
    if "maria" in dset:
        if len(dset.split("/")) >= 3:
            split = dset.split("/")[2]
        else:
            split = "melanoma"
    else:
        split = dset

    build_confusion_matrix(
        [vectors[i]["confusion_matrix"] for i in range(4)],
        split=split,
        dest_dir=dest_dir,
        annot=f"Dataset: {split}, Feature set: "
        + combination[0]
        + ", Data source: "
        + combination[1]
        + ", Layers: "
        + combination[2],
        load_data=False,
    )

    build_precision_recall_curve(
        [vectors[i]["precision_recall_curve"] for i in range(4)],
        split=split,
        dest_dir=dest_dir,
        annot="Dataset: CD4, Feature set: "
        + combination[0]
        + ", Data source: "
        + combination[1]
        + ", Layers: "
        + combination[2],
        load_data=False,
    )

    build_roc_curve(
        [vectors[i]["roc"] for i in range(4)],
        split=split,
        dest_dir=dest_dir,
        annot="Dataset: CD4, Feature set: "
        + combination[0]
        + ", Data source: "
        + combination[1]
        + ", Layers: "
        + combination[2]
        + "\n"
        + f"$ {scalars.loc['auroc']['mean']} \pm {scalars.loc['auroc']['std']} $",
        load_data=False,
    )
    return None


def main():
    datasets = [
        "cd4",
        "hold_out",
        "nod",
        "maria/K562/DRB1_0101_ligands",
        "maria/K562/DRB1_0404_ligands",
        "maria/melanoma",
        "nod",
        "xu",
        "you",
    ]

    # Load data
    feature_set = ["seq_only", "seq_mhc"]
    data_source = [
        "iedb",
        "iedb_nod",
    ]
    layers = ["4", "8"]
    subsets = [
        c
        for c in os.listdir(
            here() / "outputs/inference/hold_out/seq_only-iedb-4-0/"
        )
        if c != "hold_out.log"
        and c != ".hydra"
        and c != "performance_metrics"
        and c != "tensorboard"
        and c != "predict"
    ]

    for dset in datasets:
        if dset != "hold_out":
            for combination in itertools.product(
                feature_set, data_source, layers
            ):
                process_combination(combination, dset, subset=None)
        else:
            for combination in itertools.product(
                feature_set, data_source, layers
            ):
                for subset in subsets:
                    process_combination(combination, dset, subset=subset)


if __name__ == "__main__":
    main()
