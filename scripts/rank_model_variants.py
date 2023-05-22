# -*- coding: utf-8 -*-
"""rank_model_variants.py

This script ranks the model variants based on the performance of each model.
"""

import itertools
import os
from typing import Any, List

import numpy as np
import pandas as pd
from pyprojroot import here
from tqdm import tqdm


def flatten_lists(lists: List) -> List:
    """Removes nested lists.

    Args:
        lists (List): List of lists to flatten

    Returns:
        List: Flattened list
    """
    result = list()
    for _list in lists:
        _list = list(_list)
        if _list != []:
            result += _list
        else:
            continue
    return result


def check_versions(combinations):
    """checks if there are any different versions_* in the csv logger files."""
    for comb in combinations:
        versions = os.listdir(
            here()
            / f"outputs/variants/{comb[0]}-{comb[1]}-{comb[2]}-{comb[3]}/csv/transformer/"
        )
        if len(versions) > 1:
            print(f"Found {len(versions)} versions for {comb}")


def reposition_element(l: List, elem: Any, new_idx: int) -> List:
    """Repositions an element in a list.

    Args:
        l (List): List to reposition the element in
        elem (Any): Element to reposition
        new_idx (int): New index of the element

    Raises:
        ValueError: If the element is not in the list

    Returns:
        List: List with the element repositioned
    """

    if elem not in l:
        raise ValueError("Element not found in the list.")

    current_index = l.index(elem)
    l = l[:current_index] + l[current_index + 1 :]
    l = l[:new_idx] + [elem] + l[new_idx:]

    return l


def reorder_columns(df: pd.DataFrame) -> List:
    """Reorders the columns of a dataframe.

    Args:
        df (pd.DataFrame): Dataframe to reorder

    Returns:
        List: List of column names in the new order
    """
    new_col_order = flatten_lists(
        [
            [col + "_mean", col + "_std"]
            for col in df.columns.to_list()
            if col not in ["seed", "layers", "feature_set", "data_source"]
        ]
    )
    # Reverse order of the columns
    new_col_order = new_col_order[::-1]

    new_col_order = reposition_element(new_col_order, "epoch_mean", 0)
    new_col_order = reposition_element(new_col_order, "epoch_std", 1)
    new_col_order = reposition_element(new_col_order, "step_mean", 2)
    new_col_order = reposition_element(new_col_order, "step_std", 3)
    new_col_order = reposition_element(new_col_order, "train_loss_mean", 4)
    new_col_order = reposition_element(new_col_order, "train_loss_std", 5)
    new_col_order = [
        "data_source",
        "feature_set",
        "layers",
    ] + new_col_order

    return new_col_order


def main():  # pylint: disable=missing-docstring # pylint: disable=too-many-locals
    # Metric used to rank the models
    metric_to_select_models = "val_auroc/dataloader_idx_0"
    metric_to_sort_models = "val_auroc/dataloader_idx_1"

    feature_set = ["seq_only", "seq_mhc"]
    data_source = ["iedb", "iedb_nod", "nod"]
    layers = [2, 4, 8]
    seeds = [0, 1, 2, 3]

    combinations = list(
        itertools.product(feature_set, data_source, layers, seeds)
    )
    check_versions(combinations)
    df = pd.DataFrame()
    for comb in tqdm(combinations):
        df_comb = pd.read_csv(
            here()
            / f"outputs/variants/{comb[0]}-{comb[1]}-{comb[2]}-{comb[3]}/csv/transformer/version_0/metrics.csv"
        )
        epoch_metrics = df_comb.groupby(["epoch"]).mean(numeric_only=True)
        best_epoch = epoch_metrics[metric_to_select_models].idxmax()
        epoch_metrics = epoch_metrics.loc[best_epoch]
        relevant_fields = [
            col for col in epoch_metrics.index.to_list() if "dataloader" in col
        ] + [
            "step",
            "train_loss",
        ]
        results = dict()
        for field in relevant_fields:
            results[field] = epoch_metrics[field]
        results["epoch"] = best_epoch
        results["feature_set"] = comb[0]
        results["data_source"] = comb[1]
        results["layers"] = comb[2]
        results["seed"] = comb[3]

        df = pd.concat([pd.DataFrame(results, index=[0]), df])

    df = df.sort_values(metric_to_sort_models, ascending=False)
    new_col_order = reorder_columns(df)
    df_mean = (
        df.groupby(
            [
                "data_source",
                "feature_set",
                "layers",
            ]
        )
        .mean()
        .reset_index()
    )
    df_std = (
        df.groupby(
            [
                "data_source",
                "feature_set",
                "layers",
            ]
        )
        .std()
        .reset_index()
    )

    df = df_mean.merge(
        df_std,
        on=["data_source", "feature_set", "layers"],
        suffixes=("_mean", "_std"),
    )
    df = df[new_col_order]
    # Drop columns that are all NaN
    df = df.dropna(axis=1, how="all")
    # Create the new column order and reindex the dataframe
    df.to_csv(here() / "outputs/variants/ranked_models.csv", index=False)


if __name__ == "__main__":
    main()
