#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tree.py

Tree-based XGBoost algorithm functions to run on the random split of the public
data.

"""

import json
import pickle as pkl
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.model_selection import RandomizedSearchCV

from mhciipresentation.constants import N_JOBS, USE_SUBSET
from mhciipresentation.loaders import load_sa_el_data, load_sa_el_random_idx
from mhciipresentation.paths import CACHE_DIR, DATA_DIR
from mhciipresentation.utils import (
    check_cache,
    compute_performance_measures,
    make_dir,
    oh_encode,
    set_seeds,
)


def get_scale_pos_weight(labels: pd.DataFrame) -> float:
    """Gets the proportion of the negative class to the positive class

    Args:
        labels (pd.DataFrame): labels used to evaluate the imbalance.

    Returns:
        float: proportion of the negative class to the positive class
    """
    class_weight_values = (
        pd.DataFrame(labels).value_counts() / labels.shape[0]
    ).values

    return class_weight_values[0] / class_weight_values[1]  # type: ignore


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray):
    r"""Trains an xgboost by tuning the hyperparameters on a cross validation
        regime.

    Args:
        X_train (np.ndarray): training features of shape
            `(n_samples * n_features)`
        y_train (np.ndarray): [description]

    Returns:
        [type]: trained estimator with hyperparameters selected using the test
            set
    """
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_jobs=N_JOBS
    )
    scale_pos_weight = get_scale_pos_weight(pd.DataFrame(y_train))
    param_distributions = {
        "booster": ["dart"],
        "eta": np.arange(0, 1, 0.01),
        "scale_pos_weight": [scale_pos_weight],
        "min_child_weight": [1],
        "max_depth": range(500, 1000, 1),  # range(1, 1300),
        "gamma": np.arange(7, 15, 0.01),
        "n_estimators": range(10, 20, 1),  # range(10, 100),
        "reg_lambda": np.arange(1, 20, 0.01),  # Ridge regularization
        "reg_alpha": np.arange(1, 20, 0.01),  # Lasso regularization
        "use_label_encoder": [False],
    }

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        scoring=[
            "roc_auc",
            "f1",
            "balanced_accuracy",
            "accuracy",
            "precision",
            "recall",
        ],
        n_jobs=N_JOBS,
        cv=5,
        n_iter=30,
        verbose=3,
        return_train_score=True,
        refit="roc_auc",
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_scores = search.best_score_
    cv_results = pd.DataFrame(search.cv_results_)
    print("CV Results")
    print(cv_results)
    print("Best parameters")
    print(best_params)
    print("Best CV score with possible peptide contamination")
    print(best_scores)
    return search.best_estimator_


def main():
    set_seeds()
    use_cache = check_cache("encoded_sa_el_features.npz")
    if use_cache:
        print("Loading Data")
        sa_el_data = load_sa_el_data()
        encoded_sa_el_data = sparse.load_npz(
            CACHE_DIR + "encoded_sa_el_features.npz"
        )
    else:
        print("Load the data")
        sa_el_data = load_sa_el_data()
        print(f"Data loaded is of shape {sa_el_data.shape}")
        encoded_sa_el_data, sa_el_data_labels = oh_encode(sa_el_data)
        print(f"Final feature vectors of shape {encoded_sa_el_data.shape}")
        sparse.save_npz(
            CACHE_DIR + "encoded_sa_el_features.npz", encoded_sa_el_data,
        )

    print("Load indices and split data according to those")
    X_train_idx, X_val_idx, X_test_idx = load_sa_el_random_idx()

    X_train = encoded_sa_el_data[X_train_idx["index"].tolist()]
    X_val = encoded_sa_el_data[X_val_idx["index"].tolist()]
    X_test = encoded_sa_el_data[X_test_idx["index"].tolist()]

    y_train = sa_el_data.iloc[
        X_train_idx["index"].tolist()
    ].target_value.values
    y_val = sa_el_data.iloc[X_val_idx["index"].tolist()].target_value.values
    y_test = sa_el_data.iloc[X_test_idx["index"].tolist()].target_value.values

    if USE_SUBSET:
        X_train = X_train[:5000]
        y_train = y_train[:5000]

    print("Starting XGBoost training process")
    xgboost_model = train_xgboost(X_train, y_train)

    y_train_pred = xgboost_model.predict(X_train)
    y_val_pred = xgboost_model.predict(X_val)
    y_test_pred = xgboost_model.predict(X_test)
    train_perf = compute_performance_measures(
        y_train_pred.reshape(-1, 1), y_train
    )
    val_perf = compute_performance_measures(y_val_pred.reshape(-1, 1), y_val)
    test_perf = compute_performance_measures(
        y_test_pred.reshape(-1, 1), y_test
    )

    now = datetime.now()
    training_start_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    log_dir = DATA_DIR + "/modelling/xgboost/" + training_start_time + "/"
    make_dir(log_dir)
    make_dir(log_dir + "/metrics/")
    make_dir(log_dir + "/models/")
    with open(f"{log_dir}/metrics/train_perf.json", "w") as outfile:
        json.dump(train_perf, outfile)
    print(train_perf)

    with open(f"{log_dir}/metrics/val_perf.json", "w") as outfile:
        json.dump(val_perf, outfile)
    print(val_perf)

    with open(f"{log_dir}/metrics/test_perf.json", "w") as outfile:
        json.dump(test_perf, outfile)
    print(test_perf)

    pkl.dump(
        xgboost_model,
        open(f"{log_dir}" + "/models/" + "trained_model.pkl", "wb"),
    )
    xgboost_model = pkl.load(
        open(f"{log_dir}" + "/models/" + "trained_model.pkl", "rb")
    )
    y_train_pred = xgboost_model.predict(X_train)
    print("LOADING OK")
    print(y_train_pred)


if __name__ == "__main__":
    main()
