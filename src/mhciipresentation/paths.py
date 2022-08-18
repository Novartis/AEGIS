#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""paths.py

Provides global variables to other scripts used in the repository

    Attributes:
        DOTENV_KEY2VAL (str): loads the .env file in the root folder of the
        repository containing general paths useful for multiple subsequent
        paths.
        N_JOBS (str): number of CPU cores used by the script.
        COL_NAMES (str): column names used by the unencoded dataframes.
        TRAIN_LOG_DIR (str): directory where model training logs are stored
        RAW_DATA (str): directory where the raw data is stored
        XGBOOST_MODEL_DIRS (str): directory where the xgboost models is stored
        PROCESSED_DATA (str): directory where processed data (i.e.
        non-raw, so it includes temporary files)
        ENCODED_DATA_PATH (str): directory where the encoded data is stored.
        This can be used as input for models
        PREDICTION_PATH (str): directory to store predictions
        PERFORMANCE_PATH (str): directory to store performance measures of the
        models
        IDX_PATH (str): directory containing files that have the indices paths
        for folds
        LOMO_IDX (str): directory containing the indices for the leave one
        (hla) molecule out folds.
        PLOTS (str): directory containing the plots generated along the way
        MOUSE_PUBLIC (str): directory containing the public mouse data
"""
import os

SRC_DIR = os.path.dirname(os.path.realpath(__file__))

CACHE_DIR = "/scratch/$USER/.cache/"

DATA_DIR = os.path.abspath(
    os.path.join(SRC_DIR, "../../data/mhciipresentation/")
)

SPLITS_DIR = os.path.join(DATA_DIR, "splits/")

LOG_DIR = os.path.join(DATA_DIR, "modelling/logs/")

MODELS_DIR = os.path.join(DATA_DIR, "modelling/")

PROD_MODELS_DIR = os.path.join(DATA_DIR, "models/")

PSEUDOSEQUENCES = os.path.join(DATA_DIR, "pseudosequence_mappings/")

TRAIN_LOG = os.path.join(LOG_DIR, "training_logs/")

RAW_DATA = os.path.join(DATA_DIR, "datasets/")

EPITOPES_DIR = os.path.join(DATA_DIR, "eval/")

XGBOOST_MODEL = os.path.join(DATA_DIR, "modelling/1-xgboost/")

PROCESSED_DATA = os.path.join(DATA_DIR, "processed/0-preprocessed/")

LEVENSTEIN_DIR = os.path.join(DATA_DIR, "processed/levenstein/")

ENCODED_DATA = os.path.join(DATA_DIR, "processed/1-encoded/")

PREDICTION = os.path.join(DATA_DIR, "processed/2-predictions/")

PERFORMANCE = os.path.join(DATA_DIR, "processed/3-performance/")

PLOTS = os.path.join(DATA_DIR, "plots/")

MOUSE_PUBLIC = os.path.join(DATA_DIR, "splits/mouse/random/")

PUBLIC_VAL_PERF = os.path.join(DATA_DIR, "public/validation_performance/")
