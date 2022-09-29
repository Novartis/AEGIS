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
        non-raw) so it includes temporary files)
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
from pathlib import Path

from pyprojroot import here

SRC_DIR = os.path.dirname(os.path.realpath(__file__))

CACHE_DIR = Path(os.path.abspath(os.path.join(SRC_DIR, "../../.cache/")))
#CACHE_DIR = here() / Path(".cache/")

DATA_DIR = Path(os.path.abspath(os.path.join(SRC_DIR, "../../data/")))
#DATA_DIR = here() / Path("./data/")

LOGS_DIR = Path(os.path.abspath(os.path.join(SRC_DIR, "../../logs/")))
#LOGS_DIR = here() / Path("./logs")

SPLITS_DIR = DATA_DIR / "splits/"

LOG_DIR = DATA_DIR / "modelling/logs/"

MODELS_DIR = DATA_DIR / "modelling/"

PROD_MODELS_DIR = DATA_DIR / "models/"

TRAIN_LOG = LOGS_DIR / "training_logs/"

RAW_DATA = DATA_DIR / "raw/"

EPITOPES_DIR = DATA_DIR / "eval/"

XGBOOST_MODEL = DATA_DIR / "modelling/1-xgboost/"

PROCESSED_DATA = DATA_DIR / "processed/"

LEVENSTEIN_DIR = DATA_DIR / "processed/levenstein/"

ENCODED_DATA = DATA_DIR / "processed/1-encoded/"

PREDICTION = DATA_DIR / "processed/2-predictions/"

PERFORMANCE = DATA_DIR / "processed/3-performance/"

PLOTS = DATA_DIR / "plots/"


PUBLIC_VAL_PERF = DATA_DIR / "public/validation_performance/"
