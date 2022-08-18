#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""inference.py

Utilities useful for inference.

"""

import json
import os

import torch
from torch import nn

from mhciipresentation.constants import N_JOBS, USE_GPU
from mhciipresentation.transformer import TransformerModel
from mhciipresentation.utils import load_model_weights


def setup_model_local(device: torch.device, model_path: str):
    """Sets up the model.

    Args:
        device (torch.device): device to run the model on
        model_path (str): [description]

    Returns:
        [type]: [description]
    """
    with open(
        os.path.join(os.path.dirname(model_path), "training_params.json"), "r"
    ) as infile:
        data = json.loads(infile.read())
        input_dim = data["input_dim"]
        n_tokens = data["n_tokens"]
        embedding_size = data["embedding_size"]
        n_attn_heads = data["n_attn_heads"]
        enc_ff_hidden = data["enc_ff_hidden"]
        ff_hidden = data["ff_hidden"]
        nlayers = data["nlayers"]
        dropout = data["dropout"]

    model = TransformerModel(
        input_dim,
        n_tokens,
        embedding_size,
        n_attn_heads,
        enc_ff_hidden,
        ff_hidden,
        nlayers,
        dropout,
        device,
    )
    if USE_GPU:
        model = nn.DataParallel(model, device_ids=[0])  # type: ignore
    else:
        model = nn.DataParallel(  # type: ignore
            model, device_ids=[i for i in range(30)]
        )
    model = load_model_weights(model, model_path, device)
    model.eval()
    return model, input_dim


def setup_model(device: torch.device, model_path: str):
    """Sets up the model.

    Args:
        device (torch.device): device to run the model on
        model_path (str): [description]

    Returns:
        [type]: [description]
    """
    with open(
        "/".join(model_path.split("/")[:-2]) + "/" + "training_params.json",
        "r",
    ) as infile:
        data = json.loads(infile.read())
        input_dim = data["input_dim"]
        n_tokens = data["n_tokens"]
        embedding_size = data["embedding_size"]
        n_attn_heads = data["n_attn_heads"]
        enc_ff_hidden = data["enc_ff_hidden"]
        ff_hidden = data["ff_hidden"]
        nlayers = data["nlayers"]
        dropout = data["dropout"]

    model = TransformerModel(
        input_dim,
        n_tokens,
        embedding_size,
        n_attn_heads,
        enc_ff_hidden,
        ff_hidden,
        nlayers,
        dropout,
        device,
    )
    if USE_GPU:
        model = nn.DataParallel(model, device_ids=[0])  # type: ignore
    else:
        model = nn.DataParallel(  # type: ignore
            model, device_ids=[i for i in range(30)]
        )
    model = load_model_weights(model, model_path, device)
    model.eval()
    return model, input_dim
