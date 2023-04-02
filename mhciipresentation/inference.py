#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""inference.py

Utilities useful for inference.

"""

import json
import os
from pathlib import Path

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
        input_dim = int(data["input_dim"])
        n_tokens = int(data["n_tokens"])
        embedding_size = int(data["embedding_size"])
        n_attn_heads = int(data["n_attn_heads"])
        enc_ff_hidden = int(data["enc_ff_hidden"])
        ff_hidden = int(data["ff_hidden"])
        nlayers = int(data["nlayers"])
        dropout = float(data["dropout"])

        if "max_len" in data:
            max_len = int(data["max_len"])
        else:
            max_len = 5000

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
        max_len,
    )
    if USE_GPU:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))  # type: ignore
    else:
        model = nn.DataParallel(  # type: ignore
            model, device_ids=[i for i in range(30)]
        )
    model = load_model_weights(model, model_path, device)
    model.eval()
    return model, input_dim, max_len


def setup_model(device: torch.device, model_path: str):
    """Sets up the model.

    Args:
        device (torch.device): device to run the model on
        model_path (str): [description]

    Returns:
        [type]: [description]
    """
    with open(
        Path(model_path).parent.parent / "training_params.json",
        "r",
    ) as infile:
        data = json.loads(infile.read())
        input_dim = int(data["input_dim"])
        n_tokens = int(data["n_tokens"])
        embedding_size = int(data["embedding_size"])
        n_attn_heads = int(data["n_attn_heads"])
        enc_ff_hidden = int(data["enc_ff_hidden"])
        ff_hidden = int(data["ff_hidden"])
        nlayers = int(data["nlayers"])
        dropout = float(data["dropout"])

        if "max_len" in data:
            max_len = int(data["max_len"])
        else:
            max_len = 5000

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
        max_len,
    )
    if USE_GPU:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))  # type: ignore
    else:
        model = nn.DataParallel(  # type: ignore
           model, device_ids=[i for i in range(30)]
        )
    model = load_model_weights(model, model_path, device)
    model.eval()
    return model, input_dim, max_len
