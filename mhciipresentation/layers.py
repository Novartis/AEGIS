#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""layers.py

Layers used for the various protein language models in this package.

"""

import math

import torch
import torch.nn.functional as nn
from torch import nn
import numpy as np
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class DummyEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        all_ones: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if not all_ones:
            pe = torch.zeros(1, max_len, d_model)
        else:
            pe = torch.ones(1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Feedforward neural network using the output of the transformer for binary
    classification purposes.
    """

    def __init__(self, input_dim: int, ff_hidden: int, dropout: float = 0.0):
        """Initializes the feedforward network used to make a classification
            using the representation given by the transformer encoder layer

        Args:
            input_dim (int): size of the embedding used as input for the model
                equal to embedding_size * max_seq_len.
            ff_hidden (int): hidden neurons in the hidden layers.
            dropout (float, optional): Dropout rate of the hidden layer.
                Defaults to 0.0.
        """
        super().__init__()
        # We set ff_hidden as a default to 2048
        self.linear_1 = nn.Linear(input_dim, ff_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_hidden, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the feedforward network.

        Args:
            x (torch.Tensor): torch tensor of shape
                (n_samples, embedding_size * max_seq_len)

        Returns:
            torch.Tensor: (n_samples, 1) prediction output
        """
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        x = self.activation(x)
        return x


class TemperatureScaling(nn.Module):
    """
    Adapted from: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Perform temperature scaling on logits

        Args:
            logits (torch.Tensor): logits from model output to scale

        Returns:
            torch.Tensor: scaled logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature  # type: ignore

    def forward(self, logits):
        return self.temperature_scale(logits)
