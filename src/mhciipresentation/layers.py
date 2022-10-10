#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""layers.py

Layers used for the various protein language models in this package.

"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
        self, embedding_size: int, dropout: float = 0.0, max_len: int = 6144
    ):
        r"""Initializes the PositionalEncoding layer of the encoder network

        Args:
            embedding_size (int): embedding size of the data
            dropout (float, optional): dropout rate of the output of the pe
                layer. Defaults to 0.0.
            max_len (int, optional): maximum length of the positional encoder.
                Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2)
            * (-math.log(10000.0) / embedding_size)
        )
        pe = torch.zeros(max_len, 1, embedding_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computations performed on each input.

        Args:
            x (torch.Tensor): (batch_size, max_seq_len, embedding_size) output
                data

        Returns:
            torch.Tensor: (batch_size, max_seq_len, embedding_size) output
                positional encoding with dropout at training
        """
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)  # type: ignore


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

    def __init__(self,):
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
