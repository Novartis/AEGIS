#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""models.py

File defining all protein language models used in this package.

"""

import math

import pytorch_lightning as pl
import torch
from mhciipresentation.constants import AA_TO_INT_PTM, USE_GPU
from mhciipresentation.layers import FeedForward, PositionalEncoding
from mhciipresentation.utils import prepare_batch
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from torch imporst nn



class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        dropout,
        device,
    ):
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_dim
        self.device = device

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim * self.hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = Variable(
            torch.zeros(self.n_layers * 2, x.size(1), self.hidden_dim)
        )
        hidden_state = (
            hidden_state.cuda(device=self.device) if USE_GPU else hidden_state
        )

        cell_state = Variable(
            torch.zeros(self.n_layers * 2, x.size(1), self.hidden_dim)
        )
        cell_state = (
            cell_state.cuda(device=self.device) if USE_GPU else cell_state
        )

        embed = self.embedding(x)
        output, _ = self.lstm(embed, (hidden_state, cell_state))
        logits = self.sigmoid(
            self.fc(output.view(-1, self.input_dim * self.hidden_dim * 2))
        )
        return logits.double()  # type: ignore


class TransformerModel(pl.LightningModule):
    """Main class for the transformer encoder used in this script."""

    def __init__(
        self,
        seq_len: int,
        n_tokens: int,
        embedding_size: int,
        n_attn_heads: int,
        enc_ff_hidden: int,
        ff_hidden: int,
        n_layers: int,
        dropout: float,
        device: torch.device,
        max_len: int = 5000,
        loss_fn = nn.BCELoss()
    ):
        r"""Initializes TransformerModel, including PositionalEncoding and
            TransformerEncoderLayer

        Args:
            seq_len (int): maximum input sequence length
            embedding_size (int): embedding size of the the first layer
            n_attn_heads (int): number of attention heads in the encoder layer
            enc_ff_hidden (int): dimensionality of the feedfoward network in the
                encoder layer
            ff_hidden (int): dimensionality of the feedfoward network in the
                last layer
            n_layers (int): number of transformer layers in the encoder
            dropout (float): dropout for the final feedforward layer
            device (torch.device): device used for computation
        """
        super().__init__()
        self.model_type = "Transformer"
        self.seq_len = seq_len
        self.pos_encoder = PositionalEncoding(embedding_size, dropout, max_len)
        encoder_layers = TransformerEncoderLayer(
            embedding_size,
            n_attn_heads,
            enc_ff_hidden,
            dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=n_layers,
        )
        self.embedding = nn.Embedding(n_tokens, embedding_size)
        self.feedforward = FeedForward(
            embedding_size * self.seq_len, ff_hidden, dropout
        )
        self.embedding_size = embedding_size
        self.device = device
        self.loss_fn = loss_fn
        self.init_weights()
        

    def init_weights(self) -> None:
        """Uniform weight initialization"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, src: torch.Tensor, src_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Defines computation to be performed for each input

        Args:
            src (torch.Tensor): input data of shape (batch_size, max_seq_len)
            src_padding_mask (torch.Tensor): bool mask of padding token of shape
                (batch_size, max_seq_len)

        Returns:
            torch.Tensor: output of the model
        """
        src = self.embedding(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(
            src, src_key_padding_mask=src_padding_mask
        )
        output = self.feedforward(
            src.view(-1, self.embedding_size * self.seq_len)
        )

        return output.double()  # type: ignore

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = {
            "scheduler": ExponentialLR(optimizer, gamma=0.9), # TODO: replace with one of those oscillating schedulers
            "monitor": "val_loss",
            "interval": "step",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def evaluate_model(self, y_true, y_pred, prefix):
        for metric in self.metrics.keys():
            self.log(
                f"{prefix}_{metric}",
                self.metrics[metric](y_true, y_pred),
                batch_size=y_true.shape[0],
                sync_dist=True,
            )
        pass

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train_loss", loss, batch_size=batch.y.shape[0], sync_dist=True)
        self.evaluate_model(batch.y, y_hat, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("val_loss", loss, batch_size=batch.y.shape[0], sync_dist=True)
        self.evaluate_model(batch.y, y_hat, "validation")

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("test_loss", loss, batch_size=batch.y.shape[0], sync_dist=True)
        self.evaluate_model(batch.y, y_hat, "test")
