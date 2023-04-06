#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""models.py

File defining all protein language models used in this package.

"""

import collections
import math
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torchmetrics
from mhciipresentation.constants import AA_TO_INT_PTM, USE_GPU
from mhciipresentation.layers import FeedForward, PositionalEncoding
from mhciipresentation.utils import prepare_batch
from sklearn.preprocessing import Binarizer
from torch import nn
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import (
    AUROC,
    ROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    PrecisionRecallCurve,
    Recall,
)
from tqdm import tqdm


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
        pad_num: int,
        batch_size: int,
        start_learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        loss_fn=nn.BCELoss(),
        metrics: Dict[str, Any] = {},
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

        def update_metric_keys(metrics, prefix):
            metrics_new = metrics.copy()
            for metric in metrics.keys():
                metrics_new[f"{prefix}_{metric}"] = metrics_new.pop(metric)
            return metrics_new

        super().__init__()
        self.model_type = "Transformer"
        self.seq_len = seq_len
        self.pos_encoder = PositionalEncoding(embedding_size, dropout, seq_len)
        encoder_layers = TransformerEncoderLayer(
            embedding_size,
            n_attn_heads,
            enc_ff_hidden,
            dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(n_tokens, self.embedding_size)
        self.feedforward = FeedForward(
            self.embedding_size * self.seq_len, ff_hidden, dropout
        )
        self.loss_fn = loss_fn
        self.start_learning_rate = start_learning_rate
        self.weight_decay = weight_decay
        self.metrics = metrics
        self.pad_num = pad_num
        self.batch_size = batch_size
        train_metrics = update_metric_keys(metrics.copy(), prefix="train")
        val_metrics = update_metric_keys(metrics.copy(), prefix="val")
        test_metrics = update_metric_keys(metrics.copy(), prefix="test")
        metrics = {}
        for d in [train_metrics, val_metrics, test_metrics]:
            for k, v in d.items():
                metrics[k] = v
        self.metrics = metrics
        self.init_weights()

    def init_weights(self) -> None:
        """Uniform weight initialization"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, data, src_padding_mask) -> torch.Tensor:
        """Defines computation to be performed for each input

        Args:
            src (torch.Tensor): input data of shape (batch_size, max_seq_len)
            src_padding_mask (torch.Tensor): bool mask of padding token of shape
                (batch_size, max_seq_len)

        Returns:
            torch.Tensor: output of the model
        """
        src = data[0]
        src, src_padding_mask = self.add_dummy_sample(src, src_padding_mask)
        src = self.embedding(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(
            src, src_key_padding_mask=src_padding_mask
        )
        output = self.feedforward(src.view(src.size(0), -1))[1:]
        return output.double()  # type: ignore

    def add_dummy_sample(self, src, src_padding_mask):
        src_padding_mask = torch.vstack(
            [
                torch.full((self.seq_len,), False).to(self.device),
                src_padding_mask,
            ]
        )
        src = torch.vstack(
            [
                torch.full((self.seq_len,), AA_TO_INT_PTM["E"]).to(
                    self.device
                ),
                src,
            ]
        )
        return src, src_padding_mask

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = {
            "scheduler": ExponentialLR(
                optimizer, gamma=0.9
            ),  # TODO: replace with one of those oscillating schedulers
            "monitor": "val_loss",
            "interval": "step",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    # def update_metrics(self, y_true, y_pred, prefix):
    #     for metric in [key for key in self.metrics.keys() if prefix in key]:
    #         self.metrics[metric].update(y_pred.cpu(), y_true.cpu().int())

    def compute_metrics(self, prefix, y_pred, y_true):
        for metric in [key for key in self.metrics.keys() if prefix in key]:
            metric_res = self.metrics[metric](y_pred.cpu(), y_true.cpu().int())
            self.log(
                f"{metric}",
                metric_res.float(),
                sync_dist=True,
            )

    def reset_metrics(self, prefix):
        for metric in [key for key in self.metrics.keys() if prefix in key]:
            self.metrics[metric].reset()

    def generate_padding_mask(self, batch):
        src_padding_mask = batch[0] == self.pad_num
        src_padding_mask = src_padding_mask.bool()
        if self.batch_size != batch[0].size(0):
            src_padding_mask[:, : batch[0].size(0)]
        return src_padding_mask

    def training_step(self, batch, batch_idx):
        src_padding_mask = self.generate_padding_mask(batch)
        y_hat = self(batch, src_padding_mask)
        loss = self.loss_fn(y_hat, batch[1].view(-1, 1).double())
        self.log(
            "train_loss", loss, batch_size=batch[0].shape[0], sync_dist=True
        )
        y_true = torch.Tensor(
            Binarizer(threshold=0.5).transform(
                batch[1].view(-1, 1).double().cpu()
            )
        )
        self.compute_metrics("train", y_true, y_hat)
        return loss

    def validation_step(self, batch, batch_idx):
        src_padding_mask = self.generate_padding_mask(batch)
        y_hat = self(batch, src_padding_mask)
        loss = self.loss_fn(y_hat, batch[1].view(-1, 1).double())
        self.log(
            "val_loss", loss, batch_size=batch[0].shape[0], sync_dist=True
        )
        y_true = torch.Tensor(
            Binarizer(threshold=0.5).transform(
                batch[1].view(-1, 1).double().cpu()
            )
        )
        self.compute_metrics("val", y_true, y_hat)

    def test_step(self, batch, batch_idx):
        src_padding_mask = self.generate_padding_mask(batch)
        y_hat = self(batch, src_padding_mask)
        loss = self.loss_fn(y_hat, batch[1].view(-1, 1).double())
        self.log(
            "test_loss", loss, batch_size=batch[0].shape[0], sync_dist=True
        )
        y_true = torch.Tensor(
            Binarizer(threshold=0.5).transform(
                batch[1].view(-1, 1).double().cpu()
            )
        )
        self.compute_metrics("test", y_true, y_hat)

    # def on_train_epoch_end(self):
    #     # compute metrics
    #     self.compute_metrics("train")
    #     # reset all metrics
    #     self.reset_metrics("train")

    # def on_validation_epoch_end(self):
    #     # compute metrics
    #     self.compute_metrics("val")
    #     # reset all metrics
    #     self.reset_metrics("val")

    # def on_test_epoch_end(self):
    #     # compute metrics
    #     self.compute_metrics("test")
    #     # reset all metrics
    #     self.reset_metrics("test")
