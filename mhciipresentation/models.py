#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""models.py

File defining all protein language models used in this package.

"""
import collections
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from mhciipresentation.constants import AA_TO_INT_PTM, USE_GPU
from mhciipresentation.layers import FeedForward, PositionalEncoding, DummyEncoding
from mhciipresentation.scheduler import (
    GradualWarmupScheduler,
    NoamScheduler,
    linear_warmup_decay,
)
from mhciipresentation.utils import prepare_batch, save_obj
from sklearn.preprocessing import Binarizer
from torch import nn
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F
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
            warmup_steps: int,
            epochs: int,
            start_learning_rate: float = 0.001,
            peak_learning_rate: float = 0.01,
            weight_decay: float = 0.01,
            loss_fn=nn.BCELoss(),
            scalar_metrics: Dict[str, Any] = {},
            vector_metrics: Dict[str, Any] = {},
            steps_per_epoch: int = 100,
            n_gpu: int = 1,
            n_cpu: int = 1,
            dummy_encoding: bool=False,
            all_ones: bool=False,
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
        if dummy_encoding:
            self.pos_encoder = PositionalEncoding(embedding_size, dropout, seq_len)
        else:
            if all_ones:
                self.pos_encoder = DummyEncoding(embedding_size, dropout, seq_len, all_ones=all_ones)
            else:
                self.pos_encoder = DummyEncoding(embedding_size, dropout, seq_len, all_ones=all_ones)
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
        self.pad_num = pad_num
        self.batch_size = batch_size

        self.warmup_steps = warmup_steps
        self.epochs = epochs
        self.n_gpu = n_gpu
        self.n_cpu = n_cpu
        self.peak_learning_rate = peak_learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.init_metrics(scalar_metrics, vector_metrics)
        self.init_weights()

    def init_weights(self) -> None:
        """Uniform weight initialization"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def init_metrics(self, scalar_metrics, vector_metrics):
        def update_metric_keys(metrics, prefix):
            metrics_new = metrics.copy()
            for metric in metrics.keys():
                metrics_new[f"{prefix}_{metric}"] = metrics_new.pop(metric)
            return metrics_new

        def split_metric_builder(metrics):
            train_metrics = update_metric_keys(metrics.copy(), prefix="train")
            val_metrics = update_metric_keys(metrics.copy(), prefix="val")
            test_metrics = update_metric_keys(metrics.copy(), prefix="test")
            metrics = {}
            for d in [
                train_metrics,
                val_metrics,
                test_metrics,
            ]:
                for k, v in d.items():
                    metrics[k] = v
            return metrics

        self.scalar_metrics = split_metric_builder(scalar_metrics)
        self.vector_metrics = split_metric_builder(vector_metrics)

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
        src = self.embedding(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(
            src, src_key_padding_mask=src_padding_mask
        )
        output = self.feedforward(src.view(src.size(0), -1))
        return output.double()  # type: ignore

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = {
            "scheduler": NoamScheduler(
                optimizer,
                self.peak_learning_rate,
                warmup_steps=self.warmup_steps,
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def compute_metrics(self, prefix, y_pred, y_true):
        for metric in [
            key for key in self.scalar_metrics.keys() if prefix in key
        ]:
            self.log(
                f"{metric}",
                self.scalar_metrics[metric](y_pred.cpu(), y_true).float(),
                sync_dist=True,
                on_epoch=True,
                on_step=False,
            )

    def generate_padding_mask(self, batch):
        src_padding_mask = batch[0] == self.pad_num
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
        self.compute_metrics("train", y_hat, y_true)
        return {"loss": loss, "y_hat": y_hat, "y_true": y_true}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
        self.compute_metrics("val", y_hat, y_true)
        return {"loss": loss, "y_hat": y_hat, "y_true": y_true}

    def test_step(self, batch, batch_idx, dataloader_idx=1):
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
        self.compute_metrics("test", y_hat, y_true)
        return {"loss": loss, "y_hat": y_hat, "y_true": y_true}

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        src_padding_mask = self.generate_padding_mask(batch)
        y_hat = self(batch, src_padding_mask)
        return {"y_hat": y_hat}
