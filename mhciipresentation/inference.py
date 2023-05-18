#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""inference.py

Utilities useful for inference.

"""
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from mhciipresentation.callbacks import GPUUsageLogger, VectorLoggingCallback
from mhciipresentation.constants import AA_TO_INT
from mhciipresentation.metrics import (
    build_scalar_metrics,
    build_vector_metrics,
    compute_performance_metrics,
    save_performance_metrics,
)
from mhciipresentation.models import TransformerModel
from mhciipresentation.utils import (
    get_accelerator,
    get_hydra_logging_directory,
    load_model_weights,
)
from pyprojroot import here
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def setup_model(input_dim, n_tokens, cfg):
    return TransformerModel.load_from_checkpoint(
        cfg.paths.checkpoint,
        seq_len=input_dim,
        n_tokens=n_tokens,
        embedding_size=cfg.model.aegis.embedding.size,
        n_attn_heads=cfg.model.aegis.n_attn_heads,
        enc_ff_hidden=cfg.model.aegis.enc_ff_hidden,
        ff_hidden=cfg.model.aegis.ff_hidden,
        n_layers=cfg.model.aegis.n_layers,
        dropout=cfg.model.aegis.dropout,
        pad_num=AA_TO_INT["X"],
        batch_size=cfg.training.batch_size,
        warmup_steps=cfg.training.learning_rate.warmup_steps,
        epochs=cfg.training.epochs,
        start_learning_rate=cfg.training.learning_rate.start_learning_rate,
        peak_learning_rate=cfg.training.learning_rate.peak_learning_rate,
        weight_decay=cfg.training.optimizer.weight_decay,
        loss_fn=nn.BCELoss(),
        scalar_metrics=build_scalar_metrics(),
        vector_metrics=build_vector_metrics(),
        n_gpu=cfg.compute.n_gpu,
        n_cpu=cfg.compute.n_cpu,
        steps_per_epoch=100,
        dummy_encoding=cfg.model.aegis.embedding.dummy_embedding
        all_ones=cfg.model.aegis.embedding.all_ones
    )


def make_inference(X, y, cfg, input_dim, dest_dir):
    X = torch.from_numpy(
        np.stack(
            [
                np.pad(
                    seq,
                    pad_width=(0, input_dim - len(seq)),
                    constant_values=AA_TO_INT["X"],
                )
                for seq in X
            ]
        ).astype(int)
    ).int()
    y = torch.from_numpy(y)

    dataset = TensorDataset(
        torch.from_numpy(
            np.stack(
                [
                    np.pad(
                        seq,
                        pad_width=(0, input_dim - len(seq)),
                        constant_values=AA_TO_INT["X"],
                    )
                    for seq in X
                ]
            )
        ),
        y,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.compute.n_cpu,
    )
    # batch_size = 5000

    device = get_accelerator(
        cfg.debug.debug, cfg.compute.n_gpu, cfg.compute.mps, cfg.compute.cuda
    )
    model = setup_model(
        input_dim=input_dim,
        n_tokens=len(list(AA_TO_INT.values())),
        cfg=cfg,
    )
    save_name = "aegis_inference_cd4"
    trainer = pl.Trainer(
        default_root_dir=get_hydra_logging_directory() / "predictions_logs",
        accelerator=device.type,
        devices=cfg.compute.n_gpu,
        num_nodes=cfg.compute.num_nodes,
        max_epochs=cfg.training.epochs,
        callbacks=[
            RichProgressBar(leave=True),
            VectorLoggingCallback(
                root=Path(get_hydra_logging_directory()) / "vector_logs"
            ),
            GPUUsageLogger(
                log_dir=get_hydra_logging_directory()
                / "tensorboard"
                / "gpu_usage"
            ),
        ],
        log_every_n_steps=1,
        benchmark=cfg.debug.benchmark,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        logger=[
            pl_loggers.TensorBoardLogger(
                save_dir=get_hydra_logging_directory()
                / "predict"
                / "tensorboard",
                name="aegis_inference",
            ),
            pl_loggers.CSVLogger(
                save_dir=get_hydra_logging_directory() / "predict" / "csv",
                name=save_name,
            ),
        ],
    )
    # This is specific to our model because of what we return in the predict_step method.
    y_hat = trainer.predict(model, loader)[0]["y_hat"].reshape(-1)
    y = y.reshape(-1)
    scalar_metrics = build_scalar_metrics()
    vector_metrics = build_vector_metrics()
    scalar_metric_values, vector_metric_values = compute_performance_metrics(
        scalar_metrics,
        vector_metrics,
        y,
        y_hat,
    )
    save_performance_metrics(
        dest_dir,
        scalar_metric_values,
        vector_metric_values,
    )
