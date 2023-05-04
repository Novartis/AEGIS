from pathlib import Path

import GPUtil
import pytorch_lightning as pl
import torch
from mhciipresentation.utils import make_dir, save_obj
from torch.utils.tensorboard import SummaryWriter


class VectorLoggingCallback(pl.Callback):
    def __init__(self, root: Path):
        super().__init__()
        self.state = {"train": [], "val": [], "test": []}
        self.root = root

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ):
        self.state["train"].append(outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ):
        self.state["val"].append(outputs)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused=0
    ):
        self.state["test"].append(outputs)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        y_true = torch.cat([x["y_true"] for x in self.state["train"]])
        y_hat = torch.cat([x["y_hat"] for x in self.state["train"]])
        self.log_vectors(y_true, y_hat, "train", pl_module)
        self.state["train"] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        y_true = torch.cat([x["y_true"] for x in self.state["val"]])
        y_hat = torch.cat([x["y_hat"] for x in self.state["val"]])
        self.log_vectors(y_true, y_hat, "val", pl_module)
        self.state["val"] = []

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        y_true = torch.cat([x["y_true"] for x in self.state["test"]])
        y_hat = torch.cat([x["y_hat"] for x in self.state["test"]])
        self.log_vectors(y_true, y_hat, "test", pl_module)
        self.state["test"] = []

    def log_vectors(self, y_true, y_hat, prefix, model):
        log_dest = self.root / str(model.current_epoch)
        make_dir(log_dest)
        for metric in [
            key for key in model.vector_metrics.keys() if prefix in key
        ]:
            metric_res = model.vector_metrics[metric](
                y_hat.cpu(), y_true.cpu().int()
            )
            save_obj(
                metric_res,
                log_dest / str(prefix + "_" + metric + ".pkl"),
            )


class GPUUsageLogger(pl.Callback):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx % 100 == 0:
            gpu = GPUtil.getGPUs()[0]
            global_step = trainer.global_step
            self.summary_writer.add_scalar(
                "gpu_usage/memory_used", gpu.memoryUsed, global_step
            )
            self.summary_writer.add_scalar(
                "gpu_usage/memory_total", gpu.memoryTotal, global_step
            )
            self.summary_writer.add_scalar(
                "gpu_usage/memory_utilization", gpu.memoryUtil, global_step
            )
            self.summary_writer.add_scalar(
                "gpu_usage/gpu_utilization", gpu.load, global_step
            )
