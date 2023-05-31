from pathlib import Path

import GPUtil
import pytorch_lightning as pl
import torch
from mhciipresentation.utils import make_dir, save_obj
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter


class DelayedEarlyStopping(EarlyStopping):
    """Only start early stopping mmonitoriung after a certain number of epochs"""

    def __init__(self, delay_epochs, **kwargs):
        super().__init__(**kwargs)
        self.delay_epochs = delay_epochs

    def on_train_end(self, trainer, pl_module):
        self.wait_count = 0

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.delay_epochs:
            return
        super().on_validation_end(trainer, pl_module)


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
        def log_loader(idx, name):
            # Loop through 0 and 1 and torch.cat each vectors separately
            y_true = torch.cat(
                [
                    x["y_true"]
                    for x in [
                        s for s in self.state["val"] if s.get("idx") == idx
                    ]
                ]
            )
            y_hat = torch.cat(
                [
                    x["y_hat"]
                    for x in [
                        s for s in self.state["val"] if s.get("idx") == idx
                    ]
                ]
            )
            self.log_vectors(y_true, y_hat, name, pl_module)

        idx_values = [0, 1]
        dset_names = ["val", "test"]

        for idx, name in zip(idx_values, dset_names):
            log_loader(idx, name)

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
            key for key in model.vector_metrics.keys() if str(prefix) in key
        ]:
            metric_res = model.vector_metrics[metric](
                y_hat.cpu(), y_true.cpu().int()
            )
            save_obj(
                metric_res,
                log_dest / str(metric + ".pkl"),
            )


class GPUUsageLogger(pl.Callback):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def on_train_epoch_end(self, trainer, batch_idx):
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


class ResetProfilerCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Print profiler summary
        if trainer.current_epoch % 50 or trainer.current_epoch == 1:
            profiler_dump = trainer.profiler.summary()
            dest = (
                Path(trainer.profiler.dirpath)
                / f"profiler_dump_epoch_{trainer.current_epoch}.csv"
            )
            # Make parent dir if non-existent
            dest.parent.mkdir(parents=True, exist_ok=True)
            if profiler_dump is not None:
                profiler_dump.to_csv(
                    dest,
                    index=False,
                )

        # Reset profiler
        trainer.profiler = trainer.profiler.__class__(
            dirpath=trainer.profiler.dirpath
        )
