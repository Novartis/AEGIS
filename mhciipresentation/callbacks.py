from pathlib import Path

import pytorch_lightning as pl
import torch
from mhciipresentation.utils import make_dir, save_obj


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
