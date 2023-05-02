import math

from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1, verbose=False):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        current_step = self.last_epoch + 1
        scale = self.d_model**-0.5
        arg1 = current_step**-0.5
        arg2 = current_step * (self.warmup_steps**-1.5)

        return [base_lr * scale * min(arg1, arg2) for base_lr in self.base_lrs]


def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay to 0 at total_steps."""
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn


class GradualWarmupScheduler(_LRScheduler):
    def __init__(
        self, optimizer, multiplier, total_epoch, after_scheduler=None
    ):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * (
                (self.multiplier - 1.0) * self.last_epoch / self.total_epoch
                + 1.0
            )
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
