"""
utils.py
--------
Small helper functions shared by train.py and predict.py.
"""

import os
import torch


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, device):
    checkpoint = torch.load(path, map_location=device)
    print(f"Checkpoint loaded from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


def clip_gradients(optimizer, grad_clip: float):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter:
    """Tracks a running average of a metric (e.g. loss) across batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
