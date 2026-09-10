"""Gradient Reversal Layer (GRL) -- the mechanism that makes DANN adversarial.

Forward pass: identity (changes nothing). Backward pass: multiplies the
incoming gradient by -lambda_. So from the feature extractor's point of
view, minimizing the domain discriminator's loss through this layer is the
same as *maximizing* it directly -- the standard trick from Ganin & Lempitsky
2015 ("Unsupervised Domain Adaptation by Backpropagation") that lets one
optimizer, one backward() call, train the whole adversarial game without
alternating min/max steps by hand.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """`lambda_` is mutable (set it before each forward pass from the
    training loop) so it can follow the standard DANN warm-up schedule --
    see `grl_lambda_schedule()` below -- rather than being fixed for the
    whole run.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)


def grl_lambda_schedule(progress: float, gamma: float = 10.0) -> float:
    """progress: training progress in [0, 1] (e.g. epoch / total_epochs).

    Returns lambda = 2/(1+exp(-gamma*progress)) - 1, the schedule from the
    original DANN paper: starts at 0 (early training doesn't fight the
    still-random feature extractor with a full-strength adversarial
    signal) and saturates toward 1 as training progresses.
    """
    progress = min(max(progress, 0.0), 1.0)
    return float(2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0)
