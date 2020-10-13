from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss


class FocalLoss(nn.Module):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param input:  model's output, shape of [batch_size, num_cls]
        :param target: ground truth labels, shape of [batch_size, num_cls]
        :return: loss
        """
        epsilon = 1.0e-9

        t: torch.Tensor = targets.to(torch.float32)
        p: torch.Tensor = inputs.to(torch.float32) + epsilon

        pt: torch.Tensor = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        ce: torch.Tensor = -torch.log(pt)
        weight: torch.Tensor = (1.0 - pt) ** self.gamma
        loss: torch.Tensor = weight * self.alpha * ce

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class ImplicitFeedbackBCELoss(nn.Module):
    def __init__(
        self, confidence_weights: List[float], weight=None, reduction="mean"
    ) -> None:
        super().__init__()
        self.confidence_weights = confidence_weights
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        *confidence_targets: torch.Tensor
    ) -> torch.Tensor:
        assert len(self.confidence_weights) == len(confidence_targets)
        confidence = torch.ones_like(target)
        for confidence_target, confidence_weight in zip(
            confidence_targets, self.confidence_weights
        ):
            confidence += confidence_weight * confidence_target
        loss = confidence * F.binary_cross_entropy(
            input, target, weight=self.weight, reduction="none"
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CounterfactualRiskMinimization(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean", clip=None):
        super().__init__(size_average, reduce, reduction)
        self.reduction = reduction
        self.clip = clip

    def forward(self, prob, target, ps):
        # from IPython import embed; embed()
        ps = 1 / (ps + 0.0001)
        if self.clip is not None:
            ps = torch.clamp(ps, max=self.clip)

        loss = F.binary_cross_entropy(prob.view(-1), target, reduction="none")

        loss = loss * ps  # (/ ps.sum())#.sum()

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DummyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)

    def forward(self, loss, target):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss    
