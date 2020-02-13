from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class ImplicitFeedbackBCELoss(nn.Module):
    def __init__(self, confidence_weights: List[float], weight=None, reduction='mean') -> None:
        super().__init__()
        self.confidence_weights = confidence_weights
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, *confidence_targets: torch.Tensor) -> torch.Tensor:
        assert len(self.confidence_weights) == len(confidence_targets)
        confidence = torch.ones_like(target)
        for confidence_target, confidence_weight in zip(confidence_targets, self.confidence_weights):
            confidence += confidence_weight * confidence_target
        loss = confidence * F.binary_cross_entropy(input, target, weight=self.weight, reduction="none")
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CounterfactualRiskMinimization(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)
        self.reduction = reduction

    def forward(self, prob, target, ps):
        
        weights  = 1.0 / ps
        loss     = F.binary_cross_entropy(prob.view(-1), target, weight=weights, reduction='none')

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
