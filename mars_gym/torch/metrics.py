from typing import Sequence

import torch
import torchbearer
from torchbearer import metrics, Metric
from torchbearer.metrics import default_for_key, running_mean, mean
import torch.nn.functional as F


@metrics.default_for_key("bce")
@running_mean
@mean
@metrics.lambda_metric("bce", on_epoch=False)
def bce(y_pred: torch.Tensor, y_true: torch.Tensor, *args):
    if isinstance(y_true, Sequence) and isinstance(y_pred, torch.Tensor):
        y_true = y_true[0]
    if y_true.layout == torch.sparse_coo:
        y_true = y_true.to_dense()

    loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")

    return loss.mean()


@metrics.default_for_key("binary_accuracy")
@running_mean
@mean
@metrics.lambda_metric("binary_accuracy", on_epoch=False)
def binary_accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor, *args, threshold: float = 0.5
):
    if isinstance(y_true, Sequence) and isinstance(y_pred, torch.Tensor):
        y_true = y_true[0]
    if y_true.layout == torch.sparse_coo:
        y_true = y_true.to_dense()

    y_pred = (y_pred.float() > threshold).long()
    y_true = (y_true.float() > threshold).long()

    return torch.eq(y_pred, y_true).view(-1).float()


@metrics.default_for_key("precision")
@running_mean
@mean
@metrics.lambda_metric("precision", on_epoch=False)
def precision(
    y_pred: torch.Tensor, y_true: torch.Tensor, *args, threshold: float = 0.5, eps=1e-9
):
    if isinstance(y_true, Sequence) and isinstance(y_pred, torch.Tensor):
        y_true = y_true[0]
    if y_true.layout == torch.sparse_coo:
        y_true = y_true.to_dense()

    y_pred = (y_pred.float() > threshold).float()
    y_true = (y_true.float() > threshold).float()

    # true_positive = (y_pred * y_true).sum(dim=-1)
    # return true_positive.div(y_pred.sum(dim=-1).add(eps))

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)

    return precision


@metrics.default_for_key("recall")
@running_mean
@mean
@metrics.lambda_metric("recall", on_epoch=False)
def recall(
    y_pred: torch.Tensor, y_true: torch.Tensor, *args, threshold: float = 0.5, eps=1e-9
):
    if isinstance(y_true, Sequence) and isinstance(y_pred, torch.Tensor):
        y_true = y_true[0]
    if y_true.layout == torch.sparse_coo:
        y_true = y_true.to_dense()
    y_pred = (y_pred.float() > threshold).float()
    y_true = (y_true.float() > threshold).float()

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    recall = tp / (tp + fn + epsilon)

    return recall


@metrics.default_for_key("f1_score")
@running_mean
@mean
@metrics.lambda_metric("f1_score", on_epoch=False)
def f1_score(
    y_pred: torch.Tensor, y_true: torch.Tensor, *args, threshold: float = 0.5, eps=1e-9
):
    if isinstance(y_true, Sequence) and isinstance(y_pred, torch.Tensor):
        y_true = y_true[0]
    if y_true.layout == torch.sparse_coo:
        y_true = y_true.to_dense()
    y_pred = (y_pred.float() > threshold).float()
    y_true = (y_true.float() > threshold).float()

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1
