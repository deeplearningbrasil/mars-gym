import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import List


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

    def __init__(self, gamma: float = 2.0, alpha: float = 4.0, size_average=True):
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
        epsilon = 1.e-9

        t: torch.Tensor = targets.to(torch.float32)
        p: torch.Tensor = inputs.to(torch.float32) + epsilon

        pt: torch.Tensor = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        ce: torch.Tensor = -torch.log(pt)
        weight: torch.Tensor = (1. - pt) ** self.gamma
        loss: torch.Tensor = weight * self.alpha * ce
        if loss.dim() == 2:  # softmax
            loss: torch.Tensor = loss[:, 0]

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BayesianPersonalizedRankingTripletLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = 1 - \sigma (d(a_i, p_i) - d(a_i, n_i))

    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Args:
        margin (float, optional): Default: `1`.
        p (int, optional): The norm degree for pairwise distance. Default: `2`.
        swap (float, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'

    Shape:
        - Input: :math:`(N, D)` where `D` is the vector dimension.
        - Output: scalar. If `reduce` is False, then `(N)`.

    >>> triplet_loss = BayesianPersonalizedRankingTripletLoss(p=2)
    >>> input1 = torch.randn(100, 128, requires_grad=True)
    >>> input2 = torch.randn(100, 128, requires_grad=True)
    >>> input3 = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()

    .. _BPR: Bayesian Personalized Ranking from Implicit Feedback:
        https://arxiv.org/abs/1205.2618
    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """
    def __init__(self, p=2., eps=1e-6, swap=False, size_average=None,
                 reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, self.p, self.eps)
        negative_distance = F.pairwise_distance(anchor, negative, self.p, self.eps)

        if self.swap:
            positive_negative_distance = F.pairwise_distance(positive, negative, self.p, self.eps)
            negative_distance = torch.min(negative_distance, positive_negative_distance)

        loss = 1 - F.sigmoid(positive_distance - negative_distance)
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()


class VAELoss(nn.Module):
    def __init__(self, anneal=1.0):
        super(VAELoss, self).__init__()
        self.anneal = anneal

    def forward(self, recon_x, mu, logvar, targets):
        targets = targets.to_dense()
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * targets, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        loss = BCE + self.anneal * KLD
        return loss