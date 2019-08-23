import re
from collections import Mapping, Sequence
from typing import List, Union

import torch
import torchbearer
import math
import numpy as np
from scipy.sparse import coo_matrix
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern
from torchbearer.callbacks import Callback
import mlflow
from torchbearer.callbacks.torch_scheduler import TorchScheduler, StepLR


class MLFlowLogger(Callback):
    def on_end_epoch(self, state):
        metrics: dict = state[torchbearer.METRICS]

        for key, value in metrics.items():
            mlflow.log_metric(key, value)


class _CosineAnnealingWithRestartsLR(torch.optim.lr_scheduler._LRScheduler):
    r"""
    Forked from: https://github.com/roveo/pytorch/pull/1

    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. This implements
    the cosine annealing part of SGDR, the restarts and number of iterations multiplier.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        T_mult (float): Multiply T_max by this number after each restart. Default: 1.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.restart_every = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.restarted_at = 0
        super().__init__(optimizer, last_epoch)

    def restart(self):
        self.restart_every *= self.T_mult
        self.restarted_at = self.last_epoch

    def cosine(self, base_lr):
        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2

    @property
    def step_n(self):
        return self.last_epoch - self.restarted_at

    def get_lr(self):
        if self.step_n >= self.restart_every:
            self.restart()
        return [self.cosine(base_lr) for base_lr in self.base_lrs]


class CosineAnnealingWithRestartsLR(TorchScheduler):
    def __init__(self, T_max, eta_min=0, T_mult=1, last_epoch=-1, step_on_batch=False):
        super(CosineAnnealingWithRestartsLR, self).__init__(
            lambda opt: _CosineAnnealingWithRestartsLR(opt, T_max, eta_min, last_epoch, T_mult),
            step_on_batch=step_on_batch)


class _CyclicLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Forked from: https://github.com/pytorch/pytorch/pull/2016

    Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle.
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up.
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_idx (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 optimizer,
                 base_lr=1e-3,
                 max_lr=6e-3,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 last_batch_idx=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        base_lrs = self._format_lr('base_lr', optimizer, base_lr)
        if last_batch_idx == -1:
            for base_lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = base_lr

        self.max_lrs = self._format_lr('max_lr', optimizer, max_lr)

        step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.total_size = float(step_size_up + step_size_down)
        self.step_ratio = float(step_size_up) / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        super(_CyclicLR, self).__init__(optimizer, last_batch_idx)

    def _format_lr(self, name, optimizer, lr):
        """Return correctly formatted lr for each param group."""
        if isinstance(lr, (list, tuple)):
            if len(lr) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(lr)))
            return np.array(lr)
        else:
            return lr * np.ones(len(optimizer.param_groups))

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.
        """
        cycle = np.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)
        return lrs


class CyclicLR(TorchScheduler):
    def __init__(self, base_lr=1e-3,
                 max_lr=6e-3,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 last_batch_idx=-1):
        super(CyclicLR, self).__init__(
            lambda opt: _CyclicLR(opt, base_lr, max_lr, step_size_up, step_size_down, mode, gamma, scale_fn, scale_mode,
                                  last_batch_idx), step_on_batch=True)


class LearningRateFinder(StepLR):
    def __init__(self, iterations: int, initial_lr: float, end_lr: float = 10.0, linear=False, stop_dv=True):
        ratio = end_lr / initial_lr
        lr_mult = (ratio / iterations) if linear else ratio ** (1 / iterations)

        super().__init__(step_size=1, gamma=lr_mult, step_on_batch=True)

        self._iterations = iterations
        self._current_iteration = 0
        self._stop_dv = stop_dv
        self._best_loss = 1e9
        self.learning_rates: List[float] = []
        self.loss_values: List[float] = []

    def on_criterion(self, state):
        self._current_iteration += 1

        self.learning_rates.append(self._scheduler.get_lr()[0])
        loss = float(state[torchbearer.LOSS])
        self.loss_values.append(loss)

        if loss < self._best_loss:
            self._best_loss = loss

        if self._current_iteration >= self._iterations or (self._stop_dv and loss > 10 * self._best_loss):
            state[torchbearer.STOP_TRAINING] = True

    def get_loss_derivatives(self, sma: int = 1):
        derivatives = [0] * (sma + 1)
        for i in range(1 + sma, len(self.loss_values)):
            derivative = (self.loss_values[i] - self.loss_values[i - sma]) / sma
            derivatives.append(derivative)
        return derivatives


class MaskedZeroesLoss(nn.Module):
    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        self._wrapped_loss = loss

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if target.layout == torch.sparse_coo:
            target = target.to_dense()
        mask = target.ne(0)
        input = input.masked_select(mask)
        target = target.masked_select(mask)
        return self._wrapped_loss.forward(input, target)


def coo_matrix_to_sparse_tensor(coo: coo_matrix) -> torch.Tensor:
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.tensor(indices)
    v = torch.tensor(values, dtype=torch.float32)

    return torch.sparse_coo_tensor(i, v, coo.shape)

def collate_fn(batch, use_shared_memory=False):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        if len(batch[0].shape) > 1:
            return torch.cat(batch, 0, out=out)
        else:
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


