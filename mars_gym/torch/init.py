import torch
from math import sqrt
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out


def lecun_normal_init(tensor: torch.Tensor):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    nn.init.normal_(tensor, 0, sqrt(1.0 / fan_in))


def he_init(tensor: torch.Tensor):
    nn.init.kaiming_normal_(tensor, mode="fan_in")
