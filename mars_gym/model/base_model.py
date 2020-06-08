from typing import Tuple, Callable, Union, Type, List
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mars_gym.utils import lecun_normal_init
import numpy as np


class LogisticRegression(nn.Module):
    def __init__(self, n_factors: int, weight_init: Callable = lecun_normal_init):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_factors, 1)
        self.weight_init = weight_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def none_tensor(self, x):
        return type(x) == type(None)

    def predict(self, item_representation, user_representation, context_representation):
        x: torch.Tensor = context_representation

        if not self.none_tensor(user_representation):
            x = (
                user_representation
                if self.none_tensor(x)
                else torch.cat((x, user_representation), dim=1)
            )
        if not self.none_tensor(item_representation):
            x = (
                item_representation
                if self.none_tensor(x)
                else torch.cat((x, item_representation), dim=1)
            )

        return torch.sigmoid(self.linear(x))

class DeepFactorizationMachine(nn.Module):
    def __init__(
        self,
        n_factors: int,
        context_input_dim: int,
        item_input_dim: int,
        user_input_dim: int,
        order: int = 1,
        weight_init: Callable = lecun_normal_init,
        deep: bool = False,
        hidden_layers: List[int] = [32],
    ):
        super(DeepFactorizationMachine, self).__init__()
        input_dnn = 0

        if context_input_dim > 0:
            self.context = nn.Linear(context_input_dim, n_factors)
            input_dnn += n_factors
        if item_input_dim > 0:
            self.item = nn.Linear(item_input_dim, n_factors)
            input_dnn += n_factors
        if user_input_dim > 0:
            self.user = nn.Linear(user_input_dim, n_factors)
            input_dnn += n_factors

        self.linear = nn.Linear(context_input_dim + item_input_dim + user_input_dim, 1)
        self.weight_init = weight_init
        self.deep = deep
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(input_dnn if i == 0 else hidden_layers[i - 1], layer_size)
                for i, layer_size in enumerate(hidden_layers)
            ]
        )
        self.deep_linear_out = nn.Linear(hidden_layers[-1], 1)
        self.order = order
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def none_tensor(self, x):
        return type(x) == type(None)

    def predict(self, item_representation, user_representation, context_representation):
        latent_vectors = []

        if not self.none_tensor(context_representation):
            latent_vectors.append(self.context(context_representation))
        if not self.none_tensor(item_representation):
            latent_vectors.append(self.item(item_representation))
        if not self.none_tensor(user_representation):
            latent_vectors.append(self.user(user_representation))

        x: torch.Tensor = None

        # 1st order interactions
        for v in [item_representation, user_representation, context_representation]:
            x = v if self.none_tensor(x) else torch.cat((x, v), dim=1)

        x = self.linear(x)

        # higher order interactions
        for k in range(2, self.order + 1):
            for a, b in combinations(latent_vectors, k):
                dot = torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).squeeze(1)
                x = dot if self.none_tensor(x) else torch.cat((x, dot), dim=1)

        # deep model
        if self.deep:
            v = torch.cat(latent_vectors, dim=1)
            for layer in self.hidden_layers:
                v = F.relu(layer(v))
            v = self.deep_linear_out(v)
            x = torch.cat((x, v), dim=1)

        x = torch.sum(x, dim=1)

        return torch.sigmoid(x)

