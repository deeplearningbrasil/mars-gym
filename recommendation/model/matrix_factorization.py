from typing import Callable

import torch
import torch.nn as nn

from recommendation.utils import lecun_normal_init


class MatrixFactorization(nn.Module):

    def __init__(self, n_users: int, n_items: int, n_factors: int = 20,
                 weight_init: Callable = lecun_normal_init):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)

        self.weight_init = weight_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Embedding:
            self.weight_init(module.weight)

    def forward(self, user_item_tuple: torch.Tensor) -> torch.Tensor:
        user: torch.Tensor = user_item_tuple[:, 0].to(torch.int64)
        item: torch.Tensor = user_item_tuple[:, 1].to(torch.int64)

        return (self.user_factors(user) * self.item_factors(item)).sum(1)


class BiasedMatrixFactorization(nn.Module):

    def __init__(self, n_users: int, n_items: int, n_factors: int = 20,
                 weight_init: Callable = lecun_normal_init):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)

        self.weight_init = weight_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Embedding:
            self.weight_init(module.weight)

    def forward(self, user_item_tuple: torch.Tensor) -> torch.Tensor:
        user: torch.Tensor = user_item_tuple[:, 0].to(torch.int64)
        item: torch.Tensor = user_item_tuple[:, 1].to(torch.int64)

        pred: torch.Tensor = self.user_biases(user)[:, 0] + self.item_biases(item)[:, 0]
        pred += (self.user_factors(user) * self.item_factors(item)).sum(1)
        return pred
