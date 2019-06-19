from typing import Callable

import torch
import torch.nn as nn

from recommendation.utils import lecun_normal_init


class MatrixFactorization(nn.Module):

    def __init__(self, n_users: int, n_items: int, n_factors: int = 20,
                 weight_init: Callable = lecun_normal_init):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)

        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)

    def forward(self, user_item_tuple: torch.Tensor) -> torch.Tensor:
        user_ids: torch.Tensor = user_item_tuple[:, 0].to(torch.int64)
        item_ids: torch.Tensor = user_item_tuple[:, 1].to(torch.int64)

        return (self.user_factors(user_ids) * self.item_factors(item_ids)).sum(1)


class BiasedMatrixFactorization(nn.Module):

    def __init__(self, n_users: int, n_items: int, n_factors: int = 20,
                 weight_init: Callable = lecun_normal_init):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)

        self.user_bias.weight.data.zero_()
        self.item_bias.weight.data.zero_()

    def forward(self, user_item_tuple: torch.Tensor) -> torch.Tensor:
        user_ids: torch.Tensor = user_item_tuple[:, 0].to(torch.int64)
        item_ids: torch.Tensor = user_item_tuple[:, 1].to(torch.int64)

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_bias = self.user_bias(user_ids).squeeze(1)
        item_bias = self.item_bias(item_ids).squeeze(1)

        dot = (user_embedding * item_embedding).sum(1)

        return dot + user_bias + item_bias
