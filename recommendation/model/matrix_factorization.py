from typing import Callable, List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from recommendation.model.embedding import UserAndItemEmbedding
from recommendation.utils import lecun_normal_init


class MatrixFactorization(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int, binary: bool,
                 weight_init: Callable = lecun_normal_init):
        super().__init__(n_users, n_items, n_factors, weight_init)
        self.binary = binary

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        output = (self.user_embeddings(user_ids) * self.item_embeddings(item_ids)).sum(1)
        if self.binary:
            return torch.sigmoid(output)
        return output


class BiasedMatrixFactorization(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int, binary: bool,
                 weight_init: Callable = lecun_normal_init):
        super().__init__(n_users, n_items, n_factors, weight_init)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self.binary = binary

        self.user_bias.weight.data.zero_()
        self.item_bias.weight.data.zero_()

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_bias = self.user_bias(user_ids).squeeze(1)
        item_bias = self.item_bias(item_ids).squeeze(1)

        dot = (user_embedding * item_embedding).sum(1)

        output = dot + user_bias + item_bias
        if self.binary:
            return torch.sigmoid(output)
        return output


class DeepMatrixFactorization(UserAndItemEmbedding):

    def __init__(self, n_users: int, n_items: int, n_factors: int,
                 dense_layers: List[int], dropout_between_layers_prob: float,
                 bn_between_layers: bool, activation_function: Callable = F.selu,
                 weight_init: Callable = lecun_normal_init,
                 dropout_module: Type[Union[nn.Dropout, nn.AlphaDropout]] = nn.AlphaDropout):
        super().__init__(n_users, n_items, n_factors, weight_init)
        self.dense_layers = nn.ModuleList(
            [nn.Linear(
                2 * n_factors if i == 0 else dense_layers[i - 1],
                layer_size
            ) for i, layer_size in enumerate(dense_layers)])
        self.last_dense_layer = nn.Linear(dense_layers[-1], 1)
        if dropout_between_layers_prob:
            self.dropout: nn.Module = dropout_module(dropout_between_layers_prob)

        self.bn_between_layers = bn_between_layers
        self.activation_function = activation_function

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        x = torch.cat((user_embedding, item_embedding), 1)
        for layer in self.dense_layers:
            x = layer(x)
            if self.bn_between_layers:
                x = F.batch_norm(x)
            x = self.activation_function(x)
            if hasattr(self, "dropout"):
                x = self.dropout(x)

        x = self.last_dense_layer(x)

        return x
