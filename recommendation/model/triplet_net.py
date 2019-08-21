from typing import Tuple, Callable, Union

import torch

from recommendation.model.matrix_factorization import MatrixFactorization
from recommendation.utils import lecun_normal_init


class TripletNet(MatrixFactorization):

    def __init__(self, n_users: int, n_items: int, n_factors: int,
                 weight_init: Callable = lecun_normal_init):
        super().__init__(n_users, n_items, n_factors, True, weight_init)

    def forward(self, user_ids: torch.Tensor, positive_item_ids: torch.Tensor,
                negative_item_ids: torch.Tensor = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if negative_item_ids is None:
            return super().forward(user_ids, positive_item_ids)
        return self.user_embeddings(user_ids), self.item_embeddings(positive_item_ids), \
               self.item_embeddings(negative_item_ids)
