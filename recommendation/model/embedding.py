from typing import Callable

import torch.nn as nn

from recommendation.utils import lecun_normal_init


class UserAndItemEmbedding(nn.Module):

    def __init__(self, n_users: int, n_items: int, n_factors: int, weight_init: Callable = lecun_normal_init):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)

        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)
