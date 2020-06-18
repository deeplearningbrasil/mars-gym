from typing import Tuple, Callable, Union, Type, List, Dict, Any
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mars_gym.torch.init import lecun_normal_init
from mars_gym.model.abstract import RecommenderModule
import numpy as np
from mars_gym.meta_config import ProjectConfig, IOType, Column


class LogisticRegression(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        weight_init: Callable = lecun_normal_init,
    ):
        super().__init__(project_config, index_mapping)

        self.user_embeddings = nn.Embedding(self._n_users, n_factors)
        self.item_embeddings = nn.Embedding(self._n_items, n_factors)

        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)

        self.linear = nn.Linear(n_factors * 2, 1)
        self.weight_init = weight_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def none_tensor(self, x):
        return type(x) == type(None)

    def forward(self, user_ids, item_ids, context_representation=None):
        x: torch.Tensor = context_representation

        # Geral embs
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        x = torch.cat((user_emb, item_emb), dim=1,)

        return torch.sigmoid(self.linear(x))
