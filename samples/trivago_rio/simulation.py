from multiprocessing import Pool
from typing import Tuple, Callable, Union, Type, List, Dict, Any
import os
import json

import luigi
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchbearer import Trial
from mars_gym.meta_config import ProjectConfig, IOType, Column

from mars_gym.model.abstract import RecommenderModule


class SimpleLinearModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        metadata_size: int,
        window_hist_size: int,
    ):
        super().__init__(project_config, index_mapping)

        self.user_embeddings = nn.Embedding(self._n_users, n_factors)
        self.item_embeddings = nn.Embedding(self._n_items, n_factors)

        num_dense = n_factors + window_hist_size + 1 + metadata_size

        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense / 2)),
            nn.SELU(),
            nn.Linear(int(num_dense / 2), 1),
        )

    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

    def item_dot_history(self, itemA, itemB):
        dot = torch.matmul(
            self.normalize(itemA.unsqueeze(1)), self.normalize(itemB.permute(0, 2, 1))
        )
        return self.flatten(dot)

    def forward(
        self, user_ids, item_ids, pos_item_id, list_reference_item, list_metadata,
    ):
        # Geral embs
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Item embs
        interaction_item_emb = self.item_embeddings(list_reference_item)

        # Dot Item X History
        item_dot_interaction_item_emb = self.item_dot_history(
            item_emb, interaction_item_emb
        )

        x = torch.cat(
            (
                item_emb,
                item_dot_interaction_item_emb,
                pos_item_id.float().unsqueeze(1),
                list_metadata.float(),
            ),
            dim=1,
        )

        x = self.dense(x)
        out = torch.sigmoid(x)
        return out
