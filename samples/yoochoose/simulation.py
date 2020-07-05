from typing import Dict, Any, List, Tuple, Union
import os
import luigi
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchbearer import Trial
from mars_gym.meta_config import ProjectConfig
from mars_gym.model.abstract import RecommenderModule
from mars_gym.model.bandit import BanditPolicy
from numpy.random.mtrand import RandomState


class SimpleLinearModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
    ):
        super().__init__(project_config, index_mapping)

        self.user_embeddings = nn.Embedding(self._n_users, n_factors)
        self.item_embeddings = nn.Embedding(self._n_items, n_factors)
        self.dayofweek_embeddings = nn.Embedding(7, n_factors)

        num_dense = n_factors * (2 + 5) + 1

        self.dense = nn.Sequential(
            nn.Linear(num_dense, int(num_dense / 2)),
            nn.SELU(),
            nn.Linear(int(num_dense / 2), 1),
        )

    def flatten(self, input):
        return input.view(input.size(0), -1)

    def forward(self, session_ids, item_ids, dayofweek, step, item_history_ids):
        # Item emb
        item_emb = self.item_embeddings(item_ids)

        # Item History embs
        interaction_item_emb = self.flatten(self.item_embeddings(item_history_ids))

        # DayofWeek Emb
        dayofweek_emb = self.dayofweek_embeddings(dayofweek.long())

        x = torch.cat(
            (item_emb, interaction_item_emb, dayofweek_emb, step.float().unsqueeze(1),),
            dim=1,
        )

        out = torch.sigmoid(self.dense(x))
        return out


class RandomPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, seed: int = 42) -> None:
        super().__init__(None)
        self._rng = RandomState(seed)

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...] = None,
        arm_scores: List[float] = None,
        pos: int = 0,
    ) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)

        arm_probas = np.ones(n_arms) / n_arms

        action = self._rng.choice(n_arms, p=arm_probas)

        return action
