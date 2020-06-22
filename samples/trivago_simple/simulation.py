import luigi
from typing import Dict, Any
import torch
import torch.nn as nn
import numpy as np
from numpy.random.mtrand import RandomState

from mars_gym.model.bandit import BanditPolicy
from typing import Dict, Any, List, Tuple, Union
from mars_gym.meta_config import ProjectConfig
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

        # user + item + flatten hist + position + metadata
        num_dense = 2 * n_factors + window_hist_size * n_factors + 1 + metadata_size

        self.dense = nn.Sequential(
            nn.Linear(num_dense, 500), nn.SELU(), nn.Linear(500, 1),
        )

    def flatten(self, input: torch.Tensor):
        return input.view(input.size(0), -1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        pos_item_id: torch.Tensor,
        list_reference_item: torch.Tensor,
        list_metadata: torch.Tensor,
    ):
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        history_items_emb = self.item_embeddings(list_reference_item)

        x = torch.cat(
            (
                user_emb,
                item_emb,
                self.flatten(history_items_emb),
                pos_item_id.float().unsqueeze(1),
                list_metadata.float(),
            ),
            dim=1,
        )

        x = self.dense(x)
        return torch.sigmoid(x)


class EGreedyPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, epsilon: float = 0.1, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._epsilon = epsilon
        self._rng = RandomState(seed)

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...] = None,
        arm_scores: List[float] = None,
        pos: int = 0,
    ) -> Union[int, Tuple[int, float]]:

        if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
            action = self._rng.choice(len(arm_indices))
        else:
            action = np.argmax(arm_scores)

        return action

