import abc
from typing import Dict, Any

import torch.nn as nn

from mars_gym.meta_config import ProjectConfig


class RecommenderModule(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self, project_config: ProjectConfig, index_mapping: Dict[str, Dict[Any, int]],
    ) -> None:
        super().__init__()
        self._index_mapping = index_mapping
        self._project_config = project_config
        self._n_users = max(index_mapping[project_config.user_column.name].values()) + 1
        self._n_items = max(index_mapping[project_config.item_column.name].values()) + 1

    
    def recommendation_score(self, *args):
        return self.forward(*args)