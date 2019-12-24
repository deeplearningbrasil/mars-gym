import abc
from typing import List, Any, Union

import numpy as np
import torch.nn as nn
from numpy.random.mtrand import RandomState


class BanditPolicy(object, metaclass=abc.ABCMeta):
    def __init__(self, reward_model: nn.Module) -> None:
        self._reward_model = reward_model

    @abc.abstractmethod
    def _select_idx(self, arm_ids: List[Union[str, int]], arm_scores: List[float]) -> int:
        pass

    def select_idx(self, arm_ids: List[Union[str, int]], arm_contexts: List[Any] = None,
                   arm_scores: List[float] = None) -> int:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = [self._reward_model(*arm_context) for arm_context in arm_contexts]
        return self._select_idx(arm_ids, arm_scores)

    def select(self, arm_ids: List[Union[str, int]], arm_contexts: List[Any] = None,
               arm_scores: List[float] = None) -> Union[str, int]:
        return arm_ids[self.select_idx(arm_ids, arm_contexts, arm_scores)]

    def rank(self, arm_ids: List[Union[str, int]], arm_contexts: List[Any] = None,
                   arm_scores: List[float] = None) -> List[Union[str, int]]:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = [self._reward_model(*arm_context) for arm_context in arm_contexts]
        assert len(arm_ids) == len(arm_scores)

        ranked_arms = []
        arm_ids = list(arm_ids)
        arm_scores = list(arm_scores)
        for _ in range(len(arm_ids)):
            idx = self.select_idx(arm_ids, arm_scores=arm_scores)
            ranked_arms.append(arm_ids[idx])
            arm_ids.pop(idx)
            arm_scores.pop(idx)

        return ranked_arms


class EpsilonGreedy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, epsilon: float, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._epsilon = epsilon
        self._rng = RandomState(seed)

    def _select_idx(self, arm_ids: List[Union[str, int]], arm_scores: List[float]) -> int:
        if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
            n_arms = len(arm_ids)
            arm_probas = np.ones(n_arms)
            arm_probas = arm_probas / np.sum(arm_probas)

            return self._rng.choice(n_arms, p=arm_probas)
        else:
            return int(np.argmax(arm_scores))
