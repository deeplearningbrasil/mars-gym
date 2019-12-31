import abc
from typing import List, Any, Union, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from numpy.random.mtrand import RandomState
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class BanditPolicy(object, metaclass=abc.ABCMeta):
    def __init__(self, reward_model: nn.Module) -> None:
        self._reward_model = reward_model

    def fit(self, dataset: Dataset) -> None:
        pass

    @abc.abstractmethod
    def _select_idx(self, arm_ids: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float]) -> int:
        pass

    def _calculate_scores(self, arm_contexts: Tuple[np.ndarray, ...]) -> List[float]:
        inputs: torch.Tensor = default_convert(arm_contexts)
        scores: torch.Tensor = self._reward_model(*inputs)
        return scores.detach().cpu().numpy().tolist()

    def select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None) -> int:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = self._calculate_scores(arm_contexts)
        return self._select_idx(arm_indices, arm_contexts, arm_scores)

    def select(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
               arm_scores: List[float] = None) -> Union[str, int]:
        return arm_indices[self.select_idx(arm_indices, arm_contexts, arm_scores)]

    def rank(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
             arm_scores: List[float] = None, limit: int = None) -> List[int]:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = self._calculate_scores(arm_contexts)
        assert len(arm_indices) == len(arm_scores)

        ranked_arms = []
        arm_indices = list(arm_indices)
        arm_scores = list(arm_scores)
        n = len(arm_indices) if limit is None else min(len(arm_indices), limit)
        for _ in range(n):
            idx = self.select_idx(arm_indices, arm_scores=arm_scores)
            ranked_arms.append(arm_indices[idx])
            arm_indices.pop(idx)
            arm_scores.pop(idx)

        return ranked_arms


class EpsilonGreedy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, epsilon: float, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._epsilon = epsilon
        self._rng = RandomState(seed)

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float]) -> int:
        if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
            n_arms = len(arm_indices)
            arm_probas = np.ones(n_arms)
            arm_probas = arm_probas / np.sum(arm_probas)

            return self._rng.choice(n_arms, p=arm_probas)
        else:
            return int(np.argmax(arm_scores))


class LinUCB(BanditPolicy):
    def __init__(self, reward_model: nn.Module, alpha: float = 0.5, arm_index: int = 1) -> None:
        super().__init__(reward_model)
        self._alpha = alpha
        self._arm_index = arm_index
        self._Ainv_per_arm: Dict[int, np.ndarray] = {}

    def _sherman_morrison_update(self, Ainv: np.ndarray, x: np.ndarray) -> None:
        ## x should have shape (n, 1)
        Ainv -= np.linalg.multi_dot([Ainv, x, x.T, Ainv]) / (1.0 + np.linalg.multi_dot([x.T, Ainv, x]))

    def _flatten_input_and_extract_arm(self, input_: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, int]:
        flattened_input = np.concatenate([el.reshape(1, -1) for el in input_], axis=1)[0]

        return np.delete(flattened_input, self._arm_index), int(flattened_input[self._arm_index])

    def fit(self, dataset: Dataset) -> None:
        n = len(dataset)

        for i in tqdm(range(n), total=n):
            input_: Tuple[np.ndarray, ...] = dataset[i][0]
            x, arm = self._flatten_input_and_extract_arm(input_)

            if arm not in self._Ainv_per_arm:
                self._Ainv_per_arm[arm] = np.eye(x.shape[0])

            x = x.reshape((-1, 1))
            self._sherman_morrison_update(self._Ainv_per_arm[arm], x)

    def _calculate_confidence_bound(self, arm_context: Tuple[np.ndarray]):
        x, arm = self._flatten_input_and_extract_arm(arm_context)
        Ainv = self._Ainv_per_arm.get(arm) or np.eye(x.shape[0])
        return self._alpha * np.sqrt(np.linalg.multi_dot([x.T, Ainv, x]))

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float]) -> int:
        arm_contexts: List[Tuple[np.ndarray, ...]] = [tuple(el[i] for el in arm_contexts)
                                                      for i in range(len(arm_indices))]
        arm_scores_with_cb = [arm_score + self._calculate_confidence_bound(arm_context)
                              for arm_context, arm_score in zip(arm_contexts, arm_scores)]
        return int(np.argmax(arm_scores_with_cb))

    def rank(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
             arm_scores: List[float] = None, limit: int = None) -> List[int]:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = self._calculate_scores(arm_contexts)
        assert len(arm_indices) == len(arm_scores)

        arm_contexts: List[Tuple[np.ndarray, ...]] = [tuple(el[i] for el in arm_contexts)
                                                      for i in range(len(arm_indices))]
        arm_scores_with_cb = [arm_score + self._calculate_confidence_bound(arm_context)
                              for arm_context, arm_score in zip(arm_contexts, arm_scores)]

        ranked_list = [arm_id for _, arm_id in sorted(zip(arm_scores_with_cb, arm_indices), reverse=True)]
        return ranked_list if limit is None else ranked_list[:limit]
