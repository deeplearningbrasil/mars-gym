import abc
from typing import List, Union, Tuple, Dict

import math
import numpy as np
import torch
import torch.nn as nn
from numpy.random.mtrand import RandomState
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from recommendation.utils import chunks


class BanditPolicy(object, metaclass=abc.ABCMeta):
    def __init__(self, reward_model: nn.Module) -> None:
        self._reward_model = reward_model

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        pass

    @abc.abstractmethod
    def _select_idx(self, arm_ids: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int, with_prob: bool) -> Union[int, Tuple[int, float]]:
        pass

    def _calculate_scores(self, arm_contexts: Tuple[np.ndarray, ...]) -> List[float]:
        inputs: torch.Tensor = default_convert(arm_contexts)
        scores: torch.Tensor = self._reward_model(*inputs)
        return scores.detach().cpu().numpy().tolist()

    def select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = None,
                   with_prob: bool = False) -> Union[int, Tuple[int, float]]:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = self._calculate_scores(arm_contexts)
        return self._select_idx(arm_indices, arm_contexts, arm_scores, pos, with_prob)

    def select(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
               arm_scores: List[float] = None) -> int:
        return arm_indices[self.select_idx(arm_indices, arm_contexts, arm_scores)]

    def rank(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
             arm_scores: List[float] = None, with_probs: bool = False,
             limit: int = None) -> Union[List[int], Tuple[List[int], List[float]]]:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = self._calculate_scores(arm_contexts)
        assert len(arm_indices) == len(arm_scores)

        ranked_arms = []
        if with_probs:
            prob_arms = []
        arm_indices = list(arm_indices)
        arm_scores = list(arm_scores)
        n = len(arm_indices) if limit is None else min(len(arm_indices), limit)
        for i in range(n):
            idx = self.select_idx(arm_indices, arm_scores=arm_scores, pos=i, with_prob=with_probs)

            if with_probs:
                idx, prob = idx  # type: int, float
                prob_arms.append(prob)

            ranked_arms.append(arm_indices[idx])

            arm_indices.pop(idx)
            arm_scores.pop(idx)

        if with_probs:
            return ranked_arms, prob_arms
        else:
            return ranked_arms

class RandomPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._rng = RandomState(seed)

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int, with_prob: bool) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms

        action = self._rng.choice(n_arms, p=arm_probas)

        if with_prob:
            return action, arm_probas[action]
        else:
            return action

class ModelPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._rng = RandomState(seed)

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int, with_prob: bool) -> Union[int, Tuple[int, float]]:

        action = int(np.argmax(arm_scores))

        if with_prob:
            return action, int(pos == 0)
        else:
            return action


class EpsilonGreedy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, epsilon: float = 0.05, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._epsilon = epsilon
        self._rng = RandomState(seed)

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int, with_prob: bool) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        total_arms = (n_arms + pos)
        arm_probas = np.ones(n_arms) / n_arms

        if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
            action = self._rng.choice(len(arm_indices), p=arm_probas)
            prob = self._epsilon * arm_probas[action]
        else:
            action = int(np.argmax(arm_scores))
            prob = (1.0 - self._epsilon) + (self._epsilon * arm_probas[action])

        # If different from hit@1 use exploration probability
        if pos > 0:
            prob = self._epsilon * (1 / total_arms)

        if with_prob:
            return action, prob
        else:
            return action


class _LinBanditPolicy(BanditPolicy, metaclass=abc.ABCMeta):

    def __init__(self, reward_model: nn.Module, arm_index: int = 1) -> None:
        super().__init__(reward_model)
        self._arm_index = arm_index
        self._Ainv_per_arm: Dict[int, np.ndarray] = {}

    def _sherman_morrison_update(self, Ainv: np.ndarray, x: np.ndarray) -> None:
        ## x should have shape (n, 1)
        Ainv -= np.linalg.multi_dot([Ainv, x, x.T, Ainv]) / (1.0 + np.linalg.multi_dot([x.T, Ainv, x]))

    def _flatten_input_and_extract_arms(self, input_: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray]:
        flattened_input = np.concatenate([el.reshape(-1, 1) if len(el.shape) == 1 else el for el in input_], axis=1)

        return np.delete(flattened_input, self._arm_index, axis=1), flattened_input[:, self._arm_index]

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        n = len(dataset)

        for indices in tqdm(chunks(range(n), batch_size), total=math.ceil(n / batch_size)):
            input_: Tuple[np.ndarray, ...] = dataset[indices][0]
            X, arms = self._flatten_input_and_extract_arms(input_)

            for x, arm in zip(X, arms):
                if arm not in self._Ainv_per_arm:
                    self._Ainv_per_arm[arm] = np.eye(x.shape[0])

                x = x.reshape((-1, 1))
                self._sherman_morrison_update(self._Ainv_per_arm[arm], x)

    @abc.abstractmethod
    def _calculate_score(self, original_score: float, x: np.ndarray, arm: int) -> float:
        pass

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int, with_prob: bool) -> Union[int, Tuple[int, float]]:
        X, arms = self._flatten_input_and_extract_arms(arm_contexts)
        arm_scores = [self._calculate_score(arm_score, x, arm)
                              for x, arm, arm_score in zip(X, arms, arm_scores)]

        action = int(np.argmax(arm_scores))

        if with_prob:
            return action, int(pos == 0)
        else:
            return action

    def rank(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
             arm_scores: List[float] = None, with_probs: bool = False,
             limit: int = None) -> Union[List[int], Tuple[List[int], List[float]]]:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = self._calculate_scores(arm_contexts)
        assert len(arm_indices) == len(arm_scores)

        X, arms = self._flatten_input_and_extract_arms(arm_contexts)
        arm_scores = [self._calculate_score(arm_score, x, arm)
                              for x, arm, arm_score in zip(X, arms, arm_scores)]

        ranked_arms = [arm_id for _, arm_id in sorted(zip(arm_scores, arm_indices), reverse=True)]
        if limit is not None:
            ranked_arms = ranked_arms[:limit]

        if with_probs:
            return ranked_arms, [1.0 if i == 0 else 0.0 for i in range(len(ranked_arms))]
        else:
            return ranked_arms


class LinUCB(_LinBanditPolicy):
    def __init__(self, reward_model: nn.Module, alpha: float = 0.5, arm_index: int = 1) -> None:
        super().__init__(reward_model, arm_index)
        self._alpha = alpha

    def _calculate_score(self, original_score: float, x: np.ndarray, arm: int) -> float:
        Ainv = self._Ainv_per_arm.get(arm) or np.eye(x.shape[0])
        confidence_bound = self._alpha * np.sqrt(np.linalg.multi_dot([x.T, Ainv, x]))
        return original_score + confidence_bound


class LinThompsonSampling(_LinBanditPolicy):
    def __init__(self, reward_model: nn.Module, v_sq: float = 1.0, arm_index: int = 1) -> None:
        """
        :param v_sq: Parameter by which to multiply the covariance matrix (more means higher variance).
        """
        super().__init__(reward_model, arm_index)
        self._v_sq = v_sq

    def _calculate_score(self, original_score: float, x: np.ndarray, arm: int) -> float:
        Ainv = self._Ainv_per_arm.get(arm) or np.eye(x.shape[0])

        mu = np.random.multivariate_normal(original_score, self._v_sq * Ainv)
        return x.dot(mu)
