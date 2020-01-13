import abc
from typing import List, Union, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from numpy.random.mtrand import RandomState
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import collections 


class BanditPolicy(object, metaclass=abc.ABCMeta):
    def __init__(self, reward_model: nn.Module) -> None:
        self._reward_model = reward_model

    def fit(self, dataset: Dataset) -> None:
        pass

    @abc.abstractmethod
    def _select_idx(self, arm_ids: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:
        pass
    
    @abc.abstractmethod
    def _compute_prob(self, arm_scores: List[float]) -> List[float]:
        raise NotImplementedError

    def _calculate_scores(self, arm_contexts: Tuple[np.ndarray, ...]) -> List[float]:
        inputs: torch.Tensor = default_convert(arm_contexts)
        scores: torch.Tensor = self._reward_model(*inputs)
        return scores.detach().cpu().numpy().tolist()

    def select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = None) -> Union[int, Tuple[int, float]]:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = self._calculate_scores(arm_contexts)
        return self._select_idx(arm_indices, arm_contexts, arm_scores, pos)

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
        arm_indices = list(arm_indices)
        arm_scores = list(arm_scores)

        if with_probs:
            prob_ranked_arms = []
            arm_probs = list(self._compute_prob(arm_scores))

        n = len(arm_indices) if limit is None else min(len(arm_indices), limit)
        for i in range(n):
            idx = self.select_idx(arm_indices, arm_scores=arm_scores, pos=i)

            ranked_arms.append(arm_indices[idx])

            if with_probs:
                prob_ranked_arms.append(arm_probs[idx])
                arm_probs.pop(idx)

            arm_indices.pop(idx)
            arm_scores.pop(idx)
            

        if with_probs:
            return ranked_arms, prob_ranked_arms
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
    def __init__(self, reward_model: nn.Module, epsilon: float = 0.05, epsilon_decay: float = 1.0, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._epsilon = epsilon
        self._rng = RandomState(seed)
        self._epsilon_decay = epsilon_decay

    def _compute_prob(self, arm_scores):
        n_arms = len(arm_scores)
        arms_probs = self._epsilon * np.ones(n_arms) / n_arms

        argmax = int(np.argmax(arm_scores))

        arms_probs[argmax] += (1 - self._epsilon)

        return arms_probs


    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms

        if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
            action = self._rng.choice(len(arm_indices), p=arm_probas)
        else:
            action = int(np.argmax(arm_scores))

        #We must adapt the epsilon rate only in the beginning of each evaluation:
        if pos == 0:
            self._epsilon *= self._epsilon_decay

        return action

class PercentileAdaptiveGreedy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, window_size: int = 500, exploration_threshold: float = 0.9, percentile = 35, percentile_decay: float = 0.9998,
         seed: int = 42) -> None:
        super().__init__(reward_model)
        self._window_size = window_size
        self._initial_exploration_threshold = exploration_threshold
        self._percentile_decay = percentile_decay
        self._best_arm_history = {} # We save a deque for each pos
        self._rng = RandomState(seed)
        self._percentile = percentile
        self._t = 0
        self._first_evaluation = True

    def _compute_prob(self, arm_scores):
        max_score = max(arm_scores)

        exploration_threshold = np.percentile(self._best_arm_history[pos], self._percentile) \
            if self._t >= self._window_size else self._initial_exploration_threshold
        
        arm_probs = np.zeros(len(arm_scores))
        argmax = int(np.argmax(arm_scores))
        
        if max_score > exploration_threshold:
            arm_probs[argmax] = 1.0
        else:   
            arm_probs = exploration_threshold * np.ones(n_arms) / n_arms

        return arm_probs

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int, with_prob: bool) -> Union[int, Tuple[int, float]]:

        if self._t == 0:
            self._best_arm_history[pos] = collections.deque([])

        if pos == 0:
            if not self._first_evaluation:
                self._t += 1
            else:
                #As first evaluation, we do not need do update t
                self._first_evaluation = False

        n_arms = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms

        max_score = max(arm_scores)
    
        exploration_threshold = np.percentile(self._best_arm_history[pos], self._percentile) \
            if self._t >= self._window_size else self._initial_exploration_threshold

        if max_score > exploration_threshold:
            action = int(np.argmax(arm_scores))
        else:
            action = self._rng.choice(len(arm_indices), p=arm_probas)

        if self._t >= self._window_size:
            #update history
            self._best_arm_history[pos].append(max_score)
            self._best_arm_history[pos].popleft()
            
            #We must adapt the percentile only in the beginning of each evaluation:
            if pos == 0:
                self._percentile *= self._percentile_decay
        else:
            self._best_arm_history[pos].append(max_score)
            
        return action


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
                    arm_scores: List[float], pos: int, with_prob: bool) -> Union[int, Tuple[int, float]]:
        arm_contexts: List[Tuple[np.ndarray, ...]] = [tuple(el[i] for el in arm_contexts)
                                                      for i in range(len(arm_indices))]
        arm_scores_with_cb = [arm_score + self._calculate_confidence_bound(arm_context)
                              for arm_context, arm_score in zip(arm_contexts, arm_scores)]

        action = int(np.argmax(arm_scores_with_cb))

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

        arm_contexts: List[Tuple[np.ndarray, ...]] = [tuple(el[i] for el in arm_contexts)
                                                      for i in range(len(arm_indices))]
        arm_scores_with_cb = [arm_score + self._calculate_confidence_bound(arm_context)
                              for arm_context, arm_score in zip(arm_contexts, arm_scores)]

        ranked_arms = [arm_id for _, arm_id in sorted(zip(arm_scores_with_cb, arm_indices), reverse=True)]
        if limit is not None:
            ranked_arms = ranked_arms[:limit]

        if with_probs:
            return ranked_arms, [1.0 if i == 0 else 0.0 for i in range(len(ranked_arms))]
        else:
            return ranked_arms
