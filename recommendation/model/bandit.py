import abc
from typing import List, Union, Tuple, Dict, Type

import math
import numpy as np
import torch
import torch.nn as nn
from numpy.random.mtrand import RandomState
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import collections

from recommendation.utils import chunks


class BanditPolicy(object, metaclass=abc.ABCMeta):
    def __init__(self, reward_model: nn.Module) -> None:
        self.reward_model = reward_model
        self._limit = None

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        pass

    @abc.abstractmethod
    def _select_idx(self, arm_ids: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:
        pass

    @abc.abstractmethod
    def _compute_prob(self, arm_scores: List[float]) -> List[float]:
        pass

    def _calculate_scores(self, arm_contexts: Tuple[np.ndarray, ...]) -> List[float]:
        inputs: torch.Tensor = default_convert(arm_contexts)
        scores: torch.Tensor = self.reward_model(*inputs)
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
        self._limit = limit
        ranked_arms = []
        arm_indices = list(arm_indices)
        arm_scores  = list(arm_scores)

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
        super().__init__(None)
        self._rng = RandomState(seed)

    def _compute_prob(self, arm_scores):
        n_arms = len(arm_scores)
        arms_probs = np.ones(n_arms) / n_arms
        return arms_probs

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)

        arm_probas = np.ones(n_arms) / n_arms

        action = self._rng.choice(n_arms, p=arm_probas)

        return action

class FixedPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, arg: int = 0, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._arg = arg

    def _compute_prob(self, arm_scores):
        n_arms     = len(arm_scores)
        arms_probs = np.zeros(n_arms)
        arms_probs[self._arg] = 1.0
        return arms_probs

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        return int(self._arg)

class ModelPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._rng = RandomState(seed)

    def _compute_prob(self, arm_scores):
        n_arms = len(arm_scores)
        arms_probs = np.zeros(n_arms)
        argmax = int(np.argmax(arm_scores))
        arms_probs[argmax] = 1.0
        return arms_probs

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        action = int(np.argmax(arm_scores))

        return action

class ExploreThenExploit(BanditPolicy):
    #TODO: Tune breakpoint parameter
    def __init__(self, reward_model: nn.Module, breakpoint_explore: int = 50000, breakpoint_exploit: int = 0, 
                exploration_decay: int = 10000, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._breakpoint_explore = breakpoint_explore
        self._breakpoint_exploit = breakpoint_exploit
        self._exploration_decay = exploration_decay
        self._rng = RandomState(seed)
        self._t = 0
        self.exploring = True

    def _update_state(self):
        if self.exploring and self._t > self._breakpoint_explore:
                self._t = 0
                self.exploring = False
                if self._breakpoint_explore >= 0:
                    self._breakpoint_explore -= self._exploration_decay
        elif not self.exploring and self._t > self._breakpoint_exploit:
                self._t = 0
                self.exploring = True
                if self._breakpoint_explore >= 0:
                    self._breakpoint_exploit += self._exploration_decay

    def _compute_prob(self, arm_scores):
        n_arms = len(arm_scores)
        arm_probs = np.zeros(len(arm_scores))
        max_score = max(arm_scores)
        argmax = int(np.argmax(arm_scores))
        
        if self.exploring:
            arm_probs = np.ones(n_arms) / n_arms
        else: 
            arm_probs[argmax] = 1.0

        return arm_probs

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms
        max_score = max(arm_scores)

        if pos == 0:
            self._t += 1
            self._update_state()

        if self.exploring:
            action = self._rng.choice(len(arm_indices), p=arm_probas)
        else:
            action = int(np.argmax(arm_scores))

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

class AdaptiveGreedy(BanditPolicy):
    #TODO: Tune these parameters: exploration_threshold, decay_rate
    def __init__(self, reward_model: nn.Module, exploration_threshold: float = 0.7, decay_rate: float = 0.999997,
         seed: int = 42) -> None:
        super().__init__(reward_model)
        self._exploration_threshold = exploration_threshold
        self._decay_rate = decay_rate
        self._rng = RandomState(seed)

    def _compute_prob(self, arm_scores):
        n_arms = len(arm_scores)
        arm_probs = np.zeros(len(arm_scores))
        max_score = max(arm_scores)
        argmax = int(np.argmax(arm_scores))

        if max_score > self._exploration_threshold:
            arm_probs[argmax] = 1.0
        else:
            arm_probs = np.ones(n_arms) / n_arms

        return arm_probs

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms
        max_score = max(arm_scores)

        if max_score > self._exploration_threshold:
            action = int(np.argmax(arm_scores))
        else:
            action = self._rng.choice(len(arm_indices), p=arm_probas)

        if pos == 0:
            self._exploration_threshold *= self._decay_rate

        return action


class PercentileAdaptiveGreedy(BanditPolicy):
    #TODO: Tune these parameters: window_size, exploration_threshold, percentile, percentile_decay
    def __init__(self, reward_model: nn.Module, window_size: int = 500, exploration_threshold: float = 0.7, percentile = 35, percentile_decay: float = 1.0,
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
        n_arms = len(arm_scores)
        max_score = max(arm_scores)

        exploration_threshold = np.percentile(self._best_arm_history[0], self._percentile) \
            if self._t >= self._window_size else self._initial_exploration_threshold

        arm_probs = np.zeros(len(arm_scores))
        argmax = int(np.argmax(arm_scores))

        if max_score > exploration_threshold:
            arm_probs[argmax] = 1.0
        else:
            arm_probs = np.ones(n_arms) / n_arms

        return arm_probs

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        if pos not in self._best_arm_history:
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
            if len(self._best_arm_history[pos]) >= self._window_size else self._initial_exploration_threshold

        if max_score > exploration_threshold:
            action = int(np.argmax(arm_scores))
        else:
            action = self._rng.choice(len(arm_indices), p=arm_probas)

        if len(self._best_arm_history[pos]) >= self._window_size:
            #update history
            self._best_arm_history[pos].append(max_score)
            self._best_arm_history[pos].popleft()

            #We must adapt the percentile only in the beginning of each evaluation:
            if pos == 0:
                self._percentile *= self._percentile_decay
        else:
            self._best_arm_history[pos].append(max_score)

        return action


class _LinBanditPolicy(BanditPolicy, metaclass=abc.ABCMeta):

    def __init__(self, reward_model: nn.Module, arm_index: int = 1, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._arm_index = arm_index
        self._Ainv_per_arm: Dict[int, np.ndarray] = {}

    def _sherman_morrison_update(self, Ainv: np.ndarray, x: np.ndarray) -> None:
        ## x should have shape (n, 1)
        Ainv -= np.linalg.multi_dot([Ainv, x, x.T, Ainv]) / (1.0 + np.linalg.multi_dot([x.T, Ainv, x]))

    def _flatten_input_and_extract_arms(self, input_: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray]:
        flattened_input = np.concatenate([el.reshape(-1, 1) if len(el.shape) == 1 else el for el in input_], axis=1)
        #print(flattened_input)
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

    def _compute_prob(self, arm_scores):
        #In this case, we expected arm_scores to be arms_scores_with_cb
        n_arms = len(arm_scores)
        arms_probs = np.zeros(n_arms)
        argmax = int(np.argmax(arm_scores))
        arms_probs[argmax] = 1.0
        return arms_probs

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:
        X, arms    = self._flatten_input_and_extract_arms(arm_contexts)
        arm_scores = [self._calculate_score(arm_score, x, arm)
                              for x, arm, arm_score in zip(X, arms, arm_scores)]

        action = int(np.argmax(arm_scores))

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
            return ranked_arms, self._compute_prob(arm_scores)
        else:
            return ranked_arms


class LinUCB(_LinBanditPolicy):
    def __init__(self, reward_model: nn.Module, alpha: float = 1e-5, arm_index: int = 1, seed: int = 42) -> None:
        super().__init__(reward_model, arm_index)
        self._alpha = alpha

    def _calculate_score(self, original_score: float, x: np.ndarray, arm: int) -> float:
        Ainv = self._Ainv_per_arm.get(arm)
        if Ainv is None:
            Ainv = np.eye(x.shape[0])
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
        Ainv = self._Ainv_per_arm.get(arm)
        if Ainv is None:
            Ainv = np.eye(x.shape[0])

        mu = np.random.multivariate_normal(original_score, self._v_sq * Ainv)
        return x.dot(mu)

class SoftmaxExplorer(BanditPolicy):
    def __init__(self, reward_model: nn.Module, logit_multiplier: float = 1.0, reverse_sigmoid: bool = True, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._logit_multiplier = logit_multiplier
        self._rng = RandomState(seed)
        self._reverse_sigmoid = reverse_sigmoid

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _compute_prob(self, arm_scores):
        n_arms = len(arm_scores)
        arm_scores = np.array(arm_scores)
        if self._reverse_sigmoid:
            arm_scores = np.log(arm_scores/((1 - arm_scores) + 1e-8))

        
        arms_probs = self._softmax(self._logit_multiplier * arm_scores)

        return arms_probs


    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probs = self._compute_prob(arm_scores)

        return self._rng.choice(a=len(arm_scores), p=arm_probs)


BANDIT_POLICIES: Dict[str, Type[BanditPolicy]] = dict(
    epsilon_greedy=EpsilonGreedy, lin_ucb=LinUCB, lin_ts=LinThompsonSampling, random=RandomPolicy,
    percentile_adaptive=PercentileAdaptiveGreedy, adaptive=AdaptiveGreedy, model=ModelPolicy,
    softmax_explorer = SoftmaxExplorer, explore_then_exploit=ExploreThenExploit, fixed=FixedPolicy, none=None)
