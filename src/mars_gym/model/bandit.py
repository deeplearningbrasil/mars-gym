import abc
import collections
from typing import List, Union, Tuple, Dict, Type, Optional, Any

import math
import numpy as np
import torch
import torch.nn as nn
from numpy.random.mtrand import RandomState
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from mars_gym.utils.utils import chunks


class BanditPolicy(object, metaclass=abc.ABCMeta):
    def __init__(self, reward_model: nn.Module) -> None:
        self.reward_model = reward_model
        self._limit = None

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        pass

    @abc.abstractmethod
    def _select_idx(
        self,
        arm_ids: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: List[float],
        pos: int,
    ) -> Union[int, Tuple[int, float]]:
        pass

    def _compute_prob(
        self, arm_indices: List[int], arm_scores: List[float]
    ) -> List[float]:
        n_arms = len(arm_indices)
        arms_probs = np.zeros(n_arms)

        arms_probs[0] = 1.0
        return arms_probs.tolist()

    def calculate_scores(
        self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...]
    ) -> List[float]:
        if self.reward_model:
            inputs: torch.Tensor = default_convert(arm_contexts)
            scores: torch.Tensor = self.reward_model(*inputs)
            return scores.detach().cpu().numpy().tolist()
        else:
            return list(np.zeros(len(arm_indices)))

    def select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...] = None,
        arm_scores: List[float] = None,
        pos: int = 0,
    ) -> Union[int, Tuple[int, float]]:
        assert arm_contexts is not None or arm_scores is not None

        if arm_scores is None:
            arm_scores = self.calculate_scores(arm_indices, arm_contexts)

        return self._select_idx(arm_indices, arm_contexts, arm_scores, pos)

    def select(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...] = None,
        arm_scores: List[float] = None,
    ) -> int:
        return arm_indices[self.select_idx(arm_indices, arm_contexts, arm_scores)]

    def rank(
        self,
        arms: List[Any],
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...] = None,
        arm_scores: List[float] = None,
        with_probs: bool = False,
        limit: int = None,
    ) -> Union[List[Any], Tuple[List[Any], List[float]]]:
        assert arm_contexts is not None or arm_scores is not None

        if arm_scores is None:
            arm_scores = self.calculate_scores(arm_indices, arm_contexts)

        assert len(arm_indices) == len(arm_scores)

        self._limit = limit
        ranked_arms = []
        arms = list(arms)
        arm_indices = list(arm_indices)
        arm_scores = list(arm_scores)

        if with_probs:
            prob_ranked_arms = []
            arm_probs = list(self._compute_prob(arm_indices, arm_scores))

        n = len(arm_indices) if limit is None else min(len(arm_indices), limit)
        for i in range(n):
            idx = self.select_idx(
                arm_indices, arm_contexts=arm_contexts, arm_scores=arm_scores, pos=i
            )
            ranked_arms.append(arms[idx])

            if with_probs:
                prob_ranked_arms.append(arm_probs[idx])
                arm_probs.pop(idx)

            arms.pop(idx)
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

    def calculate_scores(
        self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...]
    ) -> List[float]:
        return list(np.random.rand(len(arm_indices)))

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...] = None,
        arm_scores: List[float] = None,
        pos: int = 0,
    ) -> Union[int, Tuple[int, float]]:

        action = int(np.argmax(arm_scores))

        return action


class FixedPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, arg: int = 1, seed: int = 42) -> None:
        super().__init__(None)
        self._arg = arg
        self._arm_index = 1

    def calculate_scores(
        self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...]
    ) -> List[float]:
        X, arms = self._flatten_input_and_extract_arms(arm_contexts)
        arm_scores = [int(x[self._arg] == arm) for x, arm in zip(X, arms)]
        return arm_scores

    def _compute_prob(
        self, arm_indices: List[int], arm_scores: List[float]
    ) -> List[float]:
        n_arms = len(arm_scores)
        arms_probs = np.zeros(n_arms)
        argmax = int(np.argmax(arm_scores))
        arms_probs[argmax] = 1.0
        return arms_probs.tolist()

    def _flatten_input_and_extract_arms(
        self, input_: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        flattened_input = np.concatenate(
            [el.reshape(-1, 1) if len(el.shape) == 1 else el for el in input_], axis=1
        )
        return (
            np.delete(flattened_input, self._arm_index, axis=1),
            flattened_input[:, self._arm_index],
        )

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: List[float],
        pos: int,
    ) -> Union[int, Tuple[int, float]]:

        action = int(np.argmax(arm_scores))

        return action


class ModelPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._rng = RandomState(seed)

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: List[float],
        pos: int,
    ) -> Union[int, Tuple[int, float]]:

        action = int(np.argmax(arm_scores))

        return action


class ExploreThenExploit(BanditPolicy):
    # TODO: Tune breakpoint parameter
    def __init__(
        self,
        reward_model: nn.Module,
        explore_rounds: int = 500,
        decay_rate: float = 0.0026456,
        seed: int = 42,
    ) -> None:
        super().__init__(reward_model)
        self._init_explore_rounds = explore_rounds
        self._explore_rounds = explore_rounds
        self._exploit_rounds = explore_rounds
        self._decay_rate = decay_rate

        self._rng = RandomState(seed)
        self._t = 0
        self._te = 0
        self.exploring = True

    def _update_state(self):
        self._t += 1
        self._te += 1

        if self._explore_rounds > 1:
            if self.exploring and self._te > self._explore_rounds:
                self._te = 0
                self._explore_rounds = self.decay(
                    self._init_explore_rounds, self._decay_rate, self._t
                )
                self.exploring = False
            elif not self.exploring and self._te > self._exploit_rounds:
                self._te = 0
                # self._exploit_rounds += (self._init_explore_rounds-self._explore_rounds)
                self.exploring = True
        else:
            self.exploring = False

    def decay(self, init, decay_rate, t):
        return init * (1 - decay_rate) ** t

    def _compute_prob(
        self, arm_indices: List[int], arm_scores: List[float]
    ) -> List[float]:
        n_arms = len(arm_scores)
        arm_probs = np.zeros(len(arm_scores))
        max_score = max(arm_scores)
        argmax = int(np.argmax(arm_scores))

        if self.exploring:
            arm_probs = np.ones(n_arms) / n_arms
        else:
            arm_probs[argmax] = 1.0

        return arm_probs.tolist()

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: List[float],
        pos: int,
    ) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms
        max_score = max(arm_scores)

        if pos == 0:
            self._update_state()

        if self.exploring:
            action = self._rng.choice(len(arm_indices), p=arm_probas)
        else:
            action = int(np.argmax(arm_scores))

        return action


class EpsilonGreedy(BanditPolicy):
    def __init__(
        self,
        reward_model: nn.Module,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(reward_model)
        self._epsilon = epsilon
        self._rng = RandomState(seed)
        self._epsilon_decay = epsilon_decay

    def _compute_prob(
        self, arm_indices: List[int], arm_scores: List[float]
    ) -> List[float]:
        n_arms = len(arm_scores)
        arms_probs = self._epsilon * np.ones(n_arms) / n_arms

        argmax = int(np.argmax(arm_scores))

        arms_probs[argmax] += 1 - self._epsilon

        return arms_probs.tolist()

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: List[float],
        pos: int,
    ) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms

        if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
            action = self._rng.choice(len(arm_indices), p=arm_probas)
        else:
            action = int(np.argmax(arm_scores))

        # We must adapt the epsilon rate only in the beginning of each evaluation:
        if pos == 0:
            self._epsilon *= self._epsilon_decay

        return action


class AdaptiveGreedy(BanditPolicy):
    # TODO: Tune these parameters: exploration_threshold, decay_rate
    def __init__(
        self,
        reward_model: nn.Module,
        exploration_threshold: float = 0.8,
        decay_rate: float = 0.0010391,
        seed: int = 42,
    ) -> None:
        super().__init__(reward_model)
        self._init_exploration_threshold = exploration_threshold
        self._exploration_threshold = exploration_threshold
        self._decay_rate = decay_rate
        self._rng = RandomState(seed)
        self._t = 0

    def _compute_prob(
        self, arm_indices: List[int], arm_scores: List[float]
    ) -> List[float]:
        n_arms = len(arm_scores)
        arm_probs = np.zeros(len(arm_scores))
        max_score = max(arm_scores)
        argmax = int(np.argmax(arm_scores))

        if max_score > self._exploration_threshold:
            arm_probs[argmax] = 1.0
        else:
            arm_probs = np.ones(n_arms) / n_arms

        return arm_probs.tolist()

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: List[float],
        pos: int,
    ) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms
        max_score = max(arm_scores)

        if max_score > self._exploration_threshold:
            action = int(np.argmax(arm_scores))
        else:
            action = self._rng.choice(len(arm_indices), p=arm_probas)

        if pos == 0:
            self._t += 1
            self._exploration_threshold = self.decay(
                self._init_exploration_threshold, self._decay_rate, self._t
            )

        return action

    def decay(self, init, decay_rate, t):
        return init * (1 - decay_rate) ** t


class PercentileAdaptiveGreedy(BanditPolicy):
    # TODO: Tune these parameters: window_size, exploration_threshold, percentile, percentile_decay
    def __init__(
        self,
        reward_model: nn.Module,
        window_size: int = 500,
        exploration_threshold: float = 0.5,
        percentile=35,
        percentile_decay: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(reward_model)
        self._window_size = window_size
        self._initial_exploration_threshold = exploration_threshold
        self._percentile_decay = percentile_decay
        self._best_arm_history = {}  # We save a deque for each pos
        self._rng = RandomState(seed)
        self._percentile = percentile
        self._t = 0

    def _compute_prob(
        self, arm_indices: List[int], arm_scores: List[float]
    ) -> List[float]:
        n_arms = len(arm_scores)
        max_score = max(arm_scores)

        exploration_threshold = (
            np.percentile(self._best_arm_history[0], self._percentile)
            if self._t >= self._window_size
            else self._initial_exploration_threshold
        )

        arm_probs = np.zeros(len(arm_scores))
        argmax = int(np.argmax(arm_scores))

        if max_score >= exploration_threshold:
            arm_probs[argmax] = 1.0
        else:
            arm_probs = np.ones(n_arms) / n_arms

        return arm_probs.tolist()

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: List[float],
        pos: int,
    ) -> Union[int, Tuple[int, float]]:

        if pos not in self._best_arm_history:
            self._best_arm_history[pos] = collections.deque([])

        if pos == 0:
            self._t += 1

        n_arms = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms

        max_score = max(arm_scores)

        exploration_threshold = (
            np.percentile(self._best_arm_history[pos], self._percentile)
            if len(self._best_arm_history[pos]) >= self._window_size
            else self._initial_exploration_threshold
        )

        if max_score >= exploration_threshold:
            action = int(np.argmax(arm_scores))
        else:
            action = self._rng.choice(len(arm_indices), p=arm_probas)

        if len(self._best_arm_history[pos]) >= self._window_size:
            # update history
            self._best_arm_history[pos].append(max_score)
            self._best_arm_history[pos].popleft()

            # We must adapt the percentile only in the beginning of each evaluation:
            if pos == 0:
                self._percentile *= self._percentile_decay
        else:
            self._best_arm_history[pos].append(max_score)

        return action


class _LinBanditPolicy(BanditPolicy, metaclass=abc.ABCMeta):
    def __init__(
        self, reward_model: nn.Module, arm_index: int = 1, scaler=False, seed: int = 42
    ) -> None:
        super().__init__(reward_model)
        self._arm_index = arm_index
        self._Ainv_per_arm: Dict[int, np.ndarray] = {}
        if reward_model is None:
            self._b_per_arm: Dict[int, np.ndarray] = {}
        # self._scaler = StandardScaler()

    def _sherman_morrison_update(self, Ainv: np.ndarray, x: np.ndarray) -> None:
        ## x should have shape (n, 1)
        Ainv -= np.linalg.multi_dot([Ainv, x, x.T, Ainv]) / (
            1.0 + np.linalg.multi_dot([x.T, Ainv, x])
        )

    def _flatten_input_and_extract_arms(
        self, input_: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        flattened_input = np.concatenate(
            [el.reshape(-1, 1) if len(el.shape) == 1 else el for el in input_], axis=1
        )
        return (
            np.delete(flattened_input, self._arm_index, axis=1),
            flattened_input[:, self._arm_index],
        )

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        self._Ainv_per_arm.clear()
        if hasattr(self, "_b_per_arm"):
            self._b_per_arm.clear()

        n = len(dataset)

        for indices in tqdm(
            chunks(range(n), batch_size), total=math.ceil(n / batch_size)
        ):
            input_: Tuple[np.ndarray, ...] = dataset[indices][0]
            output_: Tuple[np.ndarray, ...] = dataset[indices][1]

            X, arms = self._flatten_input_and_extract_arms(input_)
            output = output_[0] if isinstance(output_, tuple) else output_

            for x, arm, y in zip(X, arms, output):
                if arm not in self._Ainv_per_arm:
                    self._Ainv_per_arm[arm] = np.eye(x.shape[0])
                if hasattr(self, "_b_per_arm"):
                    if arm not in self._b_per_arm:
                        self._b_per_arm[arm] = np.zeros((x.shape[0], 1))

                x = x.reshape((-1, 1))
                self._sherman_morrison_update(self._Ainv_per_arm[arm], x)
                if hasattr(self, "_b_per_arm"):
                    self._b_per_arm[arm] += x * y

    @abc.abstractmethod
    def _calculate_score(
        self, original_score: Optional[float], x: np.ndarray, arm: int
    ) -> float:
        pass

    def _compute_prob(
        self, arm_indices: List[int], arm_scores: List[float]
    ) -> List[float]:
        # In this case, we expected arm_scores to be arms_scores_with_cb
        n_arms = len(arm_scores)
        arms_probs = np.zeros(n_arms)
        argmax = int(np.argmax(arm_scores))
        arms_probs[argmax] = 1.0
        return arms_probs.tolist()

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: Optional[List[float]],
        pos: int,
    ) -> Union[int, Tuple[int, float]]:

        X, arms = self._flatten_input_and_extract_arms(arm_contexts)

        if arm_scores:
            arm_scores = [
                self._calculate_score(arm_score, x, arm)
                for x, arm, arm_score in zip(X, arms, arm_scores)
            ]
        else:
            arm_scores = [
                self._calculate_score(None, x, arm) for x, arm in zip(X, arms)
            ]

        action = int(np.argmax(arm_scores))

        return action

    def rank(
        self,
        arms: List[Any],
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...] = None,
        arm_scores: List[float] = None,
        with_probs: bool = False,
        limit: int = None,
    ) -> Union[List[Any], Tuple[List[Any], List[float]]]:
        assert arm_contexts is not None or arm_scores is not None
        if not arm_scores:
            arm_scores = self.calculate_scores(arm_indices, arm_contexts)
        assert len(arm_indices) == len(arm_scores)

        X, context_arms = self._flatten_input_and_extract_arms(arm_contexts)
        arm_scores = [
            self._calculate_score(arm_score, x, arm)
            for x, arm, arm_score in zip(X, context_arms, arm_scores)
        ]

        ranked_arms = [
            arm for _, arm in sorted(zip(arm_scores, arms), reverse=True)
        ]
        if limit is not None:
            ranked_arms = ranked_arms[:limit]

        if with_probs:
            return ranked_arms, self._compute_prob(arm_indices, arm_scores)
        else:
            return ranked_arms


class CustomRewardModelLinUCB(_LinBanditPolicy):
    def __init__(
        self,
        reward_model: nn.Module,
        alpha: float = 1e-5,
        arm_index: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__(reward_model, arm_index)
        self._alpha = alpha

    def _calculate_score(self, original_score: float, x: np.ndarray, arm: int) -> float:
        Ainv = self._Ainv_per_arm.get(arm)
        if Ainv is None:
            Ainv = np.eye(x.shape[0])
        confidence_bound = self._alpha * np.sqrt(np.linalg.multi_dot([x.T, Ainv, x]))

        return original_score + confidence_bound


class LinUCB(CustomRewardModelLinUCB):
    def __init__(
        self,
        reward_model: nn.Module,
        alpha: float = 1e-5,
        arm_index: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__(None, arm_index)
        self._alpha = alpha

    def calculate_scores(
        self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...]
    ) -> List[float]:
        X, arms = self._flatten_input_and_extract_arms(arm_contexts)
        scores: List[float] = []

        for x, arm in zip(X, arms):
            Ainv = self._Ainv_per_arm.get(arm)
            b = self._b_per_arm.get(arm)
            if Ainv is None:
                Ainv = np.eye(x.shape[0])
            if b is None:
                b = np.zeros((x.shape[0], 1))
            scores.append(Ainv.dot(b).T.dot(x)[0])

        return scores


class LinThompsonSampling(_LinBanditPolicy):
    def __init__(
        self, reward_model: nn.Module, v_sq: float = 1.0, arm_index: int = 1
    ) -> None:
        """
        :param v_sq: Parameter by which to multiply the covariance matrix (more means higher variance).
        """
        super().__init__(None, arm_index)
        self._v_sq = v_sq

    def _calculate_score(self, original_score: float, x: np.ndarray, arm: int) -> float:
        Ainv = self._Ainv_per_arm.get(arm)
        b = self._b_per_arm.get(arm)
        if Ainv is None:
            Ainv = np.eye(x.shape[0])
        if b is None:
            b = np.zeros((x.shape[0], 1))

        mu = (Ainv.dot(b)).reshape(-1)
        mu = np.random.multivariate_normal(mu, self._v_sq * Ainv)
        return x.dot(mu)

    def calculate_scores(
        self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...]
    ) -> List[float]:
        X, arms = self._flatten_input_and_extract_arms(arm_contexts)
        scores: List[float] = []

        for x, arm in zip(X, arms):
            score = self._calculate_score(None, x, arm)
            scores.append(score)

        return scores


class SoftmaxExplorer(BanditPolicy):
    def __init__(
        self,
        reward_model: nn.Module,
        logit_multiplier: float = 1.0,
        reverse_sigmoid: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(reward_model)
        self._logit_multiplier = logit_multiplier
        self._rng = RandomState(seed)
        self._reverse_sigmoid = reverse_sigmoid

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _compute_prob(
        self, arm_indices: List[int], arm_scores: List[float]
    ) -> List[float]:
        arm_scores = np.array(arm_scores)

        if self._reverse_sigmoid:
            arm_scores = np.log(arm_scores + 1e-8 / ((1 - arm_scores) + 1e-8))

        arms_probs = self._softmax(self._logit_multiplier * arm_scores)
        return arms_probs.tolist()

    def _select_idx(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: List[float],
        pos: int,
    ) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probs = self._compute_prob(arm_indices, arm_scores)

        return self._rng.choice(a=len(arm_scores), p=arm_probs)
