import abc
import collections
from typing import List, Union, Tuple, Dict, Type, Optional
import requests

import math
import numpy as np
import torch
import torch.nn as nn
from numpy.random.mtrand import RandomState
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import json
from recommendation.utils import chunks
from typing import NamedTuple, List, Union, Dict
from scipy.special import softmax, expit
class InverseIndexMapping(NamedTuple):
    user: Dict[int, str]
    item: Dict[int, str]

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
        if self.reward_model:
            inputs: torch.Tensor = default_convert(arm_contexts)
            scores: torch.Tensor = self.reward_model(*inputs)
            return scores.detach().cpu().numpy().tolist()
        else:
            return None

    def select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:
        assert arm_contexts is not None or arm_scores is not None

        if arm_scores is None:
            arm_scores = self._calculate_scores(arm_contexts)
        return self._select_idx(arm_indices, arm_contexts, arm_scores, pos)

    def select(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
               arm_scores: List[float] = None) -> int:
        return arm_indices[self.select_idx(arm_indices, arm_contexts, arm_scores)]

    def rank(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
             arm_scores: List[float] = None, with_probs: bool = False,
             limit: int = None) -> Union[List[int], Tuple[List[int], List[float]]]:
        assert arm_contexts is not None or arm_scores is not None
        
        if arm_scores is None:
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
            idx = self.select_idx(arm_indices, arm_contexts=arm_contexts, arm_scores=arm_scores, pos=i)
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


class RemotePolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, seed: int = 42, index_data = None, endpoints=[], window_reward=500) -> None:
        super().__init__(None)
        self._rng = RandomState(seed)
        self._endpoints      = endpoints
        self._total_arms     = len(endpoints)
        self._arms_selected  = [] 
        self._arms_rewards   = {}
        self.index_mapping   = index_data
        self._window_reward = window_reward

        self.init_arms()
    
    def init_arms(self):
        if self._total_arms == 0:
            raise(Exception("endpoints empty"))

        for arm, endpoint in enumerate(self._endpoints):
            self._arms_rewards[arm] = [1]

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        i        = len(dataset)-1
        row      = dataset[i]
        
        #for i, row in enumerate(dataset):
        input_   = row[0]
        output_  = row[1]
        reward   = output_[0][0] if isinstance(output_, tuple) else output_
        #if i % 500 == 0:
        #    

        self._arms_rewards[self._arms_selected[i]].append(reward)

    @property
    def inverse_index_mapping(self) -> InverseIndexMapping:
        if not hasattr(self, "_inverse_index_mapping"):
            self._inverse_index_mapping = InverseIndexMapping(
                user=dict((v, k) for k, v in self.index_mapping.user.items()),
                item=dict((v, k) for k, v in self.index_mapping.item.items()),
            )
        return self._inverse_index_mapping

    def _calculate_scores(self, arm_contexts: Tuple[np.ndarray, ...]) -> List[float]:
        return self._compute_prob(arm_contexts[0])

    def _compute_prob(self, arm_scores) -> List[float]:
        return np.ones(len(arm_scores))

    def _reduction_rewards(self, func= np.mean):
        return np.array([func(self._arms_rewards[i][-self._window_reward:-1]) for i in range(self._total_arms)])

    def _select_best_endpoint(self):
        return 0

    def _request(self, endpoint: str, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None):

        items_idx = arm_indices
        user_idx  = arm_contexts[0][0]
        
        #{"user": "anything", "items": ["a6aa3afc-30c4-4e64-aad5-b1a6db104245", "fb42869a-088d-4b98-941f-aa15ca464128", "6a462430-96cc-424d-9f03-7c85bbdb9b1d"]}
        payload   = {"user": self.index_mapping.user[user_idx], "items": [self.index_mapping.item[i] for i in items_idx]}
        
        r = requests.post(endpoint, data = json.dumps(payload), 
                        headers={"Content-Type": "application/json"} )
        
        r = json.loads(r.text)
        
        return self.inverse_index_mapping.item[r['items'][0]]

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:

        # Select best endpoint
        arm_idx   = self._select_best_endpoint()

        # Request
        action    = self._request(self._endpoints[arm_idx], arm_indices, arm_contexts)

        # Save arm
        self._arms_selected.append(arm_idx)

        return list(arm_indices).index(action)


class RemoteEpsilonGreedy(RemotePolicy):
    def __init__(self, reward_model: nn.Module, epsilon: float = 0.1, 
                index_data = None, endpoints=[], window_reward=500,seed: int = 42) -> None:
        super().__init__(reward_model, seed, index_data, endpoints, window_reward)
        self._epsilon = epsilon
        self._rng = RandomState(seed)
    def _select_best_endpoint(self):
        return np.argmax(self._reduction_rewards())

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:
        
        # Select best endpoint
        if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
            arm_idx = self._rng.choice(self._total_arms)
        else:
            arm_idx = self._select_best_endpoint()

        # Request
        action    = self._request(self._endpoints[arm_idx], arm_indices, arm_contexts)

        # Save arm
        self._arms_selected.append(arm_idx)

        return list(arm_indices).index(action)

class RemoteEnsemble(RemotePolicy):
    def __init__(self, reward_model: nn.Module, index_data = None, endpoints=[], agg='mean', window_reward=500,seed: int = 42) -> None:
        super().__init__(reward_model, seed, index_data, endpoints, window_reward)
        self._rng = RandomState(seed)
        agg_func = {"mean": np.mean, "max": np.max, "min": np.min}
        self._agg_func = agg_func[agg]

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        pass

    def _request(self, endpoint: str, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None):

        items_idx = arm_indices
        user_idx  = arm_contexts[0][0]
        
        #{"user": "anything", "items": ["a6aa3afc-30c4-4e64-aad5-b1a6db104245", "fb42869a-088d-4b98-941f-aa15ca464128", "6a462430-96cc-424d-9f03-7c85bbdb9b1d"]}
        payload   = {"user": self.index_mapping.user[user_idx], "items": [self.index_mapping.item[i] for i in items_idx]}
        
        r = requests.post(endpoint, data = json.dumps(payload), 
                        headers={"Content-Type": "application/json"} )
        
        r = json.loads(r.text)
        
        return r['items']#

    def _get_best_item(self, list_models):
        actions_scores = {}
        list_size = len(list_models[0])

        for l in list_models:
            for i in range(list_size):
                 if l[i] in actions_scores:
                    actions_scores[l[i]].append(i)
                 else:
                    actions_scores[l[i]] = [i]
            
        actions     = list(actions_scores.keys())
        avg_pos     = [self._agg_func(l) for l in  list(actions_scores.values())]
        best_action = np.argmin(avg_pos)
        #from IPython import embed; embed()
        return self.inverse_index_mapping.item[actions[best_action]]

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:
        
        # Request
        list_models = [ ]
        for endpoint in self._endpoints:
            actions    = self._request(endpoint, arm_indices, arm_contexts)
            list_models.append(actions)

        action = self._get_best_item(list_models)

        return list(arm_indices).index(action)

class RemoteUCB(RemotePolicy):
    def __init__(self, reward_model: nn.Module, c: float = 2, index_data = None, window_reward=500,endpoints=[], seed: int = 42) -> None:
        super().__init__(reward_model, seed, index_data, endpoints, window_reward)
        self._c            = c
        self._times        = 1
        self._action_times = np.zeros(len(endpoints))        
        self._rng = RandomState(seed)

    def _select_best_endpoint(self):
        reward_mean      = self._reduction_rewards()

        confidence_bound = reward_mean + \
                            self._c * np.sqrt(\
                                  np.log(self._times) / (self._action_times + 0.1))  # c=2
        return np.argmax(confidence_bound)

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:

        # Seçect Arm
        arm_idx = self._select_best_endpoint()

        # Request
        action    = self._request(self._endpoints[arm_idx], arm_indices, arm_contexts)

        # Save arm
        self._arms_selected.append(arm_idx)
        self._times += 1
        self._action_times[arm_idx] += 1

        return list(arm_indices).index(action)

class RemoteSoftmax(RemotePolicy):
    def __init__(self, reward_model: nn.Module, logit_multiplier: float = 2.0, index_data = None, endpoints=[], seed: int = 42) -> None:
        super().__init__(reward_model, seed, index_data, endpoints)
        self._logit_multiplier = logit_multiplier   
        self._rng = RandomState(seed)
        self._total_arms = len(endpoints)

    def _select_best_endpoint(self):
        reward_mean  = self._reduction_rewards()
        
        reward_logit = expit(reward_mean)
        arms_probs   = softmax(self._logit_multiplier * reward_logit)
        print(arms_probs)
        return self._rng.choice(list(range(self._total_arms)), p = arms_probs)
        

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:

        # Seçect Arm
        arm_idx   = self._select_best_endpoint()
        
        # Request
        action    = self._request(self._endpoints[arm_idx], arm_indices, arm_contexts)

        # Save arm
        self._arms_selected.append(arm_idx)

        return list(arm_indices).index(action)

from creme import compose
from creme import linear_model, multiclass
from creme import metrics
from creme import preprocessing
from creme import optim
from creme import sampling

class RemoteContextualEpsilonGreedy(RemoteEpsilonGreedy):
    def __init__(self, reward_model: nn.Module, epsilon: float = 0.1, index_data = None, endpoints=[], seed: int = 42) -> None:
        super().__init__(reward_model, seed, index_data, endpoints)
        self._epsilon = epsilon
        self._rng     = RandomState(seed)
        self._oracle  = self.build_oracle()
        self._oracle_metric  = metrics.MacroF1()
        self._times        = 1

    def build_oracle(self):
        model = compose.Pipeline(
            ('scale', preprocessing.StandardScaler()),
            ('learn', multiclass.OneVsRestClassifier(
                binary_classifier=linear_model.LogisticRegression())
            )
        )        
        return model

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        i        = len(dataset)-1
        row      = dataset[i]

        input_   = row[0]
        output_  = row[1]
        reward   = output_[0][0] if isinstance(output_, tuple) else output_
        
        # build features
        x = np.nan_to_num(np.reshape(input_, -1))
        x = {i: e for i, e in enumerate(x)} 

        if reward:
            # fit
            self._oracle.fit_one(x, self._arms_selected[i])
        
        # if self._times % 500 == 0:
        #    

    def _flatten_input(self, input_: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray]:
        flattened_input = np.concatenate([el.reshape(-1, 1) if len(el.shape) == 1 else el for el in input_], axis=1)
        return flattened_input

    def _select_best_endpoint(self, arm_contexts):
        scores = []
        #

        input_ = self._flatten_input(arm_contexts)[0]
        
        x = np.nan_to_num(np.reshape(input_, -1))
        x = {i: e for i, e in enumerate(x)} 

        #
        
        arm = self._oracle.predict_one(x)
        
        return self._rng.choice(self._total_arms) if arm is None else arm


    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:

        # Select best endpoint
        if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
            arm_idx = self._rng.choice(self._total_arms)
        else:
            arm_idx = self._select_best_endpoint(arm_contexts)

        # Request
        action    = self._request(self._endpoints[arm_idx], arm_indices, arm_contexts)

        # Save arm
        self._arms_selected.append(arm_idx)
        self._times += 1

        return list(arm_indices).index(action)
    

class RemoteContextualSoftmax(RemoteContextualEpsilonGreedy):
    def __init__(self, reward_model: nn.Module, logit_multiplier: float = 1.0, reverse_sigmoid: bool = True, index_data = None, endpoints=[], seed: int = 42) -> None:
        super().__init__(reward_model, 0.1, index_data, endpoints)
        self._logit_multiplier = logit_multiplier
        self._reverse_sigmoid = reverse_sigmoid

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def inv_logit(self, p):
        return np.exp(p) / (1 + np.exp(p))

    def _arm_probs(self, arm_contexts):
        scores = []
        #

        input_ = self._flatten_input(arm_contexts)[0]
        
        x = np.nan_to_num(np.reshape(input_, -1))
        x = {i: e for i, e in enumerate(x)} 

        arm_probs = self._oracle.predict_proba_one(x)
    
        if len(arm_probs) != self._total_arms:
            arm_scores = np.random.random(self._total_arms)
            #return arm_probs/arm_probs.sum()
        else:
            arm_scores = [arm_probs[i] if i in arm_probs else 0.0 for i in range(self._total_arms) ] 
            #return arm_probs/np.sum(arm_probs)
        
        arm_scores = self.inv_logit(arm_scores)
        arms_probs = self._softmax(self._logit_multiplier * arm_scores)

        return arms_probs

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:
        # Select best endpoint
        probs   = self._arm_probs(arm_contexts)
        arm_idx = self._rng.choice(self._total_arms, p=probs)

        # Request
        action    = self._request(self._endpoints[arm_idx], arm_indices, arm_contexts)

        # Save arm
        self._arms_selected.append(arm_idx)
        self._times += 1

        return list(arm_indices).index(action)
    

class MetaBanditPolicy(RemoteContextualEpsilonGreedy):
    def __init__(self, reward_model: nn.Module, epsilon: float = 0.1, index_data = None, 
            endpoints="", seed: int = 42) -> None:
        super().__init__(reward_model, epsilon, index_data, endpoints, seed)
        self._rng = RandomState(seed)
        self._endpoint = endpoints
        #self._arms = arms#['random', 'most_popular', 'cvae']

    def _update(self, c, a, r):
        
        payload = {
            "context": c,
            "arm": a,
            "reward": r
        }

        r = requests.post(self._endpoint+"/update", data = json.dumps(payload), 
                        headers={"Content-Type": "application/json"} )
        

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        i        = len(dataset)-1
        row      = dataset[i]

        input_   = row[0]
        output_  = row[1]
        reward   = output_[0][0] if isinstance(output_, tuple) else output_
        
        # build features
        x = np.nan_to_num(np.reshape(input_, -1))
        x = {i: e for i, e in enumerate(x)} 

        if reward:
            #from IPython import embed; embed()
            self._update(x, self._arms_selected[i], reward)


    def _request(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None):

        # Context        
        input_ = self._flatten_input(arm_contexts)[0]
        x      = np.nan_to_num(np.reshape(input_, -1))
        x      = {i: e for i, e in enumerate(x)} 

        # input
        items_idx = arm_indices
        user_idx  = arm_contexts[0][0]
        
        payload   = {"context": x, "input": {"user": self.index_mapping.user[user_idx], "items": [self.index_mapping.item[i] for i in items_idx]}}
        
        r = requests.post(self._endpoint+"/predict", data = json.dumps(payload), headers={"Content-Type": "application/json"} )
        
        return json.loads(r.text)


    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:

        # Request
        result  = self._request(arm_indices, arm_contexts)

        action  = self.inverse_index_mapping.item[result['result']['items'][0]]
        arm     = result['bandit']['arm']
        
        # Save arm
        self._arms_selected.append(arm)
        self._times += 1

        return list(arm_indices).index(action)










# =========================================================================================================
#
#
class RandomPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, seed: int = 42) -> None:
        super().__init__(None)
        self._rng = RandomState(seed)

    def _calculate_scores(self, arm_contexts: Tuple[np.ndarray, ...]) -> List[float]:
        return self._compute_prob(arm_contexts[0])

    def _compute_prob(self, arm_scores) -> List[float]:
        n_arms = len(arm_scores)
        arms_probs = np.ones(n_arms) / n_arms
        return arms_probs.tolist()

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...] = None,
                   arm_scores: List[float] = None, pos: int = 0) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)

        arm_probas = np.ones(n_arms) / n_arms
        
        action = self._rng.choice(n_arms, p=arm_probas)

        return action

class FixedPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, arg: int = 1, seed: int = 42) -> None:
        super().__init__(None)
        self._arg = arg
        self._arm_index = 1

    def _calculate_scores(self, arm_contexts: Tuple[np.ndarray, ...]) -> List[float]:
        X, arms    = self._flatten_input_and_extract_arms(arm_contexts)
        arm_scores = [int(x[self._arg] == arm) for x, arm in zip(X, arms)]

        return arm_scores

    def _compute_prob(self, arm_scores) -> List[float]:
        n_arms      = len(arm_scores)
        arms_probs  = np.zeros(n_arms)
        argmax      = int(np.argmax(arm_scores))
        arms_probs[argmax] = 1.0
        return arms_probs.tolist()

    def _flatten_input_and_extract_arms(self, input_: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray]:
        flattened_input = np.concatenate([el.reshape(-1, 1) if len(el.shape) == 1 else el for el in input_], axis=1)
        return np.delete(flattened_input, self._arm_index, axis=1), flattened_input[:, self._arm_index]

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        action     = int(np.argmax(arm_scores))

        return action

class ModelPolicy(BanditPolicy):
    def __init__(self, reward_model: nn.Module, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._rng = RandomState(seed)

    def _compute_prob(self, arm_scores) -> List[float]:
        n_arms      = len(arm_scores)
        arms_probs  = np.zeros(n_arms)
        argmax      = int(np.argmax(arm_scores))
        arms_probs[argmax] = 1.0
        return arms_probs.tolist()

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        action = int(np.argmax(arm_scores))

        return action

class ExploreThenExploit(BanditPolicy):
    #TODO: Tune breakpoint parameter
    def __init__(self, reward_model: nn.Module, 
                 explore_rounds: int = 500, 
                 decay_rate: float = 0.0026456, 
                 seed: int = 42) -> None:
        super().__init__(reward_model)
        self._init_explore_rounds = explore_rounds
        self._explore_rounds = explore_rounds
        self._exploit_rounds = explore_rounds
        self._decay_rate     = decay_rate
        
        self._rng = RandomState(seed)
        self._t   = 0
        self._te  = 0        
        self.exploring = True

    def _update_state(self):
        self._t  += 1
        self._te += 1

        if self._explore_rounds > 1:
            if self.exploring and self._te > self._explore_rounds:
                    self._te = 0
                    self._explore_rounds = self.decay(self._init_explore_rounds, self._decay_rate, self._t)
                    self.exploring = False
            elif not self.exploring and self._te > self._exploit_rounds:
                    self._te = 0
                    #self._exploit_rounds += (self._init_explore_rounds-self._explore_rounds)
                    self.exploring = True
        else: 
            self.exploring = False

    def decay(self, init, decay_rate, t):
        return init*(1-decay_rate)**t

    def _compute_prob(self, arm_scores) -> List[float]:
        n_arms = len(arm_scores)
        arm_probs = np.zeros(len(arm_scores))
        max_score = max(arm_scores)
        argmax = int(np.argmax(arm_scores))
        
        if self.exploring:
            arm_probs = np.ones(n_arms) / n_arms
        else: 
            arm_probs[argmax] = 1.0

        return arm_probs.tolist()

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

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
    def __init__(self, reward_model: nn.Module, epsilon: float = 0.1, epsilon_decay: float = 1.0, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._epsilon = epsilon
        self._rng = RandomState(seed)
        self._epsilon_decay = epsilon_decay

    def _compute_prob(self, arm_scores) -> List[float]:
        n_arms     = len(arm_scores)
        arms_probs = self._epsilon * np.ones(n_arms) / n_arms

        argmax     = int(np.argmax(arm_scores))

        arms_probs[argmax] += (1 - self._epsilon)

        return arms_probs.tolist()


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
    def __init__(self, reward_model: nn.Module, exploration_threshold: float = 0.8, decay_rate: float = 0.0010391,
         seed: int = 42) -> None:
        super().__init__(reward_model)
        self._init_exploration_threshold = exploration_threshold
        self._exploration_threshold = exploration_threshold
        self._decay_rate = decay_rate
        self._rng = RandomState(seed)
        self._t = 0

    def _compute_prob(self, arm_scores) -> List[float]:
        n_arms = len(arm_scores)
        arm_probs = np.zeros(len(arm_scores))
        max_score = max(arm_scores)
        argmax = int(np.argmax(arm_scores))

        if max_score > self._exploration_threshold:
            arm_probs[argmax] = 1.0
        else:
            arm_probs = np.ones(n_arms) / n_arms

        return arm_probs.tolist()

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        n_arms     = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms
        max_score  = max(arm_scores)

        if max_score > self._exploration_threshold:
            action = int(np.argmax(arm_scores))
        else:
            action = self._rng.choice(len(arm_indices), p=arm_probas)

        if pos == 0:
            self._t += 1
            self._exploration_threshold = self.decay(self._init_exploration_threshold, self._decay_rate, self._t)

        return action
    
    def decay(self, init, decay_rate, t):
        return init*(1-decay_rate)**t

class PercentileAdaptiveGreedy(BanditPolicy):
    #TODO: Tune these parameters: window_size, exploration_threshold, percentile, percentile_decay
    def __init__(self, reward_model: nn.Module, window_size: int = 500, exploration_threshold: float = 0.5, percentile = 35, percentile_decay: float = 1.0,
         seed: int = 42) -> None:
        super().__init__(reward_model)
        self._window_size = window_size
        self._initial_exploration_threshold = exploration_threshold
        self._percentile_decay = percentile_decay
        self._best_arm_history = {} # We save a deque for each pos
        self._rng = RandomState(seed)
        self._percentile = percentile
        self._t = 0
    
    def _compute_prob(self, arm_scores) -> List[float]:
        n_arms = len(arm_scores)
        max_score = max(arm_scores)

        exploration_threshold = np.percentile(self._best_arm_history[0], self._percentile) \
            if self._t >= self._window_size else self._initial_exploration_threshold

        arm_probs = np.zeros(len(arm_scores))
        argmax = int(np.argmax(arm_scores))

        if max_score >= exploration_threshold:
            arm_probs[argmax] = 1.0
        else:
            arm_probs = np.ones(n_arms) / n_arms

        return arm_probs.tolist()

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        if pos not in self._best_arm_history:
            self._best_arm_history[pos] = collections.deque([])

        if pos == 0:
            self._t += 1

        n_arms     = len(arm_indices)
        arm_probas = np.ones(n_arms) / n_arms

        max_score  = max(arm_scores)

        exploration_threshold = np.percentile(self._best_arm_history[pos], self._percentile) \
            if len(self._best_arm_history[pos]) >= self._window_size else self._initial_exploration_threshold

        if max_score >= exploration_threshold:
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

    def __init__(self, reward_model: nn.Module, arm_index: int = 1, scaler=False, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._arm_index = arm_index
        self._Ainv_per_arm: Dict[int, np.ndarray] = {}
        if reward_model is None:
            self._b_per_arm: Dict[int, np.ndarray] = {}
        #self._scaler = StandardScaler()

    def _sherman_morrison_update(self, Ainv: np.ndarray, x: np.ndarray) -> None:
        ## x should have shape (n, 1)
        Ainv -= np.linalg.multi_dot([Ainv, x, x.T, Ainv]) / (1.0 + np.linalg.multi_dot([x.T, Ainv, x]))

    def _flatten_input_and_extract_arms(self, input_: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray]:
        flattened_input = np.concatenate([el.reshape(-1, 1) if len(el.shape) == 1 else el for el in input_], axis=1)
        return np.delete(flattened_input, self._arm_index, axis=1), flattened_input[:, self._arm_index]

    def fit(self, dataset: Dataset, batch_size: int = 500) -> None:
        self._Ainv_per_arm.clear()
        if hasattr(self, "_b_per_arm"):
            self._b_per_arm.clear()

        n = len(dataset)

        for indices in tqdm(chunks(range(n), batch_size), total=math.ceil(n / batch_size)):
            input_: Tuple[np.ndarray, ...] = dataset[indices][0]
            output_: Tuple[np.ndarray, ...] = dataset[indices][1]

            X, arms = self._flatten_input_and_extract_arms(input_)
            output  = output_[0] if isinstance(output_, tuple) else output_

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
    def _calculate_score(self, original_score: Optional[float], x: np.ndarray, arm: int) -> float:
        pass

    def _compute_prob(self, arm_scores) -> List[float]:
        #In this case, we expected arm_scores to be arms_scores_with_cb
        n_arms             = len(arm_scores)
        arms_probs         = np.zeros(n_arms)
        argmax             = int(np.argmax(arm_scores))
        arms_probs[argmax] = 1.0
        return arms_probs.tolist()

    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: Optional[List[float]], pos: int) -> Union[int, Tuple[int, float]]:

        X, arms    = self._flatten_input_and_extract_arms(arm_contexts)

        if arm_scores:
            arm_scores = [self._calculate_score(arm_score, x, arm)
                      for x, arm, arm_score in zip(X, arms, arm_scores)]
        else:
            arm_scores = [self._calculate_score(None, x, arm)
                          for x, arm in zip(X, arms)]

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

class CustomRewardModelLinUCB(_LinBanditPolicy):
    def __init__(self, reward_model: nn.Module, alpha: float = 1e-5, arm_index: int = 1, seed: int = 42) -> None:
        super().__init__(reward_model, arm_index)
        self._alpha = alpha

    def _calculate_score(self, original_score: float, x: np.ndarray, arm: int) -> float:
        Ainv = self._Ainv_per_arm.get(arm)
        if Ainv is None:
            Ainv = np.eye(x.shape[0])
        confidence_bound = self._alpha * np.sqrt(np.linalg.multi_dot([x.T, Ainv, x]))
        
        return original_score + confidence_bound

class LinUCB(CustomRewardModelLinUCB):
    def __init__(self, reward_model: nn.Module, alpha: float = 1e-5, arm_index: int = 1, seed: int = 42) -> None:
        super().__init__(None, arm_index)
        self._alpha = alpha

    def _calculate_scores(self, arm_contexts: Tuple[np.ndarray, ...]) -> List[float]:
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
    def __init__(self, reward_model: nn.Module, v_sq: float = 1.0, arm_index: int = 1) -> None:
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

    def _calculate_scores(self, arm_contexts: Tuple[np.ndarray, ...]) -> List[float]:
        X, arms = self._flatten_input_and_extract_arms(arm_contexts)
        scores: List[float] = []

        for x, arm in zip(X, arms):
            score = self._calculate_score(None, x, arm)
            scores.append(score)

        return scores        

class SoftmaxExplorer(BanditPolicy):
    def __init__(self, reward_model: nn.Module, logit_multiplier: float = 1.0, reverse_sigmoid: bool = True, seed: int = 42) -> None:
        super().__init__(reward_model)
        self._logit_multiplier = logit_multiplier
        self._rng = RandomState(seed)
        self._reverse_sigmoid = reverse_sigmoid

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _compute_prob(self, arm_scores) -> List[float]:
        n_arms = len(arm_scores)
        arm_scores = np.array(arm_scores)

        if self._reverse_sigmoid:
            arm_scores = np.log(arm_scores + 1e-8/((1 - arm_scores) + 1e-8))

        arms_probs = self._softmax(self._logit_multiplier * arm_scores)
        return arms_probs.tolist()


    def _select_idx(self, arm_indices: List[int], arm_contexts: Tuple[np.ndarray, ...],
                    arm_scores: List[float], pos: int) -> Union[int, Tuple[int, float]]:

        n_arms = len(arm_indices)
        arm_probs = self._compute_prob(arm_scores)

        return self._rng.choice(a=len(arm_scores), p=arm_probs)


BANDIT_POLICIES: Dict[str, Type[BanditPolicy]] = dict(
    epsilon_greedy=EpsilonGreedy, lin_ucb=LinUCB, custom_lin_ucb=CustomRewardModelLinUCB,
    lin_ts=LinThompsonSampling, random=RandomPolicy, percentile_adaptive=PercentileAdaptiveGreedy,
    adaptive=AdaptiveGreedy, model=ModelPolicy, softmax_explorer = SoftmaxExplorer,
    explore_then_exploit=ExploreThenExploit, fixed=FixedPolicy, none=None, remote=RemotePolicy, 
    remote_epsilon_greedy=RemoteEpsilonGreedy, remote_ucb=RemoteUCB, remote_softmax=RemoteSoftmax,
    remote_contextual_epsilon_greedy=RemoteContextualEpsilonGreedy, meta_bandit=MetaBanditPolicy, 
    remote_contextual_softmax=RemoteContextualSoftmax, remote_ensemble=RemoteEnsemble)
