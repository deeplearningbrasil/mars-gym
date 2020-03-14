from collections import OrderedDict
from typing import Tuple, List, Any

import gym
import numpy as np
import pandas as pd
from gym import utils
from gym import spaces


class RecSysEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': []}
    reward_range = [0.0, 1.0]

    def __init__(self, dataset: pd.DataFrame, item_column: str):
        super().__init__()
        self._dataset = dataset
        self._item_column = item_column
        self._obs_dataset: List[dict] = dataset.drop(columns=[item_column]).to_dict('records')

        self.action_space = spaces.Discrete(self._dataset[item_column].max()+1)
        self.observation_space = spaces.Dict({
            key: self._convert_value_to_space(key, value)
            for key, value in self._obs_dataset[0].items()
        })
        self._current_index = 0

    def _convert_value_to_space(self, key: str, value: Any) -> spaces.Space:
        if isinstance(value, int):
            return spaces.Discrete(self._dataset[key].max() + 1)
        elif isinstance(value, float):
            return spaces.Box(self._dataset[key].min(), self._dataset[key].max(), shape=(1,))
        elif isinstance(value, np.ndarray):
            if issubclass(value.dtype, np.integer):
                return spaces.MultiDiscrete([self._dataset[key].max() + 1] * len(value))
            elif issubclass(value.dtype, np.floating):
                return spaces.Box(self._dataset[key].min(), self._dataset[key].max(), shape=value.shape)
        raise ValueError("Unkown type in the observation space for {}:{}".format(key, value))

    def _compute_stats(self, action: float) -> dict:
        # TODO: Choose which batch metrics to return
        return {}

    def _compute_reward(self, action: int) -> float:
        return float(self._dataset.iloc[self._current_index][self._item_column] == action)

    def _next_obs(self) -> dict:
        return self._obs_dataset[self._current_index]

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        reward = self._compute_reward(action)
        info = self._compute_stats(action)

        self._current_index += 1

        done = (self._current_index + 1) == len(self._dataset)
        next_obs = self._next_obs() if not done else None

        return next_obs, reward, done, info

    def reset(self) -> dict:
        self._current_index = 0
        return self._next_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
