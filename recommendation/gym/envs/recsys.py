from collections import OrderedDict
from typing import Tuple, List

import gym
import numpy as np
import pandas as pd
from gym import utils


class RecSysEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, dataset: pd.DataFrame, item_column: str):
        self._dataset = dataset
        self._item_column = item_column
        self._obs_dataset: List[dict] = dataset.drop(columns=[item_column]).to_dict('records')

        self._current_index = 0
        self.reward_range = [0.0, 1.0]
        # TODO
        # self.observation_space
        # self.action_space

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
