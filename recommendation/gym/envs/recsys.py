from typing import Tuple

import gym
import numpy as np
import pandas as pd
from gym import utils


class RecSysEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, dataset: pd.DataFrame, item_column: str, obs_batch_size: int = 2000):
        self.init_batch = 0
        self.end_batch = 0
        self.obs_batch_size = obs_batch_size
        self.reward_range = [0.0, 1.0]
        # TODO
        # self.observation_space
        # self.action_space
        self.dataset = dataset
        self.item_column = item_column
        self.obs_dataset = dataset.drop(columns=[item_column])

    def _compute_end_batch(self):
        if self.end_batch + self.obs_batch_size < len(self.dataset):
            self.end_batch += self.obs_batch_size
        else:
            self.end_batch = len(self.dataset)

    def _compute_stats(self, action: np.ndarray) -> dict:
        # TODO: Choose which batch metrics to return
        return {}

    def _compute_rewards(self, action: np.ndarray) -> np.ndarray:
        expected_items = self.dataset[self.init_batch: self.end_batch][[self.item_column]].values.flatten()

        return (action == expected_items) * 1.0

    def _next_obs(self) -> np.ndarray:
        return self.obs_dataset[self.init_batch: self.end_batch].values

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        rewards = self._compute_rewards(action)
        info = self._compute_stats(action)
        done = False

        if self.end_batch == (len(self.dataset)):
            done = True
            next_obs = np.array([])
        else:
            self.init_batch = self.end_batch
            self._compute_end_batch()

            next_obs = self._next_obs()
        return next_obs, rewards, done, info

    def reset(self) -> np.ndarray:
        self.init_batch = 0
        self.end_batch = 0
        self._compute_end_batch()
        return self._next_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
