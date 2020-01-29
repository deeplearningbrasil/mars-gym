import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd
import numpy as np


class iFoodRecSysEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, dynamics_dataset_path, obs_batch_size = 2000):
        self.init_batch = 0
        self.end_batch = 0
        self.obs_batch_size = obs_batch_size
        self.reward_range = [0.0, 1.0]
        #TODO
        #self.observation_space
        #self.action_space
        self.dataset = pd.read_parquet(dynamics_dataset_path)[['account_id', 'merchant_id', 'click_timestamp']].sort_values("click_timestamp")
    
    def _compute_end_batch(self):
        if self.end_batch + self.obs_batch_size < len(self.dataset):
            self.end_batch += self.obs_batch_size
        else:
            self.end_batch = len(self.dataset) - 1

    def _compute_stats(self, action):
        #TODO: Choose which batch metrics to return
        return {}

    def _compute_rewards(self, action):
        merchant_list = self.dataset[self.init_batch : self.end_batch][['merchant_id']].values.flatten()
        return (action == merchant_list) * 1.0
    
    def step(self, action):
        rewards = self._compute_rewards(action)
        info = self._compute_stats(action)
        done = False
        next_obs = np.array([])
        if self.init_batch == len(self.dataset) - 1:
            done = True
        else:
            self.init_batch = self.end_batch
            self._compute_end_batch()
            
            next_obs = self.dataset[self.init_batch : self.end_batch][['account_id', 'click_timestamp']].values
        return next_obs, rewards, done, info

    def reset(self):
        self.init_batch = 0
        self.end_batch = 0
        self._compute_end_batch()
        return self.dataset[self.init_batch : self.end_batch][['account_id', 'click_timestamp']].values

    def render(self, mode='human'):
        pass
    def close(self):
        pass