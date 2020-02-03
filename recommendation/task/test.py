import gym
import numpy as np
import luigi
import pandas as pd
import os
from recommendation.gym_ifood.envs.ifood_recsys_env import iFoodRecSysEnv
from recommendation.gym_ifood.util.log_stats import LogStats
from recommendation.task.iterator_eval.base import BuildIteractionDatasetTask
from numpy.random.mtrand import RandomState
from scipy.special import softmax, expit

import json

class RandomAgent():
    def __init__(self, action_list):
        self.action_list = action_list
    def act(self, obs):
        return np.random.choice(self.action_list, obs.shape[0])

# class EgreedyAgent():
    
#     def __init__(self, epsilon=0.1, action_list=[]):
#         self.action_list = action_list
#         self._epsilon    = epsilon
#         self._rng        = RandomState(42)
#         self._arms_rewards =  dict([(a,[]) for a in action_list])

#     def act(self, obs):
#         actions = []
#         for o in obs:
#             if self._rng.choice([True, False], p=[self._epsilon, 1.0 - self._epsilon]):
#                 actions.append(np.random.choice(self.action_list))
#             else:
#                 action = actions.append(self.action_list[np.argmax(self.reduction_rewards())])

#         return actions

#     def update(self, arms, rewards):
#         for i in list(zip(arms, rewards)):
#             arm, reward = i

#             self._arms_rewards[arm].append(reward)
    
#     def reduction_rewards(self, func= np.mean):
#         return np.array([func(self._arms_rewards[a]) for a in self.action_list])


#PYTHONPATH="." luigi --module recommendation.task.test EnvironmentTestTask --local-scheduler
class EnvironmentTestTask(luigi.Task):
    obs_batch_size: int = luigi.IntParameter(default=2000)
    filter_dish: str = luigi.Parameter(default="all")
    
    def requires(self):
        return BuildIteractionDatasetTask(run_type = "reinforcement", filter_dish=self.filter_dish)

    def output(self):
        return  luigi.LocalTarget(os.path.join("output", "reinforcement", self.__class__.__name__, 
                                    "results", self.task_id, "history.csv")), \
                luigi.LocalTarget(os.path.join("output", "reinforcement", self.__class__.__name__, 
                                    "results", self.task_id, "params.json"))   

    def save_params(self):
        with open(self.output()[1].path, "w") as params_file:
            json.dump(self.param_kwargs, params_file, default=lambda o: dict(o), indent=4)

    def _get_action_list(self, path):
        return pd.read_parquet(path)['merchant_id'].unique().flatten()

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)
        log = LogStats()
        env = gym.make('ifood-recsys-v0', 
                        dynamics_dataset_path=self.input().path, 
                        obs_batch_size = self.obs_batch_size)

        action_list = self._get_action_list(self.input().path)

        agent = RandomAgent(action_list)
        #agent = EgreedyAgent(action_list=action_list)
        env.seed(0)

        episode_count = 1
        interactions = 0
        done = False

        for i in range(episode_count):
            ob = env.reset()
            while True:
                interactions += len(ob)

                action = agent.act(ob)
                ob, reward, done, info = env.step(action)

                # Log
                log.logger(ob, action, reward)
                print(interactions, log.rewards_stats)

                if done:
                    break

        
        # Save logs
        log.save_history(self.output()[0].path)
        self.save_params()

        env.close()



