import gym
import numpy as np
import luigi
import pandas as pd
from typing import List

from recommendation.gym_ifood.envs.ifood_recsys_env import IFoodRecSysEnv
from recommendation.task.data_preparation.ifood import CreateInteractionDataset, \
    IndexAccountsAndMerchantsOfSessionTrainDataset, CreateGroundTruthForInterativeEvaluation

from recommendation.task.iterator_eval.base import BuildIteractionDatasetTask

class RandomAgent():
    def __init__(self, action_list: List[int]):
        self.action_list = action_list
    def act(self, obs):
        return np.random.choice(self.action_list, obs.shape[0])

#PYTHONPATH="." luigi --module recommendation.task.test EnvironmentTestTask --local-scheduler
class EnvironmentTestTask(luigi.Task):
    obs_batch_size: int = luigi.IntParameter(default=2000)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    filter_dish: str = luigi.Parameter(default="all")
    
    def requires(self):
        return CreateGroundTruthForInterativeEvaluation(minimum_interactions=self.minimum_interactions,
                                                        filter_dish=self.filter_dish)

    def run(self):
        df = pd.read_parquet(self.input().path)
        env= gym.make('ifood-recsys-v0', dataset=df, obs_batch_size = self.obs_batch_size)

        action_list = df['merchant_idx'].unique().flatten()

        agent = RandomAgent(action_list)

        env.seed(0)

        episode_count = 1
        rewards = []
        interactions = 0
        done = False

        for i in range(episode_count):
            ob = env.reset()
            while True:
                interactions += len(ob)
                action = agent.act(ob)
                ob, reward, done, info = env.step(action)
                if done:
                    break

                rewards.append(reward)
                print(interactions)
                print(np.mean(reward), np.std(reward))
        
        env.close()



