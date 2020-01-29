import gym
import numpy as np
import luigi
import pandas as pd
from recommendation.gym_ifood.envs.ifood_recsys_env import iFoodRecSysEnv


from recommendation.task.iterator_eval.base import BuildIteractionDatasetTask

class RandomAgent():
    def __init__(self, action_list):
        self.action_list = action_list
    def act(self, obs):
        return np.random.choice(self.action_list, obs.shape[0])

#PYTHONPATH="." luigi --module recommendation.task.test EnvironmentTestTask --local-scheduler
class EnvironmentTestTask(luigi.Task):
    obs_batch_size: int = luigi.IntParameter(default=2000)
    filter_dish: str = luigi.Parameter(default="all")
    
    def requires(self):
        return BuildIteractionDatasetTask(run_type = "reinforcement", filter_dish=self.filter_dish)

    def _get_action_list(self, path):
        return pd.read_parquet(path)['merchant_id'].unique().flatten()

    def run(self):
        env= gym.make('ifood-recsys-v0', dynamics_dataset_path=self.input().path, obs_batch_size = self.obs_batch_size)

        action_list = self._get_action_list(self.input().path)

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



