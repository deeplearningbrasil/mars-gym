from typing import List, Tuple, Dict, Any

import gym
import luigi
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchbearer import Trial

from recommendation.gym_ifood.envs import IFoodRecSysEnv
from recommendation.model.bandit import BanditPolicy, BANDIT_POLICIES
from recommendation.task.data_preparation.ifood import CreateGroundTruthForInterativeEvaluation
from recommendation.task.model.matrix_factorization import MatrixFactorizationTraining


class BanditAgent(object):

    def __init__(self, bandit: BanditPolicy) -> None:
        super().__init__()

        self.bandit = bandit

    def fit(self, trial: Trial, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        trial.with_generators(train_generator=train_loader, val_generator=val_loader).run(epochs=epochs)

        self.bandit.fit(train_loader.dataset)

    def act(self, arm_indices: List[List[int]], arm_contexts: List[Tuple[np.ndarray, ...]]) -> np.ndarray:
        pass


class IteractionTraining(MatrixFactorizationTraining):
    obs_batch_size: int = luigi.IntParameter(default=2000)
    filter_dish: str = luigi.Parameter(default="all")

    num_episodes: int = luigi.IntParameter(default=1)
    full_refit: bool = luigi.BoolParameter(default=False)

    bandit_policy: str = luigi.ChoiceParameter(choices=BANDIT_POLICIES.keys(), default="model")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})

    def requires(self):
        return CreateGroundTruthForInterativeEvaluation(minimum_interactions=self.minimum_interactions,
                                                        filter_dish=self.filter_dish)

    def create_agent(self) -> BanditAgent:
        bandit = BANDIT_POLICIES[self.bandit_policy](reward_model=self.create_module(), **self.bandit_policy_params)
        return BanditAgent(bandit)

    def _get_arm_indices(self, ob: np.ndarray) -> List[List[int]]:
        pass

    def _create_arm_contexts(self, ob: np.ndarray) -> List[Tuple[np.ndarray, ...]]:
        pass

    @property
    def known_observations_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_known_observations_data_frame"):
            self._known_observations_data_frame = pd.DataFrame(columns=["account_idx", "merchant_idx", "buy"])
        return self._known_observations_data_frame

    def _accumulate_known_observations(self, ob: np.ndarray, action: np.ndarray, reward: np.ndarray):
        pass

    @property
    def train_data_frame(self) -> pd.DataFrame:
        pass

    @property
    def val_data_frame(self) -> pd.DataFrame:
        pass

    def _reset_dataset(self):
        del self._train_dataset
        del self._val_dataset

    @property
    def n_users(self) -> int:
        pass

    @property
    def n_items(self) -> int:
        pass

    def run(self):
        df = pd.read_parquet(self.input().path)
        env: IFoodRecSysEnv = gym.make('ifood-recsys-v0', dataset=df, obs_batch_size=self.obs_batch_size)

        env.seed(0)

        agent: BanditAgent = None
        if not self.full_refit:
            agent = self.create_agent()

        rewards = []
        interactions = 0
        done = False

        for i in range(self.num_episodes):
            ob = env.reset()
            while True:
                interactions += len(ob)

                if self.full_refit:
                    agent = self.create_agent()

                action = agent.act(self._get_arm_indices(ob), self._create_arm_contexts(ob))
                ob, reward, done, info = env.step(action)
                rewards.append(reward)
                if done:
                    break

                self._accumulate_known_observations(ob, action, reward)
                self._reset_dataset()
                agent.fit(self.create_trial(agent.bandit.reward_model), self.get_train_generator(),
                          self.get_val_generator(), self.epochs)

                print(interactions)
                print(np.mean(reward), np.std(reward))

        env.close()
