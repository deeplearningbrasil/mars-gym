import abc
import os
from typing import List, Tuple, Dict, Any, Union, Optional

import gym
import luigi
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchbearer import Trial
from tqdm import tqdm

from recommendation.task.model.base import BaseTorchModelTraining

tqdm.pandas()
from recommendation.gym.envs import RecSysEnv
from recommendation.model.bandit import BanditPolicy, BANDIT_POLICIES
from recommendation.torch import NoAutoCollationDataLoader, FasterBatchSampler
from recommendation.files import get_interaction_dir


class BanditAgent(object):

    def __init__(self, bandit: BanditPolicy) -> None:
        super().__init__()

        self.bandit = bandit

    def fit(self, trial: Optional[Trial], train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self.bandit.fit(train_loader.dataset)

        if trial:
            trial.with_generators(train_generator=train_loader, val_generator=val_loader).run(epochs=epochs)

    def act(self, arm_indices: List[int],
            arm_contexts: Tuple[np.ndarray, ...],
            arm_scores: Optional[List[float]]) -> int:
        return self.bandit.select(arm_indices, arm_contexts=arm_contexts, arm_scores=arm_scores)


class InteractionTraining(BaseTorchModelTraining, metaclass=abc.ABCMeta):
    test_size = 0.0

    obs_batch_size: int = luigi.IntParameter(default=10000)

    num_episodes: int = luigi.IntParameter(default=1)
    full_refit: bool = luigi.BoolParameter(default=False)

    bandit_policy: str = luigi.ChoiceParameter(choices=BANDIT_POLICIES.keys(), default="model")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})

    def output(self):
        return luigi.LocalTarget(get_interaction_dir(self.__class__, self.task_id))

    def create_agent(self) -> BanditAgent:
        bandit = BANDIT_POLICIES[self.bandit_policy](reward_model=self.create_module(), **self.bandit_policy_params)
        return BanditAgent(bandit)

    @property
    def obs_columns(self) -> List[str]:
        return [self.project_config.user_column.name] + [
            column.name for column in self.project_config.other_input_columns]

    def _get_arm_indices(self, ob: np.ndarray) -> List[int]:
        return self.unique_items

    def _get_arm_scores(self, agent: BanditAgent, ob_dataset: Dataset) -> List[float]:
        batch_sampler = FasterBatchSampler(ob_dataset, self.batch_size, shuffle=False)
        generator     = NoAutoCollationDataLoader(
                            ob_dataset,
                            batch_sampler=batch_sampler, num_workers=self.generator_workers,
                            pin_memory=self.pin_memory if self.device == "cuda" else False)

        trial = Trial(agent.bandit.reward_model, criterion=lambda *args: torch.zeros(1, device=self.torch_device,
                                                                                     requires_grad=True)) \
            .with_test_generator(generator).to(self.torch_device).eval()

        with torch.no_grad():
            model_output: Union[torch.Tensor, Tuple[torch.Tensor]] = trial.predict(verbose=2)

        scores_tensor: torch.Tensor = model_output if isinstance(model_output, torch.Tensor) else model_output[0][0]
        scores: List[float] = scores_tensor.cpu().numpy().reshape(-1).tolist()

        return scores

    def _create_ob_dataset(self, ob: np.ndarray, arm_indices: List[int]) -> Dataset:
        ob_df = pd.DataFrame(
            columns=self.obs_columns + [self.project_config.item_column.name],
            data=np.array([np.append(ob, arm_index) for arm_index in arm_indices]))

        if len(self.known_observations_data_frame) > 0:
            ob_df = ob_df.drop(columns=[self.project_config.hist_view_column_name,
                                        self.project_config.hist_output_column_name])
            hist_count_df = self.known_observations_data_frame.groupby(
                [self.project_config.user_column.name, self.project_config.item_column.name]) \
                .agg({self.project_config.hist_view_column_name: 'max',
                      self.project_config.hist_output_column_name: 'max'}).reset_index()
            ob_df = ob_df.merge(hist_count_df, how='left', on=[self.project_config.user_column.name,
                                                               self.project_config.item_column.name]).fillna(0)
        else:
            ob_df[self.project_config.hist_view_column_name] = 0
            ob_df[self.project_config.hist_output_column_name] = 0

        if self.project_config.output_column.name not in ob_df.columns:
            ob_df[self.project_config.output_column.name] = 1
        for auxiliar_output_column in self.project_config.auxiliar_output_columns:
            if auxiliar_output_column.name not in ob_df.columns:
                ob_df[auxiliar_output_column.name] = 0

        dataset = self.project_config.dataset_class(ob_df, self.metadata_data_frame, self.project_config)

        return dataset

    @property
    def known_observations_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_known_observations_data_frame"):
            columns = self.obs_columns + [self.project_config.item_column.name,
                                            self.project_config.output_column.name]

            self._known_observations_data_frame = pd.DataFrame(
                columns=columns)\
                        .astype(self.interactions_data_frame[columns].dtypes)
            
        return self._known_observations_data_frame

    def _accumulate_known_observations(self, ob: np.ndarray, action: int, reward: float):
        data      = np.expand_dims(np.append(ob, np.array([action, reward])), axis=0)

        columns   = self.obs_columns + [self.project_config.item_column.name, self.project_config.output_column.name]
        df_append = pd.DataFrame(
                    columns=columns,
                    data=data).astype(self.known_observations_data_frame[columns].dtypes)

        df = pd.concat([self.known_observations_data_frame, df_append], ignore_index=True)

        self._create_hist_columns(df)
        self._known_observations_data_frame = df

    def _create_hist_columns(self, df: pd.DataFrame):
        user_column             = self.project_config.user_column.name
        item_column             = self.project_config.item_column.name
        output_column           = self.project_config.output_column.name
        hist_view_column        = self.project_config.hist_view_column_name
        hist_output_column      = self.project_config.hist_output_column_name
        ps_column               = self.project_config.propensity_score_column_name

        df[hist_view_column]    = 1
        user_views              = df.groupby([user_column])[hist_view_column].transform("cumsum")
        df[hist_view_column]    = df.groupby([user_column, item_column])[hist_view_column].transform("cumsum")
        df[hist_output_column]  = df.groupby([user_column, item_column])[output_column].transform("cumsum")

        df[ps_column]           = df[hist_view_column] / user_views


    def _save_log(self) -> None:
        columns = [self.project_config.user_column.name, self.project_config.item_column.name,
                  self.project_config.output_column.name, self.project_config.propensity_score_column_name]

        # Save DataLog
        df = self.known_observations_data_frame.reset_index()
        df = df[columns]
        df.columns = ['user', 'item', 'reward', 'ps']

        df.reset_index().to_csv(self.output().path + '/sim-datalog.csv', index=False)

        gt_df = self.interactions_data_frame[columns]
        gt_df.reset_index().to_csv(self.output().path + '/gt-datalog.csv', index=False)

        # Train
        #history_df = pd.read_csv(get_history_path(self.output().path))
        #plot_history(history_df).savefig(os.path.join(self.output().path, "history.jpg"))

    def _save_metrics(self) -> None:
        df = self.known_observations_data_frame.reset_index()
        df[[self.project_config.output_column.name]].describe().to_csv(self.output().path + '/stats.csv', index=False)

    @property
    def interactions_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_interactions_data_frame"):
            self._interactions_data_frame = pd.concat([pd.read_csv(self.train_data_frame_path),
                                                       pd.read_csv(self.val_data_frame_path)])
            self._interactions_data_frame.sort_values(self.project_config.timestamp_column_name)

        return self._interactions_data_frame

    @property
    def train_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_train_data_frame"):
            self._train_data_frame = self.interactions_data_frame.sample(1)
        return self._train_data_frame

    @property
    def val_data_frame(self) -> pd.DataFrame:
        return self._val_data_frame

    @property
    def test_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame()

    def _reset_dataset(self):
        self._train_data_frame, self._val_data_frame = train_test_split(
                self.known_observations_data_frame, test_size=self.val_size, 
                random_state=self.seed, stratify=self.known_observations_data_frame[self.project_config.output_column.name] 
                    if np.sum(self.known_observations_data_frame[self.project_config.output_column.name]) > 1 else None)

        if hasattr(self, "_train_dataset"):
            del self._train_dataset
        
        if hasattr(self, "_val_dataset"):
            del self._val_dataset

    @property
    def n_users(self) -> int:
        if not hasattr(self, "_n_users"):
            self._n_users = self.interactions_data_frame[self.project_config.user_column.name].max() + 1
        return self._n_users

    @property
    def unique_items(self) -> List[int]:
        if not hasattr(self, "_unique_items"):
            self._unique_items = self.interactions_data_frame[self.project_config.item_column.name].unique()
        return self._unique_items

    @property
    def n_items(self) -> int:
        if not hasattr(self, "_n_items"):
            self._n_items = self.interactions_data_frame[self.project_config.item_column.name].max() + 1
        return self._n_items

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)
        self._save_params()

        env_dataset = self.interactions_data_frame.loc[
                        self.interactions_data_frame[self.project_config.output_column.name] == 1, self.obs_columns + [
                        self.project_config.item_column.name]]
        env: RecSysEnv = gym.make('recsys-v0', dataset=env_dataset, item_column=self.project_config.item_column.name)
        env.seed(42)

        agent: BanditAgent = self.create_agent()

        rewards = []
        interactions = 0
        k = 0
        for i in range(self.num_episodes):
            ob = env.reset()

            while True:
                interactions += 1

                arm_indices = self._get_arm_indices(ob)
                ob_dataset = self._create_ob_dataset(ob, arm_indices)
                arm_contexts = ob_dataset[:len(ob_dataset)][0]
                arm_scores = self._get_arm_scores(agent, ob_dataset) if agent.bandit.reward_model else None
                action = agent.act(arm_indices, arm_contexts, arm_scores)

                new_ob, reward, done, info = env.step(action)
                rewards.append(reward)
                self._accumulate_known_observations(ob, action, reward)

                if done:
                    break

                ob = new_ob

                if interactions % self.obs_batch_size == 0:
                    self._reset_dataset()

                    if agent.bandit.reward_model and self.full_refit:
                        agent.bandit.reward_model = self.create_module()

                    agent.fit(self.create_trial(agent.bandit.reward_model) if agent.bandit.reward_model else None,
                              self.get_train_generator(),
                              self.get_val_generator(),
                              self.epochs)

                    print("\n", k, ": Interaction Stats")
                    print(
                        self.known_observations_data_frame[
                            [self.project_config.output_column.name]].describe().transpose(),
                        "\n")
                    if hasattr(self, "train_data_frame"):
                        print(
                            self._train_data_frame[[self.project_config.output_column.name]].describe().transpose(),
                            "\n")
                    if hasattr(self, "val_data_frame"):
                        print(
                            self._val_data_frame[[self.project_config.output_column.name]].describe().transpose(),
                            "\n")

                    # print(k, "===>", interactions, np.mean(rewards), np.sum(rewards))
                    k += 1
                    self._save_log()

        env.close()

        # Save logs

        self._save_log()
        self._save_metrics()
