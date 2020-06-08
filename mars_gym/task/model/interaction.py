import abc
import os
from typing import List, Tuple, Dict, Any, Union, Optional

import gym
import luigi
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torchbearer
from torchbearer import Trial
from tqdm import tqdm
import time
import pickle
from gym import spaces
import gc
from mars_gym.data import preprocess_interactions_data_frame
from mars_gym.gym.envs.recsys import ITEM_METADATA_KEY
from mars_gym.model.agent import BanditAgent
from mars_gym.task.meta_config import ProjectConfig, Column, IOType
from mars_gym.task.model.base import (
    BaseTorchModelTraining,
    TORCH_LOSS_FUNCTIONS,
    BaseTorchModelWithAgentTraining,
)
from mars_gym.utils.plot import plot_history, plot_scores

tqdm.pandas()
from mars_gym.gym.envs import RecSysEnv
from mars_gym.model.bandit import BANDIT_POLICIES
from mars_gym.torch.torch import NoAutoCollationDataLoader, FasterBatchSampler
from mars_gym.utils.files import (
    get_interaction_dir,
    get_test_set_predictions_path,
    get_history_path,
)
from mars_gym.utils.files import (
    get_simulator_datalog_path,
    get_interator_datalog_path,
    get_ground_truth_datalog_path,
)
from mars_gym.utils.utils import save_trained_data

# from IPython import embed; embed()


class InteractionTraining(BaseTorchModelWithAgentTraining, metaclass=abc.ABCMeta):
    loss_function: str = luigi.ChoiceParameter(
        choices=TORCH_LOSS_FUNCTIONS.keys(), default="crm"
    )
    test_size: float = luigi.FloatParameter(default=0.0)
    test_split_type: str = luigi.ChoiceParameter(
        choices=["random", "time"], default="time"
    )
    val_split_type: str = luigi.ChoiceParameter(
        choices=["random", "time"], default="time"
    )
    crm_ps_strategy: str = luigi.ChoiceParameter(
        choices=["bandit", "dataset"], default="bandit"
    )

    obs_batch_size: int = luigi.IntParameter(default=10000)
    num_episodes: int = luigi.IntParameter(default=1)
    full_refit: bool = luigi.BoolParameter(default=False)
    output_model_dir: str = luigi.Parameter(default="")

    def create_agent(self) -> BanditAgent:
        bandit = BANDIT_POLICIES[self.bandit_policy](
            reward_model=self.create_module(), **self.bandit_policy_params
        )
        return BanditAgent(bandit)

    def output(self):
        return luigi.LocalTarget(get_interaction_dir(self.__class__, self.task_id))

    @property
    def known_observations_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_known_observations_data_frame"):
            columns = self.obs_columns + [
                self.project_config.item_column.name,
                self.project_config.output_column.name,
            ]

            self._known_observations_data_frame = pd.DataFrame(columns=columns).astype(
                self.interactions_data_frame[columns].dtypes
            )

        return self._known_observations_data_frame

    @property
    def hist_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_hist_data_frame"):
            self._hist_data_frame = pd.DataFrame(
                columns=[
                    self.project_config.user_column.name,
                    self.project_config.item_column.name,
                    self.project_config.hist_view_column_name,
                    self.project_config.hist_output_column_name,
                ],
                dtype=np.int,
            ).set_index(
                [
                    self.project_config.user_column.name,
                    self.project_config.item_column.name,
                ]
            )
        return self._hist_data_frame

    def _fill_hist_columns(self, ob_df: pd.DataFrame) -> pd.DataFrame:
        if len(self.known_observations_data_frame) > 0:
            ob_df = ob_df.drop(
                columns=[
                    self.project_config.hist_view_column_name,
                    self.project_config.hist_output_column_name,
                ],
                errors="ignore",
            )
            ob_df = ob_df.merge(
                self.hist_data_frame,
                how="left",
                left_on=[
                    self.project_config.user_column.name,
                    self.project_config.item_column.name,
                ],
                right_index=True,
            ).fillna(0)
        else:
            ob_df[self.project_config.hist_view_column_name] = 0
            ob_df[self.project_config.hist_output_column_name] = 0
        return ob_df

    def _accumulate_known_observations(
        self, ob: dict, action: int, prob: float, reward: float
    ):
        user_column = self.project_config.user_column.name
        item_column = self.project_config.item_column.name
        output_column = self.project_config.output_column.name
        hist_view_column = self.project_config.hist_view_column_name
        hist_output_column = self.project_config.hist_output_column_name
        ps_column = self.project_config.propensity_score_column_name

        if self.crm_ps_strategy == "bandit":
            ps_val = self._calulate_propensity_score(ob, prob)
        else:
            ps_val = self._calulate_propensity_score_with_probs(ob, action)

        new_row = {**ob, item_column: action, output_column: reward, ps_column: ps_val}

        self._known_observations_data_frame = self.known_observations_data_frame.append(
            new_row, ignore_index=True
        )

        user_index = ob[user_column]
        if (user_index, action) not in self.hist_data_frame.index:
            self.hist_data_frame.loc[(user_index, action), hist_view_column] = 1
            self.hist_data_frame.loc[(user_index, action), hist_output_column] = int(
                reward
            )
        else:
            self.hist_data_frame.loc[(user_index, action), hist_view_column] += 1
            self.hist_data_frame.loc[(user_index, action), hist_output_column] += int(
                reward
            )

    def _calulate_propensity_score(self, ob: dict, prob: float) -> float:
        df = self.known_observations_data_frame

        n = np.sum(
            ob[self.project_config.available_arms_column_name]
        )  # Binary Array [0,0,1,0,0,1...]
        prob += 0.001  # error
        ps = (1 / n) / prob
        return ps

    def _calulate_propensity_score_with_probs(self, ob: dict, action: int):
        df = self.known_observations_data_frame
        try:
            prob = self._known_observations_data_frame.item_idx.value_counts(
                normalize=True
            )[action]
        except IndexError:
            prob = 0
        except KeyError:
            prob = 0

        n = np.sum(
            ob[self.project_config.available_arms_column_name]
        )  # Binary Array [0,0,1,0,0,1...]
        prob += 0.001  # error
        ps = (1 / n) / prob

        return ps

    def _create_hist_columns(self):
        df = self.known_observations_data_frame

        user_column = self.project_config.user_column.name
        item_column = self.project_config.item_column.name
        output_column = self.project_config.output_column.name
        hist_view_column = self.project_config.hist_view_column_name
        hist_output_column = self.project_config.hist_output_column_name
        ps_column = self.project_config.propensity_score_column_name

        # Method 1
        # df[hist_view_column]   = 1
        # user_views             = df.groupby([user_column])[hist_view_column].transform("cumsum")
        # df[hist_view_column]   = df.groupby([user_column, item_column])[hist_view_column].transform("cumsum")
        # df[hist_output_column] = df.groupby([user_column, item_column])[output_column].transform("cumsum")
        # ps_value               = df[hist_view_column] / user_views

        # Method 2
        # ground_truth_df = df[df[self.project_config.output_column.name] == 1]
        # ps_per_pos_item_idx: Dict[int, float] = {
        #     pos_item_idx: np.sum(ground_truth_df["pos_item_idx"] == pos_item_idx) / len(ground_truth_df)
        #     for pos_item_idx in df["pos_item_idx"].unique()
        # }
        # ps_value = df["pos_item_idx"].apply(lambda pos_item_idx: ps_per_pos_item_idx[pos_item_idx] + 0.01)
        # from IPython import embed; embed()

        # ----
        df[ps_column] = ps_value

    # def _calcule_propensity_score(self, df) -> None:

    def _save_result(self) -> None:
        print("Saving logs...")

        self._save_params()
        self._save_log()
        self._save_metrics()
        # self._save_bandit_model()

        if self.test_size > 0:
            self._save_test_set_predictions(self.agent)

        if self.output_model_dir:
            save_trained_data(self.output().path, self.output_model_dir)

    def _save_bandit_model(self):
        # Save Bandit Object
        with open(os.path.join(self.output().path, "bandit.pkl"), "wb") as bandit_file:
            pickle.dump(self.agent.bandit, bandit_file)

    def _save_log(self) -> None:
        columns = [
            self.project_config.user_column.name,
            self.project_config.item_column.name,
            self.project_config.output_column.name,
            self.project_config.propensity_score_column_name,
        ]

        # Env Dataset
        env_data_df = self.env_data_frame.reset_index()
        env_data_duplicate_df = pd.concat(
            [env_data_df] * self.num_episodes, ignore_index=True
        )

        # Simulator Dataset
        sim_df = self.known_observations_data_frame.reset_index(drop=True)
        sim_df = sim_df[columns]
        sim_df.columns = ["user", "item", "reward", "ps"]
        sim_df["index_env"] = env_data_duplicate_df["index"]

        # All Dataset
        gt_df = self.interactions_data_frame[columns].reset_index()

        sim_df.to_csv(get_simulator_datalog_path(self.output().path), index=False)
        gt_df.to_csv(get_interator_datalog_path(self.output().path), index=False)
        env_data_df.to_csv(
            get_ground_truth_datalog_path(self.output().path), index=False
        )

        # Train
        # history_df = pd.read_csv(get_history_path(self.output().path))
        # plot_history(history_df).savefig(os.path.join(self.output().path, "history.jpg"))

    def _save_metrics(self) -> None:
        df = self.known_observations_data_frame.reset_index()
        df_metric = df[[self.project_config.output_column.name]].describe().transpose()
        df_metric["time"] = self.end_time - self.start_time

        df_metric.transpose().reset_index().to_csv(
            self.output().path + "/stats.csv", index=False
        )

    def _save_trial_log(self, i, trial) -> None:
        os.makedirs(os.path.join(self.output().path, "plot_history"), exist_ok=True)

        if trial:
            history_df = pd.read_csv(get_history_path(self.output().path))
            plot_history(history_df).savefig(
                os.path.join(
                    self.output().path, "plot_history", "history_{}.jpg".format(i)
                )
            )
            self._save_score_log(i, trial)

    def _save_score_log(self, i, trial) -> None:
        val_loader = self.get_val_generator()
        trial = (
            Trial(
                self.agent.bandit.reward_model,
                criterion=lambda *args: torch.zeros(
                    1, device=self.torch_device, requires_grad=True
                ),
            )
            .with_generators(val_generator=val_loader)
            .to(self.torch_device)
            .eval()
        )

        with torch.no_grad():
            model_output: Union[torch.Tensor, Tuple[torch.Tensor]] = trial.predict(
                verbose=0, data_key=torchbearer.VALIDATION_DATA
            )

        scores_tensor: torch.Tensor = model_output if isinstance(
            model_output, torch.Tensor
        ) else model_output[0][0]
        scores: List[float] = scores_tensor.cpu().numpy().reshape(-1).tolist()

        plot_scores(scores).savefig(
            os.path.join(self.output().path, "plot_history", "scores_{}.jpg".format(i))
        )

    @property
    def interactions_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_interactions_data_frame"):
            self._interactions_data_frame = preprocess_interactions_data_frame(
                pd.concat(
                    [
                        pd.read_csv(self.train_data_frame_path),
                        pd.read_csv(self.val_data_frame_path),
                    ],
                    ignore_index=True,
                ),
                self.project_config,
            )
            self._interactions_data_frame.sort_values(
                self.project_config.timestamp_column_name
            ).reset_index(drop=True)

        return self._interactions_data_frame

    @property
    def train_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_train_data_frame"):
            self._train_data_frame = self.interactions_data_frame.sample(1)
        return self._train_data_frame

    @property
    def val_data_frame(self) -> pd.DataFrame:
        return self._val_data_frame

    def _reset_dataset(self):
        # Random Split
        if self.val_split_type == "random":
            self._train_data_frame, self._val_data_frame = train_test_split(
                self.known_observations_data_frame,
                test_size=self.val_size,
                random_state=self.seed,
                stratify=self.known_observations_data_frame[
                    self.project_config.output_column.name
                ]
                if np.sum(
                    self.known_observations_data_frame[
                        self.project_config.output_column.name
                    ]
                )
                > 1
                else None,
            )
        else:
            # Time Split
            df = self.known_observations_data_frame
            size = len(df)
            cut = int(size - size * self.val_size)
            self._train_data_frame, self._val_data_frame = df.iloc[:cut], df.iloc[cut:]

        if hasattr(self, "_train_dataset"):
            del self._train_dataset

        if hasattr(self, "_val_dataset"):
            del self._val_dataset

    def clean(self):
        super().clean()
        if hasattr(self, "_interactions_data_frame"):
            del self._interactions_data_frame

        if hasattr(self, "_known_observations_data_frame"):
            del self._known_observations_data_frame

        gc.collect()

    @property
    def n_users(self) -> int:
        if not hasattr(self, "_n_users"):
            self._n_users = max(
                self.interactions_data_frame[self.project_config.user_column.name].max()
                + 1,
                self.test_data_frame[self.project_config.user_column.name].max() + 1,
            )
        return self._n_users

    @property
    def n_items(self) -> int:
        if not hasattr(self, "_n_items"):
            self._n_items = max(
                self.interactions_data_frame[self.project_config.item_column.name].max()
                + 1,
                self.test_data_frame[self.project_config.item_column.name].max() + 1,
            )
        return self._n_items

    @property
    def env_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_env_data_frame"):
            env_columns = self.obs_columns + [self.project_config.item_column.name]
            if self.project_config.available_arms_column_name:
                env_columns += [self.project_config.available_arms_column_name]

            self._env_data_frame = self.interactions_data_frame.loc[
                self.interactions_data_frame[self.project_config.output_column.name]
                == 1,
                env_columns,
            ]
        return self._env_data_frame

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)
        self.start_time = time.time()

        self._save_params()

        self.env: RecSysEnv = gym.make(
            "recsys-v0",
            dataset=self.env_data_frame,
            available_items_column=self.project_config.available_arms_column_name,
            item_column=self.project_config.item_column.name,
            number_of_items=self.interactions_data_frame[
                self.project_config.item_column.name
            ].max()
            + 1,
            item_metadata=self.embeddings_for_metadata,
        )
        self.env.seed(42)

        self.agent: BanditAgent = self.create_agent()

        rewards = []
        interactions = 0
        k = 0
        for i in range(self.num_episodes):
            ob = self.env.reset()

            while True:
                interactions += 1

                action, prob = self._act(self.agent, ob)

                new_ob, reward, done, info = self.env.step(action)
                rewards.append(reward)
                self._accumulate_known_observations(ob, action, prob, reward)

                if done:
                    break

                ob = new_ob

                if interactions % self.obs_batch_size == 0:
                    # self._create_hist_columns()
                    self._reset_dataset()

                    if self.agent.bandit.reward_model:
                        if self.full_refit:
                            self.agent.bandit.reward_model = self.create_module()

                        trial = self.create_trial(self.agent.bandit.reward_model)
                    else:
                        trial = None

                    self.agent.fit(
                        trial,
                        self.get_train_generator(),
                        self.get_val_generator(),
                        self.epochs,
                    )
                    self._save_trial_log(interactions, trial)

                    print("\n", k, ": Interaction Stats")
                    print(
                        self.known_observations_data_frame[
                            [self.project_config.output_column.name]
                        ]
                        .describe()
                        .transpose(),
                        "\n",
                    )
                    if hasattr(self, "train_data_frame"):
                        print(
                            self._train_data_frame[
                                [self.project_config.output_column.name]
                            ]
                            .describe()
                            .transpose(),
                            "\n",
                        )
                    if hasattr(self, "val_data_frame"):
                        print(
                            self._val_data_frame[
                                [self.project_config.output_column.name]
                            ]
                            .describe()
                            .transpose(),
                            "\n",
                        )

                    # print(k, "===>", interactions, np.mean(rewards), np.sum(rewards))
                    k += 1
                    self._save_log()

        self.env.close()
        self.end_time = time.time()
        # Save logs
        self._save_result()
