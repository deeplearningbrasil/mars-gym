import abc
import os
from typing import List, Tuple, Union, Type, Any

import functools
import gym
import luigi
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torchbearer
from torchbearer import Trial
from tqdm import tqdm
import time
import pickle
import gc
from mars_gym.data.dataset import preprocess_interactions_data_frame
from mars_gym.model.agent import BanditAgent
from mars_gym.model.bandit import BanditPolicy
from mars_gym.simulation.training import (
    TORCH_LOSS_FUNCTIONS,
    SupervisedModelTraining,
)
from mars_gym.utils.index_mapping import transform_with_indexing, map_array
from mars_gym.utils.plot import plot_history, plot_scores
from mars_gym.utils.reflection import load_attr

tqdm.pandas()
from mars_gym.gym.envs import RecSysEnv
from mars_gym.utils.files import (
    get_interaction_dir,
    get_history_path,
)
from mars_gym.utils.files import (
    get_simulator_datalog_path,
    get_interator_datalog_path,
    get_ground_truth_datalog_path,
)
from mars_gym.utils.utils import save_trained_data


# from IPython import embed; embed()


class InteractionTraining(SupervisedModelTraining, metaclass=abc.ABCMeta):
    loss_function: str = luigi.ChoiceParameter(
        choices=TORCH_LOSS_FUNCTIONS.keys(), default="crm"
    )
    test_size: float = luigi.FloatParameter(default=0.2)
    test_split_type: str = luigi.ChoiceParameter(
        choices=["random", "time"], default="time"
    )
    val_split_type: str = luigi.ChoiceParameter(
        choices=["random", "time"], default="time"
    )
    crm_ps_strategy: str = luigi.ChoiceParameter(
        choices=["bandit", "dataset"], default="bandit"
    )

    obs_batch_size: int = luigi.IntParameter(default=1000)
    num_episodes: int = luigi.IntParameter(default=1)
    sample_size: int = luigi.IntParameter(default=-1)
    full_refit: bool = luigi.BoolParameter(default=False)
    output_model_dir: str = luigi.Parameter(default="")

    def create_agent(self) -> BanditAgent:
        bandit_class = load_attr(self.bandit_policy_class, Type[BanditPolicy])
        bandit = bandit_class(
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

        if self.project_config.available_arms_column_name is None:
            n = 1
        else:
            n = len(ob[self.project_config.available_arms_column_name])
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

        n = len(ob[self.project_config.available_arms_column_name])
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

        # ----
        df[ps_column] = ps_value

    # def _calcule_propensity_score(self, df) -> None:
    def _save_result(self) -> None:
        print("Saving logs...")

        self._save_params()
        self._save_log()
        self._save_metrics()
        self._save_bandit_model()

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
        if (
            self.project_config.propensity_score_column_name
            not in self.interactions_data_frame
        ):
            self.interactions_data_frame[
                self.project_config.propensity_score_column_name
            ] = None

        gt_df = self.interactions_data_frame[columns].reset_index()

        sim_df.to_csv(get_simulator_datalog_path(self.output().path), index=False)
        gt_df.to_csv(get_interator_datalog_path(self.output().path), index=False)
        env_data_df.to_csv(
            get_ground_truth_datalog_path(self.output().path), index=False
        )

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

    def get_data_frame_for_indexing(self) -> pd.DataFrame:
        return self.interactions_data_frame

    @property
    def interactions_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_interactions_data_frame"):
            data = pd.concat(
                [
                    pd.read_csv(self.train_data_frame_path),
                    pd.read_csv(self.val_data_frame_path),
                ],
                ignore_index=True,
            )
            if self.sample_size > 0:
                data = data[-self.sample_size :]

            self._interactions_data_frame = preprocess_interactions_data_frame(
                data, self.project_config,
            )
            self._interactions_data_frame.sort_values(
                self.project_config.timestamp_column_name
            ).reset_index(drop=True)

        # Needed in case index_mapping was invoked before
        if not hasattr(self, "_creating_index_mapping") and not hasattr(
            self, "_interactions_data_frame_indexed"
        ):
            transform_with_indexing(
                self._interactions_data_frame, self.index_mapping, self.project_config
            )
            self._interactions_data_frame_indexed = True
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
    def env_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_env_data_frame"):
            env_columns = self.obs_columns + [self.project_config.item_column.name]
            if self.project_config.available_arms_column_name:
                env_columns += [self.project_config.available_arms_column_name]

            df = self.interactions_data_frame.loc[
                self.interactions_data_frame[self.project_config.output_column.name]
                == 1,
                env_columns,
            ]

            if self.project_config.available_arms_column_name:
                df[self.project_config.available_arms_column_name] = df[
                    self.project_config.available_arms_column_name
                ].map(
                    functools.partial(
                        map_array,
                        mapping=self.index_mapping[
                            self.project_config.item_column.name
                        ],
                    )
                )
            self._env_data_frame = df
        return self._env_data_frame

    def _print_hist(self):
        stats = pd.concat(
            [
                self.known_observations_data_frame[
                    [self.project_config.output_column.name]
                ]
                .describe()
                .transpose(),
                self._train_data_frame[[self.project_config.output_column.name]]
                .describe()
                .transpose(),
                self._val_data_frame[[self.project_config.output_column.name]]
                .describe()
                .transpose(),
            ]
        )
        stats["dataset"] = ["all", "train", "valid"]
        stats = stats.set_index("dataset")

        percent = len(self.known_observations_data_frame) / (len(self.env_data_frame) * self.num_episodes)

        print("\nInteraction Stats ({}%)".format(np.round(percent * 100, 2)))
        print(stats[["count", "mean", "std"]], "\n")

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)
        self.start_time = time.time()

        self._save_params()
        print("DataFrame: env_data_frame, ", self.env_data_frame.shape)
        print("DataFrame: interactions_data_frame, ", self.interactions_data_frame.shape)

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

                if self.project_config.available_arms_column_name in ob:
                    # The Env returns a binary array to be compatible with OpenAI Gym API but the actual items are needed
                    ob[self.project_config.available_arms_column_name] = [
                        self.reverse_index_mapping[self.project_config.item_column.name][index]
                        for index in np.flatnonzero(
                            ob[self.project_config.available_arms_column_name]
                        ).tolist()
                    ]
                #from IPython import embed; embed()
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
                    self._print_hist()

                    # print(k, "===>", interactions, np.mean(rewards), np.sum(rewards))
                    k += 1
                    self._save_log()

        self.env.close()
        self.end_time = time.time()
        # Save logs
        self._save_result()
