import abc
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Union, Optional
import os
import gym
import luigi
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torchbearer import Trial
import pytz
from tqdm import tqdm

from recommendation.task.model.base import BaseTorchModelTraining

tqdm.pandas()
from recommendation.gym.envs import IFoodRecSysEnv
from recommendation.model.bandit import BanditPolicy, BANDIT_POLICIES
from recommendation.task.data_preparation.ifood import CreateGroundTruthForInterativeEvaluation, \
    GenerateIndicesForAccountsAndMerchantsDataset, AddAdditionallInformationDataset, ProcessRestaurantContentDataset
from recommendation.torch import NoAutoCollationDataLoader
from recommendation.utils import get_scores_per_tuples_with_click_timestamp, datetime_to_shift
from recommendation.files import get_interaction_dir


class BanditAgent(object):

    def __init__(self, bandit: BanditPolicy) -> None:
        super().__init__()

        self.bandit = bandit

    def fit(self, trial: Optional[Trial], train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self.bandit.fit(train_loader.dataset)

        if trial:
            trial.with_generators(train_generator=train_loader, val_generator=val_loader).run(epochs=epochs)

    def act(self, batch_of_arm_indices: List[List[int]], 
                  batch_of_arm_context: Tuple[np.ndarray, ...],
                  batch_of_arm_scores: Optional[List[List[float]]]) -> np.ndarray:

        if batch_of_arm_scores:
            return np.array([self.bandit.select(arm_indices, arm_contexts=arm_contexts, arm_scores=arm_scores)
                        for arm_indices, arm_contexts, arm_scores in zip(batch_of_arm_indices, batch_of_arm_context,
                                                                         batch_of_arm_scores)])
        else:
            return np.array([self.bandit.select(arm_indices, arm_contexts=arm_contexts)
                             for arm_indices, arm_contexts in zip(batch_of_arm_indices, batch_of_arm_context)])


class InteractionTraining(BaseTorchModelTraining, metaclass=abc.ABCMeta):
    project              = "ifood_contextual_bandit"
    test_size            = 0.0

    obs_batch_size: int = luigi.IntParameter(default=10000)
    filter_dish:    str = luigi.Parameter(default="all")

    num_episodes:   int = luigi.IntParameter(default=1)
    full_refit:    bool = luigi.BoolParameter(default=False)

    bandit_policy: str = luigi.ChoiceParameter(choices=BANDIT_POLICIES.keys(), default="model")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})

    def requires(self):
        return CreateGroundTruthForInterativeEvaluation(minimum_interactions=self.minimum_interactions,
                                                        filter_dish=self.filter_dish), \
               GenerateIndicesForAccountsAndMerchantsDataset(
                   test_size=self.test_size,
                   minimum_interactions=self.minimum_interactions,
                   sample_size=self.sample_size), \
               AddAdditionallInformationDataset(
                   test_size=self.test_size,
                   sample_size=self.sample_size,
                   minimum_interactions=self.minimum_interactions),\
               ProcessRestaurantContentDataset()

    def output(self):
        return luigi.LocalTarget(get_interaction_dir(self.__class__, self.task_id))

    def create_agent(self) -> BanditAgent:
        bandit = BANDIT_POLICIES[self.bandit_policy](reward_model=self.create_module(), **self.bandit_policy_params)
        return BanditAgent(bandit)

    @property
    def shift_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_shift_data_frame"):
            self._shift_data_frame = pd.read_csv(self.input()[2][0].path)

        return self._shift_data_frame

    @property
    def metadata_data_frame_path(self) -> str:
        return self.input()[1][1].path

    @property
    def account_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_account_data_frame"):
            self._account_data_frame = pd.read_csv(self.input()[1][0].path)

        return self._account_data_frame

    @property
    def merchant_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_merchant_data_frame"):
            self._merchant_data_frame = pd.read_csv(self.input()[1][1].path)

        return self._merchant_data_frame

    @property
    def availability_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_availability_data_frame"):
            df    = pd.read_parquet(self.input()[2][1].path)

            # Merge with merchant availability information and merchant Ground-truth 
            df = df.merge(self.merchant_data_frame[["merchant_id", "merchant_idx", 'dish_description']], on="merchant_id")
            
            if self.filter_dish != "all":
                dish_df = pd.read_csv(self.input()[3][2].path)
                dish_df = dish_df[dish_df.dish_description == self.filter_dish]
                df      = df[df.dish_description.isin(dish_df['id'].values)]

            df["open_time"]               = df["open"].apply(datetime.time)
            df["close_time"]              = df["close"].apply(datetime.time)
            df["open_time_minus_30_min"]  = df["open"].apply(lambda dt: datetime.time(dt - timedelta(minutes=30)))
            df["close_time_plus_30_min"]  = df["close"].apply(lambda dt: datetime.time(dt + timedelta(minutes=30)))

            self._availability_data_frame = df

        return self._availability_data_frame

    def _get_batch_of_arm_indices(self, ob: np.ndarray) -> List[List[int]]:
        df   = self.availability_data_frame
        tmz  = pytz.timezone("America/Sao_Paulo")

        days_of_week = [int(click_timestamp.strftime('%w')) for click_timestamp in ob[:, 1]]
        click_times  = [datetime.time(click_timestamp - timedelta(minutes=180)) for click_timestamp in ob[:, 1]]
        
        return [
            np.unique(
                df["merchant_idx"].values.tolist())
            for click_time, day_of_week in zip(click_times, days_of_week)
        ]

        # return [
        #     np.unique(
        #         df[
        #             (df["day_of_week"] == day_of_week) &
        #             ((df["open_time_minus_30_min"] <= click_time) | (df["open_time"] <= click_time)) &
        #             ((df["close_time_plus_30_min"] >= click_time) | (df["close_time"] >= click_time))
        #             ]["merchant_idx"].values.tolist())
        #     for click_time, day_of_week in zip(click_times, days_of_week)
        # ]

    def _get_scores_from_reward_model(self, agent: BanditAgent, ob: np.ndarray,
                                      batch_of_arm_indices: List[List[int]]) -> Dict[Tuple[int, int, datetime], float]:

        tuples_df, dataset = self._create_batch_dataset(ob, batch_of_arm_indices)
        generator = NoAutoCollationDataLoader(
                        dataset, batch_size=self.batch_size, shuffle=False,
                        num_workers=self.generator_workers,
                        pin_memory=self.pin_memory if self.device == "cuda" else False)
        
        trial = Trial(agent.bandit.reward_model, criterion=lambda *args: torch.zeros(1, device=self.torch_device,
                                                                                     requires_grad=True)) \
                    .with_test_generator(generator).to(self.torch_device).eval()

        with torch.no_grad():
            model_output: Union[torch.Tensor, Tuple[torch.Tensor]] = trial.predict(verbose=2)
        
        scores_tensor: torch.Tensor = model_output if isinstance(model_output, torch.Tensor) else model_output[0][0]
        scores: np.ndarray = scores_tensor.cpu().numpy().reshape(-1)
        
        
        return {
            (account_idx, merchant_idx, click_timestamp): score
            for account_idx, merchant_idx, click_timestamp, score in
            zip(tuples_df["account_idx"], tuples_df["merchant_idx"],
                tuples_df["click_timestamp"], scores)
        }

    def _get_batch_of_arm_scores(self, agent: BanditAgent, ob: np.ndarray,
                                 arm_indices: List[List[int]]) -> List[List[float]]:
        scores_per_tuple = self._get_scores_from_reward_model(agent, ob, arm_indices)

        return [get_scores_per_tuples_with_click_timestamp(account_idx, merchant_idx_list, click_timestamp,
                                                           scores_per_tuple)
                for account_idx, merchant_idx_list, click_timestamp in zip(ob[:, 0], arm_indices, ob[:, 1])]

    def _create_batch_dataset(self, ob: np.ndarray,
                                    batch_of_arm_indices: List[List[int]]) -> Dict[Tuple[int, int, datetime], float]:

        tuples_df = pd.DataFrame(columns=["account_idx", "merchant_idx", "click_timestamp"],
                                 data=[(account_idx, merchant_idx, click_timestamp)
                                        for account_idx, merchant_idx_list, click_timestamp
                                            in zip(ob[:, 0], batch_of_arm_indices, ob[:, 1])
                                                for merchant_idx in merchant_idx_list])

        if len(self.known_observations_data_frame) > 0:
            hist_count_df = self.known_observations_data_frame.groupby(["account_idx", "merchant_idx"])\
                                    .agg({"hist_visits": 'max', 'hist_buys': 'max'}).reset_index()
            tuples_df     = tuples_df.merge(hist_count_df, how='left', on=["account_idx", "merchant_idx"]).fillna(0)
        else:
            tuples_df["hist_visits"] = 0
            tuples_df["hist_buys"]   = 0

        # Add shift information
        tuples_df['shift'] = tuples_df.click_timestamp.apply(lambda c: datetime_to_shift(c - timedelta(minutes=180))) 
        tuples_df = tuples_df.merge(self.shift_data_frame, how='left', on='shift').fillna(0)


        if self.project_config.output_column.name not in tuples_df.columns:
            tuples_df[self.project_config.output_column.name] = 1
        for auxiliar_output_column in self.project_config.auxiliar_output_columns:
            if auxiliar_output_column.name not in tuples_df.columns:
                tuples_df[auxiliar_output_column.name] = 0
        
        dataset   = self.project_config.dataset_class(tuples_df, self.metadata_data_frame, self.project_config)

        return tuples_df, dataset

    def _create_arm_contexts(self, ob: np.ndarray,
                                    batch_of_arm_indices: List[List[int]]) -> Dict[Tuple[int, int, datetime], float]:
        

        tuples_df, dataset =self._create_batch_dataset(ob, batch_of_arm_indices)

        # _create_dictionary_of_dataset_indices
        dataset_indices_per_tuple = {(account_idx, merchant_idx): i for account_idx, merchant_idx, i
                                    in tqdm(zip(tuples_df["account_idx"], tuples_df["merchant_idx"],
                                                range(len(tuples_df))),
                                            total=len(tuples_df))}

        # create dataset per batch
        batch_contexts = []
        for account_idx, merchant_idx_list in zip(ob[:, 0], batch_of_arm_indices):
            dataset_indices = [dataset_indices_per_tuple[(account_idx, merchant_idx)] for merchant_idx in merchant_idx_list]
            batch_contexts.append(dataset[dataset_indices][0]) 

        return batch_contexts
        
    @property
    def known_observations_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_known_observations_data_frame"):
            df = pd.DataFrame(columns=["account_idx", "merchant_idx", "click_timestamp", "shift_idx", "ps", "buy"])
            df = df.astype({
                "account_idx": np.int,
                "merchant_idx": np.int,
                "shift_idx": np.int,
                "click_timestamp": np.datetime64,
                "buy": np.int,
            })
            df = df.set_index(["account_idx", "merchant_idx", "click_timestamp"], drop=False).sort_index()
            df.index.set_names(["index_account_idx", "index_merchant_idx", "index_click_timestamp"], inplace=True)
            self._known_observations_data_frame = df
        return self._known_observations_data_frame

    def _accumulate_known_observations(self, ob: np.ndarray, action: np.ndarray, reward: np.ndarray):
        df        = self.known_observations_data_frame
        df_append = pd.DataFrame(columns=["account_idx", "merchant_idx", "click_timestamp", "buy"],
                                data={"account_idx": ob[:, 0], "merchant_idx": action, "click_timestamp": ob[:, 1],
                                        "buy": reward})
        
        # Add shift information
        df_append['shift'] = df_append.click_timestamp.apply(lambda c: datetime_to_shift(c - timedelta(minutes=180))) 
        df_append = df_append.merge(self.shift_data_frame, on='shift')


        df = pd.concat([df, df_append]).reset_index(drop=True)
        
        df["hist_visits"]  = 1
        account_visits     = df.groupby(["account_idx"])["hist_visits"].transform("cumsum")
        df["hist_visits"]  = df.groupby(["account_idx", "merchant_idx"])["hist_visits"].transform("cumsum")
        df["hist_buys"]    = df.groupby(["account_idx", "merchant_idx"])["buy"].transform("cumsum")

        df["ps"]           = df["hist_visits"]/account_visits

        df['account_idx']  = df['account_idx'].astype(int)
        df['merchant_idx'] = df['merchant_idx'].astype(int)
        

        self._known_observations_data_frame = df

    def _save_log(self) -> None:
        df = self.known_observations_data_frame.reset_index()
        df = df[["account_idx", "merchant_idx", "buy"]]
        df.columns = ['context', 'arm', 'reward']

        df.reset_index().to_csv(self.output().path+'/data_log.csv', index=False)

    def _save_metrics(self) -> None:
        df = self.known_observations_data_frame.reset_index()
        df[["buy"]].describe().to_csv(self.output().path+'/stats.csv', index=False)

    @property
    def train_data_frame(self) -> pd.DataFrame:
        return self._train_data_frame

    @property
    def val_data_frame(self) -> pd.DataFrame:
        return self._val_data_frame

    @property
    def test_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame()

    def _reset_dataset(self):
        self._train_data_frame, self._val_data_frame = train_test_split(
            self.known_observations_data_frame, test_size=self.val_size, random_state=self.seed)
        if hasattr(self, "_train_dataset"):
            del self._train_dataset
        if hasattr(self, "_val_dataset"):
            del self._val_dataset

    @property
    def n_users(self) -> int:
        if not hasattr(self, "_n_users"):
            self._n_users = len(self.account_data_frame)
        return self._n_users

    @property
    def n_items(self) -> int:
        if not hasattr(self, "_n_items"):
            self._n_items = len(self.merchant_data_frame)
        return self._n_items

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)
        self._save_params()
                
        self.ground_truth_data_frame = pd.read_parquet(self.input()[0].path)
        env: IFoodRecSysEnv = gym.make('ifood-recsys-v0', dataset=self.ground_truth_data_frame,
                                       obs_batch_size=self.obs_batch_size)
        env.seed(42)

        agent: BanditAgent = self.create_agent()

        rewards      = []
        interactions = 0
        done         = False
        k            = 0
        for i in range(self.num_episodes):
            ob = env.reset()

            while True:
                interactions += len(ob)

                batch_of_arm_indices = self._get_batch_of_arm_indices(ob)
                batch_of_arm_context = self._create_arm_contexts(ob, batch_of_arm_indices)
                batch_of_arm_scores  = self._get_batch_of_arm_scores(agent, ob, batch_of_arm_indices) \
                    if agent.bandit.reward_model else None

                action = agent.act(batch_of_arm_indices, batch_of_arm_context, batch_of_arm_scores)

                new_ob, reward, done, info = env.step(action)
                rewards.extend(reward)
                self._accumulate_known_observations(ob, action, reward)

                if done:
                    break

                ob = new_ob
                self._reset_dataset()

                if agent.bandit.reward_model and self.full_refit:
                    agent.bandit.reward_model = self.create_module()

                agent.fit(self.create_trial(agent.bandit.reward_model) if agent.bandit.reward_model else None,
                          self.get_train_generator(),
                          self.get_val_generator(),
                          self.epochs)

                
                print("\n", k,": Interaction Stats")
                print(self.known_observations_data_frame[['buy']].describe().transpose(), "\n")
                #print(k, "===>", interactions, np.mean(rewards), np.sum(rewards))
                k+=1
                self._save_log()

        env.close()

        # Save logs

        self._save_log()
        self._save_metrics()