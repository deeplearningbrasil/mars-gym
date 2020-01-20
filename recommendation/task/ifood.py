import functools
import json
import math
import os
import pickle
import pprint
from itertools import starmap
from multiprocessing.pool import Pool
from time import time
from typing import Dict, Tuple, List, Any, Type, Union

import luigi
import numpy as np
import pandas as pd

import scipy
import torch
from sklearn import manifold
from torch.utils.data.dataset import Dataset
from torchbearer import Trial
from tqdm import tqdm
tqdm.pandas()
import gc

from recommendation.data import literal_eval_array_columns
from recommendation.model.bandit import BanditPolicy, EpsilonGreedy, LinUCB, RandomPolicy, ModelPolicy, \
    PercentileAdaptiveGreedy, AdaptiveGreedy, LinThompsonSampling
from recommendation.plot import plot_histogram, plot_tsne
from recommendation.offpolicy_metrics import DirectEstimator, eval_IPS, eval_CIPS, eval_SNIPS, eval_doubly_robust
from recommendation.rank_metrics import average_precision, ndcg_at_k, prediction_coverage_at_k, personalization_at_k, precision_at_k
from recommendation.task.data_preparation.ifood import PrepareIfoodIndexedOrdersTestData, \
    PrepareIfoodIndexedSessionTestData, \
    ListAccountMerchantTuplesForIfoodIndexedOrdersTestData, ProcessRestaurantContentDataset, \
    PrepareRestaurantContentDataset, \
    CreateInteractionDataset, GenerateIndicesForAccountsAndMerchantsDataset, \
    IndexAccountsAndMerchantsOfSessionTrainDataset, LoggingPolicyPsDataset,\
        DummyTask
from recommendation.task.evaluation import BaseEvaluationTask
from recommendation.torch import NoAutoCollationDataLoader
from recommendation.utils import chunks, parallel_literal_eval
from recommendation.task.model.contextual_bandits import DirectEstimatorTraining

_BANDIT_POLICIES: Dict[str, Type[BanditPolicy]] = dict(
    epsilon_greedy=EpsilonGreedy, lin_ucb=LinUCB, lin_ts=LinThompsonSampling, random=RandomPolicy,
    percentile_adaptive=PercentileAdaptiveGreedy, adaptive=AdaptiveGreedy, model=ModelPolicy, none=None)


def _get_scores_per_tuple(account_idx: int, merchant_idx_list: List[int],
                          scores_per_tuple: Dict[Tuple[int, int], float]) -> List[float]:
    return list(map(lambda merchant_idx: scores_per_tuple.get((account_idx, merchant_idx), -1.0), merchant_idx_list))


def _sort_merchants_by_tuple_score(account_idx: int, merchant_idx_list: List[int],
                                       scores_per_tuple: Dict[Tuple[int, int], float], limit: int = None) -> List[int]:
    scores = _get_scores_per_tuple(account_idx, merchant_idx_list, scores_per_tuple)
    ranked_list = [merchant_idx for _, merchant_idx in sorted(zip(scores, merchant_idx_list), reverse=True)]
    return ranked_list if limit is None else ranked_list[:limit]


def _sort_merchants_by_tuple_score_with_bandit_policy(account_idx: int, merchant_idx_list: List[int],
                                                      scores_per_tuple: Dict[Tuple[int, int], float],
                                                      dataset_indices_per_tuple: Dict[Tuple[int, int], int],
                                                      dataset: Dataset, bandit_policy: BanditPolicy,
                                                      limit: int = None) -> Tuple[List[int], List[float]]:
    scores = _get_scores_per_tuple(account_idx, merchant_idx_list, scores_per_tuple)
    
    if dataset is None or dataset_indices_per_tuple is None:
        arm_contexts = None
    else:
        dataset_indices = [dataset_indices_per_tuple[(account_idx, merchant_idx)] for merchant_idx in merchant_idx_list]
        arm_contexts    =  dataset[dataset_indices][0]

    return bandit_policy.rank(merchant_idx_list, arm_scores=scores, arm_contexts=arm_contexts, limit=limit,
                              with_probs=True)


def _get_scores_per_merchant(merchant_idx_list: List[int], scores_per_merchant: Dict[int, float]) -> List[float]:
    return list(map(lambda merchant_idx: scores_per_merchant[merchant_idx], merchant_idx_list))


def _sort_merchants_by_merchant_score(merchant_idx_list: List[int], scores_per_merchant: Dict[int, float]) -> List[int]:
    scores = _get_scores_per_merchant(merchant_idx_list, scores_per_merchant)
    return [merchant_idx for _, merchant_idx in sorted(zip(scores, merchant_idx_list), reverse=True)]


def _sort_merchants_by_merchant_score_with_bandit_policy(merchant_idx_list: List[int],
                                                         scores_per_merchant: Dict[int, float],
                                                         bandit_policy: BanditPolicy) -> List[int]:
    scores = _get_scores_per_merchant(merchant_idx_list, scores_per_merchant)
    return bandit_policy.rank(merchant_idx_list, arm_scores=scores, with_probs=True)


def _ps_policy_eval(relevance_list: List[int], prob_merchant_idx_list: List[int]) -> List[int]:
    return np.sum(np.array(relevance_list) * np.array(prob_merchant_idx_list[:len(relevance_list)]))


def _get_rhat_scores(relevance_list: List[int], scores_merchant_idx_list: List[int]) -> List[int]:
    return np.sum(np.array(relevance_list) * np.array(scores_merchant_idx_list[:len(relevance_list)]))


def _get_rhat_rewards(relevance_list: List[int], rewards_merchant_idx_list: List[int]) -> List[int]:
    return rewards_merchant_idx_list[0]


def _create_relevance_list(sorted_merchant_idx_list: List[int], ordered_merchant_idx: int) -> List[int]:
    return [1 if merchant_idx == ordered_merchant_idx else 0 for merchant_idx in sorted_merchant_idx_list]


def _generate_relevance_list(account_idx: int, ordered_merchant_idx: int, merchant_idx_list: List[int],
                             scores_per_tuple: Dict[Tuple[int, int], float]) -> List[int]:
    scores = list(map(lambda merchant_idx: scores_per_tuple.get((account_idx, merchant_idx), -1.0), merchant_idx_list))
    sorted_merchant_idx_list = [merchant_idx for _, merchant_idx in
                                sorted(zip(scores, merchant_idx_list), reverse=True)]
    return [1 if merchant_idx == ordered_merchant_idx else 0 for merchant_idx in sorted_merchant_idx_list]


def _generate_random_relevance_list(ordered_merchant_idx: int, merchant_idx_list: List[int]) -> List[int]:
    np.random.shuffle(merchant_idx_list)
    return _create_relevance_list(merchant_idx_list, ordered_merchant_idx)


def _generate_relevance_list_from_merchant_scores(ordered_merchant_idx: int, merchant_idx_list: List[int],
                                                  scores_per_merchant: Dict[int, float]) -> List[int]:
    scores = list(map(lambda merchant_idx: scores_per_merchant[merchant_idx], merchant_idx_list))
    sorted_merchant_idx_list = [merchant_idx for _, merchant_idx in
                                sorted(zip(scores, merchant_idx_list), reverse=True)]
    return [1 if merchant_idx == ordered_merchant_idx else 0 for merchant_idx in sorted_merchant_idx_list]



class SortMerchantListsForIfoodModel(BaseEvaluationTask):
    batch_size: int = luigi.IntParameter(default=100000)
    plot_histogram: bool = luigi.BoolParameter(default=False)
    bandit_policy: str = luigi.ChoiceParameter(choices=_BANDIT_POLICIES.keys(), default="none")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})
    bandit_weights: str = luigi.Parameter(default='none')
    limit_list_size: int = luigi.IntParameter(default=50)
    pin_memory: bool = luigi.BoolParameter(default=False)
    num_processes: int = luigi.IntParameter(default=os.cpu_count())
    no_offpolicy_eval: bool = luigi.BoolParameter(default=False)

    def requires(self):
        test_size            = self.model_training.requires().session_test_size
        minimum_interactions = self.model_training.requires().minimum_interactions
        sample_size          = self.model_training.requires().sample_size

        train_dataset_split = {'test_size': test_size, 
                                'minimum_interactions':minimum_interactions,
                                'sample_size':sample_size}
        
        return PrepareIfoodIndexedOrdersTestData(**train_dataset_split, nofilter_iteractions_test=self.nofilter_iteractions_test), \
               ListAccountMerchantTuplesForIfoodIndexedOrdersTestData(**train_dataset_split, nofilter_iteractions_test=self.nofilter_iteractions_test), \
               LoggingPolicyPsDataset(**train_dataset_split),\
               self.load_direct_estimator(), \
               IndexAccountsAndMerchantsOfSessionTrainDataset(**train_dataset_split), \
               CreateInteractionDataset(**train_dataset_split)               

    def output(self):
        return luigi.LocalTarget(os.path.join("output", "evaluation", self.__class__.__name__, "results", self.task_name))

    def load_bandit_model(self) -> BanditPolicy:
        # Load Bandit Modle
        if self.bandit_weights == 'none':
            bandit_policy = _BANDIT_POLICIES[self.bandit_policy](reward_model=None, **self.bandit_policy_params)
        else:
            with open(self.bandit_weights, 'rb') as f:
                bandit_policy = pickle.load(f)            

        # Train Bandit Model
        bandit_policy.fit(self.train_dataset)        

        return bandit_policy

    def save_bandit_model(self, bandit_model: BanditPolicy):
        # Save Bandit Object
        with open(os.path.join(self.output().path, "bandit.pkl"), 'wb') as bandit_file:
            pickle.dump(bandit_model, bandit_file)        

    def load_direct_estimator(self) -> DirectEstimatorTraining:
        if self.no_offpolicy_eval:
            return DummyTask()

        test_size            = self.model_training.requires().session_test_size
        minimum_interactions = self.model_training.requires().minimum_interactions
        sample_size          = self.model_training.requires().sample_size

        return DirectEstimatorTraining(project='ifood_offpolicy_direct_estimator', 
                                       session_test_size=test_size, 
                                       minimum_interactions=minimum_interactions, 
                                       sample_size=sample_size,
                                       task_hash=self.task_name)        


    def _read_test_data_frame(self) -> pd.DataFrame:
        tuples_df = pd.read_parquet(self.input()[1].path) #, columns=['account_idx', 'merchant_idx', 'shift_idx']

        return tuples_df

    def _transform_scores(self, scores: np.ndarray) -> np.ndarray:
        return scores.reshape(-1)

    @property
    def test_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_test_data_frame"):
            self._test_data_frame = self._read_test_data_frame()
        return self._test_data_frame

    @property
    def tuple_data_frame(self) -> pd.DataFrame:
        return self.test_data_frame

    @property
    def test_direct_estimator_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_test_direct_estimator_data_frame"):
            tuples_df = self.tuple_data_frame

            train_interactions_df = pd.read_parquet(self.input()[-1].path, 
                                        columns=['account_idx', 'merchant_idx', 'visits', 'buys'])

            train_interactions_df['buys']   = train_interactions_df['buys'].astype(float)
            train_interactions_df['visits'] = train_interactions_df['visits'].astype(float)

            tuples_df = tuples_df.merge(train_interactions_df, on=['account_idx', 'merchant_idx'], how='outer')
            #tuples_df.dropna(subset=['session_id'], how='all', inplace=True)
            tuples_df.fillna(0.0, inplace=True)
            tuples_df.rename(columns={"buys": "hist_buys", "visits": "hist_visits"}, inplace=True)
                        
            self._test_direct_estimator_data_frame = tuples_df
        return self._test_direct_estimator_data_frame

    @property
    def train_dataset(self) -> Dataset:
        return self.model_training.train_dataset

    @property
    def dataset(self) -> Dataset:
        if not hasattr(self, "_dataset"):
            print("Reading tuples files...")
            if self.model_training.project_config.output_column.name not in self.test_data_frame.columns:
                self.test_data_frame[self.model_training.project_config.output_column.name] = 1
            for auxiliar_output_column in self.model_training.project_config.auxiliar_output_columns:
                if auxiliar_output_column.name not in self.test_data_frame.columns:
                    self.test_data_frame[auxiliar_output_column.name] = 0

            self._dataset = self.model_training.project_config.dataset_class(self.test_data_frame,
                                                                             self.model_training.metadata_data_frame,
                                                                             self.model_training.project_config,
                                                                             negative_proportion=0.0)

        return self._dataset

    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        generator = NoAutoCollationDataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                              num_workers=self.model_training.generator_workers,
                                              pin_memory=self.pin_memory if self.model_training.device == "cuda" else False)

        print("Loading trained model...", self.batch_size)
        module = self.model_training.get_trained_module()
        print(module)
        trial = Trial(module,
                      criterion=lambda *args:
                      torch.zeros(1, device=self.model_training.torch_device, requires_grad=True)) \
            .with_test_generator(generator).to(self.model_training.torch_device)
        model_output: Union[torch.Tensor, Tuple[torch.Tensor]] = trial.predict(verbose=2)

        # TODO else with batch size
        scores_tensor: torch.Tensor = model_output if isinstance(model_output, torch.Tensor) else model_output[0][0]
        scores: np.ndarray = scores_tensor.detach().cpu().numpy()

        scores = self._transform_scores(scores)

        return self._create_dictionary_of_scores(scores)

    def _direct_estimator_rewards_merchant_tuples(self):
        print("Create the direct estimator rewards")
        module_training      = self.load_direct_estimator()
        
        print("Loading trained model (DE)... ", module_training.task_id)            
        module = module_training.get_trained_module()

        print("Loading Dataset from DE model...")
        _dataset = module_training.project_config.dataset_class(self.test_direct_estimator_data_frame,
                                                                    module_training.metadata_data_frame,
                                                                    module_training.project_config,
                                                                    negative_proportion=0.0)

        generator = NoAutoCollationDataLoader(_dataset, batch_size=module_training.batch_size, shuffle=False,
                                              num_workers=module_training.generator_workers,
                                              pin_memory=module_training.pin_memory if module_training.device == "cuda" else False)

        trial = Trial(module,
                      criterion=lambda *args:
                      torch.zeros(1, device=module_training.torch_device, requires_grad=True)) \
            .with_test_generator(generator).to(module_training.torch_device)

        scores_tensor: torch.Tensor = trial.predict(verbose=2)
        scores: np.ndarray = scores_tensor.detach().cpu().numpy().reshape(-1)
        scores = self._transform_scores(scores)

        return {(account_idx, merchant_idx): score for account_idx, merchant_idx, score
                in tqdm(zip(self.test_direct_estimator_data_frame["account_idx"], 
                            self.test_direct_estimator_data_frame["merchant_idx"], scores),
                        total=len(scores))}

    def _create_dictionary_of_scores(self, scores: np.ndarray) -> Dict[Tuple[int, int], float]:
        print("Creating the dictionary of scores...")
        return {(account_idx, merchant_idx): score for account_idx, merchant_idx, score
                in tqdm(zip(self.test_data_frame["account_idx"], self.test_data_frame["merchant_idx"], 
                            scores),
                        total=len(scores))}

    def _create_dictionary_of_dataset_indices(self) -> Dict[Tuple[int, int], int]:
        print("Creating the dictionary of dataset indices...")
        return {(account_idx, merchant_idx): i for account_idx, merchant_idx, i
                in tqdm(zip(self.test_data_frame["account_idx"], self.test_data_frame["merchant_idx"],
                            range(len(self.dataset))),
                        total=len(self.dataset))}

    def run(self):
        scores_per_tuple     = self._evaluate_account_merchant_tuples()

        de_rewards_per_tuple = scores_per_tuple.copy()

        print("Reading the orders DataFrame...")
        orders_df: pd.DataFrame = pd.read_parquet(self.input()[0].path, 
                                        columns=['session_id', 'account_id', 'dt_partition', 'account_idx', 'merchant_idx', 
                                                'shift_idx', 'day_of_week', 'count_buys', 'count_visits', 'buy', 'merchant_idx_list'])

        print("Join with LogPolicyProb...")
        logpolicy_df: pd.DataFrame = pd.read_csv(self.input()[2].path)

        orders_df = orders_df.merge(logpolicy_df, how='left', on=['merchant_idx', 'account_idx']).fillna(0)

        print("Filtering orders where the ordered merchant isn't in the list...")
        orders_df = orders_df[orders_df.progress_apply(lambda row: row["merchant_idx"] in row["merchant_idx_list"], axis=1)]

        print("Sorting merchnat_idx_list per user.")
        if self.bandit_policy == "none":
            bandit_model  = None
            sort_function = functools.partial(_sort_merchants_by_tuple_score, scores_per_tuple=scores_per_tuple,
                                              limit=self.limit_list_size)

            sorted_merchant_idx_list =  list(tqdm(starmap(functools.partial(sort_function, scores_per_tuple=scores_per_tuple),
                                                    zip(orders_df["account_idx"], orders_df["merchant_idx_list"])),
                                            total=len(orders_df)))

            orders_df["sorted_merchant_idx_list"] = sorted_merchant_idx_list
            orders_df["prob_merchant_idx_list"]   = [list(np.ones(self.limit_list_size)) for _ in range(len(orders_df))] 
        else:
            # BanditPolicy
            bandit_model = self.load_bandit_model()

            # DirectEstimator
            if not self.no_offpolicy_eval:
                de_rewards_per_tuple = self._direct_estimator_rewards_merchant_tuples()

            # Sort Function
            dataset_indices_per_tuple = self._create_dictionary_of_dataset_indices()
            sort_function = functools.partial(_sort_merchants_by_tuple_score_with_bandit_policy,
                                              scores_per_tuple=scores_per_tuple,
                                              dataset_indices_per_tuple=dataset_indices_per_tuple,
                                              dataset=self.dataset, bandit_policy=bandit_model,
                                              limit=self.limit_list_size)

            sorted_merchant_idx_list =  list(tqdm(starmap(sort_function,
                                                zip(orders_df["account_idx"], orders_df["merchant_idx_list"])),
                                        total=len(orders_df)))

            # unzip (sorted, prob)
            sorted, prob = zip(*sorted_merchant_idx_list)

            orders_df["sorted_merchant_idx_list"] = sorted
            orders_df["prob_merchant_idx_list"]   = prob

        print("Creating the relevance lists...")
        orders_df["relevance_list"] = list(tqdm(
            starmap(_create_relevance_list, zip(orders_df["sorted_merchant_idx_list"], orders_df["merchant_idx"])),
            total=len(orders_df)))

        print("Calculate ps policy eval...")
        orders_df["ps_eval"] = list(tqdm(
            starmap(_ps_policy_eval, zip(orders_df["relevance_list"], orders_df["prob_merchant_idx_list"])),
            total=len(orders_df)))

        print("Creating scores merchant list...")
        orders_df["scores_merchant_idx_list"] = list(tqdm(
            starmap(functools.partial(_get_scores_per_tuple, scores_per_tuple=scores_per_tuple), 
                        zip(orders_df["account_idx"], orders_df["sorted_merchant_idx_list"])),
                    total=len(orders_df)))

        orders_df["rhat_scores"] = list(tqdm(
            starmap(_get_rhat_scores, zip(orders_df["relevance_list"], orders_df["scores_merchant_idx_list"])),
            total=len(orders_df)))

        orders_df["rhat_merchant_idx"] = list(tqdm(
            starmap(lambda sorted_merchant_idx_list, scores_merchant_idx_list: sorted_merchant_idx_list[0], zip(orders_df["sorted_merchant_idx_list"], orders_df["scores_merchant_idx_list"])),
            total=len(orders_df)))

        print("Creating direct estimator rewards merchant list...")
        orders_df["rewards_merchant_idx_list"] = list(tqdm(
            starmap(functools.partial(_get_scores_per_tuple, scores_per_tuple=de_rewards_per_tuple), 
                        zip(orders_df["account_idx"], orders_df["sorted_merchant_idx_list"])),
                    total=len(orders_df)))

        orders_df["rhat_rewards"] = list(tqdm(
            starmap(_get_rhat_rewards, zip(orders_df["relevance_list"], orders_df["rewards_merchant_idx_list"])),
            total=len(orders_df)))

        print("Calcule rewards...")
        orders_df["rewards"] = orders_df["buy"]

        print("Saving the output file...")
        os.makedirs(self.output().path, exist_ok=True)

        # Save Bandit Object
        if bandit_model:
            self.save_bandit_model(bandit_model)

        if self.plot_histogram:
            plot_histogram(de_rewards_per_tuple.values()).savefig(
               os.path.join(self.output().path, "DE_rewards_histogram.jpg"))

            plot_histogram(scores_per_tuple.values()).savefig(
               os.path.join(self.output().path, "scores_histogram.jpg"))
            
            plot_histogram(orders_df["ps_eval"].values).savefig(
                os.path.join(self.output().path, "ps_eval.jpg"))

            plot_histogram(orders_df["ps"].values).savefig(
                os.path.join(self.output().path, "ps.jpg"))

        gc.collect()

        orders_df[["session_id", "account_id", "dt_partition", "account_idx", "merchant_idx", 
                   "rhat_merchant_idx", "shift_idx", "day_of_week", 
                   "sorted_merchant_idx_list", "scores_merchant_idx_list", "rhat_scores", 
                   "rewards_merchant_idx_list", "prob_merchant_idx_list", "relevance_list", 
                   "ps", "ps_eval", "count_visits", "count_buys", "rhat_rewards", "rewards"]].to_csv(
            os.path.join(self.output().path, "orders_with_sorted_merchants.csv"), index=False)


class SortMerchantListsForAutoEncoderIfoodModel(SortMerchantListsForIfoodModel):
    batch_size: int = luigi.IntParameter(default=100000)

    def _read_test_data_frame(self) -> pd.DataFrame:
        print("Reading train, val and test DataFrames...")
        train_df = self._eval_buys_per_merchant_column(pd.read_csv(self.model_training.input()[0].path))
        val_df   = self._eval_buys_per_merchant_column(pd.read_csv(self.model_training.input()[1].path))
        test_df  = self._eval_buys_per_merchant_column(pd.read_csv(self.model_training.input()[2].path))
        df: pd.DataFrame = pd.concat((train_df, val_df, test_df))

        # Needed if split_per_user=False
        df = df.groupby(["account_idx", "n_users", "n_items"])["buys_per_merchant"] \
            .apply(lambda lists: [inner for outer in lists for inner in outer]).reset_index()

        return df

    def _transform_scores(self, scores: np.ndarray) -> np.ndarray:
        return scores

    def _eval_buys_per_merchant_column(self, df: pd.DataFrame):
        if len(df) > 0 and type(df.iloc[0]["buys_per_merchant"]) is str:
            df["buys_per_merchant"] = parallel_literal_eval(df["buys_per_merchant"])
        return df

    @property
    def tuple_data_frame(self) -> pd.DataFrame:
        tuples_df = pd.read_parquet(self.input()[1].path)

        return tuples_df

    def _create_dictionary_of_scores(self, scores: np.ndarray) -> Dict[Tuple[int, int], float]:
        tuples_df = self.tuple_data_frame
        df        = self.test_data_frame

        print("Grouping by account index...")
        merchant_indices_per_account_idx: pd.Series = tuples_df.groupby('account_idx')['merchant_idx'].apply(list)

        scores_per_tuple: List[Tuple[Tuple[int, int], float]] = []

        for account_idx, score in zip(df["account_idx"], scores):
            if account_idx in merchant_indices_per_account_idx:
                merchant_indices = merchant_indices_per_account_idx[account_idx]
                scores_per_tuple.extend([((account_idx, merchant_idx), score[merchant_idx])
                                         for merchant_idx in merchant_indices])

        print("Creating the dictionary of scores...")
        return dict(scores_per_tuple)

    def _create_dictionary_of_dataset_indices(self) -> Dict[Tuple[int, int], int]:
        return None

class EvaluateIfoodModel(BaseEvaluationTask):
    num_processes: int = luigi.IntParameter(default=os.cpu_count())
    bandit_policy: str = luigi.ChoiceParameter(choices=_BANDIT_POLICIES.keys(), default="none")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})
    bandit_weights: str = luigi.Parameter(default='none')
    batch_size: int = luigi.IntParameter(default=100000)
    plot_histogram: bool = luigi.BoolParameter(default=False)
    no_offpolicy_eval: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return SortMerchantListsForIfoodModel(model_module=self.model_module, 
                                              model_cls=self.model_cls,
                                              model_task_id=self.model_task_id, 
                                              bandit_policy=self.bandit_policy,
                                              bandit_policy_params=self.bandit_policy_params,
                                              bandit_weights=self.bandit_weights,
                                              batch_size=self.batch_size,
                                              plot_histogram=self.plot_histogram,
                                              limit_list_size=self.limit_list_size,
                                              nofilter_iteractions_test=self.nofilter_iteractions_test,
                                              no_offpolicy_eval=self.no_offpolicy_eval,
                                              task_hash=self.task_hash)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_path, "orders_with_metrics.csv")), \
               luigi.LocalTarget(os.path.join(self.output_path, "metrics.json")),

    @property
    def sort_merchant_list_path(self):
        return self.input().path

    @property
    def evaluation_data_frame_path(self):
        return os.path.join(self.sort_merchant_list_path, "orders_with_sorted_merchants.csv")

    @property
    def bandit_path(self):
        return os.path.join(self.sort_merchant_list_path, "bandit.pkl")

    def read_evaluation_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.evaluation_data_frame_path)

    def _mean_personalization(self, df: pd.DataFrame, k: int):
        grouped_df = df.groupby(["shift_idx", "day_of_week"])
        personalization_per_shift: List[float] = []
        for _, group_df in grouped_df:
            if len(group_df["sorted_merchant_idx_list"]) > 1:
                personalization_per_shift.append(personalization_at_k(group_df["sorted_merchant_idx_list"], k))
        return np.mean(personalization_per_shift)

    def _offpolicy_eval(self, df: pd.DataFrame):
        # Filter df used in offpolicy evaluation
        df_offpolicy  = df[df.ps > 0]

        # Adiciona zeros que são das visitas sem compra, recompensas zeradas para o calculo geral
        #
        rewards      = np.concatenate((df_offpolicy['rewards'], np.zeros(df.iloc[0].count_visits)), axis=None)
        rhat_rewards = np.concatenate((df_offpolicy['rhat_rewards'], np.zeros(df.iloc[0].count_visits)), axis=None)
        ps_eval      = np.concatenate((df_offpolicy['ps_eval'], np.ones(df.iloc[0].count_visits)), axis=None)
        ps           = np.concatenate((df_offpolicy['ps'], np.ones(df.iloc[0].count_visits)), axis=None)
        
        return rhat_rewards, rewards, ps_eval, ps

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        df: pd.DataFrame = self.read_evaluation_data_frame()

        with Pool(self.num_processes) as p:
            df["sorted_merchant_idx_list"]  = parallel_literal_eval(df["sorted_merchant_idx_list"], pool=p)
            df["scores_merchant_idx_list"]  = parallel_literal_eval(df["scores_merchant_idx_list"], pool=p)
            df["rewards_merchant_idx_list"] = parallel_literal_eval(df["rewards_merchant_idx_list"], pool=p)
            df["prob_merchant_idx_list"]    = parallel_literal_eval(df["relevance_list"], pool=p)
            df["relevance_list"]            = parallel_literal_eval(df["relevance_list"], pool=p)

            df["average_precision"] = list(
                tqdm(p.map(average_precision, df["relevance_list"]), total=len(df)))

            df["precision_at_1"] = list(
                tqdm(p.map(functools.partial(precision_at_k, k=1), df["relevance_list"]), total=len(df)))

            df["ndcg_at_5"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=5), df["relevance_list"]), total=len(df)))
            df["ndcg_at_10"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=10), df["relevance_list"]), total=len(df)))
            df["ndcg_at_15"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=15), df["relevance_list"]), total=len(df)))
            df["ndcg_at_20"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=20), df["relevance_list"]), total=len(df)))
            df["ndcg_at_50"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=50), df["relevance_list"]), total=len(df)))

        catalog = range(self.n_items)

        metrics = {
            "model_task": self.model_task_id,
            "count": len(df),
            "count_percent": len(df)/df.iloc[0].count_buys,
            "mean_average_precision": df["average_precision"].mean(),
            "precision_at_1": df["precision_at_1"].mean(),
            "ndcg_at_5": df["ndcg_at_5"].mean(),
            "ndcg_at_10": df["ndcg_at_10"].mean(),
            "ndcg_at_15": df["ndcg_at_15"].mean(),
            "ndcg_at_20": df["ndcg_at_20"].mean(),
            "ndcg_at_50": df["ndcg_at_50"].mean(),
            "coverage_at_5": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 5),
            "coverage_at_10": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 10),
            "coverage_at_15": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 15),
            "coverage_at_20": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 20),
            "coverage_at_50": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 50),
            "personalization_at_5": self._mean_personalization(df, 5),
            "personalization_at_10": self._mean_personalization(df, 10),
            "personalization_at_15": self._mean_personalization(df, 15),
            "personalization_at_20": self._mean_personalization(df, 20),
            "personalization_at_50": self._mean_personalization(df, 50)
        }

        if self.bandit_policy != "none" and not self.no_offpolicy_eval:
            rhat_rewards, rewards, ps_eval, ps = self._offpolicy_eval(df)

            metrics["IPS"]   = eval_IPS(rewards, ps_eval, ps)
            metrics["CIPS"]  = eval_CIPS(rewards, ps_eval, ps)
            metrics["SNIPS"] = eval_SNIPS(rewards, ps_eval, ps)            
            metrics["DirectEstimator"] = np.mean(rhat_rewards)
            metrics["DoublyRobust"]    = eval_doubly_robust(rhat_rewards, rewards, ps_eval, ps)

        print("\n====================")
        print("Metrics")
        print("====================\n")
        pprint.pprint(metrics)
        print("")

        df = df.drop(columns=["sorted_merchant_idx_list", "scores_merchant_idx_list",
                                "rewards_merchant_idx_list", "prob_merchant_idx_list", 
                                "relevance_list"])

        df.to_csv(self.output()[0].path)
        
        with open(self.output()[1].path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        
        self._save_params()

class SortMerchantListsRandomly(SortMerchantListsForIfoodModel):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    @property
    def dataset(self) -> Dataset:
        return None

    @property
    def train_dataset(self) -> Dataset:
        return None

    def _create_dictionary_of_dataset_indices(self) -> Dict[Tuple[int, int], int]:
        return None

    def load_direct_estimator(self) -> DirectEstimatorTraining:
        if self.no_offpolicy_eval:
            return DummyTask()

        return DirectEstimatorTraining(project='ifood_offpolicy_direct_estimator', 
                                       session_test_size=self.test_size, 
                                       minimum_interactions=self.minimum_interactions, 
                                       sample_size=self.sample_size,
                                       task_hash=self.task_name)    

    def requires(self):
        train_dataset_split = {'test_size': self.test_size, 'minimum_interactions':self.minimum_interactions, 'sample_size':self.sample_size}

        return PrepareIfoodIndexedOrdersTestData(**train_dataset_split, nofilter_iteractions_test=self.nofilter_iteractions_test), \
               ListAccountMerchantTuplesForIfoodIndexedOrdersTestData(**train_dataset_split, nofilter_iteractions_test=self.nofilter_iteractions_test), \
               LoggingPolicyPsDataset(**train_dataset_split), \
               self.load_direct_estimator(), \
               IndexAccountsAndMerchantsOfSessionTrainDataset(**train_dataset_split), \
               CreateInteractionDataset(**train_dataset_split)                            


    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        scores: np.array  = np.random.rand(len(self.test_data_frame))

        return self._create_dictionary_of_scores(scores)

class EvaluateAutoEncoderIfoodModel(EvaluateIfoodModel):
    def requires(self):
        return SortMerchantListsForAutoEncoderIfoodModel(model_module=self.model_module,
                                                         model_cls=self.model_cls,
                                                         model_task_id=self.model_task_id,
                                                         bandit_policy=self.bandit_policy,
                                                         bandit_policy_params=self.bandit_policy_params,
                                                         bandit_weights=self.bandit_weights,
                                                         plot_histogram=self.plot_histogram,
                                                         limit_list_size=self.limit_list_size,
                                                         nofilter_iteractions_test=self.nofilter_iteractions_test,
                                                         task_hash=self.task_hash)


class EvaluateRandomIfoodModel(EvaluateIfoodModel):
    model_task_id: str = luigi.Parameter(default="RandomIfoodModel")
    
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    bandit_policy: str = luigi.ChoiceParameter(choices=_BANDIT_POLICIES.keys(), default="none")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})
    bandit_weights: str = luigi.Parameter(default='none')
    plot_histogram: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return SortMerchantListsRandomly(test_size=self.test_size, 
                                        sample_size=self.sample_size,
                                        minimum_interactions=self.minimum_interactions,
                                        bandit_policy=self.bandit_policy, 
                                        bandit_policy_params=self.bandit_policy_params,
                                        bandit_weights=self.bandit_weights,
                                        plot_histogram=self.plot_histogram,
                                        model_task_id=self.model_task_id,
                                        limit_list_size=self.limit_list_size,
                                        nofilter_iteractions_test=self.nofilter_iteractions_test,
                                        task_hash=self.task_hash), \
               GenerateIndicesForAccountsAndMerchantsDataset(
                                        test_size=self.test_size, 
                                        minimum_interactions=self.minimum_interactions, 
                                        sample_size=self.sample_size)

    @property
    def sort_merchant_list_path(self):
        return self.input()[0].path

    @property
    def n_items(self):
        return len(pd.read_csv(self.input()[1][1].path))


class SortMerchantListsByMostPopular(SortMerchantListsForIfoodModel):
    test_size: float = luigi.FloatParameter(default=0.10)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def _read_test_data_frame(self) -> pd.DataFrame:
        tuples_df = pd.read_parquet(self.input()[1].path) # columns=['account_idx', 'merchant_idx'])

        return tuples_df

    @property
    def dataset(self) -> Dataset:
        return None

    @property
    def train_dataset(self) -> Dataset:
        return None

    def _create_dictionary_of_dataset_indices(self) -> Dict[Tuple[int, int], int]:
        return None

    def load_direct_estimator(self) -> DirectEstimatorTraining:
        if self.no_offpolicy_eval:
            return DummyTask()

        return DirectEstimatorTraining(project='ifood_offpolicy_direct_estimator', 
                                       session_test_size=self.test_size, 
                                       minimum_interactions=self.minimum_interactions, 
                                       sample_size=self.sample_size,
                                       task_hash=self.task_name)    
    def requires(self):
        train_dataset_split = {'test_size': self.test_size, 
                                'minimum_interactions':self.minimum_interactions,
                                'sample_size':self.sample_size}

        return PrepareIfoodIndexedOrdersTestData(**train_dataset_split, nofilter_iteractions_test=self.nofilter_iteractions_test), \
               ListAccountMerchantTuplesForIfoodIndexedOrdersTestData(**train_dataset_split, nofilter_iteractions_test=self.nofilter_iteractions_test), \
               LoggingPolicyPsDataset(**train_dataset_split), \
               self.load_direct_estimator(), \
               IndexAccountsAndMerchantsOfSessionTrainDataset(**train_dataset_split), \
               CreateInteractionDataset(**train_dataset_split)                            


    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        
        interactions_df = pd.read_parquet(self.input()[-1].path)
        test_df         = self.test_data_frame

        # Group by Merchant a sum results
        scores: pd.DataFrame = interactions_df\
                                .groupby("merchant_idx")["buys"].sum().reset_index()\
                                .rename(columns = {'buys': 'total_buy'})

        merchant_tuples_scores: pd.DataFrame = test_df.merge(scores, how='left', on='merchant_idx').fillna(0)

        return self._create_dictionary_of_scores(merchant_tuples_scores['total_buy'])


class EvaluateMostPopularIfoodModel(EvaluateIfoodModel):
    model_task_id: str = luigi.Parameter(default="MostPopularIfoodModel")
    test_size: float = luigi.FloatParameter(default=0.1)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    bandit_policy: str = luigi.ChoiceParameter(choices=_BANDIT_POLICIES.keys(), default="none")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})
    bandit_weights: str = luigi.Parameter(default='none')
    plot_histogram: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return SortMerchantListsByMostPopular(test_size=self.test_size, 
                                              sample_size=self.sample_size,
                                              minimum_interactions=self.minimum_interactions,
                                              bandit_policy=self.bandit_policy, 
                                              bandit_policy_params=self.bandit_policy_params,
                                              bandit_weights=self.bandit_weights,
                                              plot_histogram=self.plot_histogram,
                                              model_task_id=self.model_task_id,
                                              limit_list_size=self.limit_list_size,
                                              nofilter_iteractions_test=self.nofilter_iteractions_test,
                                              task_hash=self.task_hash), \
               GenerateIndicesForAccountsAndMerchantsDataset(
                                            test_size=self.test_size, 
                                            sample_size=self.sample_size,
                                            minimum_interactions=self.minimum_interactions)

    @property
    def sort_merchant_list_path(self):
        return self.input()[0].path

    @property
    def n_items(self):
        return len(pd.read_csv(self.input()[1][1].path))

    @property
    def output_path(self):
        return os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                    self.task_id)


class SortMerchantListsByMostPopularPerUser(SortMerchantListsForIfoodModel):
    test_size: float = luigi.FloatParameter(default=0.1)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    buy_importance: float = luigi.FloatParameter(default=1.0)
    visit_importance: float = luigi.FloatParameter(default=0.0)

    def _read_test_data_frame(self) -> pd.DataFrame:
        tuples_df = pd.read_parquet(self.input()[1].path) # columns=['account_idx', 'merchant_idx', 'count_buys', 'count_visits', 'buy'])

        return tuples_df

    @property
    def dataset(self) -> Dataset:
        return None

    @property
    def train_dataset(self) -> Dataset:
        return None

    def _create_dictionary_of_dataset_indices(self) -> Dict[Tuple[int, int], int]:
        return None

    def load_direct_estimator(self) -> DirectEstimatorTraining:
        if self.no_offpolicy_eval:
            return DummyTask()

        return DirectEstimatorTraining(project='ifood_offpolicy_direct_estimator', 
                                       session_test_size=self.test_size, 
                                       minimum_interactions=self.minimum_interactions, 
                                       sample_size=self.sample_size,
                                       task_hash=self.task_name)       
    def requires(self):
        train_dataset_split = {'test_size': self.test_size, 'minimum_interactions':self.minimum_interactions, 'sample_size':self.sample_size}

        return PrepareIfoodIndexedOrdersTestData(**train_dataset_split, nofilter_iteractions_test=self.nofilter_iteractions_test), \
               ListAccountMerchantTuplesForIfoodIndexedOrdersTestData(**train_dataset_split, nofilter_iteractions_test=self.nofilter_iteractions_test), \
               LoggingPolicyPsDataset(**train_dataset_split), \
               self.load_direct_estimator(), \
               IndexAccountsAndMerchantsOfSessionTrainDataset(**train_dataset_split), \
               CreateInteractionDataset(**train_dataset_split)                            


    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        print("Evaluate account X merchant tuples...")

        interactions_df    = pd.read_parquet(self.input()[-1].path, 
                                columns=["account_idx", "merchant_idx", "buys", "visits"])

        test_df            = self.test_data_frame
        #print(test_df.info(memory_usage='deep'))

        count_buys_visits  = interactions_df.groupby(["account_idx", "merchant_idx"])\
                                        .agg({'buys': 'sum', 'visits': 'sum'}).reset_index()

        count_buys_visits['score'] = count_buys_visits.progress_apply(lambda row: row['buys'] * self.buy_importance + row['visits']*self.visit_importance, axis=1) 

        merchant_tuples_scores: pd.DataFrame = test_df.merge(count_buys_visits, how='left', 
                                                            on=['account_idx', 'merchant_idx']).fillna(-1)

        return self._create_dictionary_of_scores(merchant_tuples_scores['score'])



class EvaluateMostPopularPerUserIfoodModel(EvaluateIfoodModel):
    model_task_id: str = luigi.Parameter(default="MostPopularPerUserIfoodModel")
    test_size: float = luigi.FloatParameter(default=0.1)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    bandit_policy: str = luigi.ChoiceParameter(choices=_BANDIT_POLICIES.keys(), default="none")
    bandit_policy_params: Dict[str, Any] = luigi.DictParameter(default={})
    bandit_weights: str = luigi.Parameter(default='none')
    buy_importance: float = luigi.FloatParameter(default=1.0)
    visit_importance: float = luigi.FloatParameter(default=0.0)

    def requires(self):
        return SortMerchantListsByMostPopularPerUser(test_size=self.test_size, 
                                                    sample_size=self.sample_size,
                                                    minimum_interactions=self.minimum_interactions,
                                                    bandit_policy=self.bandit_policy, 
                                                    bandit_policy_params=self.bandit_policy_params,
                                                    bandit_weights=self.bandit_weights,
                                                    plot_histogram=self.plot_histogram,
                                                    model_task_id=self.model_task_id,
                                                    limit_list_size=self.limit_list_size,
                                                    buy_importance=self.buy_importance,
                                                    visit_importance=self.visit_importance,
                                                    nofilter_iteractions_test=self.nofilter_iteractions_test,
                                                    task_hash=self.task_hash), \
               GenerateIndicesForAccountsAndMerchantsDataset(
                                                    test_size=self.test_size,
                                                    sample_size=self.sample_size,
                                                    minimum_interactions=self.minimum_interactions)

    @property
    def sort_merchant_list_path(self):
        return self.input()[0].path

    @property
    def n_items(self):
        return len(pd.read_csv(self.input()[1][1].path))


class GenerateUserEmbeddingsFromContentModel(BaseEvaluationTask):
    group_last_k_merchants: int = luigi.FloatParameter(default=20)

    # use_visit_interactions: bool

    def requires(self):
        test_size            = self.model_training.requires().session_test_size
        minimum_interactions = self.model_training.requires().minimum_interactions
        sample_size          = self.model_training.requires().sample_size

        return (GenerateContentEmbeddings(
            model_module=self.model_module,
            model_cls=self.model_cls,
            model_task_id=self.model_task_id),
                IndexAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=test_size,
                    minimum_interactions=minimum_interactions,
                    sample_size=sample_size),
                ProcessRestaurantContentDataset(),
                AddAdditionallInformationDataset())

    def output(self):
        return (luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.task_name, "user_embeddings_{}.tsv".format(self.group_last_k_merchants))), \
                luigi.LocalTarget(
                    os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                 self.task_name,
                                 "user_embeddings_by_shift_{}.pkl".format(self.group_last_k_merchants))))

    def _generate_content_tensors(self, rows):
        inputs = []

        for input_column in self.model_training.project_config.metadata_columns:
            dtype = torch.float32 if input_column.name == "restaurant_complete_info" else torch.int64
            values = rows[input_column.name].values.tolist()
            inputs.append(torch.tensor(values, dtype=dtype).to(self.model_training.torch_device))

        return inputs

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        processed_content_df = pd.read_csv(self.input()[2][0].path)  # .set_index('merchant_idx')
        shifts_df = pd.read_csv(self.input()[3][0].path).set_index('shift')
        # print(processed_content_df.head())
        # print(processed_content_df.columns)

        # d
        literal_eval_array_columns(processed_content_df,
                                   self.model_training.project_config.input_columns
                                   + [self.model_training.project_config.output_column]
                                   + self.model_training.project_config.metadata_columns)

        print("Loading trained model...")
        module = self.model_training.get_trained_module()
        print(module)

        restaurant_embs = np.genfromtxt(self.input()[0][0].path, dtype='float')
        restaurant_df = pd.read_csv(self.input()[0][1].path, sep='\t').reset_index().set_index("merchant_id")
        tuples_df = pd.read_parquet(self.input()[1].path)
        # print(restaurant_df.reset_index().set_index("merchant_id").columns)
        # d
        # filter only buy
        tuples_df = tuples_df[tuples_df.buy > 0].sort_values('click_timestamp', ascending=False)

        print("Generating embeddings for each account...")
        embeddings: List[float] = []

        # Predict embeddings for merchants's user
        #
        embeddings = {}
        i = 0
        for name, group in tqdm(tuples_df.groupby(['account_idx', 'shift'])):
            account_idx, shift = name
            if account_idx not in embeddings:
                embeddings[account_idx] = {}

            merchant_id = group.head(int(self.group_last_k_merchants)).merchant_id.values
            merchant_emb_idx = restaurant_df.loc[merchant_id]['index'].values
            emb = restaurant_embs[merchant_emb_idx]
            # merchant_idx  = group.head(int(self.group_last_k_merchants)).merchant_idx.values
            # rows          = processed_content_df.iloc[merchant_idx]

            # inputs                         = self._generate_content_tensors(rows)
            # batch_embeddings: torch.Tensor = module.compute_item_embeddings(inputs)
            # emb                            = batch_embeddings.detach().cpu().numpy()
            embeddings[account_idx][shift] = emb
            # i = i + 1
            # if i > 10:
            #     break 
        # Shape numpy embeddings
        #
        # (account, shift, features)
        account_embeddings_by_shift = np.zeros((len(embeddings.keys()), len(shifts_df), emb.shape[1]))

        # (account, features)
        account_embeddings_geral = np.zeros((len(embeddings.keys()), emb.shape[1]))

        for account_idx, shifts in embeddings.items():
            account_geral_emb = []

            for shift, embeddings in shifts.items():
                shift_emb = embeddings.mean(0)
                shift_idx = shifts_df.loc[shift].shift_idx
                account_embeddings_by_shift[account_idx][shift_idx] = shift_emb
                account_geral_emb.append(shift_emb)
            account_embeddings_geral[account_idx] = np.array(account_geral_emb).mean(0)

        # fillzero to geral emb
        for i in range(len(account_embeddings_geral)):
            for s in range(len(shifts_df)):
                if account_embeddings_by_shift[i][s].sum() == 0:
                    account_embeddings_by_shift[i][s] = account_embeddings_geral[i]

        np.savetxt(os.path.join(self.output()[0].path), account_embeddings_geral, delimiter="\t")

        with open(self.output()[1].path, 'wb') as output:
            pickle.dump(account_embeddings_by_shift, output)
        # np.savetxt(os.path.join(self.output()[1].path), account_embeddings_by_shift, delimiter="\t")


class SortMerchantListsTripletNetInfoContent(SortMerchantListsForIfoodModel):
    batch_size: int = luigi.IntParameter(default=10000)
    group_last_k_merchants: int = luigi.FloatParameter(default=20)

    def requires(self):
        test_size            = self.model_training.requires().session_test_size
        minimum_interactions = self.model_training.requires().minimum_interactions
        sample_size          = self.model_training.requires().sample_size

        return super().requires() + \
               (GenerateContentEmbeddings(
                   model_module=self.model_module,
                   model_cls=self.model_cls,
                   model_task_id=self.model_task_id,
                   batch_size=self.batch_size),
                IndexAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=test_size, minimum_interactions=minimum_interactions, sample_size=self.sample_size),
                GenerateIndicesForAccountsAndMerchantsDataset(
                    test_size=test_size, minimum_interactions=minimum_interactions, sample_size=self.sample_size),
                GenerateUserEmbeddingsFromContentModel(
                    group_last_k_merchants=self.group_last_k_merchants,
                    model_module=self.model_module,
                    model_cls=self.model_cls,
                    model_task_id=self.model_task_id))

    def _generate_content_tensors(self, rows):
        inputs = []

        for input_column in self.model_training.project_config.metadata_columns:
            dtype = torch.float32 if input_column.name == "restaurant_complete_info" else torch.int64
            values = rows[input_column.name].values.tolist()
            inputs.append(torch.tensor(values, dtype=dtype).to(self.model_training.torch_device))

        return inputs

    def _generate_batch_tensors(self, rows: pd.DataFrame, pool: Pool) -> List[torch.Tensor]:
        metadata_columns_name = [input_column.name for input_column in
                                 self.model_training.project_config.metadata_columns]
        # 'session_id', 'account_idx', 'merchant_idx', 'shift', 'shift_idx',
        # 'mode_shift_idx', 'mode_day_of_week', 'day_of_week'

        # assert metadata_columns_name == ['trading_name', 'description', 'category_names', 'restaurant_complete_info']

        # account_idxs = torch.tensor(rows["account_idx"].values, dtype=torch.int64) \
        #     .to(self.model_training.torch_device)

        # account_embeddings = []
        # for k, row in rows.iterrows():
        #     account_embeddings.append(self.account_embeddings_by_shift[row.account_idx][row.shift_idx])

        # self.account_embeddings_geral
        # account_embeddings = torch.from_numpy(self.account_embeddings_geral[rows["account_idx"].values])\
        #                        .to(self.model_training.torch_device)

        account_embeddings = self.account_embeddings_by_shift[rows["account_idx"].values, rows['shift_idx'].values]

        account_embeddings = torch.from_numpy(np.array(account_embeddings)).to(self.model_training.torch_device)
        merchant_rows = self.merchant_df.loc[rows["merchant_idx"]]
        inputs = self._generate_content_tensors(merchant_rows)

        return [account_embeddings, inputs]

    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        print("Reading merchant data frame...")
        self.merchant_df = pd.read_csv(self.input()[4][1].path).set_index('merchant_idx')

        print("Reading account embeddings...")
        self.account_embeddings_geral = np.loadtxt(self.input()[5][0].path, delimiter="\t")
        with open(self.input()[5][1].path, 'rb') as file:
            self.account_embeddings_by_shift = pickle.load(file, encoding="utf8")

        literal_eval_array_columns(self.merchant_df,
                                   self.model_training.project_config.input_columns
                                   + [self.model_training.project_config.output_column]
                                   + self.model_training.project_config.metadata_columns)

        tuples_df = pd.read_parquet(self.input()[1].path)

        # assert self.model_training.project_config.input_columns[0].name == "account_idx"
        # assert self.model_training.project_config.input_columns[1].name == "merchant_idx"
        print("Loading trained model...")
        module = self.model_training.get_trained_module()
        print(module)

        scores: List[float] = []
        print("Running the model for every account and merchant tuple...")
        with Pool(os.cpu_count()) as pool:
            for indices in tqdm(chunks(range(len(tuples_df)), self.batch_size),
                                total=math.ceil(len(tuples_df) / self.batch_size)):
                rows: pd.DataFrame = tuples_df.iloc[indices]
                account_embeddings, inputs = self._generate_batch_tensors(rows, pool)

                item_embeddings: torch.Tensor = module.compute_item_embeddings(inputs)
                batch_scores: torch.Tensor = module.similarity(account_embeddings, item_embeddings)
                batch_scores = batch_scores.detach().cpu().numpy()

                scores.extend(batch_scores)

        print("Creating the dictionary of scores...")
        return {(account_idx, merchant_idx): score for account_idx, merchant_idx, score
                in tqdm(zip(tuples_df["account_idx"], tuples_df["merchant_idx"], scores), total=len(scores))}


class SortMerchantListsFullContentModel(SortMerchantListsForIfoodModel):
    batch_size: int = luigi.IntParameter(default=10000)

    @property
    def tuple_data_frame(self) -> pd.DataFrame:
        tuples_df = pd.read_parquet(self.input()[1].path)

        return tuples_df
        
    def _read_test_data_frame(self) -> pd.DataFrame:
        tuples_df = pd.read_parquet(self.input()[1].path)#.sample(100)
        #print(tuples_df.info(memory_usage='deep'))

        train_interactions_df = pd.read_parquet(self.input()[-1].path,
                                    columns=['account_idx', 'merchant_idx', 
                                            'visits', 'buys'])

        tuples_df = tuples_df.merge(train_interactions_df, 
                                on=['account_idx', 'merchant_idx'], how='outer')
        #tuples_df.dropna(subset=['session_id'], how='all', inplace=True)
        tuples_df.fillna(0.0, inplace=True)
        tuples_df.rename(columns={"buys": "hist_buys", "visits": "hist_visits"}, inplace=True)
        return tuples_df


class EvaluateIfoodTripletNetInfoContent(EvaluateIfoodModel):
    batch_size: int = luigi.IntParameter(default=10000)
    model_task_id: str = luigi.Parameter(default="none")
    test_size: float = luigi.FloatParameter(default=0.1)
    sample_size: int = luigi.IntParameter(default=-1)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    group_last_k_merchants: int = luigi.FloatParameter(default=20)

    def requires(self):
        return [
            SortMerchantListsTripletNetInfoContent(
                model_module=self.model_module, 
                model_cls=self.model_cls,
                model_task_id=self.model_task_id, 
                bandit_policy=self.bandit_policy,
                bandit_policy_params=self.bandit_policy_params, 
                bandit_weights=self.bandit_weights,
                batch_size=self.batch_size,
                group_last_k_merchants=self.group_last_k_merchants),
            ProcessRestaurantContentDataset(),
            GenerateIndicesForAccountsAndMerchantsDataset(
                test_size=self.test_size,
                sample_size=self.sample_size,
                minimum_interactions=self.minimum_interactions)]

    @property
    def n_items(self):
        return self.model_training.n_items

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        df: pd.DataFrame = self.read_evaluation_data_frame()

        with Pool(self.num_processes) as p:
            df["sorted_merchant_idx_list"] = parallel_literal_eval(df["sorted_merchant_idx_list"], pool=p)
            df["relevance_list"] = parallel_literal_eval(df["relevance_list"], pool=p)

            df["average_precision"] = list(
                tqdm(p.map(average_precision, df["relevance_list"]), total=len(df)))

            df["ndcg_at_5"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=5), df["relevance_list"]), total=len(df)))
            df["ndcg_at_10"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=10), df["relevance_list"]), total=len(df)))
            df["ndcg_at_15"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=15), df["relevance_list"]), total=len(df)))
            df["ndcg_at_20"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=20), df["relevance_list"]), total=len(df)))
            df["ndcg_at_50"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=50), df["relevance_list"]), total=len(df)))

        # TODO mudar hardcode
        catalog = range(1664)

        metrics = {
            "model_task": self.model_task_id,
            "count": len(df),
            "mean_average_precision": df["average_precision"].mean(),
            "ndcg_at_5": df["ndcg_at_5"].mean(),
            "ndcg_at_10": df["ndcg_at_10"].mean(),
            "ndcg_at_15": df["ndcg_at_15"].mean(),
            "ndcg_at_20": df["ndcg_at_20"].mean(),
            "ndcg_at_50": df["ndcg_at_50"].mean(),
            "coverage_at_5": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 5),
            "coverage_at_10": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 10),
            "coverage_at_15": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 15),
            "coverage_at_20": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 20),
            "coverage_at_50": prediction_coverage_at_k(df["sorted_merchant_idx_list"], catalog, 50),
            "personalization_at_5": self._mean_personalization(df, 5),
            "personalization_at_10": self._mean_personalization(df, 10),
            "personalization_at_15": self._mean_personalization(df, 15),
            "personalization_at_20": self._mean_personalization(df, 20),
            "personalization_at_50": self._mean_personalization(df, 50),
        }

        print("Metrics")
        pprint.pprint(metrics)
        print("")

        df = df.drop(columns=["sorted_merchant_idx_list", "relevance_list"])
        df.to_csv(self.output()[0].path)
        with open(self.output()[1].path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)

    @property
    def sort_merchant_list_path(self):
        return self.input()[0].path


class EvaluateIfoodFullContentModel(EvaluateIfoodModel):
    batch_size: int = luigi.IntParameter(default=10000)

    def requires(self):
        return SortMerchantListsFullContentModel(model_module=self.model_module, 
                                                 model_cls=self.model_cls,
                                                 model_task_id=self.model_task_id, 
                                                 bandit_policy=self.bandit_policy,
                                                 bandit_policy_params=self.bandit_policy_params,
                                                 bandit_weights=self.bandit_weights,
                                                 plot_histogram=self.plot_histogram,
                                                 limit_list_size=self.limit_list_size,
                                                 batch_size=self.batch_size,
                                                 nofilter_iteractions_test=self.nofilter_iteractions_test,
                                                 no_offpolicy_eval=self.no_offpolicy_eval,
                                                 task_hash=self.task_hash)


class GenerateContentEmbeddings(BaseEvaluationTask):
    batch_size: int = luigi.IntParameter(default=10000)
    export_tsne: bool = luigi.BoolParameter(default=False)
    tsne_column_plot: str = luigi.Parameter(default="dish_description")

    def requires(self):
        return ProcessRestaurantContentDataset(), PrepareRestaurantContentDataset()

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.task_name, "restaurant_embeddings.tsv")), \
               luigi.LocalTarget(
                   os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                self.task_name, "restaurant_metadata.tsv"))

    def _generate_content_tensors(self, rows):
        return SortMerchantListsTripletNetInfoContent._generate_content_tensors(self, rows)

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        processed_content_df = pd.read_csv(self.input()[0][0].path)

        literal_eval_array_columns(processed_content_df,
                                   self.model_training.project_config.input_columns
                                   + [self.model_training.project_config.output_column]
                                   + self.model_training.project_config.metadata_columns)

        print("Loading trained model...")
        module = self.model_training.get_trained_module()

        print("Generating embeddings for each merchant...")
        embeddings: List[float] = []
        for indices in tqdm(chunks(range(len(processed_content_df)), self.batch_size),
                            total=math.ceil(len(processed_content_df) / self.batch_size)):
            rows: pd.DataFrame = processed_content_df.iloc[indices]
            inputs = self._generate_content_tensors(rows)
            batch_embeddings: torch.Tensor = module.compute_item_embeddings(inputs)
            embeddings.extend(batch_embeddings.detach().cpu().numpy())

        restaurant_df = pd.read_csv(self.input()[1].path).replace(['\n', '\t'], ' ', regex=True)
        del restaurant_df['item_imagesurl']

        print("Saving the output file...")

        if self.export_tsne:
            self.export_tsne_file(embeddings, restaurant_df)

        np.savetxt(os.path.join(self.output()[0].path), embeddings, delimiter="\t")
        restaurant_df.to_csv(os.path.join(self.output()[1].path), sep='\t', index=False)

    def export_tsne_file(self, embs, metadata):
        t0 = time()
        tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
        Y = tsne.fit_transform(embs)
        t1 = time()
        print("circles in %.2g sec" % (t1 - t0))

        plot_tsne(Y[:, 0], Y[:, 1], metadata[self.tsne_column_plot].reset_index().index).savefig(
            os.path.join(os.path.split(self.output()[0].path)[0], "tsne.jpg"))


class GenerateEmbeddings(BaseEvaluationTask):
    user_embeddings = luigi.BoolParameter(default=False)
    item_embeddings = luigi.BoolParameter(default=False)
    test_size: float = luigi.FloatParameter(default=0.1)

    # def requires(self):
    #     return GenerateIndicesForAccountsAndMerchantsDataset(test_size=self.test_size)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.task_name, "user_embeddings.tsv")), \
               luigi.LocalTarget(
                   os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                self.task_name, "restaurant_embeddings.tsv")), \
               luigi.LocalTarget(
                   os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                self.task_name, "restaurant_metadata.tsv"))

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        print("Loading trained model...")
        module = self.model_training.get_trained_module()

        restaurant_df = pd.read_csv(self.input()[1].path)

        if self.user_embeddings:
            user_embeddings: np.ndarray = module.user_embeddings.weight.data.cpu().numpy()
            np.savetxt(self.output()[0].path, user_embeddings, delimiter="\t")

        if self.item_embeddings:
            item_embeddings: np.ndarray = module.item_embeddings.weight.data.cpu().numpy()
            np.savetxt(self.output()[1].path, item_embeddings, delimiter="\t")

        restaurant_df.to_csv(os.path.join(self.output()[2].path), sep='\t', index=False)
