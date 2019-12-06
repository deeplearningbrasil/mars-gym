import functools
import functools
import json
import math
import os
from itertools import starmap
from multiprocessing.pool import Pool
from typing import Dict, Tuple, List
from sklearn import manifold
import pprint
import luigi
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random
from time import time
from recommendation.data import literal_eval_array_columns
from recommendation.plot import plot_histogram, plot_tsne
from recommendation.rank_metrics import average_precision, ndcg_at_k, prediction_coverage_at_k, personalization_at_k
from recommendation.task.data_preparation.ifood import PrepareIfoodIndexedOrdersTestData, \
    ListAccountMerchantTuplesForIfoodIndexedOrdersTestData, ProcessRestaurantContentDataset, \
    PrepareRestaurantContentDataset, CreateShiftIndices, \
    CreateInteractionDataset, GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset,\
    IndexAccountsAndMerchantsOfSessionTrainDataset
from recommendation.task.evaluation import BaseEvaluationTask
from recommendation.utils import chunks, parallel_literal_eval


def _sort_merchants_by_tuple_score(account_idx: int, merchant_idx_list: List[int],
                                   scores_per_tuple: Dict[Tuple[int, int], float]) -> List[int]:
    scores = list(map(lambda merchant_idx: scores_per_tuple.get((account_idx, merchant_idx), -1.0), merchant_idx_list))
    return [merchant_idx for _, merchant_idx in sorted(zip(scores, merchant_idx_list), reverse=True)]


def _sort_merchants_by_merchant_score(merchant_idx_list: List[int], scores_per_merchant: Dict[int, float]) -> List[int]:
    scores = list(map(lambda merchant_idx: scores_per_merchant[merchant_idx], merchant_idx_list))
    return [merchant_idx for _, merchant_idx in
            sorted(zip(scores, merchant_idx_list), reverse=True)]


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

    # num_processes: int = luigi.IntParameter(default=os.cpu_count())

    def requires(self):
        test_size = self.model_training.requires().session_test_size
        minimum_interactions = self.model_training.requires().minimum_interactions
        return PrepareIfoodIndexedOrdersTestData(test_size=test_size, minimum_interactions=minimum_interactions), \
               ListAccountMerchantTuplesForIfoodIndexedOrdersTestData(test_size=test_size,
                                                                      minimum_interactions=minimum_interactions)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.model_task_id, "orders_with_sorted_merchants.csv"))

    def _generate_batch_tensors(self, rows: pd.DataFrame, pool: Pool) -> List[torch.Tensor]:
        return [torch.tensor(rows[input_column.name].values, dtype=torch.int64)
                    .to(self.model_training.torch_device)
                for input_column in self.model_training.project_config.input_columns]

    def _read_and_transform_tuples(self):
        tuples_df = pd.read_parquet(self.input()[1].path)
        return tuples_df

    def _get_batch_score(self, batch_scores):
        return batch_scores

    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        print("Reading tuples files...")
        tuples_df = self._read_and_transform_tuples()

        assert self.model_training.project_config.input_columns[0].name == "account_idx"
        assert self.model_training.project_config.input_columns[1].name == "merchant_idx"

        print("Loading trained model...")
        module = self.model_training.get_trained_module()
        scores: List[float] = []
        print("Running the model for every account and merchant tuple...")
        with Pool(os.cpu_count()) as pool:
            for indices in tqdm(chunks(range(len(tuples_df)), self.batch_size),
                                total=math.ceil(len(tuples_df) / self.batch_size)):
                rows: pd.DataFrame = tuples_df.iloc[indices]
                inputs = self._generate_batch_tensors(rows, pool)
                batch_scores: torch.Tensor = self._get_batch_score(module(*inputs))
                scores.extend(batch_scores.detach().cpu().numpy())
                 
        print("Creating the dictionary of scores...")
        return {(account_idx, merchant_idx): score for account_idx, merchant_idx, score
                in tqdm(zip(tuples_df["account_idx"], tuples_df["merchant_idx"], scores), total=len(scores))}

    def run(self):
        os.makedirs(os.path.split(self.output().path)[0], exist_ok=True)

        scores_per_tuple = self._evaluate_account_merchant_tuples()

        print("Reading the orders DataFrame...")
        orders_df: pd.DataFrame = pd.read_parquet(self.input()[0].path)

        print("Filtering orders where the ordered merchant isn't in the list...")
        orders_df = orders_df[orders_df.apply(lambda row: row["merchant_idx"] in row["merchant_idx_list"], axis=1)]

        print("Sorting the merchant lists")
        orders_df["sorted_merchant_idx_list"] = list(tqdm(
            starmap(functools.partial(_sort_merchants_by_tuple_score, scores_per_tuple=scores_per_tuple),
                    zip(orders_df["account_idx"], orders_df["merchant_idx_list"])),
            total=len(orders_df)))

        print("Creating the relevance lists...")
        orders_df["relevance_list"] = list(tqdm(
            starmap(_create_relevance_list, zip(orders_df["sorted_merchant_idx_list"], orders_df["merchant_idx"])),
            total=len(orders_df)))

        # with mp.Manager() as manager:
        #     shared_scores_per_tuple: Dict[Tuple[int, int], float] = manager.dict(scores_per_tuple)
        #     with manager.Pool(self.num_processes) as p:
        #         orders_df["relevance_list"] = list(tqdm(
        #             starmap(functools.partial(_generate_relevance_list, scores_per_tuple=shared_scores_per_tuple),
        #                     zip(orders_df["account_idx"], orders_df["merchant_idx"], orders_df["merchant_idx_list"])),
        #             total=len(orders_df)))

        print("Saving the output file...")

        if self.plot_histogram:
            plot_histogram(scores_per_tuple.values()).savefig(
                os.path.join(os.path.split(self.output().path)[0], "scores_histogram.jpg"))

        orders_df[["session_id", "sorted_merchant_idx_list", "relevance_list", "shift_idx", "day_of_week"]].to_csv(
            self.output().path, index=False)


class SortMerchantListsForAutoEncoderIfoodModel(SortMerchantListsForIfoodModel):
    variational = luigi.BoolParameter(default=False)
    attentive = luigi.BoolParameter(default=False)
    context = luigi.BoolParameter(default=False)

    batch_size: int = luigi.IntParameter(default=500)

    def _eval_buys_per_merchant_column(self, df: pd.DataFrame):
        if len(df) > 0 and type(df.iloc[0]["buys_per_merchant"]) is str:
            df["buys_per_merchant"] = parallel_literal_eval(df["buys_per_merchant"])
        return df

    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        assert self.model_training.project_config.input_columns[0].name == "buys_per_merchant"
        assert self.model_training.project_config.output_column.name == "buys_per_merchant"

        print("Reading tuples files...")
        tuples_df = pd.read_parquet(self.input()[1].path)

        print("Grouping by account index...")
        merchant_indices_per_account_idx: pd.Series = tuples_df.groupby('account_idx')['merchant_idx'].apply(list)
        del tuples_df

        print("Reading train, val and test DataFrames...")
        train_df = self._eval_buys_per_merchant_column(pd.read_csv(self.model_training.input()[0].path))
        val_df = self._eval_buys_per_merchant_column(pd.read_csv(self.model_training.input()[1].path))
        test_df = self._eval_buys_per_merchant_column(pd.read_csv(self.model_training.input()[2].path))
        df: pd.DataFrame = pd.concat((train_df, val_df, test_df))

        # Needed if split_per_user=False
        df = df.groupby("account_idx")["buys_per_merchant"] \
            .apply(lambda lists: [inner for outer in lists for inner in outer]).reset_index()

        print("Loading trained model...")
        module = self.model_training.get_trained_module()

        scores_per_tuple: List[Tuple[Tuple[int, int], float]] = []

        print("Running the model for every account and merchant tuple...")
        for indices in tqdm(chunks(range(len(df)), self.batch_size),
                            total=math.ceil(len(df) / self.batch_size)):
            rows: pd.DataFrame = df.iloc[indices]

            i, j, data = zip(
                *((index, int(t[0]), t[1]) for index, row in enumerate(rows["buys_per_merchant"])
                  for t in row))
            batch_tensor = torch.sparse_coo_tensor(
                indices=torch.tensor([i, j]),
                values=torch.tensor(data),
                size=[len(rows), self.model_training.n_items]).to(self.model_training.torch_device)

            batch_output_tensor = None
            if self.context:
                batch_context = torch.tensor(rows['account_idx'].values).to(self.model_training.torch_device)
                batch_output_tensor = module(batch_tensor, batch_context)
            else:
                batch_output_tensor = module(batch_tensor)

            if self.attentive:
                batch_output_tensor, _, _, _, _ = batch_output_tensor
            elif self.variational:
                batch_output_tensor, _, _ = batch_output_tensor

            batch_output: np.ndarray = batch_output_tensor.detach().cpu().numpy()

            for account_idx, row in zip(rows["account_idx"], batch_output):
                if account_idx in merchant_indices_per_account_idx:
                    merchant_indices = merchant_indices_per_account_idx[account_idx]
                    scores_per_tuple.extend([((account_idx, merchant_idx), row[merchant_idx])
                                             for merchant_idx in merchant_indices])

        print("Creating the dictionary of scores...")
        return dict(scores_per_tuple)


class SortMerchantListsTripletWeightedModel(SortMerchantListsForIfoodModel):

    def _generate_batch_tensors(self, rows: pd.DataFrame, pool: Pool) -> List[torch.Tensor]:
        assert self.model_training.project_config.input_columns[0].name == 'account_idx'
        assert self.model_training.project_config.input_columns[1].name == 'merchant_idx'

        return [torch.tensor(rows[column].values, dtype=torch.int64)
                    .to(self.model_training.torch_device)
                for column in ['account_idx', 'merchant_idx']]


class EvaluateIfoodModel(BaseEvaluationTask):
    num_processes: int = luigi.IntParameter(default=os.cpu_count())

    def requires(self):
        return SortMerchantListsForIfoodModel(model_module=self.model_module, model_cls=self.model_cls,
                                              model_task_id=self.model_task_id)

    def output(self):
        model_path = os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                  self.model_task_id)
        return luigi.LocalTarget(os.path.join(model_path, "orders_with_metrics.csv")), \
               luigi.LocalTarget(os.path.join(model_path, "metrics.json")),

    def read_evaluation_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.input().path)

    @property
    def n_items(self):
        return self.model_training.n_items

    def _mean_personalization(self, df: pd.DataFrame, k: int):
        grouped_df = df.groupby(["shift_idx", "day_of_week"])
        personalization_per_shift: List[float] = []
        for _, group_df in grouped_df:
            if len(group_df["sorted_merchant_idx_list"]) > 1:
                personalization_per_shift.append(personalization_at_k(group_df["sorted_merchant_idx_list"], k))
        return np.mean(personalization_per_shift)

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

        catalog = range(self.n_items)

        metrics = {
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
        print("Metrics")
        pprint.pprint(metrics)
        print("")

        print("")

        df = df.drop(columns=["sorted_merchant_idx_list", "relevance_list"])
        df.to_csv(self.output()[0].path)
        with open(self.output()[1].path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)


class SortMerchantListsRandomly(luigi.Task):
    test_size: float = luigi.FloatParameter(default=0.1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return PrepareIfoodIndexedOrdersTestData(test_size=self.test_size,
                                                 minimum_interactions=self.minimum_interactions)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.task_id, "orders_with_sorted_merchants.csv"))

    def random(self, l):
        np.random.shuffle(l)
        return list(l)

    def run(self):
        os.makedirs(os.path.split(self.output().path)[0], exist_ok=True)

        print("Reading the orders DataFrame...")
        orders_df: pd.DataFrame = pd.read_parquet(self.input().path)

        print("Filtering orders where the ordered merchant isn't in the list...")
        orders_df = orders_df[orders_df.apply(lambda row: row["merchant_idx"] in row["merchant_idx_list"], axis=1)]

        print("Sorting the merchant lists")
        orders_df["sorted_merchant_idx_list"] = list(tqdm(
            map(self.random, orders_df["merchant_idx_list"]),
            total=len(orders_df)))

        print("Creating the relevance lists...")
        orders_df["relevance_list"] = list(tqdm(
            starmap(_create_relevance_list, zip(orders_df["sorted_merchant_idx_list"], orders_df["merchant_idx"])),
            total=len(orders_df)))

        print("Saving the output file...")
        orders_df[["session_id", "sorted_merchant_idx_list", "relevance_list", "shift_idx", "day_of_week"]].to_csv(
            self.output().path, index=False)


class EvaluateIfoodCDAEModel(EvaluateIfoodModel):
    def requires(self):
        return SortMerchantListsForAutoEncoderIfoodModel(model_module=self.model_module, model_cls=self.model_cls,
                                                         model_task_id=self.model_task_id)


class EvaluateIfoodCVAEModel(EvaluateIfoodModel):
    def requires(self):
        return SortMerchantListsForAutoEncoderIfoodModel(model_module=self.model_module, model_cls=self.model_cls,
                                                         model_task_id=self.model_task_id,
                                                         variational=True)

class EvaluateIfoodAttCVAEModel(EvaluateIfoodModel):
    def requires(self):
        return SortMerchantListsForAutoEncoderIfoodModel(model_module=self.model_module, model_cls=self.model_cls,
                                                         model_task_id=self.model_task_id,
                                                         attentive=True)


class EvaluateIfoodHybridCVAEModel(EvaluateIfoodModel):
    def requires(self):
        return SortMerchantListsForAutoEncoderIfoodModel(model_module=self.model_module, model_cls=self.model_cls,
                                                         model_task_id=self.model_task_id, context=True,
                                                         variational=True)


class EvaluateRandomIfoodModel(EvaluateIfoodModel):
    model_task_id: str = luigi.Parameter(default="none")
    test_size: float = luigi.FloatParameter(default=0.1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return SortMerchantListsRandomly(test_size=self.test_size, minimum_interactions=self.minimum_interactions), \
               GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(test_size=self.test_size,
                                                                           minimum_interactions=self.minimum_interactions)

    def read_evaluation_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.input()[0].path)

    @property
    def n_items(self):
        return len(pd.read_csv(self.input()[1][1].path))


class SortMerchantListsByMostPopular(luigi.Task):
    model_task_id: str = luigi.Parameter(default="none")
    test_size: float = luigi.FloatParameter(default=0.1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return CreateInteractionDataset(test_size=self.test_size, minimum_interactions=self.minimum_interactions), \
               PrepareIfoodIndexedOrdersTestData(test_size=self.test_size,
                                                 minimum_interactions=self.minimum_interactions)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.task_id, "orders_with_sorted_merchants.csv"))

    def run(self):
        os.makedirs(os.path.split(self.output().path)[0], exist_ok=True)

        print("Reading the interactions DataFrame...")
        interactions_df: pd.DataFrame = pd.read_parquet(self.input()[0].path)
        print("Generating the scores")
        scores: pd.Series = interactions_df.groupby("merchant_idx")["buys"].sum()
        scores_dict: Dict[int, float] = {merchant_idx: score for merchant_idx, score
                                         in tqdm(zip(scores.index, scores),
                                                 total=len(scores))}

        print("Reading the orders DataFrame...")
        orders_df: pd.DataFrame = pd.read_parquet(self.input()[1].path)

        print("Filtering orders where the ordered merchant isn't in the list...")
        orders_df = orders_df[orders_df.apply(lambda row: row["merchant_idx"] in row["merchant_idx_list"], axis=1)]

        print("Sorting the merchant lists")
        orders_df["sorted_merchant_idx_list"] = list(tqdm(
            starmap(functools.partial(_sort_merchants_by_merchant_score, scores_per_merchant=scores_dict),
                    zip(orders_df["merchant_idx_list"])),
            total=len(orders_df)))

        print("Creating the relevance lists...")
        orders_df["relevance_list"] = list(tqdm(
            starmap(_create_relevance_list, zip(orders_df["sorted_merchant_idx_list"], orders_df["merchant_idx"])),
            total=len(orders_df)))

        print("Saving the output file...")
        orders_df[["session_id", "sorted_merchant_idx_list", "relevance_list", "shift_idx", "day_of_week"]].to_csv(
            self.output().path, index=False)


class EvaluateMostPopularIfoodModel(EvaluateIfoodModel):
    model_task_id: str = luigi.Parameter(default="none")
    test_size: float = luigi.FloatParameter(default=0.1)
    minimum_interactions: int = luigi.FloatParameter(default=5)

    def requires(self):
        return SortMerchantListsByMostPopular(test_size=self.test_size, minimum_interactions=self.minimum_interactions), \
               GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                   test_size=self.test_size, minimum_interactions=self.minimum_interactions)

    def read_evaluation_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.input()[0].path)

    @property
    def n_items(self):
        return len(pd.read_csv(self.input()[1][1].path))


class SortMerchantListsByMostPopularPerUser(luigi.Task):
    model_task_id: str = luigi.Parameter(default="none")
    test_size: float = luigi.FloatParameter(default=0.1)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    buy_importance: float = luigi.FloatParameter(default=1.0)
    visit_importance: float = luigi.FloatParameter(default=0.0)

    def requires(self):
        return CreateInteractionDataset(test_size=self.test_size, minimum_interactions=self.minimum_interactions), \
               PrepareIfoodIndexedOrdersTestData(test_size=self.test_size,
                                                 minimum_interactions=self.minimum_interactions)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.task_id + "GenerateMostPopularPerUserRelevanceLists_buy_importance=%.2f_visit_importance=%.2f" % (
                             self.buy_importance,
                             self.visit_importance),
                         "orders_with_sorted_merchants.csv"))

    def run(self):
        os.makedirs(os.path.split(self.output().path)[0], exist_ok=True)

        print("Reading the interactions DataFrame...")
        interactions_df: pd.DataFrame = pd.read_parquet(self.input()[0].path)
        print("Generating the scores")
        grouped_interactions = interactions_df.groupby(["account_idx", "merchant_idx"])
        scores: pd.Series = grouped_interactions["buys"].sum() * self.buy_importance \
                            + grouped_interactions["visits"].sum() * self.visit_importance
        scores_per_tuple: Dict[Tuple[int, int], float] = {(account_idx, merchant_idx): score
                                                          for (account_idx, merchant_idx), score
                                                          in tqdm(zip(scores.index, scores), total=len(scores))}

        print("Reading the orders DataFrame...")
        orders_df: pd.DataFrame = pd.read_parquet(self.input()[1].path)

        print("Filtering orders where the ordered merchant isn't in the list...")
        orders_df = orders_df[orders_df.apply(lambda row: row["merchant_idx"] in row["merchant_idx_list"], axis=1)]

        print("Sorting the merchant lists")
        orders_df["sorted_merchant_idx_list"] = list(tqdm(
            starmap(functools.partial(_sort_merchants_by_tuple_score, scores_per_tuple=scores_per_tuple),
                    zip(orders_df["account_idx"], orders_df["merchant_idx_list"])),
            total=len(orders_df)))

        print("Creating the relevance lists...")
        orders_df["relevance_list"] = list(tqdm(
            starmap(_create_relevance_list, zip(orders_df["sorted_merchant_idx_list"], orders_df["merchant_idx"])),
            total=len(orders_df)))

        print("Saving the output file...")
        orders_df[["session_id", "sorted_merchant_idx_list", "relevance_list", "shift_idx", "day_of_week"]].to_csv(
            self.output().path, index=False)


class EvaluateMostPopularPerUserIfoodModel(EvaluateIfoodModel):
    model_task_id: str = luigi.Parameter(default="none")
    test_size: float = luigi.FloatParameter(default=0.1)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    buy_importance: float = luigi.FloatParameter(default=1.0)
    visit_importance: float = luigi.FloatParameter(default=0.0)

    def requires(self):
        return SortMerchantListsByMostPopularPerUser(test_size=self.test_size,
                                                     minimum_interactions=self.minimum_interactions,
                                                     buy_importance=self.buy_importance,
                                                     visit_importance=self.visit_importance), \
               GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                   test_size=self.test_size,
                   minimum_interactions=self.minimum_interactions)

    def read_evaluation_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.input()[0].path)

    @property
    def n_items(self):
        return len(pd.read_csv(self.input()[1][1].path))

    def output(self):
        model_path = os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                  self.task_id + "_buy_importance=%.6f_visit_importance=%.6f" % (
                                      self.buy_importance,
                                      self.visit_importance))
        return luigi.LocalTarget(os.path.join(model_path, "orders_with_metrics.csv")), \
               luigi.LocalTarget(os.path.join(model_path, "metrics.json")),

import pickle
class GenerateUserEmbeddingsFromContentModel(BaseEvaluationTask):
    test_size:              float  = luigi.FloatParameter(default=0.10)
    minimum_interactions:   int    = luigi.FloatParameter(default=10)
    group_last_k_merchants: int    = luigi.FloatParameter(default=20)
    #use_visit_interactions: bool

    def requires(self):
        test_size            = self.model_training.requires().session_test_size
        minimum_interactions = self.model_training.requires().minimum_interactions   
        
        return (GenerateContentEmbeddings(
                    model_module=self.model_module, 
                    model_cls=self.model_cls,
                    model_task_id=self.model_task_id), 
                IndexAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=test_size, 
                    minimum_interactions=minimum_interactions),
                ProcessRestaurantContentDataset(),
                CreateShiftIndices())

    def output(self):
        return (luigi.LocalTarget(
                    os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.model_task_id, "user_embeddings_{}.tsv".format(self.group_last_k_merchants))), \
               luigi.LocalTarget(
                   os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                self.model_task_id, "user_embeddings_by_shift_{}.pkl".format(self.group_last_k_merchants))))

    def _generate_content_tensors(self, rows):
        inputs = []

        for input_column in self.model_training.project_config.metadata_columns:
            dtype = torch.float32 if input_column.name == "restaurant_complete_info" else torch.int64
            values = rows[input_column.name].values.tolist()
            inputs.append(torch.tensor(values, dtype=dtype).to(self.model_training.torch_device))

        return inputs

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        processed_content_df = pd.read_csv(self.input()[2][0].path)#.set_index('merchant_idx')
        shifts_df            = pd.read_csv(self.input()[3].path).set_index('shift')
        #print(processed_content_df.head())
        #print(processed_content_df.columns)

        #d
        literal_eval_array_columns(processed_content_df,
                                   self.model_training.project_config.input_columns
                                   + [self.model_training.project_config.output_column]
                                   + self.model_training.project_config.metadata_columns)

        print("Loading trained model...")
        module = self.model_training.get_trained_module()

        restaurant_embs = np.genfromtxt(self.input()[0][0].path, dtype='float')
        restaurant_df   = pd.read_csv(self.input()[0][1].path, sep='\t').reset_index().set_index("merchant_id")
        tuples_df       = pd.read_parquet(self.input()[1].path)
        #print(restaurant_df.reset_index().set_index("merchant_id").columns)
        #d
        # filter only buy
        tuples_df       = tuples_df[tuples_df.buy > 0].sort_values('click_timestamp', ascending=False)

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
            
            merchant_id      = group.head(int(self.group_last_k_merchants)).merchant_id.values
            merchant_emb_idx = restaurant_df.loc[merchant_id]['index'].values
            emb              = restaurant_embs[merchant_emb_idx]
            #merchant_idx  = group.head(int(self.group_last_k_merchants)).merchant_idx.values
            #rows          = processed_content_df.iloc[merchant_idx]

            #inputs                         = self._generate_content_tensors(rows)
            #batch_embeddings: torch.Tensor = module.compute_item_embeddings(inputs)
            #emb                            = batch_embeddings.detach().cpu().numpy()
            embeddings[account_idx][shift] = emb
            # i = i + 1
            # if i > 10:
            #     break 
        # Shape numpy embeddings
        #
        # (account, shift, features)
        account_embeddings_by_shift = np.zeros((len(embeddings.keys()), len(shifts_df), emb.shape[1]))
        
        # (account, features)
        account_embeddings_geral    = np.zeros((len(embeddings.keys()), emb.shape[1]))

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
        #np.savetxt(os.path.join(self.output()[1].path), account_embeddings_by_shift, delimiter="\t")

class SortMerchantListsTripletNetInfoContent(SortMerchantListsForIfoodModel):
    batch_size: int = luigi.IntParameter(default=10000)
    group_last_k_merchants: int    = luigi.FloatParameter(default=20)

    def requires(self):
        test_size = self.model_training.requires().session_test_size
        minimum_interactions = self.model_training.requires().minimum_interactions        
        return super().requires() + \
                (GenerateContentEmbeddings(
                    model_module=self.model_module, 
                    model_cls=self.model_cls,
                    model_task_id=self.model_task_id,
                    batch_size=self.batch_size),
                IndexAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=test_size, minimum_interactions=minimum_interactions),
                GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                    test_size=test_size, minimum_interactions=minimum_interactions),
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

        #assert metadata_columns_name == ['trading_name', 'description', 'category_names', 'restaurant_complete_info']

        # account_idxs = torch.tensor(rows["account_idx"].values, dtype=torch.int64) \
        #     .to(self.model_training.torch_device)

        # account_embeddings = []
        # for k, row in rows.iterrows():
        #     account_embeddings.append(self.account_embeddings_by_shift[row.account_idx][row.shift_idx])


        #self.account_embeddings_geral    
        # account_embeddings = torch.from_numpy(self.account_embeddings_geral[rows["account_idx"].values])\
        #                        .to(self.model_training.torch_device)
        
        account_embeddings = self.account_embeddings_by_shift[rows["account_idx"].values, rows['shift_idx'].values]

        account_embeddings = torch.from_numpy(np.array(account_embeddings)).to(self.model_training.torch_device)
        merchant_rows      = self.merchant_df.loc[rows["merchant_idx"]]
        inputs             = self._generate_content_tensors(merchant_rows)

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

        tuples_df        = pd.read_parquet(self.input()[1].path)

        #assert self.model_training.project_config.input_columns[0].name == "account_idx"
        #assert self.model_training.project_config.input_columns[1].name == "merchant_idx"
        print("Loading trained model...")
        module = self.model_training.get_trained_module()
        scores: List[float] = []
        print("Running the model for every account and merchant tuple...")
        with Pool(os.cpu_count()) as pool:
            for indices in tqdm(chunks(range(len(tuples_df)), self.batch_size),
                                total=math.ceil(len(tuples_df) / self.batch_size)):
                rows: pd.DataFrame = tuples_df.iloc[indices]
                account_embeddings, inputs = self._generate_batch_tensors(rows, pool)

                item_embeddings: torch.Tensor = module.compute_item_embeddings(inputs)
                batch_scores:    torch.Tensor = module.similarity(account_embeddings, item_embeddings)
                batch_scores = batch_scores.detach().cpu().numpy()

                scores.extend(batch_scores)

        print("Creating the dictionary of scores...")
        return {(account_idx, merchant_idx): score for account_idx, merchant_idx, score
                in tqdm(zip(tuples_df["account_idx"], tuples_df["merchant_idx"], scores), total=len(scores))}

class SortMerchantListsFullContentModel(SortMerchantListsForIfoodModel):
    batch_size: int = luigi.IntParameter(default=10000)

    def requires(self):
        test_size = self.model_training.requires().session_test_size
        minimum_interactions = self.model_training.requires().minimum_interactions
        return super().requires() + (GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
            test_size=test_size, minimum_interactions=minimum_interactions), \
            CreateInteractionDataset(test_size=test_size))

    def _read_and_transform_tuples(self):
        tuples_df = pd.read_parquet(self.input()[1].path)

        train_interactions_df = pd.read_parquet(self.input()[-1].path)[['account_idx', 'merchant_idx', 'visits', 'buys']]
        train_interactions_df['buys'] = train_interactions_df['buys'].astype(float)
        train_interactions_df['visits'] = train_interactions_df['visits'].astype(float)

        tuples_df = tuples_df.merge(train_interactions_df, on=['account_idx', 'merchant_idx'], how='outer')
        tuples_df.dropna(subset=['session_id'], how='all', inplace=True)
        tuples_df.fillna(0.0)

        return tuples_df

    def _get_batch_score(self, batch_scores):
        return batch_scores[0]

    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        print("Reading merchant data frame...")
        self.merchant_df = pd.read_csv(self.input()[-2][1].path)

        literal_eval_array_columns(self.merchant_df,
                                   self.model_training.project_config.input_columns
                                   + [self.model_training.project_config.output_column]
                                   + self.model_training.project_config.metadata_columns)

        self.merchant_df = self.merchant_df.set_index("merchant_idx")

        return super()._evaluate_account_merchant_tuples()

    def _generate_content_tensors(self, rows, interaction_rows):
        inputs = []

        inputs.append(torch.tensor(interaction_rows["merchant_idx"].values, dtype=torch.int64).to(self.model_training.torch_device))
        
        visits = interaction_rows['visits'].fillna(0).values
        buys = interaction_rows['buys'].fillna(0).values
        
        for input_column in self.model_training.project_config.metadata_columns:
            dtype = torch.float32 if input_column.name == "restaurant_complete_info" else torch.int64
            values = rows[input_column.name].values.tolist()
            inputs.append(torch.tensor(values, dtype=dtype).to(self.model_training.torch_device))

        inputs.append(torch.tensor(visits, dtype=torch.float32).to(self.model_training.torch_device))
        inputs.append(torch.tensor(buys, dtype=torch.float32).to(self.model_training.torch_device))

        return inputs

    def _generate_batch_tensors(self, rows: pd.DataFrame, pool: Pool) -> List[torch.Tensor]:
        metadata_columns_name = [input_column.name for input_column in
                                 self.model_training.project_config.metadata_columns]

        assert metadata_columns_name == ['trading_name', 'description', 'category_names', 'restaurant_complete_info']

        account_idxs = torch.tensor(rows["account_idx"].values, dtype=torch.int64) \
            .to(self.model_training.torch_device)

        merchant_rows = self.merchant_df.loc[rows["merchant_idx"]]

        inputs = self._generate_content_tensors(merchant_rows, rows)

        return [account_idxs, inputs]

class EvaluateIfoodTripletNetInfoContent(EvaluateIfoodModel):
    batch_size: int = luigi.IntParameter(default=10000)
    model_task_id: str = luigi.Parameter(default="none")
    test_size: float = luigi.FloatParameter(default=0.1)
    minimum_interactions: int = luigi.FloatParameter(default=5)
    group_last_k_merchants: int    = luigi.FloatParameter(default=20)

    def requires(self):
        return [SortMerchantListsTripletNetInfoContent(
                    model_module=self.model_module, model_cls=self.model_cls,
                    model_task_id=self.model_task_id, batch_size = self.batch_size,
                    group_last_k_merchants=self.group_last_k_merchants),
                ProcessRestaurantContentDataset(),
                GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(
                   test_size=self.test_size, 
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
            
    def read_evaluation_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.input()[0].path)

class EvaluateIfoodFullContentModel(EvaluateIfoodModel):
    def requires(self):
        return SortMerchantListsFullContentModel(model_module=self.model_module, model_cls=self.model_cls,
                                                    model_task_id=self.model_task_id)

class EvaluateIfoodTripletNetWeightedModel(EvaluateIfoodModel):
    def requires(self):
        return SortMerchantListsTripletWeightedModel(model_module=self.model_module, model_cls=self.model_cls,
                                                     model_task_id=self.model_task_id)

class GenerateContentEmbeddings(BaseEvaluationTask):
    batch_size: int = luigi.IntParameter(default=10000)
    export_tsne: bool = luigi.BoolParameter(default=False)
    tsne_column_plot: str = luigi.Parameter(default="dish_description")

    def requires(self):
        return ProcessRestaurantContentDataset(), PrepareRestaurantContentDataset()

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.model_task_id, "restaurant_embeddings.tsv")), \
               luigi.LocalTarget(
                   os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                self.model_task_id, "restaurant_metadata.tsv"))

    def _generate_content_tensors(self, rows):
        return SortMerchantListsTripletContentModel._generate_content_tensors(self, rows)

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
        tsne    = manifold.TSNE(n_components=2, init='random', random_state=0)
        Y       = tsne.fit_transform(embs)
        t1 = time()
        print("circles in %.2g sec" % (t1 - t0))
        
        plot_tsne(Y[:, 0], Y[:, 1], metadata[self.tsne_column_plot].reset_index().index).savefig(
            os.path.join(os.path.split(self.output()[0].path)[0], "tsne.jpg"))   

class GenerateEmbeddings(BaseEvaluationTask):
    user_embeddings = luigi.BoolParameter(default=False)
    item_embeddings = luigi.BoolParameter(default=False)
    test_size: float = luigi.FloatParameter(default=0.1)

    # def requires(self):
    #     return GenerateIndicesForAccountsAndMerchantsOfSessionTrainDataset(test_size=self.test_size)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.model_task_id, "user_embeddings.tsv")), \
               luigi.LocalTarget(
                   os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                self.model_task_id, "restaurant_embeddings.tsv")), \
               luigi.LocalTarget(
                   os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                self.model_task_id, "restaurant_metadata.tsv"))

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
