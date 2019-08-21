import ast
import functools
import json
import math
import os
from itertools import starmap
import multiprocessing as mp
from multiprocessing.pool import Pool
from typing import Dict, Tuple, List

import luigi
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import torch
from tqdm import tqdm

from recommendation.rank_metrics import precision_at_k, average_precision, ndcg_at_k
from recommendation.task.data_preparation.ifood import PrepareIfoodIndexedOrdersTestData, \
    ListAccountMerchantTuplesForIfoodIndexedOrdersTestData
from recommendation.task.evaluation import BaseEvaluationTask
from recommendation.utils import chunks

tqdm.pandas()
ProgressBar().register()


class GenerateRelevanteListFunction(object):

    def __init__(self, scores_per_tuple: Dict[Tuple[int, int], float]) -> None:
        self.scores_per_tuple = scores_per_tuple

    def generate(self, account_idx: int, ordered_merchant_idx: int, merchant_idx_list: List[int]) -> List[int]:
        scores = list(map(lambda merchant_idx: self.scores_per_tuple[(account_idx, merchant_idx)], merchant_idx_list))
        sorted_merchant_idx_list = [merchant_idx for _, merchant_idx in
                                    sorted(zip(scores, merchant_idx_list), reverse=True)]
        return [1 if merchant_idx == ordered_merchant_idx else 0 for merchant_idx in sorted_merchant_idx_list]


class GenerateRelevanceListsForIfoodModel(BaseEvaluationTask):
    batch_size: int = luigi.IntParameter(default=100000)

    # num_processes: int = luigi.IntParameter(default=os.cpu_count())

    def requires(self):
        return PrepareIfoodIndexedOrdersTestData(), ListAccountMerchantTuplesForIfoodIndexedOrdersTestData()

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.model_task_id, "orders_with_relevance_lists.csv"))

    def _evaluate_account_merchant_tuples(self) -> Dict[Tuple[int, int], float]:
        print("Reading tuples files...")
        tuples_df = pd.read_parquet(self.input()[1].path)

        assert self.model_training.project_config.input_columns[0].name == "account_idx"
        assert self.model_training.project_config.input_columns[1].name == "merchant_idx"

        print("Loading trained model...")
        module = self.model_training.get_trained_module()
        scores: List[float] = []
        print("Running the model for every account and merchant tuple...")
        for indices in tqdm(chunks(range(len(tuples_df)), self.batch_size),
                            total=math.ceil(len(tuples_df) / self.batch_size)):
            rows: pd.DataFrame = tuples_df.iloc[indices]
            batch_scores: torch.Tensor = module(
                torch.tensor(rows["account_idx"].values, dtype=torch.int64).to(self.model_training.torch_device),
                torch.tensor(rows["merchant_idx"].values, dtype=torch.int64).to(self.model_training.torch_device))
            scores.extend(batch_scores.detach().cpu().numpy().tolist())

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

        print("Generating the relevance lists...")
        orders_df["relevance_list"] = list(tqdm(
            starmap(GenerateRelevanteListFunction(scores_per_tuple).generate,
                    zip(orders_df["account_idx"], orders_df["merchant_idx"], orders_df["merchant_idx_list"])),
            total=len(orders_df)))

        # with mp.Manager() as manager:
        #     shared_scores_per_tuple: Dict[Tuple[int, int], float] = manager.dict(scores_per_tuple)
        #     with manager.Pool(self.num_processes) as p:
        #         orders_df["relevance_list"] = list(tqdm(
        #                     p.starmap(GenerateRelevanteListFunction(shared_scores_per_tuple).generate,
        #                               zip(orders_df["account_idx"], orders_df["merchant_idx"], orders_df["merchant_idx_list"])),
        #                     total=len(orders_df)))

        print("Saving the output file...")
        orders_df[["order_id", "relevance_list"]].to_csv(self.output().path, index=False)


class EvaluateIfoodModel(BaseEvaluationTask):
    num_processes: int = luigi.IntParameter(default=os.cpu_count())

    def requires(self):
        return GenerateRelevanceListsForIfoodModel(model_module=self.model_module, model_cls=self.model_cls,
                                                   model_task_id=self.model_task_id)

    def output(self):
        model_path = os.path.join("output", "evaluation", self.__class__.__name__, "results",
                                  self.model_task_id)
        return luigi.LocalTarget(os.path.join(model_path, "orders_with_metrics.csv")), \
               luigi.LocalTarget(os.path.join(model_path, "metrics.json")),

    def run(self):
        os.makedirs(os.path.split(self.output()[0].path)[0], exist_ok=True)

        df: pd.DataFrame = pd.read_csv(self.input().path)

        with Pool(self.num_processes) as p:
            df["relevance_list"] = list(tqdm(p.map(ast.literal_eval, df["relevance_list"]), total=len(df)))

            # df["precision_at_5"] = list(
            #     tqdm(p.map(functools.partial(precision_at_k, k=5), df["relevance_list"]), total=len(df)))
            # df["precision_at_10"] = list(
            #     tqdm(p.map(functools.partial(precision_at_k, k=10), df["relevance_list"]), total=len(df)))
            # df["precision_at_15"] = list(
            #     tqdm(p.map(functools.partial(precision_at_k, k=15), df["relevance_list"]), total=len(df)))

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

        df = df.drop(columns=["relevance_list"])

        metrics = {
            # "precision_at_5": df["precision_at_5"].mean(),
            # "precision_at_10": df["precision_at_10"].mean(),
            # "precision_at_15": df["precision_at_15"].mean(),
            "average_precision": df["average_precision"].mean(),
            "ndcg_at_5": df["ndcg_at_5"].mean(),
            "ndcg_at_10": df["ndcg_at_10"].mean(),
            "ndcg_at_15": df["ndcg_at_15"].mean(),
            "ndcg_at_20": df["ndcg_at_20"].mean(),
            "ndcg_at_50": df["ndcg_at_50"].mean(),
        }

        df.to_csv(self.output()[0].path)
        with open(self.output()[1].path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
