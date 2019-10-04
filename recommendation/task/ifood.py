import ast
import functools
import json
import math
import os
from itertools import starmap
import multiprocessing as mp
from multiprocessing.pool import Pool
from typing import Dict, Tuple, List
import numpy as np

import luigi
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm
import timeit

from recommendation.rank_metrics import precision_at_k, average_precision, ndcg_at_k
from recommendation.task.data_preparation.ifood import PrepareIfoodIndexedOrdersTestData, \
    ListAccountMerchantTuplesForIfoodIndexedOrdersTestData, IndexAccountsAndMerchantsOfSessionTrainDataset, \
    CreateInteractionDataset
from recommendation.task.evaluation import BaseEvaluationTask
from recommendation.utils import chunks
from recommendation.plot import plot_histogram


def _generate_relevance_list(account_idx: int, ordered_merchant_idx: int, merchant_idx_list: List[int],
                             scores_per_tuple: Dict[Tuple[int, int], float]) -> List[int]:
    scores = list(map(lambda merchant_idx: scores_per_tuple.get((account_idx, merchant_idx), -1.0), merchant_idx_list))
    sorted_merchant_idx_list = [merchant_idx for _, merchant_idx in
                                sorted(zip(scores, merchant_idx_list), reverse=True)]
    return [1 if merchant_idx == ordered_merchant_idx else 0 for merchant_idx in sorted_merchant_idx_list]


def _generate_random_relevance_list(ordered_merchant_idx: int, merchant_idx_list: List[int]) -> List[int]:
    np.random.shuffle(merchant_idx_list)
    return [1 if merchant_idx == ordered_merchant_idx else 0 for merchant_idx in merchant_idx_list]


def _generate_relevance_list_from_merchant_scores(ordered_merchant_idx: int, merchant_idx_list: List[int],
                                                  scores_per_merchant: Dict[int, float]) -> List[int]:
    scores = list(map(lambda merchant_idx: scores_per_merchant[merchant_idx], merchant_idx_list))
    sorted_merchant_idx_list = [merchant_idx for _, merchant_idx in
                                sorted(zip(scores, merchant_idx_list), reverse=True)]
    return [1 if merchant_idx == ordered_merchant_idx else 0 for merchant_idx in sorted_merchant_idx_list]


class GenerateRelevanceListsForIfoodModel(BaseEvaluationTask):
    batch_size: int = luigi.IntParameter(default=100000)

    # num_processes: int = luigi.IntParameter(default=os.cpu_count())

    def requires(self):
        test_size = self.model_training.requires().session_test_size
        return PrepareIfoodIndexedOrdersTestData(test_size=test_size), \
               ListAccountMerchantTuplesForIfoodIndexedOrdersTestData(test_size=test_size)

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
            inputs = [torch.tensor(rows[input_column.name].values, dtype=torch.int64)
                          .to(self.model_training.torch_device)
                      for input_column in self.model_training.project_config.input_columns]
            batch_scores: torch.Tensor = module(*inputs)
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

        print("Generating the relevance lists...")
        orders_df["relevance_list"] = list(tqdm(
            starmap(functools.partial(_generate_relevance_list, scores_per_tuple=scores_per_tuple),
                    zip(orders_df["account_idx"], orders_df["merchant_idx"], orders_df["merchant_idx_list"])),
            total=len(orders_df)))

        # with mp.Manager() as manager:
        #     shared_scores_per_tuple: Dict[Tuple[int, int], float] = manager.dict(scores_per_tuple)
        #     with manager.Pool(self.num_processes) as p:
        #         orders_df["relevance_list"] = list(tqdm(
        #             starmap(functools.partial(_generate_relevance_list, scores_per_tuple=shared_scores_per_tuple),
        #                     zip(orders_df["account_idx"], orders_df["merchant_idx"], orders_df["merchant_idx_list"])),
        #             total=len(orders_df)))

        print("Saving the output file...")

        plot_histogram(scores_per_tuple.values()).savefig(os.path.join(os.path.split(self.output().path)[0], "scores_histogram.jpg"))
        orders_df[["session_id", "relevance_list"]].to_csv(self.output().path, index=False)


class GenerateReconstructedInteractionMatrix(GenerateRelevanceListsForIfoodModel):
    variational = luigi.BoolParameter(default=False)
    attentive = luigi.BoolParameter(default=False)
    context = luigi.BoolParameter(default=False)
    
    batch_size: int = luigi.IntParameter(default=500)

    def _eval_buys_per_merchant_column(self, df: pd.DataFrame):
        if type(df.iloc[0]["buys_per_merchant"]) is str:
            df["buys_per_merchant"] = df["buys_per_merchant"].apply(lambda value: ast.literal_eval(value))
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
            "count": len(df),
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


class GenerateRandomRelevanceLists(luigi.Task):
    test_size: float = luigi.FloatParameter(default=0.2)

    def requires(self):
        return PrepareIfoodIndexedOrdersTestData(test_size=self.test_size)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.task_id, "orders_with_relevance_lists.csv"))

    def run(self):
        os.makedirs(os.path.split(self.output().path)[0], exist_ok=True)

        print("Reading the orders DataFrame...")
        orders_df: pd.DataFrame = pd.read_parquet(self.input().path)

        print("Filtering orders where the ordered merchant isn't in the list...")
        orders_df = orders_df[orders_df.apply(lambda row: row["merchant_idx"] in row["merchant_idx_list"], axis=1)]

        print("Generating the relevance lists...")
        orders_df["relevance_list"] = list(tqdm(
            starmap(_generate_random_relevance_list,
                    zip(orders_df["merchant_idx"], orders_df["merchant_idx_list"])),
            total=len(orders_df)))

        print("Saving the output file...")
        orders_df[["session_id", "relevance_list"]].to_csv(self.output().path, index=False)


class EvaluateIfoodCDAEModel(EvaluateIfoodModel):
    def requires(self):
        return GenerateReconstructedInteractionMatrix(model_module=self.model_module, model_cls=self.model_cls,
                                                      model_task_id=self.model_task_id)
                        
class EvaluateIfoodCVAEModel(EvaluateIfoodModel):
    def requires(self):
        return GenerateReconstructedInteractionMatrix(model_module=self.model_module, model_cls=self.model_cls,
                                                      model_task_id=self.model_task_id,
                                                      variational=True)

class EvaluateIfoodAttCVAEModel(EvaluateIfoodModel):
    def requires(self):
        return GenerateReconstructedInteractionMatrix(model_module=self.model_module, model_cls=self.model_cls,
                                                      model_task_id=self.model_task_id,
                                                      attentive=True)

class EvaluateIfoodHybridCVAEModel(EvaluateIfoodModel):
    def requires(self):
        return GenerateReconstructedInteractionMatrix(model_module=self.model_module, model_cls=self.model_cls,
                                                      model_task_id=self.model_task_id, context=True,
                                                      variational=True)

class EvaluateRandomIfoodModel(EvaluateIfoodModel):
    model_task_id: str = luigi.Parameter(default="none")

    def requires(self):
        return GenerateRandomRelevanceLists()


class GenerateMostPopularRelevanceLists(luigi.Task):
    model_task_id: str = luigi.Parameter(default="none")
    test_size: float = luigi.FloatParameter(default=0.2)

    def requires(self):
        return CreateInteractionDataset(test_size=self.test_size), \
               PrepareIfoodIndexedOrdersTestData(test_size=self.test_size)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", "evaluation", self.__class__.__name__, "results",
                         self.task_id, "orders_with_relevance_lists.csv"))

    def run(self):
        os.makedirs(os.path.split(self.output().path)[0], exist_ok=True)

        print("Reading the interactions DataFrame...")
        interactions_df: pd.DataFrame = pd.read_csv(self.input()[0].path)
        print("Generating the scores")
        scores: pd.Series = interactions_df.groupby("merchant_idx")["buys"].sum()
        scores_dict: Dict[int, float] = {merchant_idx: score for merchant_idx, score
                                         in tqdm(zip(scores.index, scores),
                                                 total=len(scores))}

        print("Reading the orders DataFrame...")
        orders_df: pd.DataFrame = pd.read_parquet(self.input()[1].path)

        print("Filtering orders where the ordered merchant isn't in the list...")
        orders_df = orders_df[orders_df.apply(lambda row: row["merchant_idx"] in row["merchant_idx_list"], axis=1)]

        print("Generating the relevance lists...")
        orders_df["relevance_list"] = list(tqdm(
            starmap(functools.partial(_generate_relevance_list_from_merchant_scores, scores_per_merchant=scores_dict),
                    zip(orders_df["merchant_idx"], orders_df["merchant_idx_list"])),
            total=len(orders_df)))

        print("Saving the output file...")
        orders_df[["session_id", "relevance_list"]].to_csv(self.output().path, index=False)


class EvaluateMostPopularIfoodModel(EvaluateIfoodModel):
    model_task_id: str = luigi.Parameter(default="none")

    def requires(self):
        return GenerateMostPopularRelevanceLists()
