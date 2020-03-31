import functools
import os
from multiprocessing.pool import Pool
from typing import List

import json
import luigi
import numpy as np
import pandas as pd
from tqdm import tqdm

from recommendation.fairness_metrics import calculate_fairness_metrics
from recommendation.files import get_test_set_predictions_path
from recommendation.rank_metrics import average_precision, precision_at_k, ndcg_at_k, prediction_coverage_at_k, \
    personalization_at_k
from recommendation.task.model.base import BaseEvaluationTask
from recommendation.utils import parallel_literal_eval


class EvaluateTestSetPredictions(BaseEvaluationTask):
    num_processes: int = luigi.IntParameter(default=os.cpu_count())

    fairness_columns: List[str] = luigi.ListParameter()

    def run(self):
        os.makedirs(self.output().path)

        df: pd.DataFrame = pd.read_csv(get_test_set_predictions_path(self.model_training.output().path))
        df["sorted_actions"] = parallel_literal_eval(df["sorted_actions"])
        df["prob_actions"] = parallel_literal_eval(df["prob_actions"])
        df["action"] = df["sorted_actions"].apply(lambda sorted_actions: sorted_actions[0])

        if self.model_training.metadata_data_frame is not None:
            df = pd.merge(df, self.model_training.metadata_data_frame, left_on="action",
                          right_on=self.model_training.project_config.item_column.name, suffixes=("", "_action"))

        fairness_metrics = calculate_fairness_metrics(df, self.fairness_columns,
                                                      self.model_training.project_config.item_column.name, "action")
        fairness_metrics.to_csv(os.path.join(self.output().path, "fairness_metrics.csv"), index=False)

        with Pool(self.num_processes) as p:
            print("Creating the relevance lists...")
            df["relevance_list"] = list(
                tqdm(p.starmap(_create_relevance_list, zip(df["sorted_actions"],
                           df[self.model_training.project_config.item_column.name])), total=len(df)))

            print("Calculating average precision...")
            df["average_precision"] = list(
                tqdm(p.map(average_precision, df["relevance_list"]), total=len(df)))

            print("Calculating precision at 1...")
            df["precision_at_1"] = list(
                tqdm(p.map(functools.partial(precision_at_k, k=1), df["relevance_list"]), total=len(df)))

            print("Calculating nDCG at 5...")
            df["ndcg_at_5"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=5), df["relevance_list"]), total=len(df)))
            print("Calculating nDCG at 10...")
            df["ndcg_at_10"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=10), df["relevance_list"]), total=len(df)))
            print("Calculating nDCG at 15...")
            df["ndcg_at_15"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=15), df["relevance_list"]), total=len(df)))
            print("Calculating nDCG at 20...")
            df["ndcg_at_20"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=20), df["relevance_list"]), total=len(df)))
            print("Calculating nDCG at 50...")
            df["ndcg_at_50"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=50), df["relevance_list"]), total=len(df)))

            print("Calculate ps policy eval...")
            df["ps_eval"] = list(tqdm(
                p.starmap(_ps_policy_eval, zip(df["relevance_list"], df["prob_actions"])), total=len(df)))

        catalog = list(range(df.iloc[0]["n_items"]))

        metrics = {
            "model_task": self.model_task_id,
            "count": len(df),
            "mean_average_precision": df["average_precision"].mean(),
            "precision_at_1": df["precision_at_1"].mean(),
            "ndcg_at_5": df["ndcg_at_5"].mean(),
            "ndcg_at_10": df["ndcg_at_10"].mean(),
            "ndcg_at_15": df["ndcg_at_15"].mean(),
            "ndcg_at_20": df["ndcg_at_20"].mean(),
            "ndcg_at_50": df["ndcg_at_50"].mean(),
            "coverage_at_5": prediction_coverage_at_k(df["sorted_actions"], catalog, 5),
            "coverage_at_10": prediction_coverage_at_k(df["sorted_actions"], catalog, 10),
            "coverage_at_15": prediction_coverage_at_k(df["sorted_actions"], catalog, 15),
            "coverage_at_20": prediction_coverage_at_k(df["sorted_actions"], catalog, 20),
            "coverage_at_50": prediction_coverage_at_k(df["sorted_actions"], catalog, 50),
            "personalization_at_5": personalization_at_k(df["sorted_actions"], 5),
            "personalization_at_10": personalization_at_k(df["sorted_actions"], 10),
            "personalization_at_15": personalization_at_k(df["sorted_actions"], 15),
            "personalization_at_20": personalization_at_k(df["sorted_actions"], 20),
            "personalization_at_50": personalization_at_k(df["sorted_actions"], 50),
        }

        with open(os.path.join(self.output().path, "metrics.json"), "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)


def _create_relevance_list(sorted_actions: List[int], expected_action: int) -> List[int]:
    return [1 if action == expected_action else 0 for action in sorted_actions]


def _ps_policy_eval(relevance_list: List[int], prob_actions: List[float]) -> List[float]:
    return np.sum(np.array(relevance_list) * np.array(prob_actions[:len(relevance_list)])).tolist()

