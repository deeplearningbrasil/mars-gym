import functools
import json
import os
from itertools import starmap
from multiprocessing.pool import Pool
from typing import List, Tuple

import luigi
import numpy as np
import pandas as pd
from tqdm import tqdm

from recommendation.fairness_metrics import calculate_fairness_metrics
from recommendation.files import get_test_set_predictions_path
from recommendation.offpolicy_metrics import eval_IPS, eval_CIPS, eval_SNIPS, eval_doubly_robust
from recommendation.rank_metrics import average_precision, precision_at_k, ndcg_at_k, prediction_coverage_at_k, \
    personalization_at_k
from recommendation.task.model.base import BaseEvaluationTask
from recommendation.utils import parallel_literal_eval


class EvaluateTestSetPredictions(BaseEvaluationTask):
    num_processes: int = luigi.IntParameter(default=os.cpu_count())

    fairness_columns: List[str] = luigi.ListParameter()

    def run(self):
        os.makedirs(self.output().path)

        df: pd.DataFrame     = pd.read_csv(get_test_set_predictions_path(self.model_training.output().path))


        df["sorted_actions"] = parallel_literal_eval(df["sorted_actions"])
        df["prob_actions"]   = parallel_literal_eval(df["prob_actions"])
        df["action_scores"]  = parallel_literal_eval(df["action_scores"])
        df["action"]         = df["sorted_actions"].apply(lambda sorted_actions: sorted_actions[0])

        if self.model_training.metadata_data_frame is not None:
            df = pd.merge(df, self.model_training.metadata_data_frame, left_on="action",
                          right_on=self.model_training.project_config.item_column.name, suffixes=("", "_action"))

        ground_truth_df  = df[df[self.model_training.project_config.output_column.name] == 1]

        with Pool(self.num_processes) as p:
            print("Creating the relevance lists...")
            ground_truth_df["relevance_list"] = list(
                tqdm(p.starmap(_create_relevance_list,
                               zip(ground_truth_df["sorted_actions"], ground_truth_df[self.model_training.project_config.item_column.name],
                                   ground_truth_df[self.model_training.project_config.output_column.name])),
                     total=len(ground_truth_df)))

            print("Calculating average precision...")
            ground_truth_df["average_precision"] = list(
                tqdm(p.map(average_precision, ground_truth_df["relevance_list"]), total=len(ground_truth_df)))

            print("Calculating precision at 1...")
            ground_truth_df["precision_at_1"] = list(
                tqdm(p.map(functools.partial(precision_at_k, k=1), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))

            print("Calculating nDCG at 5...")
            ground_truth_df["ndcg_at_5"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=5), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))
            print("Calculating nDCG at 10...")
            ground_truth_df["ndcg_at_10"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=10), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))
            print("Calculating nDCG at 15...")
            ground_truth_df["ndcg_at_15"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=15), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))
            print("Calculating nDCG at 20...")
            ground_truth_df["ndcg_at_20"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=20), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))
            print("Calculating nDCG at 50...")
            ground_truth_df["ndcg_at_50"] = list(
                tqdm(p.map(functools.partial(ndcg_at_k, k=50), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))

            if not self.no_offpolicy_eval:
                ground_truth_df["rhat_scores"] = list(
                    tqdm(p.starmap(_get_rhat_scores, zip(ground_truth_df["relevance_list"], ground_truth_df["action_scores"])),
                         total=len(ground_truth_df)))

                # The ground truth of the dataset is the Direct Estimator
                ground_truth_df["rhat_rewards"] = list(
                    tqdm(p.map(_get_rhat_rewards, ground_truth_df["relevance_list"]),
                         total=len(ground_truth_df)))

                ground_truth_df["rewards"] =  ground_truth_df["rhat_rewards"] # ground_truth_df[self.model_training.project_config.output_column.name]

                print("Calculate ps policy eval...")
                ground_truth_df["ps_eval"] = list(tqdm(
                    p.starmap(_ps_policy_eval, zip(ground_truth_df["relevance_list"], ground_truth_df["prob_actions"])), total=len(ground_truth_df)))

        catalog          = list(range(ground_truth_df.iloc[0]["n_items"]))


        fairness_df      = ground_truth_df[[self.model_training.project_config.item_column.name, "action", "rewards", "rhat_scores", *self.fairness_columns]]
        fairness_metrics = calculate_fairness_metrics(fairness_df, self.fairness_columns,
                                                      self.model_training.project_config.item_column.name, "action")
        fairness_metrics.to_csv(os.path.join(self.output().path, "fairness_metrics.csv"), index=False)
        fairness_df.to_csv(os.path.join(self.output().path, "fairness_df.csv"), index=False)

        metrics = {
            "model_task": self.model_task_id,
            "count": len(ground_truth_df),
            "mean_average_precision": ground_truth_df["average_precision"].mean(),
            "precision_at_1": ground_truth_df["precision_at_1"].mean(),
            "ndcg_at_5":  ground_truth_df["ndcg_at_5"].mean(),
            "ndcg_at_10": ground_truth_df["ndcg_at_10"].mean(),
            "ndcg_at_15": ground_truth_df["ndcg_at_15"].mean(),
            "ndcg_at_20": ground_truth_df["ndcg_at_20"].mean(),
            "ndcg_at_50": ground_truth_df["ndcg_at_50"].mean(),
            "coverage_at_5":  prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 5),
            "coverage_at_10": prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 10),
            "coverage_at_15": prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 15),
            "coverage_at_20": prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 20),
            "coverage_at_50": prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 50),
            "personalization_at_5":  personalization_at_k(ground_truth_df["sorted_actions"], 5),
            "personalization_at_10": personalization_at_k(ground_truth_df["sorted_actions"], 10),
            "personalization_at_15": personalization_at_k(ground_truth_df["sorted_actions"], 15),
            "personalization_at_20": personalization_at_k(ground_truth_df["sorted_actions"], 20),
            "personalization_at_50": personalization_at_k(ground_truth_df["sorted_actions"], 50),
        }

        if not self.no_offpolicy_eval:
            rhat_rewards, rewards, ps_eval, ps = self._offpolicy_eval(ground_truth_df)

            metrics["IPS"]   = eval_IPS(rewards, ps_eval, ps)
            metrics["CIPS"]  = eval_CIPS(rewards, ps_eval, ps)
            metrics["SNIPS"] = eval_SNIPS(rewards, ps_eval, ps)
            metrics["DirectEstimator"] = np.mean(rhat_rewards)
            metrics["DoublyRobust"]    = eval_doubly_robust(rhat_rewards, rewards, ps_eval, ps)

        with open(os.path.join(self.output().path, "metrics.json"), "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)

    def _offpolicy_eval(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Filter df used in offpolicy evaluation
        ps_column    = self.model_training.project_config.propensity_score_column_name
        df_offpolicy = df[df[ps_column] > 0]

        rewards      = df_offpolicy["rewards"].values
        rhat_rewards = df_offpolicy["rhat_rewards"].values
        ps_eval      = df_offpolicy["ps_eval"].values
        ps           = df_offpolicy[ps_column].values

        return rhat_rewards, rewards, ps_eval, ps


def _create_relevance_list(sorted_actions: List[int], expected_action: int, reward: int) -> List[int]:
    if reward == 1:
        return [1 if action == expected_action else 0 for action in sorted_actions]
    else:
        return [0 for _ in sorted_actions]


def _ps_policy_eval(relevance_list: List[int], prob_actions: List[float]) -> List[float]:
    return np.sum(np.array(relevance_list) * np.array(prob_actions[:len(relevance_list)])).tolist()


def _get_rhat_scores(relevance_list: List[int], action_scores: List[float]) -> List[float]:
    return np.sum(np.array(relevance_list) * np.array(action_scores[:len(relevance_list)])).tolist()


def _get_rhat_rewards(relevance_list: List[int]) -> float:
    return relevance_list[0]
