import functools
import importlib
import inspect
import json
import os
from itertools import starmap
from multiprocessing.pool import Pool
from typing import List, Tuple
import pprint
import luigi
import numpy as np
import pandas as pd
import torch
import torchbearer
from torchbearer import Trial
from tqdm import tqdm

from recommendation.data import preprocess_interactions_data_frame
from recommendation.fairness_metrics import calculate_fairness_metrics
from recommendation.files import get_test_set_predictions_path
from recommendation.offpolicy_metrics import eval_IPS, eval_CIPS, eval_SNIPS, eval_doubly_robust
from recommendation.rank_metrics import average_precision, precision_at_k, ndcg_at_k, prediction_coverage_at_k, \
    personalization_at_k
from recommendation.task.model.base import BaseEvaluationTask, BaseTorchModelTraining
from recommendation.task.model.policy_estimator import PolicyEstimatorTraining
from recommendation.torch import FasterBatchSampler, NoAutoCollationDataLoader
from recommendation.utils import parallel_literal_eval, JsonEncoder


class EvaluateTestSetPredictions(BaseEvaluationTask):
    direct_estimator_module: str = luigi.Parameter(default=None)
    direct_estimator_cls: str = luigi.Parameter(default=None)

    policy_estimator_extra_params: dict = luigi.DictParameter(default={})

    num_processes: int = luigi.IntParameter(default=os.cpu_count())

    fairness_columns: List[str] = luigi.ListParameter()

    def get_direct_estimator(self, extra_params: dict) -> BaseTorchModelTraining:
        assert self.direct_estimator_module is not None
        assert self.direct_estimator_cls is not None

        estimator_module = importlib.import_module(self.direct_estimator_module)
        estimator_class = getattr(estimator_module, self.direct_estimator_cls)

        attribute_names = set(list(zip(*(
            inspect.getmembers(estimator_class, lambda a: not (inspect.isroutine(a))))))[0])

        params = {key: value for key, value in self.model_training.param_kwargs.items()
                  if key in attribute_names}
        return estimator_class(**{**params, **extra_params})

    @property
    def direct_estimator(self):
        if not hasattr(self, "_direct_estimator"):
            self._direct_estimator = self.get_direct_estimator({"loss_function": "bce"})
        return self._direct_estimator

    @property
    def policy_estimator(self):
        if not hasattr(self, "_policy_estimator"):
            self._policy_estimator = PolicyEstimatorTraining(
                project=self.model_training.project,
                data_frames_preparation_extra_params=self.model_training.data_frames_preparation_extra_params,
                **self.policy_estimator_extra_params,
            )
        return self._policy_estimator

    def requires(self):
        if not self.no_offpolicy_eval:
            return [self.direct_estimator, self.policy_estimator]
        return []

    def run(self):
        os.makedirs(self.output().path)

        df: pd.DataFrame     = preprocess_interactions_data_frame(
            pd.read_csv(get_test_set_predictions_path(self.model_training.output().path)),
            self.model_training.project_config)

        df["sorted_actions"] = parallel_literal_eval(df["sorted_actions"])
        df["prob_actions"]   = parallel_literal_eval(df["prob_actions"])
        df["action_scores"]  = parallel_literal_eval(df["action_scores"])
        df["action"]         = df["sorted_actions"].apply(lambda sorted_actions: sorted_actions[0])

        with Pool(self.num_processes) as p:
            print("Creating the relevance lists...")
            df["relevance_list"] = list(
                tqdm(p.starmap(_create_relevance_list,
                                zip(df["sorted_actions"], df[self.model_training.project_config.item_column.name],
                                    df[self.model_training.project_config.output_column.name])),
                        total=len(df)))

        if self.model_training.metadata_data_frame is not None:
            df = pd.merge(df, self.model_training.metadata_data_frame, left_on="action",
                          right_on=self.model_training.project_config.item_column.name, suffixes=("", "_action"))

        # Ground Truth
        ground_truth_df = df[df[self.model_training.project_config.output_column.name] == 1]

        df_rank, dict_rank               = self.rank_metrics(ground_truth_df)
        df_offpolicy, dict_offpolice     = self.offpolice_metrics(df)
        df_fairness, df_fairness_metrics = self.fairness_metrics(ground_truth_df)
        
        # Save Logs
        metrics = {**dict_rank, **dict_offpolice}
        pprint.pprint(metrics)
        with open(os.path.join(self.output().path, "metrics.json"), "w") as metrics_file:
            json.dump(metrics, metrics_file, cls=JsonEncoder, indent=4)
                 
        df_fairness_metrics.to_csv(os.path.join(self.output().path, "fairness_metrics.csv"), index=False)
        df_fairness.to_csv(os.path.join(self.output().path, "fairness_df.csv"), index=False)


    def rank_metrics(self, df: pd.DataFrame):
        with Pool(self.num_processes) as p:
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

        catalog          = list(range(df.iloc[0]["n_items"]))

        metrics = {
            "model_task": self.model_task_id,
            "count": len(df),
            "mean_average_precision": df["average_precision"].mean(),
            "precision_at_1": df["precision_at_1"].mean(),
            "ndcg_at_5":      df["ndcg_at_5"].mean(),
            "ndcg_at_10":     df["ndcg_at_10"].mean(),
            "ndcg_at_15":     df["ndcg_at_15"].mean(),
            "ndcg_at_20":     df["ndcg_at_20"].mean(),
            "ndcg_at_50":     df["ndcg_at_50"].mean(),
            "coverage_at_5":  prediction_coverage_at_k(df["sorted_actions"], catalog, 5),
            "coverage_at_10": prediction_coverage_at_k(df["sorted_actions"], catalog, 10),
            "coverage_at_15": prediction_coverage_at_k(df["sorted_actions"], catalog, 15),
            "coverage_at_20": prediction_coverage_at_k(df["sorted_actions"], catalog, 20),
            "coverage_at_50": prediction_coverage_at_k(df["sorted_actions"], catalog, 50),
            "personalization_at_5":  personalization_at_k(df["sorted_actions"], 5),
            "personalization_at_10": personalization_at_k(df["sorted_actions"], 10),
            "personalization_at_15": personalization_at_k(df["sorted_actions"], 15),
            "personalization_at_20": personalization_at_k(df["sorted_actions"], 20),
            "personalization_at_50": personalization_at_k(df["sorted_actions"], 50),
        }

        return df, metrics
        # with Pool(self.num_processes) as p:
        #     print("Creating the relevance lists...")
        #     df["relevance_list"] = list(
        #         tqdm(p.starmap(_create_relevance_list,
        #                        zip(df["sorted_actions"], df[self.model_training.project_config.item_column.name],
        #                            df[self.model_training.project_config.output_column.name])),
        #              total=len(df)))

        #     ground_truth_df = df[df[self.model_training.project_config.output_column.name] == 1]

        #     print("Calculating average precision...")
        #     ground_truth_df["average_precision"] = list(
        #         tqdm(p.map(average_precision, ground_truth_df["relevance_list"]), total=len(ground_truth_df)))

        #     print("Calculating precision at 1...")
        #     ground_truth_df["precision_at_1"] = list(
        #         tqdm(p.map(functools.partial(precision_at_k, k=1), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))

        #     print("Calculating nDCG at 5...")
        #     ground_truth_df["ndcg_at_5"] = list(
        #         tqdm(p.map(functools.partial(ndcg_at_k, k=5), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))
        #     print("Calculating nDCG at 10...")
        #     ground_truth_df["ndcg_at_10"] = list(
        #         tqdm(p.map(functools.partial(ndcg_at_k, k=10), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))
        #     print("Calculating nDCG at 15...")
        #     ground_truth_df["ndcg_at_15"] = list(
        #         tqdm(p.map(functools.partial(ndcg_at_k, k=15), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))
        #     print("Calculating nDCG at 20...")
        #     ground_truth_df["ndcg_at_20"] = list(
        #         tqdm(p.map(functools.partial(ndcg_at_k, k=20), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))
        #     print("Calculating nDCG at 50...")
        #     ground_truth_df["ndcg_at_50"] = list(
        #         tqdm(p.map(functools.partial(ndcg_at_k, k=50), ground_truth_df["relevance_list"]), total=len(ground_truth_df)))

        #     if not self.no_offpolicy_eval:
        #         self.fill_rhat_rewards(df)
        #         self.fill_ps(df, p)

        #         df["rhat_scores"] = list(
        #             tqdm(p.starmap(_get_rhat_scores, zip(df["relevance_list"], df["action_scores"])), total=len(df)))

        #         df["rewards"] = df[self.model_training.project_config.output_column.name]

        #         print("Calculate ps policy eval...")
        #         df["ps_eval"] = list(tqdm(
        #             p.starmap(_ps_policy_eval, zip(df["relevance_list"], df["prob_actions"])), total=len(df)))

        # catalog          = list(range(ground_truth_df.iloc[0]["n_items"]))


        # fairness_df      = ground_truth_df[[self.model_training.project_config.item_column.name, "action", "rewards", "rhat_scores", *self.fairness_columns]]
        # fairness_metrics = calculate_fairness_metrics(fairness_df, self.fairness_columns,
        #                                               self.model_training.project_config.item_column.name, "action")
        # fairness_metrics.to_csv(os.path.join(self.output().path, "fairness_metrics.csv"), index=False)
        # fairness_df.to_csv(os.path.join(self.output().path, "fairness_df.csv"), index=False)

        # metrics = {
        #     "model_task": self.model_task_id,
        #     "count": len(ground_truth_df),
        #     "mean_average_precision": ground_truth_df["average_precision"].mean(),
        #     "precision_at_1": ground_truth_df["precision_at_1"].mean(),
        #     "ndcg_at_5":  ground_truth_df["ndcg_at_5"].mean(),
        #     "ndcg_at_10": ground_truth_df["ndcg_at_10"].mean(),
        #     "ndcg_at_15": ground_truth_df["ndcg_at_15"].mean(),
        #     "ndcg_at_20": ground_truth_df["ndcg_at_20"].mean(),
        #     "ndcg_at_50": ground_truth_df["ndcg_at_50"].mean(),
        #     "coverage_at_5":  prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 5),
        #     "coverage_at_10": prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 10),
        #     "coverage_at_15": prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 15),
        #     "coverage_at_20": prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 20),
        #     "coverage_at_50": prediction_coverage_at_k(ground_truth_df["sorted_actions"], catalog, 50),
        #     "personalization_at_5":  personalization_at_k(ground_truth_df["sorted_actions"], 5),
        #     "personalization_at_10": personalization_at_k(ground_truth_df["sorted_actions"], 10),
        #     "personalization_at_15": personalization_at_k(ground_truth_df["sorted_actions"], 15),
        #     "personalization_at_20": personalization_at_k(ground_truth_df["sorted_actions"], 20),
        #     "personalization_at_50": personalization_at_k(ground_truth_df["sorted_actions"], 50),
        # }

        # if not self.no_offpolicy_eval:
        #     rhat_rewards, rewards, ps_eval, ps = self._offpolicy_eval(df)

        #     metrics["IPS"]   = eval_IPS(rewards, ps_eval, ps)
        #     metrics["CIPS"]  = eval_CIPS(rewards, ps_eval, ps)
        #     metrics["SNIPS"] = eval_SNIPS(rewards, ps_eval, ps)
        #     metrics["DirectEstimator"] = np.mean(rhat_rewards)
        #     metrics["DoublyRobust"]    = eval_doubly_robust(rhat_rewards, rewards, ps_eval, ps)

        # with open(os.path.join(self.output().path, "metrics.json"), "w") as metrics_file:
        #     json.dump(metrics, metrics_file, indent=4)

    def offpolice_metrics(self, df: pd.DataFrame):
        metrics = {}
        
        if self.no_offpolicy_eval:
            return metrics

        df["rewards"] = df[self.model_training.project_config.output_column.name]

        with Pool(self.num_processes) as p:
            self.fill_rhat_rewards(df)
            self.fill_ps(df, p)

            print("Calculate ps policy eval...")
            df["ps_eval"] = list(tqdm(
                p.starmap(_ps_policy_eval, zip(df["relevance_list"], df["prob_actions"])), total=len(df)))

            rhat_rewards, rewards, ps_eval, ps = self._offpolicy_eval(df)

            l_ips, ips, h_ips       = eval_IPS(rewards, ps_eval, ps)
            l_cips, cips, h_cips    = eval_CIPS(rewards, ps_eval, ps)
            l_snips, snips, h_snips = eval_SNIPS(rewards, ps_eval, ps)

            metrics["IPS_L"]        = l_ips
            metrics["IPS"]          = ips
            metrics["IPS_H"]        = h_ips
            metrics["CIPS_L"]       = l_cips
            metrics["CIPS"]         = cips
            metrics["CIPS_H"]       = h_cips
            metrics["SNIPS_L"]      = l_snips
            metrics["SNIPS"]        = snips
            metrics["SNIPS_H"]      = h_snips

            metrics["DirectEstimator"] = np.mean(rhat_rewards)
            metrics["DoublyRobust"]    = eval_doubly_robust(rhat_rewards, rewards, ps_eval, ps)

        return df, metrics

    def fairness_metrics(self, df: pd.DataFrame):
        
        df["action"]      = df["sorted_actions"].apply(lambda sorted_actions: sorted_actions[0])
        df["rhat_scores"] = df["action_scores"].apply(lambda action_scores: action_scores[0])
        df["rewards"]     = df["relevance_list"].apply(lambda relevance_list: relevance_list[0])

        fairness_df       = df[[self.model_training.project_config.item_column.name, "action", "rewards", "rhat_scores", *self.fairness_columns]]
        fairness_metrics  = calculate_fairness_metrics(fairness_df, self.fairness_columns,
                                                      self.model_training.project_config.item_column.name, "action")

        return fairness_df, fairness_metrics

    def fill_ps(self, df: pd.DataFrame, pool: Pool):
        dataset = self.policy_estimator.project_config.dataset_class(df, None, self.policy_estimator.project_config)
        batch_sampler = FasterBatchSampler(dataset, self.policy_estimator.batch_size, shuffle=False)
        data_loader = NoAutoCollationDataLoader(dataset, batch_sampler=batch_sampler)

        trial = Trial(self.policy_estimator.get_trained_module(),
                      criterion=lambda *args: torch.zeros(1, device=self.policy_estimator.torch_device, requires_grad=True)) \
            .with_generators(val_generator=data_loader).to(self.policy_estimator.torch_device).eval()

        with torch.no_grad():
            log_probas: torch.Tensor = trial.predict(verbose=0, data_key=torchbearer.VALIDATION_DATA)
        probas: np.ndarray = torch.exp(log_probas).cpu().numpy()

        item_indices = df[self.model_training.project_config.item_column.name]

        params = zip(item_indices, probas, df[self.model_training.project_config.available_arms_column_name]) \
            if self.model_training.project_config.available_arms_column_name else zip(item_indices, probas)
        df[self.model_training.project_config.propensity_score_column_name] = list(
            tqdm(pool.starmap(_get_ps_from_probas, params), total=len(df)))

    def fill_rhat_rewards(self, df: pd.DataFrame):
        dataset = self.direct_estimator.project_config.dataset_class(df, self.direct_estimator.embeddings_for_metadata,
                                                                     self.direct_estimator.project_config)
        batch_sampler = FasterBatchSampler(dataset, self.direct_estimator.batch_size, shuffle=False)
        data_loader = NoAutoCollationDataLoader(dataset, batch_sampler=batch_sampler)

        trial = Trial(self.direct_estimator.get_trained_module(),
                      criterion=lambda *args: torch.zeros(1, device=self.direct_estimator.torch_device, requires_grad=True)) \
            .with_generators(val_generator=data_loader).to(self.direct_estimator.torch_device).eval()

        with torch.no_grad():
            rewards_tensor: torch.Tensor = trial.predict(verbose=0, data_key=torchbearer.VALIDATION_DATA)
        rewards: np.ndarray = rewards_tensor[:, 0].cpu().numpy()

        df["rhat_rewards"] = rewards

    def _offpolicy_eval(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Filter df used in offpolicy evaluation
        ps_column    = self.model_training.project_config.propensity_score_column_name
        df_offpolicy = df[df[ps_column] > 0]

        rewards      = df_offpolicy["rewards"].values
        rhat_rewards = df_offpolicy["rhat_rewards"].values
        ps_eval      = df_offpolicy["ps_eval"].values
        ps           = df_offpolicy[ps_column].values

        return rhat_rewards, rewards, ps_eval, ps


def _get_ps_from_probas(item_idx: int, probas: np.ndarray, available_item_indices: List[int] = None) -> float:
    if available_item_indices:
        probas /= np.sum(probas[available_item_indices])
    return probas[item_idx]


def _create_relevance_list(sorted_actions: List[int], expected_action: int, reward: int) -> List[int]:
    return [1 if action == expected_action else 0 for action in sorted_actions]


def _ps_policy_eval(relevance_list: List[int], prob_actions: List[float]) -> List[float]:
    return np.sum(np.array(relevance_list) * np.array(prob_actions[:len(relevance_list)])).tolist()


def _get_rhat_scores(relevance_list: List[int], action_scores: List[float]) -> List[float]:
    return np.sum(np.array(relevance_list) * np.array(action_scores[:len(relevance_list)])).tolist()


def _get_rhat_rewards(relevance_list: List[int]) -> float:
    return relevance_list[0]
