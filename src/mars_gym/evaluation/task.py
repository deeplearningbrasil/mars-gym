import functools
import json
import os
from multiprocessing.pool import Pool
from typing import List, Tuple, Type, Any
import pprint

import abc
import luigi
import numpy as np
import pandas as pd
import torch
import torchbearer
from torchbearer import Trial
from tqdm import tqdm
import gc
from mars_gym.data.dataset import (
    preprocess_interactions_data_frame,
    InteractionsDataset,
)
from mars_gym.evaluation.propensity_score import FillPropensityScoreMixin
from mars_gym.evaluation.metrics.fairness import calculate_fairness_metrics
from mars_gym.utils import files
from mars_gym.utils.files import get_test_set_predictions_path, get_params_path
from mars_gym.evaluation.metrics.offpolicy import (
    eval_IPS,
    eval_CIPS,
    eval_SNIPS,
    eval_doubly_robust,
)
from mars_gym.evaluation.metrics.rank import (
    mean_reciprocal_rank,
    average_precision,
    precision_at_k,
    ndcg_at_k,
    personalization_at_k,
    prediction_coverage_at_k,
)
from mars_gym.simulation.training import (
    TorchModelTraining,
    load_torch_model_training_from_task_id,
)
from mars_gym.evaluation.policy_estimator import PolicyEstimatorTraining
from mars_gym.torch.data import FasterBatchSampler, NoAutoCollationDataLoader
from mars_gym.utils.reflection import load_attr, get_attribute_names
from mars_gym.utils.utils import parallel_literal_eval, JsonEncoder
from mars_gym.utils.index_mapping import (
    create_index_mapping,
    create_index_mapping_from_arrays,
    transform_with_indexing,
    map_array,
)

class BaseEvaluationTask(luigi.Task, metaclass=abc.ABCMeta):
    model_task_class: str = luigi.Parameter(
        default="mars_gym.simulation.interaction.InteractionTraining"
    )
    model_task_id: str = luigi.Parameter()
    offpolicy_eval: bool = luigi.BoolParameter(default=False)
    task_hash: str = luigi.Parameter(default="none")

    @property
    def cache_attr(self):
        return [""]

    @property
    def task_name(self):
        return self.model_task_id + "_" + self.task_id.split("_")[-1]

    @property
    def model_training(self) -> TorchModelTraining:
        if not hasattr(self, "_model_training"):
            class_ = load_attr(self.model_task_class, Type[TorchModelTraining])
            self._model_training = load_torch_model_training_from_task_id(
                class_, self.model_task_id
            )

        return self._model_training

    @property
    def n_items(self):
        return self.model_training.n_items

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                files.OUTPUT_PATH,
                "evaluation",
                self.__class__.__name__,
                "results",
                self.task_name,
            )
        )

    def cache_cleanup(self):
        for a in self.cache_attrs:
            if hasattr(self, a):
                delattr(self, a)

    def _save_params(self):
        with open(get_params_path(self.output().path), "w") as params_file:
            json.dump(
                self.param_kwargs, params_file, default=lambda o: dict(o), indent=4
            )


class EvaluateTestSetPredictions(FillPropensityScoreMixin, BaseEvaluationTask):
    # TODO transform this params in a dict params
    direct_estimator_class: str = luigi.Parameter(default="mars_gym.simulation.training.SupervisedModelTraining")
    direct_estimator_negative_proportion: int = luigi.FloatParameter(0)
    direct_estimator_batch_size: int = luigi.IntParameter(default=500)
    direct_estimator_epochs: int = luigi.IntParameter(default=50)
    direct_estimator_extra_params: dict = luigi.DictParameter(default={})

    eval_cips_cap: int = luigi.IntParameter(default=15)
    policy_estimator_extra_params: dict = luigi.DictParameter(default={})

    num_processes: int = luigi.IntParameter(default=os.cpu_count())

    fairness_columns: List[str] = luigi.ListParameter(default=[])
    rank_metrics: List[str] = luigi.ListParameter(default=[])

    only_new_interactions: bool = luigi.BoolParameter(default=False)
    only_exist_items: bool = luigi.BoolParameter(default=False)
    only_exist_users: bool = luigi.BoolParameter(default=False)

    def get_direct_estimator(self, extra_params: dict) -> TorchModelTraining:
        assert self.direct_estimator_class is not None
        estimator_class = load_attr(
            self.direct_estimator_class, Type[TorchModelTraining]
        )

        attribute_names = get_attribute_names(estimator_class)

        params = {
            key: value
            for key, value in self.model_training.param_kwargs.items()
            if key in attribute_names
        }
        
        return estimator_class(**{**params, **extra_params})

    #TODO We need change it
    @property
    def direct_estimator(self):
        if not hasattr(self, "_direct_estimator"):
            self._direct_estimator = self.get_direct_estimator(
                {**{
                    "project": self.model_training.project,
                    "learning_rate": 0.0001,
                    "test_size": 0.0,
                    "epochs": self.direct_estimator_epochs,
                    "batch_size": self.direct_estimator_batch_size,
                    "loss_function": "bce",
                    "loss_function_params": {},
                    "observation": "All Data",
                    "negative_proportion": self.direct_estimator_negative_proportion,
                    "policy_estimator_extra_params": {},
                    "metrics": ["loss"],
                    "seed": 51,
                }, **self.direct_estimator_extra_params}
            )
        return self._direct_estimator

    @property
    def policy_estimator(self) -> PolicyEstimatorTraining:
        if not hasattr(self, "_policy_estimator"):
            self._policy_estimator = PolicyEstimatorTraining(
                project=self.model_training.project,
                data_frames_preparation_extra_params=self.model_training.data_frames_preparation_extra_params,
                **self.policy_estimator_extra_params,
            )
        return self._policy_estimator

    def requires(self):
        if self.offpolicy_eval:
            return [self.direct_estimator, self.policy_estimator]
        return []

    @property
    def item_column(self) -> str:
        return self.model_training.project_config.item_column.name

    @property
    def available_arms_column(self) -> str:
        return self.model_training.project_config.available_arms_column_name

    @property
    def propensity_score_column(self) -> str:
        return self.model_training.project_config.propensity_score_column_name

    def get_item_index(self)-> List[str]:
        indexed_list = list(self.model_training.index_mapping[self.model_training.project_config.item_column.name].keys())
        indexed_list = [x for x in indexed_list if x is not None and str(x) != 'nan']
        return indexed_list

    def get_catalog(self, df: pd.DataFrame) -> List[str]:
        indexed_list = self.get_item_index()

        test_list    = list(df["sorted_actions"])
        test_list.append(indexed_list)

        all_items = sum(test_list, [])
        unique_items = list(np.unique(all_items))
        return unique_items

    def run(self):
        os.makedirs(self.output().path)

        # df: pd.DataFrame = preprocess_interactions_data_frame(
        #     pd.read_csv(
        #         get_test_set_predictions_path(self.model_training.output().path)
        #     ),
        #     self.model_training.project_config,
        # )  # .sample(10000)

        df: pd.DataFrame = pd.read_csv(
            get_test_set_predictions_path(self.model_training.output().path),
            dtype = {self.model_training.project_config.item_column.name : "str"}
        )  # .sample(10000)

        df["sorted_actions"] = parallel_literal_eval(df["sorted_actions"])
        df["prob_actions"]   = parallel_literal_eval(df["prob_actions"])
        df["action_scores"]  = parallel_literal_eval(df["action_scores"])

        df["action"] = df["sorted_actions"].apply(
            lambda sorted_actions: str(sorted_actions[0])
        )

        with Pool(self.num_processes) as p:
            print("Creating the relevance lists...")
            # from IPython import embed; embed()
            df["relevance_list"] = list(
                tqdm(
                    p.starmap(
                        _create_relevance_list,
                        zip(
                            df["sorted_actions"],
                            df[self.model_training.project_config.item_column.name],
                            df[self.model_training.project_config.output_column.name],
                        ),
                    ),
                    total=len(df),
                )
            )
        
        
        if self.model_training.metadata_data_frame is not None:
            df = pd.merge(
                df,
                pd.read_csv(self.model_training.metadata_data_frame_path, dtype = {self.model_training.project_config.item_column.name : "str"}),
                left_on="action",
                right_on=self.model_training.project_config.item_column.name,
                suffixes=("", "_action"),
            )

        ground_truth_df = df[
            ~(df[self.model_training.project_config.output_column.name] == 0)
        ]
        
        print("Rank Metrics...")
        df_rank, dict_rank = self.rank_metrics(ground_truth_df)
        gc.collect()

        print("Fairness Metrics")
        df_fairness, df_fairness_metrics = self.fairness_metrics(ground_truth_df)
        gc.collect()


        print("Offpolice Metrics")
        df_offpolicy, dict_offpolice = self.offpolice_metrics(df)
        gc.collect()

        # dict_offpolice = {}
        # Save Logs
        metrics = {**dict_rank, **dict_offpolice}
        pprint.pprint(metrics)
        with open(
            os.path.join(self.output().path, "metrics.json"), "w"
        ) as metrics_file:
            json.dump(metrics, metrics_file, cls=JsonEncoder, indent=4)

        df_offpolicy.to_csv(
            os.path.join(self.output().path, "df_offpolicy.csv"), index=False
        )
        df_rank.to_csv(
            os.path.join(self.output().path, "rank_metrics.csv"), index=False
        )
        df_fairness_metrics.to_csv(
            os.path.join(self.output().path, "fairness_metrics.csv"), index=False
        )
        df_fairness.to_csv(
            os.path.join(self.output().path, "fairness_df.csv"), index=False
        )


    def rank_metrics(self, df: pd.DataFrame):
        df = df.copy()

        # Filter only disponível interaction 
        df = df[df.relevance_list.apply(max) > 0]
        
        # Filter only new interactions, them not appear in trained dataset
        if self.only_new_interactions:
            df = df[df['trained'] == 0]
        
        # Filter only item indexed
        if self.only_exist_items:
            df = df[df['item_indexed']]

        with Pool(self.num_processes) as p:
            print("Calculating average precision...")
            df["average_precision"] = list(
                tqdm(p.map(average_precision, df["relevance_list"]), total=len(df))
            )

            print("Calculating precision at 1...")
            df["precision_at_1"] = list(
                tqdm(
                    p.map(functools.partial(precision_at_k, k=1), df["relevance_list"]),
                    total=len(df),
                )
            )

            
            print("Calculating MRR at 5 ...")
            df["mrr_at_5"] = list(
                tqdm(
                    p.map(functools.partial(mean_reciprocal_rank, k=5), df["relevance_list"]),
                    total=len(df),
                )
            )


            print("Calculating MRR at 10 ...")
            df["mrr_at_10"] = list(
                tqdm(
                    p.map(functools.partial(mean_reciprocal_rank, k=10), df["relevance_list"]),
                    total=len(df),
                )
            )

            print("Calculating nDCG at 5...")
            df["ndcg_at_5"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=5), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 10...")
            df["ndcg_at_10"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=10), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 15...")
            df["ndcg_at_15"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=15), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 20...")
            df["ndcg_at_20"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=20), df["relevance_list"]),
                    total=len(df),
                )
            )
            print("Calculating nDCG at 50...")
            df["ndcg_at_50"] = list(
                tqdm(
                    p.map(functools.partial(ndcg_at_k, k=50), df["relevance_list"]),
                    total=len(df),
                )
            )
        #
        catalog = self.get_catalog(df)
        
        metrics = {
            "model_task": self.model_task_id,
            "count": len(df),
            "mean_average_precision": df["average_precision"].mean(),

            "precision_at_1": df["precision_at_1"].mean(),
            "mrr_at_5": df["mrr_at_5"].mean(),
            "mrr_at_10": df["mrr_at_10"].mean(),
            "ndcg_at_5": df["ndcg_at_5"].mean(),
            "ndcg_at_10": df["ndcg_at_10"].mean(),
            "ndcg_at_15": df["ndcg_at_15"].mean(),
            "ndcg_at_20": df["ndcg_at_20"].mean(),
            "ndcg_at_50": df["ndcg_at_50"].mean(),
            "coverage_at_5": prediction_coverage_at_k(df["sorted_actions"], catalog, 5),
            #"coverage_at_10": prediction_coverage_at_k(
            #    df["sorted_actions"], catalog, 10
            #),
            #"coverage_at_15": prediction_coverage_at_k(
            #    df["sorted_actions"], catalog, 15
            #),
            "coverage_at_20": prediction_coverage_at_k(
                df["sorted_actions"], catalog, 20
            ),
            #"coverage_at_50": prediction_coverage_at_k(
            #    df["sorted_actions"], catalog, 50
            #),
            #"personalization_at_5": personalization_at_k(df["sorted_actions"], 5),
            #"personalization_at_10": personalization_at_k(df["sorted_actions"], 10),
            #"personalization_at_15": personalization_at_k(df["sorted_actions"], 15),
            #"personalization_at_20": personalization_at_k(df["sorted_actions"], 20),
            #"personalization_at_50": personalization_at_k(df["sorted_actions"], 50),
        }

        return df, metrics

    def offpolice_metrics(self, df: pd.DataFrame):
        metrics = {}

        if not self.offpolicy_eval:
            return pd.DataFrame(), metrics
        
        df["rewards"] = df[self.model_training.project_config.output_column.name]
        with Pool(self.num_processes) as p:
            self.fill_rhat_rewards(df, p)
            self.fill_ps(df, p)
            self.fill_item_rhat_rewards(df)

            print("Calculate ps policy eval...")
            df["ps_eval"] = list(
                tqdm(
                    p.starmap(
                        _ps_policy_eval, zip(df["relevance_list"], df["prob_actions"])
                    ),
                    total=len(df),
                )
            )

            (
                action_rhat_rewards,
                item_idx_rhat_rewards,
                rewards,
                ps_eval,
                ps_eval_i,
                ps,
            ) = self._offpolicy_eval(df)


            ips, c_ips = eval_IPS(rewards, ps_eval_i, ps)
            cips, c_cips = eval_CIPS(rewards, ps_eval_i, ps, cap=self.eval_cips_cap)
            snips, c_snips = eval_SNIPS(rewards, ps_eval_i, ps)
            doubly, c_doubly = eval_doubly_robust(
                action_rhat_rewards, item_idx_rhat_rewards, rewards, ps_eval_i, ps
            )

            metrics["IPS"] = ips
            metrics["IPS_C"] = c_ips
            metrics["CIPS"] = cips
            metrics["CIPS_C"] = c_cips
            metrics["SNIPS"] = snips
            metrics["SNIPS_C"] = c_snips

            metrics["DirectEstimator"] = np.mean(action_rhat_rewards)
            metrics["DoublyRobust"] = doubly
            metrics["DoublyRobust_C"] = c_doubly

        return df, metrics

    def fairness_metrics(self, df: pd.DataFrame):
        if len(self.fairness_columns) == 0:
            return pd.DataFrame(), pd.DataFrame()

        df["action"] = df["sorted_actions"].apply(
            lambda sorted_actions: sorted_actions[0]
        )
        df["rhat_scores"] = df["action_scores"].apply(
            lambda action_scores: action_scores[0]
        )
        df["rewards"] = df["relevance_list"].apply(
            lambda relevance_list: relevance_list[0]
        )

        fairness_df = df[
            [
                self.model_training.project_config.item_column.name,
                "action",
                "rewards",
                "rhat_scores",
                *self.fairness_columns,
            ]
        ]

        fairness_metrics = calculate_fairness_metrics(
            fairness_df.fillna("-"),
            self.fairness_columns,
            self.model_training.project_config.item_column.name,
            "action",
        )

        return fairness_df, fairness_metrics

    def fill_rhat_rewards(self, df: pd.DataFrame, pool: Pool):
        # from IPython import embed; embed()

        # df['item_idx_action'] item_idx_action

        # #TODO
        # # Explode
        # df_exploded = df.reset_index().explode('sorted_actions')
        # df_exploded['item_idx'] = df_exploded['sorted_actions']

        # # Predict
        # rewards     = self._direct_estimator_predict(df_exploded)
        # df_exploded["actions_rhat_rewards"] = rewards

        # # implode
        # df_implode = df_exploded.groupby('index').actions_rhat_rewards.apply(list)

        # df['actions_rhat_rewards'] = df_implode
        # df["action_rhat_rewards"]  = list(tqdm(
        #     pool.starmap(_get_rhat_rewards, zip(df["prob_actions"], df["actions_rhat_rewards"])), total=len(df)))

        df_action = df.copy()
        # set item_idx for direct_estimator
        df_action[self.model_training.project_config.item_column.name] = df["action"]

        rewards = self._direct_estimator_predict(df_action)
        del df_action

        df["action_rhat_rewards"] = rewards

    def fill_item_rhat_rewards(self, df: pd.DataFrame):
        rewards = self._direct_estimator_predict(df)

        df["item_idx_rhat_rewards"] = rewards

    def _direct_estimator_predict(self, df):
        _df = preprocess_interactions_data_frame(
            df.copy(), 
            self.direct_estimator.project_config
        )
        transform_with_indexing(
            _df, 
            self.direct_estimator.index_mapping, 
            self.direct_estimator.project_config
        )

        dataset = InteractionsDataset(
            data_frame=_df,
            embeddings_for_metadata=self.direct_estimator.embeddings_for_metadata,
            project_config=self.direct_estimator.project_config,
            index_mapping=self.direct_estimator.index_mapping
        )
        batch_sampler = FasterBatchSampler(
            dataset, self.direct_estimator.batch_size, shuffle=False
        )
        data_loader = NoAutoCollationDataLoader(dataset, batch_sampler=batch_sampler)

        trial = (
            Trial(
                self.direct_estimator.get_trained_module(),
                criterion=lambda *args: torch.zeros(
                    1, device=self.direct_estimator.torch_device, requires_grad=True
                ),
            )
            .with_generators(val_generator=data_loader)
            .to(self.direct_estimator.torch_device)
            .eval()
        )

        with torch.no_grad():
            rewards_tensor: torch.Tensor = trial.predict(
                verbose=0, data_key=torchbearer.VALIDATION_DATA
            )
        rewards: np.ndarray = rewards_tensor[:, 0].cpu().numpy()

        return rewards

    def _offpolicy_eval(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Filter df used in offpolicy evaluation
        ps_column = self.model_training.project_config.propensity_score_column_name
        e = 0.001
        df_offpolicy = df[df[ps_column] > e]

        rewards = df_offpolicy["rewards"].values
        ps_eval = df_offpolicy["ps_eval"].values
        ps_eval_i = df_offpolicy.apply(
            lambda row: int(
                row[self.model_training.project_config.item_column.name]
                == row["action"]
            ),
            axis=1,
        )

        ps = df_offpolicy[ps_column].values
        action_rhat_rewards = df_offpolicy["action_rhat_rewards"].values
        item_idx_rhat_rewards = df_offpolicy["item_idx_rhat_rewards"].values

        return (
            action_rhat_rewards,
            item_idx_rhat_rewards,
            rewards,
            ps_eval,
            ps_eval_i,
            ps,
        )


def _create_relevance_list(
    sorted_actions: List[Any], expected_action: Any, reward: int
) -> List[int]:
    return [1 if str(action) == str(expected_action) else 0 for action in sorted_actions]


def _ps_policy_eval(
    relevance_list: List[int], prob_actions: List[float]
) -> List[float]:
    return np.sum(
        np.array(relevance_list) * np.array(prob_actions[: len(relevance_list)])
    ).tolist()


def _get_rhat_scores(
    relevance_list: List[int], action_scores: List[float]
) -> List[float]:
    return np.sum(
        np.array(relevance_list) * np.array(action_scores[: len(relevance_list)])
    ).tolist()


def _get_rhat_rewards(prob_actions: List[int], action_scores: List[float]) -> float:
    return np.sum(np.array(prob_actions) * np.array(action_scores))
