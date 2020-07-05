from multiprocessing import Pool
from typing import Union, Tuple, Optional, List
import os
import json

import luigi
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchbearer import Trial

from mars_gym.data.dataset import InteractionsDataset
from mars_gym.evaluation.metrics.rank import (
    mean_reciprocal_rank,
    ndcg_at_k,
    reciprocal_rank_at_k,
    precision_at_k,
)
from mars_gym.model.trivago.trivago_models import SimpleLinearModel
from mars_gym.simulation.training import (
    TORCH_ACTIVATION_FUNCTIONS,
    TORCH_DROPOUT_MODULES,
    TORCH_LOSS_FUNCTIONS,
    TORCH_WEIGHT_INIT,
    SupervisedModelTraining,
)
from mars_gym.simulation.interaction import InteractionTraining
from mars_gym.evaluation.policy_estimator import PolicyEstimatorTraining
from mars_gym.evaluation.propensity_score import FillPropensityScoreMixin


class TrivagoModelTrainingMixin(object):
    recommender_module_class: str = None
    recommender_extra_params: dict = None

    loss_function: str = luigi.ChoiceParameter(
        choices=TORCH_LOSS_FUNCTIONS.keys(), default="crm"
    )
    n_factors: int = luigi.IntParameter(default=128)
    weight_init: str = luigi.ChoiceParameter(
        choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal"
    )
    dropout_prob: float = luigi.FloatParameter(default=0.1)
    dropout_module: str = luigi.ChoiceParameter(
        choices=TORCH_DROPOUT_MODULES.keys(), default="alpha"
    )
    activation_function: str = luigi.ChoiceParameter(
        choices=TORCH_ACTIVATION_FUNCTIONS.keys(), default="selu"
    )
    filter_sizes: List[int] = luigi.ListParameter(default=[1, 3, 5])
    num_filters: int = luigi.IntParameter(default=64)

    @property
    def window_hist_size(self):
        if not hasattr(self, "_window_hist_size"):
            self._window_hist_size = int(
                self.train_data_frame.iloc[0]["window_hist_size"]
            )
        return self._window_hist_size

    @property
    def metadata_size(self):
        if not hasattr(self, "_meta_data_size"):
            self._meta_data_size = int(self.metadata_data_frame.shape[1] - 3)
        return self._meta_data_size

    def create_module(self) -> nn.Module:

        return SimpleLinearModel(
            project_config=self.project_config,
            index_mapping=self.index_mapping,
            window_hist_size=self.window_hist_size,
            vocab_size=self.vocab_size,
            metadata_size=self.metadata_size,
            n_factors=self.n_factors,
            filter_sizes=self.filter_sizes,
            num_filters=self.num_filters,
            dropout_prob=self.dropout_prob,
            dropout_module=TORCH_DROPOUT_MODULES[self.dropout_module],
        )


class TrivagoModelInteraction(TrivagoModelTrainingMixin, InteractionTraining):
    pass


class TrivagoModelTraining(
    FillPropensityScoreMixin, TrivagoModelTrainingMixin, SupervisedModelTraining,
):
    def requires(self):
        required_tasks = [super().requires()]
        required_tasks.append(self.policy_estimator)
        return required_tasks

    @property
    def policy_estimator(self) -> PolicyEstimatorTraining:
        if not hasattr(self, "_policy_estimator"):
            self._policy_estimator = PolicyEstimatorTraining(
                project=self.project,
                data_frames_preparation_extra_params=self.data_frames_preparation_extra_params,
                **self.policy_estimator_extra_params,
            )
        return self._policy_estimator

    @property
    def train_data_frame_path(self) -> str:
        return self.input()[0][0].path

    @property
    def val_data_frame_path(self) -> str:
        return self.input()[0][1].path

    @property
    def test_data_frame_path(self) -> str:
        return self.input()[0][2].path

    @property
    def metadata_data_frame_path(self) -> Optional[str]:
        if len(self.input()[0]) > 3:
            return self.input()[0][3].path
        else:
            return None

    @property
    def train_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_train_data_frame"):
            self._train_data_frame_indexed = True  # To delay the index mapping
            self._train_data_frame = super().train_data_frame
            with Pool(max(os.cpu_count(), 10)) as p:
                self.fill_ps(self._train_data_frame, p)
            del self._train_data_frame_indexed
        return super().train_data_frame  # To invoke the index mapping if necessary

    @property
    def val_data_frame(self) -> pd.DataFrame:
        if not hasattr(self, "_val_data_frame"):
            self._val_data_frame_indexed = True  # To delay the index mapping
            self._val_data_frame = super().val_data_frame
            with Pool(max(os.cpu_count(), 10)) as p:
                self.fill_ps(self._val_data_frame, p)
            del self._val_data_frame_indexed
        return super().val_data_frame  # To invoke the index mapping if necessary

    @property
    def ps_base_df(self) -> pd.DataFrame:
        if not hasattr(self, "_base_df"):
            self._base_df = pd.concat(
                [
                    pd.read_csv(self.train_data_frame_path),
                    pd.read_csv(self.val_data_frame_path),
                    pd.read_csv(self.test_data_frame_path),
                ],
                ignore_index=True,
            )
        return self._base_df

    @property
    def output_column(self) -> str:
        return self.project_config.output_column.name

    @property
    def item_column(self) -> str:
        return self.project_config.item_column.name

    @property
    def available_arms_column(self) -> str:
        return self.project_config.available_arms_column_name

    @property
    def propensity_score_column(self) -> str:
        return self.project_config.propensity_score_column_name

    def evaluate(self):
        if self.project_config.dataset_class == InteractionsDataset:
            module = self.get_trained_module()
            val_loader = self.get_val_generator()

            print("================== Evaluate ========================")
            trial = (
                Trial(
                    module,
                    self._get_optimizer(module),
                    self._get_loss_function(),
                    callbacks=[],
                    metrics=self.metrics,
                )
                .to(self.torch_device)
                .with_test_generator(val_loader)
            )  # .eval()

            scores_tensor: Union[torch.Tensor, Tuple[torch.Tensor]] = trial.predict(
                verbose=2
            )
            scores: np.ndarray = scores_tensor.detach().cpu().numpy().reshape(-1)

            df_eval = self.val_data_frame
            df_eval["score"] = scores

            group = df_eval.sample(frac=1).groupby(
                ["timestamp", "user_idx", "session_idx", "step"]
            )

            df_eval = group.agg({"impressions": "first"})
            df_eval["list_item_idx"] = group["item_idx"].apply(list)
            df_eval["list_score"] = group["score"].apply(list)
            df_eval["pos_item_idx"] = group["pos_item_idx"].apply(list)
            df_eval["clicked"] = group["clicked"].apply(list)
            df_eval["item_idx"] = df_eval.apply(
                lambda row: int(
                    np.max(np.array(row["clicked"]) * np.array(row["list_item_idx"]))
                ),
                axis=1,
            )

            def sort_and_bin(row):
                list_sorted, score = zip(
                    *sorted(
                        zip(row["list_item_idx"], row["list_score"]),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )
                list_sorted = (np.array(list_sorted) == row["item_idx"]).astype(int)

                return list(list_sorted)

            df_eval["sorted_list"] = df_eval.apply(sort_and_bin, axis=1)
            df_eval = df_eval.reset_index()

            df_eval.head()

            metric = {
                "reciprocal_rank@5": np.mean(
                    [reciprocal_rank_at_k(l, 5) for l in list(df_eval["sorted_list"])]
                ),
                "precision@1": np.mean(
                    [precision_at_k(l, 1) for l in list(df_eval["sorted_list"])]
                ),
                "ndcg@5": np.mean(
                    [ndcg_at_k(l, 5) for l in list(df_eval["sorted_list"])]
                ),
                "MRR": mean_reciprocal_rank(list(df_eval["sorted_list"])),
            }

            with open(
                os.path.join(self.output().path, "metric.json"), "w"
            ) as params_file:
                json.dump(metric, params_file, default=lambda o: dict(o), indent=4)

            print(json.dumps(metric, indent=4))
            df_eval.to_csv(os.path.join(self.output().path, "df_eval.csv"))
