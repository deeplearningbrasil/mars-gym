from multiprocessing.pool import Pool
from typing import Dict, Tuple

import luigi
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from recommendation.task.model.evaluation import EvaluateTestSetPredictions


class EvaluateTrivagoTestSetPredictions(EvaluateTestSetPredictions):
    fill_ps_strategy: str = luigi.ChoiceParameter(choices=[
        "per_pos_item_idx",
        "per_item",
        "per_item_in_first_pos",
        "per_item_given_pos",
        "per_logistic_regression_of_pos_item_idx_and_item",
        "per_logistic_regression_of_pos_item_idx_and_item_ps",
    ], default="per_pos_item_idx")

    def output(self):
        return luigi.LocalTarget(super().output().path + "_ps_" + self.fill_ps_strategy)

    def requires(self):
        if not self.no_offpolicy_eval:
            return [self.direct_estimator]
        return []

    def _get_all_df(self):
        all_df = pd.concat([pd.read_csv(self.model_training.train_data_frame_path),
                            pd.read_csv(self.model_training.val_data_frame_path),
                            pd.read_csv(self.model_training.test_data_frame_path)],
                           ignore_index=True)
        return all_df

    def _get_ground_truth(self, all_df: pd.DataFrame = None) -> pd.DataFrame:
        if all_df is None:
            all_df = self._get_all_df()
        return all_df[all_df[self.model_training.project_config.output_column.name] == 1]

    def fill_ps_per_pos_item_idx(self, df: pd.DataFrame, pool: Pool):
        ground_truth_df = self._get_ground_truth()
        ps_per_pos_item_idx = self._create_dict_of_ps_per_pos_item_idx(ground_truth_df)

        df[self.model_training.project_config.propensity_score_column_name] = df["pos_item_idx"].apply(
            lambda pos_item_idx: ps_per_pos_item_idx.get(pos_item_idx, 0.0))

    def _create_dict_of_ps_per_pos_item_idx(self, ground_truth_df: pd.DataFrame) -> Dict[int, float]:
        return {
            pos_item_idx: np.sum(ground_truth_df["pos_item_idx"] == pos_item_idx) / len(ground_truth_df)
            for pos_item_idx in ground_truth_df["pos_item_idx"].unique()
        }

    def fill_ps_per_item(self, df: pd.DataFrame, pool: Pool):
        ground_truth_df = self._get_ground_truth()
        ps_per_item = self._create_dict_of_ps_per_item(ground_truth_df)

        item_column = self.model_training.project_config.item_column.name
        df[self.model_training.project_config.propensity_score_column_name] = df[item_column].apply(
            lambda item_idx: ps_per_item.get(item_idx, 0.0))

    def _create_dict_of_ps_per_item(self, ground_truth_df: pd.DataFrame) -> Dict[int, float]:
        item_column = self.model_training.project_config.item_column.name
        return {
            item: np.sum(ground_truth_df[item_column] == item) / len(ground_truth_df)
            for item in ground_truth_df[item_column].unique()
        }

    def fill_ps_per_item_in_first_pos(self, df: pd.DataFrame, pool: Pool):
        ground_truth_df = self._get_ground_truth()
        item_column = self.model_training.project_config.item_column.name
        ps_per_item_in_first_pos = {
            item: np.sum((ground_truth_df[item_column] == item) & (ground_truth_df["pos_item_idx"] == 0))
                  / np.sum(ground_truth_df[item_column] == item)
            for item in ground_truth_df[item_column].unique()
        }

        df[self.model_training.project_config.propensity_score_column_name] = df[item_column].apply(
            lambda item_idx: ps_per_item_in_first_pos.get(item_idx, 0.0))

    def fill_ps_per_item_given_pos(self, df: pd.DataFrame, pool: Pool):
        ground_truth_df = self._get_ground_truth()
        item_column = self.model_training.project_config.item_column.name
        ps_per_item_given_pos: Dict[Tuple[int, int], float] = {
            (pos_item_idx, item):
                np.sum((ground_truth_df[item_column] == item) & (ground_truth_df["pos_item_idx"] == pos_item_idx))
                / np.sum(ground_truth_df["pos_item_idx"] == pos_item_idx)
            for pos_item_idx in ground_truth_df["pos_item_idx"].unique()
            for item in ground_truth_df[item_column].unique()
        }

        df[self.model_training.project_config.propensity_score_column_name] = df.apply(
            lambda row: ps_per_item_given_pos.get((row["pos_item_idx"], row[item_column]), 0.0), axis=1)

    def fill_ps_per_logistic_regression_of_pos_item_idx_and_item(self, df: pd.DataFrame, pool: Pool):
        all_df = self._get_all_df()
        item_column = self.model_training.project_config.item_column.name

        train_df = all_df[["pos_item_idx", item_column]]

        encoder = OneHotEncoder()
        encoder.fit(train_df.values)

        model = LogisticRegression(class_weight="balanced")
        model.fit(encoder.transform(train_df.values),
                  all_df[self.model_training.project_config.output_column.name].values)

        df[self.model_training.project_config.propensity_score_column_name] = model.predict_proba(
            encoder.transform(df[["pos_item_idx", item_column]].values))[:, 1]

    def fill_ps_per_logistic_regression_of_pos_item_idx_and_item_ps(self, df: pd.DataFrame, pool: Pool):
        all_df = self._get_all_df()
        ground_truth_df = self._get_ground_truth(all_df)
        item_column = self.model_training.project_config.item_column.name

        ps_per_pos_item_idx = self._create_dict_of_ps_per_pos_item_idx(ground_truth_df)
        ps_per_item = self._create_dict_of_ps_per_item(ground_truth_df)

        train_df = pd.DataFrame(data={
            "ps_per_pos_item_idx": all_df["pos_item_idx"].apply(lambda pos_item_idx:
                                                            ps_per_pos_item_idx.get(pos_item_idx, 0.0)),
            "ps_per_item": all_df[item_column].apply(lambda item_idx: ps_per_item.get(item_idx, 0.0))
        })

        train_y = all_df[self.model_training.project_config.output_column.name].values
        model = LogisticRegression(class_weight="balanced")
        model.fit(train_df.values, train_y)

        test_df = pd.DataFrame(data={
            "ps_per_pos_item_idx": df["pos_item_idx"].apply(lambda pos_item_idx:
                                                            ps_per_pos_item_idx.get(pos_item_idx, 0.0)),
            "ps_per_item": df[item_column].apply(lambda item_idx: ps_per_item.get(item_idx, 0.0))
        })

        df[self.model_training.project_config.propensity_score_column_name] = model.predict_proba(test_df.values)[:, 1]

    def fill_ps(self, df: pd.DataFrame, pool: Pool):
        {
            "per_pos_item_idx": self.fill_ps_per_pos_item_idx,
            "per_item": self.fill_ps_per_item,
            "per_item_in_first_pos": self.fill_ps_per_item_in_first_pos,
            "per_item_given_pos": self.fill_ps_per_item_given_pos,
            "per_logistic_regression_of_pos_item_idx_and_item":
                self.fill_ps_per_logistic_regression_of_pos_item_idx_and_item,
            "per_logistic_regression_of_pos_item_idx_and_item_ps":
                self.fill_ps_per_logistic_regression_of_pos_item_idx_and_item_ps,
        }[self.fill_ps_strategy](df, pool)
