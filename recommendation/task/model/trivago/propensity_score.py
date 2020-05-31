import abc
from multiprocessing import Pool

import luigi
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


# Should be used with FillPropensityScoreMixin
class FillTrivagoPropensityScoreMixin(object, metaclass=abc.ABCMeta):
    fill_ps_strategy: str = luigi.ChoiceParameter(choices=[
        "per_pos_item_idx",
        "per_item",
        "per_item_in_first_pos",
        "per_item_given_pos",
        "per_logistic_regression_of_pos_item_idx_and_item",
        "per_logistic_regression_of_pos_item_idx_and_item_ps",
        "model",
        "dummy",
        "per_prob",
    ], default="per_logistic_regression_of_pos_item_idx_and_item_ps")

    @property
    @abc.abstractmethod
    def ps_base_df(self) -> pd.DataFrame:
        pass

    @property
    @abc.abstractmethod
    def output_column(self) -> str:
        pass

    def fill_ps_per_pos_item_idx(self, df: pd.DataFrame, pool: Pool):
        ps_per_pos_item_idx = self._create_dict_of_ps_per_pos_item_idx(self.ps_base_df)

        df[self.propensity_score_column] = df["pos_item_idx"].apply(
            lambda pos_item_idx: ps_per_pos_item_idx.get(pos_item_idx, 0.0))

    def _create_dict_of_ps_per_pos_item_idx(self, base_df: pd.DataFrame) -> Dict[int, float]:
        return {
            pos_item_idx: np.sum(base_df["pos_item_idx"] == pos_item_idx) / len(base_df)
            for pos_item_idx in base_df["pos_item_idx"].unique()
        }

    def fill_ps_per_item(self, df: pd.DataFrame, pool: Pool):
        ps_per_item = self._create_dict_of_ps_per_item(self.ps_base_df)

        df[self.propensity_score_column] = df[self.item_column].apply(
            lambda item_idx: ps_per_item.get(item_idx, 0.0))

    def _create_dict_of_ps_per_item(self, base_df: pd.DataFrame) -> Dict[int, float]:
        return {
            item: np.sum(base_df[self.item_column] == item) / len(base_df)
            for item in base_df[self.item_column].unique()
        }

    def fill_ps_per_item_in_first_pos(self, df: pd.DataFrame, pool: Pool):
        ps_per_item_in_first_pos = {
            item: np.sum((self.ps_base_df[self.item_column] == item) & (self.ps_base_df["pos_item_idx"] == 0))
                  / np.sum(self.ps_base_df[self.item_column] == item)
            for item in self.ps_base_df[self.item_column].unique()
        }

        df[self.propensity_score_column] = df[self.item_column].apply(
            lambda item_idx: ps_per_item_in_first_pos.get(item_idx, 0.0))

    def fill_ps_per_item_given_pos(self, df: pd.DataFrame, pool: Pool):
        ps_per_item_given_pos: Dict[Tuple[int, int], float] = {
            (pos_item_idx, item):
                np.sum((self.ps_base_df[self.item_column] == item) & (self.ps_base_df["pos_item_idx"] == pos_item_idx))
                / np.sum(self.ps_base_df["pos_item_idx"] == pos_item_idx)
            for pos_item_idx in self.ps_base_df["pos_item_idx"].unique()
            for item in self.ps_base_df[self.item_column].unique()
        }

        df[self.propensity_score_column] = df.apply(
            lambda row: ps_per_item_given_pos.get((row["pos_item_idx"], row[self.item_column]), 0.0), axis=1)

    def fill_ps_per_logistic_regression_of_pos_item_idx_and_item(self, df: pd.DataFrame, pool: Pool):
        train_df = self.ps_base_df[["pos_item_idx", self.item_column]]

        encoder = OneHotEncoder()
        encoder.fit(train_df.values)

        model = LogisticRegression(class_weight="balanced")
        model.fit(encoder.transform(train_df.values),
                  self.ps_base_df[self.output_column].values)

        df[self.propensity_score_column] = model.predict_proba(
            encoder.transform(df[["pos_item_idx", self.item_column]].values))[:, 1]

    def fill_ps_per_logistic_regression_of_pos_item_idx_and_item_ps(self, df: pd.DataFrame, pool: Pool):
        ps_per_pos_item_idx = self._create_dict_of_ps_per_pos_item_idx(self.ps_base_df)
        ps_per_item = self._create_dict_of_ps_per_item(self.ps_base_df)

        train_df = pd.DataFrame(data={
            "ps_per_pos_item_idx": self.ps_base_df["pos_item_idx"].apply(lambda pos_item_idx:
                                                                      ps_per_pos_item_idx.get(pos_item_idx, 0.0)),
            "ps_per_item": self.ps_base_df[self.item_column].apply(lambda item_idx: ps_per_item.get(item_idx, 0.0))
        })

        train_y = self.ps_base_df[self.output_column].values
        model = LogisticRegression(class_weight="balanced")
        model.fit(train_df.values, train_y)

        test_df = pd.DataFrame(data={
            "ps_per_pos_item_idx": df["pos_item_idx"].apply(lambda pos_item_idx:
                                                            ps_per_pos_item_idx.get(pos_item_idx, 0.0)),
            "ps_per_item": df[self.item_column].apply(lambda item_idx: ps_per_item.get(item_idx, 0.0))
        })

        df[self.propensity_score_column] = model.predict_proba(test_df.values)[:, 1]

    def fill_ps_per_prob(self, df: pd.DataFrame, pool: Pool):
        df_all  =  self.ps_base_df
        #
        df_prob = df_all.item_idx.value_counts(normalize=True).reset_index()
        df_prob.columns = ['item_idx', 'ps_prob']

        _df = df.merge(df_prob, on='item_idx').fillna(0)
        df[self.propensity_score_column] = _df['ps_prob']
        

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
            "per_prob": self.fill_ps_per_prob,
            "model": super().fill_ps,
            "dummy": lambda x,y: 1,
        }[self.fill_ps_strategy](df, pool)

        #df[self.propensity_score_column] = 1/df[self.propensity_score_column]
