import os
from typing import List

import luigi
import pandas as pd

from recommendation.fairness_metrics import calculate_fairness_metrics
from recommendation.files import get_test_set_predictions_path
from recommendation.task.model.base import BaseEvaluationTask
from recommendation.utils import parallel_literal_eval


class EvaluateTestSetPredictions(BaseEvaluationTask):
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
