from multiprocessing.pool import Pool
from typing import Dict

import numpy as np
import pandas as pd

from recommendation.task.model.evaluation import EvaluateTestSetPredictions


class EvaluateTrivagoTestSetPredictions(EvaluateTestSetPredictions):

    def requires(self):
        if not self.no_offpolicy_eval:
            return [self.direct_estimator]
        return []

    def fill_ps(self, df: pd.DataFrame, pool: Pool):
        all_df = pd.concat([pd.read_csv(self.model_training.train_data_frame_path),
                            pd.read_csv(self.model_training.val_data_frame_path),
                            pd.read_csv(self.model_training.test_data_frame_path)],
                          ignore_index=True)
        ground_truth_df = all_df[all_df[self.model_training.project_config.output_column.name] == 1]
        ps_per_pos_item_idx: Dict[int, float] = {
            pos_item_idx: np.sum(ground_truth_df["pos_item_idx"] == pos_item_idx) / len(ground_truth_df)
            for pos_item_idx in df["pos_item_idx"].unique()
        }

        df[self.model_training.project_config.propensity_score_column_name] = df["pos_item_idx"].apply(
            lambda pos_item_idx: ps_per_pos_item_idx[pos_item_idx])

