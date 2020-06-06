import luigi
import pandas as pd

from mars_gym.task.model.evaluation import EvaluateTestSetPredictions
from mars_gym.task.model.trivago.propensity_score import (
    FillTrivagoPropensityScoreMixin,
)


class EvaluateTrivagoTestSetPredictions(
    FillTrivagoPropensityScoreMixin, EvaluateTestSetPredictions
):
    def output(self):
        return luigi.LocalTarget(super().output().path + "_ps_" + self.fill_ps_strategy)

    def requires(self):
        if self.fill_ps_strategy == "model":
            return super(EvaluateTrivagoTestSetPredictions, self).requires()
        if not self.no_offpolicy_eval:
            return [self.direct_estimator]
        return []

    @property
    def ps_base_df(self) -> pd.DataFrame:
        base_df = pd.concat(
            [
                pd.read_csv(self.model_training.train_data_frame_path),
                pd.read_csv(self.model_training.val_data_frame_path),
                pd.read_csv(self.model_training.test_data_frame_path),
            ],
            ignore_index=True,
        )
        return base_df

    @property
    def output_column(self) -> str:
        return self.model_training.project_config.output_column.name
