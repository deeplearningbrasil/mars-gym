import abc
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torchbearer
from torchbearer import Trial
from tqdm import tqdm
from typing import List

from mars_gym.data import InteractionsDataset
from mars_gym.task.evaluation.policy_estimator import PolicyEstimatorTraining
from mars_gym.torch.torch import FasterBatchSampler, NoAutoCollationDataLoader


class FillPropensityScoreMixin(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def policy_estimator(self) -> PolicyEstimatorTraining:
        pass

    @property
    @abc.abstractmethod
    def item_column(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def available_arms_column(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def propensity_score_column(self) -> str:
        pass

    def fill_ps(self, df: pd.DataFrame, pool: Pool):
        dataset = InteractionsDataset(df, None, self.policy_estimator.project_config)
        batch_sampler = FasterBatchSampler(
            dataset, self.policy_estimator.batch_size, shuffle=False
        )
        data_loader = NoAutoCollationDataLoader(dataset, batch_sampler=batch_sampler)

        trial = (
            Trial(
                self.policy_estimator.get_trained_module(),
                criterion=lambda *args: torch.zeros(
                    1, device=self.policy_estimator.torch_device, requires_grad=True
                ),
            )
            .with_generators(val_generator=data_loader)
            .to(self.policy_estimator.torch_device)
            .eval()
        )

        with torch.no_grad():
            log_probas: torch.Tensor = trial.predict(
                verbose=0, data_key=torchbearer.VALIDATION_DATA
            )
        probas: np.ndarray = torch.exp(log_probas).cpu().numpy()

        item_indices = df[self.item_column]

        params = (
            zip(item_indices, probas, df[self.available_arms_column])
            if self.available_arms_column
            else zip(item_indices, probas)
        )

        # from IPython import embed; embed()
        df[self.propensity_score_column] = list(
            tqdm(pool.starmap(_get_ps_from_probas, params), total=len(df))
        )


def _get_ps_from_probas(
    item_idx: int, probas: np.ndarray, available_item_indices: List[int] = None
) -> float:
    if available_item_indices:
        probas /= np.sum(probas[available_item_indices])
    return probas[item_idx]
