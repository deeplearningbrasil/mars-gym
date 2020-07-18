from typing import Optional, List, Tuple, Any

import numpy as np
from torchbearer import Trial, DataLoader

from mars_gym.model.bandit import BanditPolicy


class BanditAgent(object):
    def __init__(self, bandit: BanditPolicy) -> None:
        super().__init__()

        self.bandit = bandit

    def fit(
        self,
        trial: Optional[Trial],
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        self.bandit.fit(train_loader.dataset)

        if trial:
            trial.with_generators(
                train_generator=train_loader, val_generator=val_loader
            ).run(epochs=epochs)

    def act(
        self,
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: Optional[List[float]],
        with_probs=True,
    ) -> int:

        if with_probs:
            arm_idx = self.bandit.select_idx(arm_indices, arm_contexts, arm_scores)
            prob = self.bandit._compute_prob(arm_indices, arm_scores)
            return arm_indices[arm_idx], prob[arm_idx]
        else:
            return self.bandit.select(
                arm_indices, arm_contexts=arm_contexts, arm_scores=arm_scores
            )

    def rank(
        self,
        arms: List[Any],
        arm_indices: List[int],
        arm_contexts: Tuple[np.ndarray, ...],
        arm_scores: Optional[List[float]],
    ) -> Tuple[List[Any], List[float]]:
        return self.bandit.rank(
            arms=arms,
            arm_indices=arm_indices,
            arm_contexts=arm_contexts,
            arm_scores=arm_scores,
            with_probs=True,
        )
