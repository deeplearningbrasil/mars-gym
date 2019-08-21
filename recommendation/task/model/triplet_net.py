import luigi
import torch.nn as nn

from recommendation.model.triplet_net import TripletNet
from recommendation.task.model.base import TORCH_WEIGHT_INIT
from recommendation.task.model.embedding import UserAndItemEmbeddingTraining


class TripletNetTraining(UserAndItemEmbeddingTraining):
    loss_function: str = luigi.ChoiceParameter(choices=["triplet_margin", "bpr_triplet"], default="triplet_margin")

    def create_module(self) -> nn.Module:
        return TripletNet(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )
