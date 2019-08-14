import torch.nn as nn
import luigi
from recommendation.model.triplet_net import TripletNet
from recommendation.task.model.base import BaseTorchModelTraining, TORCH_WEIGHT_INIT


class TripletNetTraining(BaseTorchModelTraining):
    loss_function: str = luigi.ChoiceParameter(choices=["triplet_margin", "bpr_triplet"], default="triplet_margin")

    n_factors: int = luigi.IntParameter(default=20)
    weight_init: str = luigi.ChoiceParameter(choices=TORCH_WEIGHT_INIT.keys(), default="lecun_normal")

    def create_module(self) -> nn.Module:
        return TripletNet(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            weight_init=TORCH_WEIGHT_INIT[self.weight_init],
        )
