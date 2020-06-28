import torch.nn as nn
from mars_gym.model.trivago.trivago_models import SimpleLinearModel
from mars_gym.simulation.training import TORCH_DROPOUT_MODULES
from mars_gym.simulation.trivago.trivago_models import (
    TrivagoModelTraining,
    TrivagoModelInteraction,
)


class TrivagoLogisticModelInteraction(TrivagoModelInteraction):
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


class TrivagoLogisticModelTraining(TrivagoModelTraining):
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
