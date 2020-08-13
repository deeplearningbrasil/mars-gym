import torch
from torch import nn as nn
from typing import List, Callable, Dict, Any

from mars_gym.meta_config import Column, IOType, ProjectConfig
from mars_gym.model.abstract import RecommenderModule
from mars_gym.torch.init import lecun_normal_init


class PolicyEstimator(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        embedding_dim: int,
        layers: List[int],
        sample_batch: List[torch.Tensor],
        weight_init: Callable = lecun_normal_init,
    ) -> None:
        super().__init__(project_config, index_mapping)

        num_elements_per_embeddings = [
            max(self._index_mapping[input_column.name].values()) + 1
            for input_column in self._project_config.input_columns
            if input_column.type in (IOType.INDEXABLE, IOType.INDEXABLE_ARRAY)
        ]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, embedding_dim) for n in num_elements_per_embeddings]
        )

        sample_transformed_inputs = self.transform_inputs(sample_batch)

        input_dim = sample_transformed_inputs.shape[1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim if i == 0 else layers[i - 1], layer_size)
                for i, layer_size in enumerate(layers)
            ]
        )
        self.output = nn.Linear(
            layers[-1] if len(layers) > 0 else input_dim, self._n_items
        )

        self.weight_init = weight_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)
        if type(module) == nn.Embedding:
            self.weight_init(module.weight)

    def flatten(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.size(0), -1)

    def transform_inputs(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        transformed_inputs: List[torch.Tensor] = []
        embeddings_idx = 0
        for input, input_column in zip(inputs, self._project_config.input_columns):
            #from IPython import embed; embed()
            # if embeddings_idx == 0:
            #    embeddings_idx += 1
            #    continue
            if input_column.type == IOType.INDEXABLE:
                transformed_inputs.append(self.embeddings[embeddings_idx](input))
                embeddings_idx += 1
            elif input_column.type == IOType.INDEXABLE_ARRAY:
                transformed_inputs.append(
                    self.flatten(self.embeddings[embeddings_idx](input))
                )
                embeddings_idx += 1
            elif input_column.type in (IOType.FLOAT_ARRAY, IOType.INT_ARRAY):
                transformed_inputs.append(input.float())
            else:
                transformed_inputs.append(input.view(-1, 1).float())

        return torch.cat(transformed_inputs, dim=1)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        out = self.transform_inputs(inputs)

        for layer in self.layers:
            out = torch.selu(layer(out))
        out = self.output(out)
        return torch.log_softmax(out, dim=1)
