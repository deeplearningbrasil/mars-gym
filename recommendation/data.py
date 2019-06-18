import abc
import ast
from typing import Tuple, List, Iterator, Union, Sized, Callable

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix

from recommendation.task.meta_config import ProjectConfig, IOType
from recommendation.torch import coo_matrix_to_sparse_tensor


class CorruptionTransformation(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, seed: int = 42) -> None:
        self._random_state = np.random.RandomState(seed)

    def setup(self, data: csr_matrix):
        pass

    @abc.abstractmethod
    def __call__(self, data: csr_matrix) -> np.ndarray:
        pass


class SupportBasedCorruptionTransformation(CorruptionTransformation):
    def setup(self, data: csr_matrix):
        self._supports = np.asarray(data.astype(bool).sum(axis=0)).flatten() / data.shape[0]

    def __call__(self, data: csr_matrix) -> np.ndarray:
        u = self._random_state.uniform(0, 1, len(data.indices))
        support: np.ndarray = self._supports[data.indices]
        # Normalize
        support = support / support.sum()

        indices_kept = data.indices[u > support]

        if len(indices_kept) == 0:
            return data.indices
        return np.array(indices_kept, dtype=np.int64)


class MaskingNoiseCorruptionTransformation(CorruptionTransformation):
    def __init__(self, fraction: float = 0.25, seed: int = 42) -> None:
        super().__init__(seed)
        self.fraction = fraction

    def __call__(self, data: csr_matrix) -> np.ndarray:
        n = len(data.indices)
        indices_kept = self._random_state.choice(data.indices, n - round(self.fraction * n), replace=False)

        return np.array(indices_kept, dtype=np.int64)


class SaltAndPepperNoiseCorruptionTransformation(CorruptionTransformation):
    def __init__(self, fraction: float = 0.25, seed: int = 42) -> None:
        super().__init__(seed)
        self.fraction = fraction

    def __call__(self, data: csr_matrix) -> np.ndarray:
        removed_indices = self._random_state.choice(data.indices, round(self.fraction * len(data.indices)),
                                                    replace=False)
        removed_indices = removed_indices[self._random_state.uniform(0, 1, len(removed_indices)) > 0.5]

        indices_kept = np.setdiff1d(data.indices, removed_indices)

        return np.array(indices_kept, dtype=np.int64)


class RatingsDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, project_config: ProjectConfig,
                 transformation: Union[Callable] = None) -> None:
        assert len(project_config.input_columns) == 2
        assert all(input_column.type == IOType.INDEX for input_column in project_config.input_columns)

        self._data_frame = data_frame
        self._input_columns = [input_column.name for input_column in project_config.input_columns]
        self._output_column = project_config.output_column.name

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row: pd.Series = self._data_frame.iloc[index]
        return torch.tensor([row[self._input_columns[0]], row[self._input_columns[1]]]), torch.tensor(row[self._output_column])


class RatingsArrayDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, project_config: ProjectConfig,
                 transformation: Union[CorruptionTransformation, Callable] = None) -> None:
        assert len(project_config.input_columns) == 1
        assert project_config.input_columns[0].name == project_config.output_column.name

        target_col = project_config.output_column.name
        if type(data_frame.iloc[0][target_col]) is str:
            data_frame[target_col] = data_frame[target_col].apply(lambda value: ast.literal_eval(value))

        i, j, data = zip(*((i, int(t[0]), t[1]) for i, row in enumerate(data_frame[target_col]) for t in row))
        self._data = csr_matrix((data, (i, j)), shape=(max(i) + 1, project_config.input_columns[0].length))

        if type(transformation) is CorruptionTransformation:
            transformation.setup(self._data)
        self._transformation = transformation

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row: csr_matrix = self._data[index]

        input_row, output_row = row, row
        if self._transformation:
            input_row = self._transformation(row)

        return coo_matrix_to_sparse_tensor(input_row.tocoo()), coo_matrix_to_sparse_tensor(output_row.tocoo())
