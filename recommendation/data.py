import abc
from typing import Tuple, List, Union, Callable, Optional, Set, Dict

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csr import csr_matrix
from torch.utils.data import Dataset

from recommendation.task.meta_config import ProjectConfig, IOType, RecommenderType, Column
from recommendation.torch import coo_matrix_to_sparse_tensor
from recommendation.utils import parallel_literal_eval


class CorruptionTransformation(object, metaclass=abc.ABCMeta):
    def __init__(self, seed: int = 42) -> None:
        self._random_state = np.random.RandomState(seed)

    def setup(self, data: csr_matrix):
        pass

    @abc.abstractmethod
    def __call__(self, data: csr_matrix) -> csr_matrix:
        pass


class RemovalCorruptionTransformation(CorruptionTransformation, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def select_col_indices_to_remove(self, row_index: int, col_indices: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, data: csr_matrix) -> csr_matrix:
        removed_row_indices = []
        removed_col_indices = []
        for i in range(data.shape[0]):
            col_indices = data.indices[data.indptr[i]:data.indptr[i + 1]]
            col_indices_to_remove = self.select_col_indices_to_remove(i, col_indices)
            if len(col_indices_to_remove) > 0:
                removed_row_indices.extend([i for _ in range(len(col_indices_to_remove))])
                removed_col_indices.extend(col_indices_to_remove)

        if len(removed_row_indices) == 0:
            return data

        data = data.copy()
        data[removed_row_indices, removed_col_indices] = 0
        data.eliminate_zeros()

        return data


class SupportBasedCorruptionTransformation(RemovalCorruptionTransformation):
    def __init__(self, intensity: float = 1.0, seed: int = 42) -> None:
        super().__init__(seed)
        self._intensity = intensity

    def setup(self, data: csr_matrix):
        self._supports = np.asarray(data.astype(bool).sum(axis=0)).flatten() / data.shape[0]

    def select_col_indices_to_remove(self, row_index: int, col_indices: np.ndarray) -> np.ndarray:
        u = self._random_state.uniform(0, 1, len(col_indices))
        support: np.ndarray = self._supports[col_indices]
        support = support / support.sum()
        return col_indices[u / self._intensity < support]


class MaskingNoiseCorruptionTransformation(RemovalCorruptionTransformation):
    def __init__(self, fraction: float = 0.25, seed: int = 42) -> None:
        super().__init__(seed)
        self.fraction = fraction

    def select_col_indices_to_remove(self, row_index: int, col_indices: np.ndarray) -> np.ndarray:
        return self._random_state.choice(col_indices, round(self.fraction * len(col_indices)), replace=False)


class SaltAndPepperNoiseCorruptionTransformation(MaskingNoiseCorruptionTransformation):
    def __init__(self, fraction: float = 0.25, seed: int = 42) -> None:
        super().__init__(seed)
        self.fraction = fraction

    def select_col_indices_to_remove(self, row_index: int, col_indices: np.ndarray) -> np.ndarray:
        removed_indices = super().select_col_indices_to_remove(row_index, col_indices)
        return removed_indices[self._random_state.uniform(0, 1, len(removed_indices)) > 0.5]


class CriteoDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, project_config: ProjectConfig,
                 metadata_data_frame: Optional[pd.DataFrame], transformation: Union[Callable] = None,
                 negative_indices_generator: 'NegativeIndicesGenerator' = None) -> None:

        self._data_frame = data_frame

        self._dense_columns = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13']
        self._categorical_columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                                     'C14',
                                     'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']

        self._input_columns = [input_column.name for input_column in project_config.input_columns]
        self._output_column = project_config.output_column.name
        self._train = True

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        rows: pd.Series = self._data_frame.iloc[indices]

        rows_dense = rows[self._dense_columns].values  # )#.astype(float)
        rows_categories = rows[self._categorical_columns].values  # )#.astype(float)

        if self._train:
            return (rows_dense, rows_categories), torch.FloatTensor(rows[self._output_column].values)
        else:
            return (rows_dense, rows_categories)


def literal_eval_array_columns(data_frame: pd.DataFrame, columns: List[Column]):
    for column in columns:
        if column.type == IOType.ARRAY and column.name in data_frame:
            data_frame[column.name] = parallel_literal_eval(data_frame[column.name])


class InteractionsDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig, transformation: Union[Callable] = None,
                 negative_indices_generator: 'NegativeIndicesGenerator' = None) -> None:
        assert len(project_config.input_columns) >= 2
        assert all(input_column.type == IOType.INDEX or input_column.type == IOType.ARRAY for input_column in
                   project_config.input_columns)

        literal_eval_array_columns(data_frame, project_config.input_columns + [project_config.output_column])
        literal_eval_array_columns(metadata_data_frame, project_config.input_columns + [
            project_config.output_column] + project_config.metadata_columns)

        self._data_frame = data_frame
        self._metadata_data_frame = metadata_data_frame
        self._input_columns = [input_column.name for input_column in project_config.input_columns]
        self._index_input_columns = [input_column.name for input_column in project_config.input_columns
                                     if input_column.type == IOType.INDEX]
        self._metadata_columns = [metadata_column.name for metadata_column in project_config.metadata_columns]
        self._output_column = project_config.output_column.name

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        rows: pd.Series = self._data_frame.iloc[indices]
        return tuple(rows[input_column].values for input_column in self._input_columns), \
               rows[self._output_column].values


class InteractionsMatrixDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig,
                 transformation: Union[CorruptionTransformation, Callable] = None,
                 negative_indices_generator: 'NegativeIndicesGenerator' = None) -> None:
        assert len(project_config.input_columns) == 1
        assert project_config.input_columns[0].name == project_config.output_column.name

        self._n_users: int = data_frame.iloc[0][project_config.n_users_column]
        self._n_items: int = data_frame.iloc[0][project_config.n_items_column]
        dim = self._n_items if project_config.recommender_type == RecommenderType.USER_BASED_COLLABORATIVE_FILTERING \
            else self._n_users

        literal_eval_array_columns(data_frame, project_config.input_columns + [project_config.output_column])

        target_col = project_config.output_column.name

        i, j, data = zip(
            *((index, int(t[0]), t[1]) for index, row in enumerate(data_frame[target_col])
              for t in row))
        self._data = csr_matrix((data, (i, j)), shape=(max(i) + 1, dim))

        if isinstance(transformation, CorruptionTransformation):
            transformation.setup(self._data)
        self._transformation = transformation

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(indices, int):
            indices = [indices]
        rows: csr_matrix = self._data[indices]

        if self._transformation:
            input_rows = self._transformation(rows)
            return coo_matrix_to_sparse_tensor(input_rows.tocoo()), coo_matrix_to_sparse_tensor(rows.tocoo())
        return (coo_matrix_to_sparse_tensor(rows.tocoo()),) * 2


class InteractionsAndContentDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig,
                 transformation: Union[CorruptionTransformation, Callable] = None,
                 negative_indices_generator: 'NegativeIndicesGenerator' = None) -> None:
        assert len(project_config.input_columns) >= 1

        self._input_columns = [input_column.name for input_column in project_config.input_columns]
        self._metadata_columns = [metadata_column.name for metadata_column in project_config.metadata_columns]
        self._output_column = project_config.output_column.name

        self._n_users: int = data_frame.iloc[0][project_config.n_users_column]
        self._n_items: int = data_frame.iloc[0][project_config.n_items_column]

        dim = self._n_items if project_config.recommender_type == RecommenderType.USER_BASED_COLLABORATIVE_FILTERING \
            else self._n_users

        literal_eval_array_columns(data_frame, project_config.input_columns + [project_config.output_column])
        literal_eval_array_columns(metadata_data_frame, project_config.input_columns + [
            project_config.output_column] + project_config.metadata_columns)

        target_col = project_config.output_column.name
        i, j, data = zip(
            *((index, int(t[0]), t[1]) for index, row in enumerate(data_frame[target_col])
              for t in row))
        self._interaction_matrix = csr_matrix((data, (i, j)), shape=(max(i) + 1, dim))
        self._content = metadata_data_frame.set_index(self._input_columns[1])

        if isinstance(transformation, CorruptionTransformation):
            transformation.setup(self._interaction_matrix)
        self._transformation = transformation

    def __len__(self) -> int:
        return self._interaction_matrix.shape[0]

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        rows: csr_matrix = self._interaction_matrix[indices]
        content_rows = self._content.loc[rows[self._input_columns[1]]]

        if self._transformation:
            input_rows = self._transformation(rows)
            return tuple([coo_matrix_to_sparse_tensor(input_rows.tocoo()), content_rows.values]), \
                   coo_matrix_to_sparse_tensor(rows.tocoo())

        return tuple(coo_matrix_to_sparse_tensor(rows.tocoo()), content_rows.values), \
               coo_matrix_to_sparse_tensor(rows.tocoo())


class NegativeIndicesGenerator(object):
    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 input_columns: List[str], possible_negative_indices_columns: Dict[str, List[str]] = None) -> None:
        self._data_frame = data_frame
        self._metadata_data_frame = metadata_data_frame
        self._input_columns = input_columns

        if possible_negative_indices_columns:
            self._data_frame = self._data_frame.set_index(input_columns, drop=False).sort_index()

            assert all(column in self._input_columns
                       for column in possible_negative_indices_columns.keys())
            self._possible_negative_indices: Dict[str, Dict[str, Set[int]]] = {}

            for input_column, pivot_columns in possible_negative_indices_columns.items():
                possible_negative_indices_for_input_column: Dict[str, Set[int]] = {}

                for pivot_column in pivot_columns:
                    possible_negative_indices_for_input_column[pivot_column] = set(
                        metadata_data_frame.loc[metadata_data_frame[pivot_column] == 1][input_column].values)

                self._possible_negative_indices[input_column] = possible_negative_indices_for_input_column

        self._non_zero_indices = set(
            data_frame[[input_column for input_column in self._input_columns]].itertuples(index=False, name=None))
        self._max_values = [data_frame[input_column].max() for input_column in self._input_columns]

    def _get_pivot_index(self, input_column: str, previous_indices: List[int]):
        positive_examples_df = self._data_frame.loc[tuple(previous_indices)]
        random_positive_row = positive_examples_df.iloc[np.random.randint(0, len(positive_examples_df))]
        pivot_index = random_positive_row[input_column]
        return pivot_index

    def _generate_random_index_by_pivot(self, input_column: str, pivot_index: int):
        possible_indices: List[int] = []
        for pivot_column, pivot_possible_indices in self._possible_negative_indices[input_column].items():
            if pivot_index in pivot_possible_indices:
                possible_indices.extend([index for index in pivot_possible_indices if index != pivot_index])
        return np.random.choice(possible_indices)

    def _generate_random_index(self, input_column: str, max_value: int, previous_indices: List[int]):
        if hasattr(self, "_possible_negative_indices"):
            if input_column in self._possible_negative_indices:
                pivot_index = self._get_pivot_index(input_column, previous_indices)

                return self._generate_random_index_by_pivot(input_column, pivot_index)
            else:
                return self._data_frame.iloc[np.random.randint(0, len(self._data_frame))][input_column]
        else:
            return np.random.randint(0, max_value + 1)

    def generate_negative_indices(self, fixed_indices: List[int] = None) -> Tuple[int, ...]:
        while True:
            indices = fixed_indices or []
            num_fixed_indices = len(fixed_indices) if fixed_indices is not None else 0

            for input_column, max_value in \
                    zip(self._input_columns[num_fixed_indices:], self._max_values[num_fixed_indices:]):
                indices.append(self._generate_random_index(input_column, max_value, indices))

            indices = tuple(indices)
            if indices not in self._non_zero_indices:
                return indices

    def generate_negative_indices_given_positive(self, positive_indices: List[int]) -> Tuple[int, ...]:
        while True:
            indices = positive_indices[:-1]

            if hasattr(self, "_possible_negative_indices"):
                indices.append(self._generate_random_index_by_pivot(self._input_columns[-1], positive_indices[-1]))
            else:
                indices.append(np.random.randint(0, self._max_values[-1] + 1))

            indices = tuple(indices)
            if indices not in self._non_zero_indices:
                return indices


class BinaryInteractionsWithOnlineRandomNegativeGenerationDataset(InteractionsDataset):

    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig, transformation: Union[Callable] = None,
                 negative_indices_generator: NegativeIndicesGenerator = None) -> None:
        data_frame = data_frame[data_frame[project_config.output_column.name] > 0]
        super().__init__(data_frame, metadata_data_frame, project_config, transformation)

        self._negative_indices_generator = negative_indices_generator

    def __len__(self) -> int:
        return super().__len__() * 2

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]

        n = super().__len__()

        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)

        positive_batch: Tuple[Tuple[np.ndarray, ...], np.ndarray] = super().__getitem__(positive_indices)
        if num_of_negatives > 0:
            negative_batch: Tuple[Tuple[List[int], ...], np.ndarray] = (tuple(zip(*[
                self._negative_indices_generator.generate_negative_indices()
                for _ in
                range(num_of_negatives)])), np.zeros(num_of_negatives))
            return tuple(np.append(positive_batch[0][i], negative_batch[0][i])
                         for i, _ in enumerate(self._input_columns)), np.append(positive_batch[1], negative_batch[1])
        else:
            return tuple(positive_batch[0][i] for i, _ in enumerate(self._input_columns)), positive_batch[1]


class UserTripletWithOnlineRandomNegativeGenerationDataset(BinaryInteractionsWithOnlineRandomNegativeGenerationDataset):
    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                                   list]:
        if isinstance(indices, int):
            indices = [indices]

        rows: pd.Series = self._data_frame.iloc[indices]
        user_indices = rows[self._input_columns[0]].values
        positive_item_indices = rows[self._input_columns[1]].values
        negative_item_indices = np.array(
            [self._negative_indices_generator.generate_negative_indices_given_positive([user_index, item_index])[-1]
             for user_index, item_index in zip(user_indices, positive_item_indices)], dtype=np.int64).flatten()
        return (user_indices, positive_item_indices, negative_item_indices), []


class UserTripletContentWithOnlineRandomNegativeGenerationDataset(InteractionsDataset):

    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig, transformation: Union[Callable] = None,
                 negative_indices_generator: NegativeIndicesGenerator = None) -> None:

        data_frame = data_frame[data_frame[project_config.output_column.name] > 0].reset_index()
        super().__init__(data_frame, metadata_data_frame, project_config, transformation)

        self._negative_indices_generator = negative_indices_generator

        self._items_df = self._metadata_data_frame[self._input_columns[1:] + self._metadata_columns].set_index(
            self._input_columns[1],
            drop=False).sort_index()

        self._users = self._data_frame[self._input_columns[0]].unique()
        self._items = self._data_frame[self._input_columns[1]].unique()

        self._n_users: int = self._users.shape[0]
        self._n_items: int = self._items.shape[0]
        self._vocab_size: int = metadata_data_frame.iloc[0]["vocab_size"]
        self._non_text_input_dim: int = metadata_data_frame.iloc[0]["non_textual_input_dim"]

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _get_items(self, item_indices: List[int]) -> Tuple[torch.Tensor, ...]:
        res = []
        df_items = self._items_df.loc[item_indices]
        for column_name in self._metadata_columns:
            c = df_items[column_name].values.tolist()
            dtype = torch.float32 if column_name == "restaurant_complete_info" else torch.int64
            res.append(torch.tensor(np.array(c), dtype=dtype))
        return tuple(res)

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, Tuple[np.ndarray, ...],
                                                                         Tuple[np.ndarray, ...]], list]:
        if isinstance(indices, int):
            indices = [indices]

        rows: pd.Series = self._data_frame.iloc[indices]
        user_indices = rows[self._input_columns[0]].values
        positive_item_indices = rows[self._input_columns[1]].values
        negative_item_indices = np.array(
            [self._negative_indices_generator.generate_negative_indices_given_positive([user_index, item_index])[-1]
             for user_index, item_index in zip(user_indices, positive_item_indices)], dtype=np.int64).flatten()

        # raise Exception("indices: {} | {}".format(
        #    positive_item_indices, negative_item_indices))

        positive_items = self._get_items(positive_item_indices)
        negative_items = self._get_items(negative_item_indices)

        return (user_indices, positive_items, negative_items), []
