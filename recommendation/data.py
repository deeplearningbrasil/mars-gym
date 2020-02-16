import abc
from typing import Tuple, List, Union, Callable, Optional, Set, Dict

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csr import csr_matrix
from torch.utils.data import Dataset

from recommendation.task.meta_config import ProjectConfig, IOType, RecommenderType, Column
from recommendation.utils import parallel_literal_eval

def literal_eval_array_columns(data_frame: pd.DataFrame, columns: List[Column]):
    for column in columns:
        if column.type == IOType.ARRAY and column.name in data_frame:
            data_frame[column.name] = parallel_literal_eval(data_frame[column.name])


class InteractionsDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig, transformation: Union[Callable] = None,
                 negative_indices_generator: 'NegativeIndicesGenerator' = None,
                 negative_proportion: float = 1.0) -> None:
        assert len(project_config.input_columns) >= 2
        assert all(
            input_column.type == IOType.INDEX or input_column.type == IOType.ARRAY or input_column.type == IOType.NUMBER
            for input_column in
            project_config.input_columns)

        literal_eval_array_columns(data_frame, project_config.input_columns + [project_config.output_column])
        literal_eval_array_columns(metadata_data_frame, project_config.input_columns + [
            project_config.output_column] + project_config.metadata_columns)

        self._metadata_data_frame = metadata_data_frame
        self._input_columns = [input_column.name for input_column in project_config.input_columns]
        self._index_input_columns = [input_column.name for input_column in project_config.input_columns
                                     if input_column.type == IOType.INDEX]
        self._metadata_columns = [metadata_column.name for metadata_column in project_config.metadata_columns]
        self._output_column = project_config.output_column.name
        self._auxiliar_output_columns = [auxiliar_output_column.name
                                         for auxiliar_output_column in project_config.auxiliar_output_columns]

        self._data_frame = data_frame[set(self._input_columns + [self._output_column] + self._auxiliar_output_columns)
            .intersection(data_frame.columns)]

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, ...],
                                                                   Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        if isinstance(indices, int):
            indices = [indices]
        rows: pd.Series = self._data_frame.iloc[indices]
        output = rows[self._output_column].values
        if self._auxiliar_output_columns:
            output = tuple([output]) + tuple(rows[auxiliar_output_column].values
                                             for auxiliar_output_column in self._auxiliar_output_columns)
        
        #raise(Exception(tuple(rows[input_column].values for input_column in self._input_columns)))
        return tuple(rows[input_column].values for input_column in self._input_columns), \
               output


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
        if possible_indices:
            return np.random.choice(possible_indices)
        else:
            print("No possible indices for %s=%d" % (input_column, pivot_index))
            return pivot_index

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

from sklearn.preprocessing import StandardScaler
class ContextualBanditsDataset(InteractionsDataset):

    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig, transformation: Union[Callable] = None,
                 negative_indices_generator: NegativeIndicesGenerator = None,
                 negative_proportion: float = 1.0) -> None:

        super().__init__(data_frame, metadata_data_frame, project_config, transformation)

        self._items_df = self._metadata_data_frame[[self._input_columns[1]] + self._metadata_columns].set_index(
            self._input_columns[1],
            drop=False).sort_index()

        # Asserting index is contiguous to extract the embeddings
        assert all(self._items_df.reset_index(drop=True).index == self._items_df.index)
        self._embeddings_for_metadata_columns: Dict[str, np.ndarray] = {}
        for column_name in self._metadata_columns:
            emb = self._items_df[column_name].values.tolist()
            dtype = np.float32 if column_name == "restaurant_complete_info" else np.int64
            self._embeddings_for_metadata_columns[column_name] = np.array(emb, dtype=dtype)
            self._items_df.drop(columns=[column_name], inplace=True)
    
        self._users = self._data_frame[self._input_columns[0]].unique()
        self._items = self._data_frame[self._input_columns[1]].unique()

        self._n_users: int = self._users.shape[0]
        self._n_items: int = self._items.shape[0]
        self._vocab_size: int = metadata_data_frame.iloc[0]["vocab_size"]
        self._non_text_input_dim: int = metadata_data_frame.iloc[0]["non_textual_input_dim"]


    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _get_items(self, item_indices: np.ndarray, visits: np.ndarray, buys: np.ndarray) -> Tuple[np.ndarray, ...]:
        res = []

        res.append(item_indices.astype(np.int64))
        for column_name in self._metadata_columns:
            res.append(self._embeddings_for_metadata_columns[column_name][item_indices])

        res.append(visits.astype(np.float32))
        res.append(buys.astype(np.float32))
        return tuple(res)

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, Tuple[np.ndarray, ...],
                                                                         Tuple[np.ndarray, ...]]]:
        if isinstance(indices, int):
            indices = [indices]

        rows: pd.Series = self._data_frame.iloc[indices]

        user_indices    = rows[self._input_columns[0]].values
        item_indices    = rows[self._input_columns[1]].values
        shift_indices   = rows[self._input_columns[2]].values
        user_item_visits = rows[self._input_columns[3]].values
        user_item_buys  = rows[self._input_columns[4]].values

        # Propensity Score - Probability
        positive_items = self._get_items(item_indices, user_item_visits, user_item_buys)
        ps             = rows[self._auxiliar_output_columns[0]].values.astype(float)
        output         = rows[self._output_column].values
        
        return (user_indices, *positive_items, shift_indices), (output, ps)

class DirectEstimatorDataset(InteractionsDataset):

    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig, transformation: Union[Callable] = None,
                 negative_indices_generator: NegativeIndicesGenerator = None,
                 negative_proportion: float = 1.0) -> None:

        super().__init__(data_frame, metadata_data_frame, project_config, transformation)

        self._items_df = self._metadata_data_frame[[self._input_columns[1]] + self._metadata_columns].set_index(
            self._input_columns[1],
            drop=False).sort_index()

        # Asserting index is contiguous to extract the embeddings
        assert all(self._items_df.reset_index(drop=True).index == self._items_df.index)
        self._embeddings_for_metadata_columns: Dict[str, np.ndarray] = {}
        for column_name in self._metadata_columns:
            emb = self._items_df[column_name].values.tolist()
            dtype = np.float32 if column_name == "restaurant_complete_info" else np.int64
            self._embeddings_for_metadata_columns[column_name] = np.array(emb, dtype=dtype)
            self._items_df.drop(columns=[column_name], inplace=True)
    
        self._users = self._data_frame[self._input_columns[0]].unique()
        self._items = self._data_frame[self._input_columns[1]].unique()

        self._n_users: int = self._users.shape[0]
        self._n_items: int = self._items.shape[0]
        self._vocab_size: int = metadata_data_frame.iloc[0]["vocab_size"]
        self._non_text_input_dim: int = metadata_data_frame.iloc[0]["non_textual_input_dim"]

        self._auxiliar_output_columns = [auxiliar_output_column.name
                                         for auxiliar_output_column in project_config.auxiliar_output_columns]

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _get_items(self, item_indices: np.ndarray, visits: np.ndarray, buys: np.ndarray) -> Tuple[np.ndarray, ...]:
        res = []

        res.append(item_indices.astype(np.int64))
        for column_name in self._metadata_columns:
            res.append(self._embeddings_for_metadata_columns[column_name][item_indices])

        res.append(visits.astype(np.float32))
        res.append(buys.astype(np.float32))
        return tuple(res)

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, Tuple[np.ndarray, ...],
                                                                         Tuple[np.ndarray, ...]]]:
        if isinstance(indices, int):
            indices = [indices]

        rows: pd.Series = self._data_frame.iloc[indices]
        user_indices    = rows[self._input_columns[0]].values
        item_indices    = rows[self._input_columns[1]].values
        shift_indices   = rows[self._input_columns[2]].values
        user_item_visits = rows[self._input_columns[3]].values
        user_item_buys  = rows[self._input_columns[4]].values

        positive_items = self._get_items(item_indices, user_item_visits, user_item_buys)
        output         = rows[self._output_column].values

        if self._auxiliar_output_columns:
            output = tuple([output]) + tuple(rows[auxiliar_output_column].values
                                             for auxiliar_output_column in self._auxiliar_output_columns)
                
        return (user_indices, shift_indices, *positive_items), (output)

