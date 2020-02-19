from typing import Tuple, List, Union, Optional, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from recommendation.task.meta_config import ProjectConfig, IOType, Column
from recommendation.utils import parallel_literal_eval


def literal_eval_array_columns(data_frame: pd.DataFrame, columns: List[Column]):
    for column in columns:
        if column.type == IOType.ARRAY and column.name in data_frame:
            data_frame[column.name] = parallel_literal_eval(data_frame[column.name])


class InteractionsDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig) -> None:
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

        return tuple(rows[input_column].values for input_column in self._input_columns), \
               output


class ContextualBanditsDataset(InteractionsDataset):

    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig) -> None:

        super().__init__(data_frame, metadata_data_frame, project_config)

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

        user_indices = rows[self._input_columns[0]].values
        item_indices = rows[self._input_columns[1]].values
        shift_indices = rows[self._input_columns[2]].values
        user_item_visits = rows[self._input_columns[3]].values
        user_item_buys = rows[self._input_columns[4]].values

        # Propensity Score - Probability
        positive_items = self._get_items(item_indices, user_item_visits, user_item_buys)
        ps = rows[self._auxiliar_output_columns[0]].values.astype(float)
        output = rows[self._output_column].values

        return (user_indices, *positive_items, shift_indices), (output, ps)


class DirectEstimatorDataset(InteractionsDataset):

    def __init__(self, data_frame: pd.DataFrame, metadata_data_frame: Optional[pd.DataFrame],
                 project_config: ProjectConfig) -> None:

        super().__init__(data_frame, metadata_data_frame, project_config)

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
        user_indices = rows[self._input_columns[0]].values
        item_indices = rows[self._input_columns[1]].values
        shift_indices = rows[self._input_columns[2]].values
        user_item_visits = rows[self._input_columns[3]].values
        user_item_buys = rows[self._input_columns[4]].values

        positive_items = self._get_items(item_indices, user_item_visits, user_item_buys)
        output = rows[self._output_column].values

        if self._auxiliar_output_columns:
            output = tuple([output]) + tuple(rows[auxiliar_output_column].values
                                             for auxiliar_output_column in self._auxiliar_output_columns)

        return (user_indices, shift_indices, *positive_items), (output)
