from typing import Tuple, List, Union, Optional, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from recommendation.task.meta_config import ProjectConfig, IOType, Column
from recommendation.utils import parallel_literal_eval

def literal_eval_array_columns(data_frame: pd.DataFrame, columns: List[Column]):
    for column in columns:
        if (column.type == IOType.FLOAT_ARRAY or column.type == IOType.INT_ARRAY) and column.name in data_frame:
            data_frame[column.name] = parallel_literal_eval(data_frame[column.name])


def preprocess_interactions_data_frame(data_frame: pd.DataFrame, project_config: ProjectConfig):
    data_frame[project_config.user_column.name] = data_frame[project_config.user_column.name].astype(int)
    data_frame[project_config.item_column.name] = data_frame[project_config.item_column.name].astype(int)
    literal_eval_array_columns(
        data_frame, [project_config.user_column, project_config.item_column, project_config.output_column]
                    + [input_column for input_column in project_config.other_input_columns])
    if project_config.available_arms_column_name and isinstance(
            data_frame.iloc[0][project_config.available_arms_column_name], str):
        data_frame[project_config.available_arms_column_name] = parallel_literal_eval(
            data_frame[project_config.available_arms_column_name])
    return data_frame


def preprocess_metadata_data_frame(metadata_data_frame: pd.DataFrame,
                                   project_config: ProjectConfig) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    literal_eval_array_columns(metadata_data_frame, project_config.metadata_columns)
    metadata_data_frame = metadata_data_frame.set_index(project_config.item_column.name, drop=False).sort_index()

    embeddings_for_metadata_columns: Dict[str, np.ndarray] = {}
    for metadata_column in project_config.metadata_columns:
        if metadata_column.type in (IOType.FLOAT_ARRAY, IOType.INT_ARRAY):
            emb = metadata_data_frame[metadata_column.name].values.tolist()
            dtype = np.float32 if metadata_column.type == IOType.FLOAT_ARRAY else np.int64
            embeddings_for_metadata_columns[metadata_column.name] = np.array(emb, dtype=dtype)
            metadata_data_frame.drop(columns=[metadata_column.name], inplace=True)

    return metadata_data_frame, embeddings_for_metadata_columns


class InteractionsDataset(Dataset):
    def __init__(self,  data_frame: pd.DataFrame, 
                        metadata_data_frame: Optional[pd.DataFrame],
                        embeddings_for_metadata_columns: Optional[Dict[str, np.ndarray]],
                        project_config: ProjectConfig) -> None:
        self._project_config = project_config
        self._input_columns = [project_config.user_column, project_config.item_column] + [
            input_column for input_column in project_config.other_input_columns]

        input_column_names = [input_column.name for input_column in self._input_columns]
        auxiliar_output_column_names = [auxiliar_output_column.name
                                        for auxiliar_output_column in project_config.auxiliar_output_columns]

        self._data_frame = data_frame[
            set(input_column_names + [project_config.output_column.name] + auxiliar_output_column_names)
                .intersection(data_frame.columns)]
        self._metadata_data_frame = metadata_data_frame
        self._embeddings_for_metadata_columns = embeddings_for_metadata_columns

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _convert_dtype(self, value: np.ndarray, type: IOType) -> np.ndarray:
        if type == IOType.INDEX:
            return value.astype(np.int64)
        if type == IOType.NUMBER:
            return value.astype(np.float64)            
        if type == IOType.INT_ARRAY:
            return np.array([np.array(v, dtype=np.int64) for v in value])
        if type == IOType.FLOAT_ARRAY:
            return np.array([np.array(v, dtype=np.float64) for v in value])
        return value

    def _get_metadata(self, item_indices: np.ndarray, column: Column) -> np.ndarray:
        if column.type in (IOType.FLOAT_ARRAY, IOType.INT_ARRAY):
            return self._embeddings_for_metadata_columns[column.name][item_indices]
        else:
            return self._convert_dtype(self._metadata_data_frame[column.name].values, column.type)

    def __getitem__(self, indices: Union[int, List[int]]) -> Tuple[Tuple[np.ndarray, ...],
                                                                   Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        if isinstance(indices, int):
            indices = [indices]
        rows: pd.Series = self._data_frame.iloc[indices]

        inputs = tuple(self._convert_dtype(rows[column.name].values, column.type) for column in self._input_columns)
        if self._metadata_data_frame is not None:
            item_indices = inputs[1]
            inputs += tuple(
                self._get_metadata(item_indices, column) for column in self._project_config.metadata_columns)

        output = self._convert_dtype(rows[self._project_config.output_column.name].values,
                                      self._project_config.output_column.type)
        if self._project_config.auxiliar_output_columns:
            output = tuple([output]) + tuple(self._convert_dtype(rows[column.name].values, column.type)
                                             for column in self._project_config.auxiliar_output_columns)
        return inputs, output
