from enum import Enum, auto
from typing import List, Type, Dict, Optional

from torch.utils.data import Dataset
import numpy as np

from mars_gym.data.task import BasePrepareDataFrames


class IOType(Enum):
    INDEXABLE = auto()
    NUMBER = auto()
    INDEXABLE_ARRAY = auto()
    FLOAT_ARRAY = auto()
    INT_ARRAY = auto()

    @property
    def dtype(self):
        return {
            self.INDEXABLE.name: np.int64,
            self.NUMBER.name: np.float32,
            self.FLOAT_ARRAY.name: np.float32,
            self.INT_ARRAY.name: np.int64,
        }[self.name]


class RecommenderType(Enum):
    USER_BASED_COLLABORATIVE_FILTERING = auto()
    ITEM_BASED_COLLABORATIVE_FILTERING = auto()
    CONTENT_BASED = auto()


class Column(object):
    def __init__(self, name: str, type: IOType, same_index_as: str = None) -> None:
        self.name = name
        self.type = type
        self.same_index_as = same_index_as


class ProjectConfig(object):
    def __init__(
        self,
        base_dir: str,
        prepare_data_frames_task: Type[BasePrepareDataFrames],
        dataset_class: Type[Dataset],
        user_column: Column,
        item_column: Column,
        other_input_columns: List[Column],
        output_column: Column,
        recommender_type: RecommenderType = RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
        dataset_extra_params: dict = {},
        user_is_input: bool = True,
        item_is_input: bool = True,
        hist_view_column_name: str = "hist_view",
        hist_output_column_name: str = "hist_output",
        timestamp_column_name: str = "timestamp",
        available_arms_column_name: str = "available_arms",
        propensity_score_column_name: str = "ps",
        n_users_column: str = "n_users",
        n_items_column: str = "n_items",
        default_balance_fields: List[str] = [],
        metadata_columns: List[Column] = [],
        auxiliar_output_columns: List[Column] = [],
        possible_negative_indices_columns: Dict[str, List[str]] = None,
    ) -> None:
        self.base_dir = base_dir
        self.prepare_data_frames_task = prepare_data_frames_task
        self.dataset_class = dataset_class
        self.dataset_extra_params = dataset_extra_params
        self.user_column = user_column
        self.item_column = item_column
        self.other_input_columns = other_input_columns
        self.output_column = output_column
        self.user_is_input = user_is_input
        self.item_is_input = item_is_input
        self.hist_view_column_name = hist_view_column_name
        self.hist_output_column_name = hist_output_column_name
        self.timestamp_column_name = timestamp_column_name
        self.available_arms_column_name = available_arms_column_name
        self.auxiliar_output_columns = auxiliar_output_columns
        self.recommender_type = recommender_type
        self.propensity_score_column_name = propensity_score_column_name
        self.n_users_column = n_users_column
        self.n_items_column = n_items_column
        self.default_balance_fields = default_balance_fields
        self.metadata_columns = metadata_columns
        self.possible_negative_indices_columns = possible_negative_indices_columns

    @property
    def input_columns(self) -> List[Column]:
        input_columns: List[Column] = []
        if self.user_is_input:
            input_columns.append(self.user_column)
        if self.item_is_input:
            self._item_input_index = len(input_columns)
            input_columns.append(self.item_column)
        input_columns.extend(self.other_input_columns)
        input_columns.extend(self.metadata_columns)
        return input_columns

    @property
    def all_columns(self) -> List[Column]:
        return [
            self.user_column,
            self.item_column,
            *self.other_input_columns,
            *self.auxiliar_output_columns,
            *self.metadata_columns
        ]

    def get_column_by_name(self, name: str) -> Optional[Column]:
        for column in self.all_columns:
            if column.name == name:
                return column
        return None
