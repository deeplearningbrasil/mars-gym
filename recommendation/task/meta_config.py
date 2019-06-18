from enum import Enum, auto
from typing import List, Type

from torch.utils.data import Dataset

from recommendation.task.data_preparation.base import BasePrepareDataFrames

class IOType(Enum):
    INDEX = auto()
    NUMBER = auto()
    ARRAY = auto()


class Column(object):
    def __init__(self, name: str, type: IOType, length: int = 1) -> None:
        self.name = name
        self.type = type
        self.length = length


class ProjectConfig(object):
    def __init__(self, base_dir: str,
                 prepare_data_frames_task: Type[BasePrepareDataFrames],
                 dataset_class: Type[Dataset],
                 input_columns: List[Column],
                 output_column: Column,
                 n_users: int,
                 n_items: int,
                 default_balance_fields: List[str] = [],
                 ) -> None:
        self.base_dir = base_dir
        self.prepare_data_frames_task = prepare_data_frames_task
        self.dataset_class = dataset_class
        self.input_columns = input_columns
        self.output_column = output_column
        self.n_users = n_users
        self.n_items = n_items
        self.default_balance_fields = default_balance_fields
