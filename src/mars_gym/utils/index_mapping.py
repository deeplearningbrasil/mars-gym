import functools
from collections import defaultdict
from typing import Dict, Any, List, Iterable

import numpy as np
import pandas as pd

from mars_gym.meta_config import ProjectConfig, IOType
from multiprocessing import Pool
import os

def create_index_mapping(
    indexable_values: Iterable, include_unkown: bool = True, include_none: bool = True
) -> Dict[Any, int]:
    indexable_values = list(
        sorted(set(value for value in indexable_values if not pd.isnull(value)))
    )
    if include_none:
        indexable_values = [None] + indexable_values
    if include_unkown:
        indices = np.arange(1, len(indexable_values) + 1)
        mapping = defaultdict(int, zip(indexable_values, indices))  # Unkown = 0
    else:
        indices = np.arange(0, len(indexable_values))
        mapping = dict(zip(indexable_values, indices))
    if include_none:
        mapping[np.nan] = 1  # Both None and nan are treated the same
    return mapping


def create_index_mapping_from_arrays(
    indexable_arrays: Iterable[list],
    include_unkown: bool = True,
    include_none: bool = True,
) -> Dict[Any, int]:
    all_values = set(value for values in indexable_arrays for value in values)
    return create_index_mapping(all_values, include_unkown, include_none)


def map_array(values: list, mapping: dict) -> List[int]:
    return [int(mapping[str(value)]) for value in values]

def transform_with_indexing(
    df: pd.DataFrame, index_mapping: Dict[str, dict], project_config: ProjectConfig,
):
    print("transform_with_indexing...")
    if df is None:
        return None

    with Pool(os.cpu_count()) as pool:
        for key, mapping in index_mapping.items():
            
            column = project_config.get_column_by_name(key)
            if column and key in df:
                if column.type == IOType.INDEXABLE:
                    df[key] = df[key].astype(str).map(mapping).astype(int)
                elif column.type == IOType.INDEXABLE_ARRAY:
                    df[key] = pool.map(functools.partial(map_array, mapping=mapping), df[key])
                    
