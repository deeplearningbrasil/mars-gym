import os
from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
from mars_gym.meta_config import *
from samples.trivago_rio import data

trivago_rio = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.PrepareTrivagoDataFrame,
    dataset_class=InteractionsDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("pos_item_id", IOType.NUMBER),
        Column("list_reference_item", IOType.INDEXABLE_ARRAY, same_index_as="item_id"),
    ],
    metadata_columns=[Column("list_metadata", IOType.INT_ARRAY),],
    output_column=Column("clicked", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)
