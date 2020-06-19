from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
from mars_gym.meta_config import *
from samples.yoochoose import data

sample_yoochoose = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.InteractionDataFrame,
    dataset_class=InteractionsDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="available_items",
    other_input_columns=[
        Column("Timestamp_dayofweek", IOType.NUMBER),
        Column("Step", IOType.NUMBER),
        Column("ItemID_history", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
    ],
    output_column=Column("buy", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

sample_yoochoose_with_negative_sample = ProjectConfig(
    base_dir=data.BASE_DIR,
    prepare_data_frames_task=data.InteractionDataFrame,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("SessionID", IOType.INDEXABLE),
    item_column=Column("ItemID", IOType.INDEXABLE),
    timestamp_column_name="Timestamp",
    available_arms_column_name="available_items",
    other_input_columns=[
        Column("Timestamp_dayofweek", IOType.NUMBER),
        Column("Step", IOType.NUMBER),
        Column("ItemID_history", IOType.INDEXABLE_ARRAY, same_index_as="ItemID"),
    ],
    output_column=Column("buy", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)
