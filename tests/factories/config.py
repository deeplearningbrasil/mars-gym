import os
from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
from mars_gym.meta_config import *
from tests.factories.data import UnitTestDataFrames

test_base_training = ProjectConfig(
    base_dir=os.path.join("tests", "output", "test"),
    prepare_data_frames_task=UnitTestDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user", IOType.INDEXABLE),
    item_column=Column("item", IOType.INDEXABLE),
    other_input_columns=[],
    metadata_columns=[],
    output_column=Column("reward", IOType.NUMBER),
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)
