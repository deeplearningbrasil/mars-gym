import os
from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
from mars_gym.meta_config import *
from samples.exp_trivago_rio import data as exp_trivago_data

PROJECTS: Dict[str, ProjectConfig] = {
    "trivago_rio": ProjectConfig(
        base_dir=exp_trivago_data.BASE_DIR,
        prepare_data_frames_task=exp_trivago_data.PrepareTrivagoSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("user_id", IOType.INDEXABLE),
        item_column=Column("item_id", IOType.INDEXABLE),
        available_arms_column_name="impressions",
        other_input_columns=[
            Column("pos_item_id", IOType.NUMBER),
            Column("list_reference_item", IOType.INDEXABLE_ARRAY),
        ],
        metadata_columns=[Column("list_metadata", IOType.INT_ARRAY),],
        output_column=Column("clicked", IOType.NUMBER),
        auxiliar_output_columns=[],
    )
}
