from typing import Dict

from recommendation.data import RatingsArrayDataset, RatingsDataset
from recommendation.task.data_preparation import yelp
from recommendation.task.meta_config import *

PROJECTS: Dict[str, ProjectConfig] = {
    "yelp": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=yelp.PrepareYelpRatingsDataFrames,
        dataset_class=RatingsDataset,
        input_columns=[Column("user_idx", IOType.INDEX), Column("business_idx", IOType.INDEX)],
        output_column=Column("stars", IOType.NUMBER),
    ),
    "yelp_user_autoencoder": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=yelp.PrepareYelpAllUserRatingsDataFrames,
        dataset_class=RatingsArrayDataset,
        input_columns=[Column("stars_per_business", IOType.ARRAY, length=192606)],
        output_column=Column("stars_per_business", IOType.ARRAY, length=192606),
    ),
    "yelp_business_autoencoder": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=yelp.PrepareYelpAllBusinessRatingsDataFrames,
        dataset_class=RatingsArrayDataset,
        input_columns=[Column("stars_per_user", IOType.ARRAY, length=1637138)],
        output_column=Column("stars_per_user", IOType.ARRAY, length=1637138),
    ),
}
