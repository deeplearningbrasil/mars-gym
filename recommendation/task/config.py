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
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "yelp_user_autoencoder": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=yelp.PrepareYelpAllUserRatingsDataFrames,
        dataset_class=RatingsArrayDataset,
        input_columns=[Column("stars_per_business", IOType.ARRAY)],
        output_column=Column("stars_per_business", IOType.ARRAY),
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "yelp_business_autoencoder": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=yelp.PrepareYelpAllBusinessRatingsDataFrames,
        dataset_class=RatingsArrayDataset,
        input_columns=[Column("stars_per_user", IOType.ARRAY)],
        output_column=Column("stars_per_user", IOType.ARRAY),
        recommender_type=RecommenderType.ITEM_BASED_COLLABORATIVE_FILTERING,
    ),
}
