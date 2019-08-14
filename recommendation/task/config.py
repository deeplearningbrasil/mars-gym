from typing import Dict

from recommendation.data import InteractionsMatrixDataset, InteractionsDataset, \
    BinaryInteractionsWithOnlineRandomNegativeGenerationDataset, UserTripletWithOnlineRandomNegativeGenerationDataset
from recommendation.task.data_preparation import yelp, ifood
from recommendation.task.data_preparation.ifood import PrepareIfoodAccountMatrixWithBinaryBuysDataFrames
from recommendation.task.meta_config import *

PROJECTS: Dict[str, ProjectConfig] = {
    "yelp": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=yelp.PrepareYelpRatingsDataFrames,
        dataset_class=InteractionsDataset,
        input_columns=[Column("user_idx", IOType.INDEX), Column("business_idx", IOType.INDEX)],
        output_column=Column("stars", IOType.NUMBER),
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "yelp_user_autoencoder": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=yelp.PrepareYelpAllUserRatingsDataFrames,
        dataset_class=InteractionsMatrixDataset,
        input_columns=[Column("stars_per_business", IOType.ARRAY)],
        output_column=Column("stars_per_business", IOType.ARRAY),
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "yelp_business_autoencoder": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=yelp.PrepareYelpAllBusinessRatingsDataFrames,
        dataset_class=InteractionsMatrixDataset,
        input_columns=[Column("stars_per_user", IOType.ARRAY)],
        output_column=Column("stars_per_user", IOType.ARRAY),
        recommender_type=RecommenderType.ITEM_BASED_COLLABORATIVE_FILTERING,
    ),
    "ifood_binary_buys_cf": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=ifood.PrepareIfoodBinaryBuysInteractionsDataFrames,
        dataset_class=InteractionsDataset,
        input_columns=[Column("account_idx", IOType.INDEX), Column("merchant_idx", IOType.INDEX)],
        output_column=Column("buys", IOType.NUMBER),
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "ifood_binary_buys_cf_with_random_negative": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=ifood.PrepareIfoodBinaryBuysInteractionsDataFrames,
        dataset_class=BinaryInteractionsWithOnlineRandomNegativeGenerationDataset,
        input_columns=[Column("account_idx", IOType.INDEX), Column("merchant_idx", IOType.INDEX)],
        output_column=Column("buys", IOType.NUMBER),
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "ifood_binary_buys_triplet_with_random_negative": ProjectConfig(
        base_dir=yelp.BASE_DIR,
        prepare_data_frames_task=ifood.PrepareIfoodBinaryBuysInteractionsDataFrames,
        dataset_class=UserTripletWithOnlineRandomNegativeGenerationDataset,
        input_columns=[Column("account_idx", IOType.INDEX), Column("merchant_idx", IOType.INDEX)],
        output_column=Column("buys", IOType.NUMBER),
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
}
