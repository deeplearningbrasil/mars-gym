from recommendation.data import InteractionsDataset, \
    ContextualBanditsDataset, DirectEstimatorDataset
from recommendation.task.data_preparation import ifood
from recommendation.task.meta_config import *

PROJECTS: Dict[str, ProjectConfig] = {
    "ifood_contextual_bandit": ProjectConfig(
        base_dir=ifood.BASE_DIR,
        prepare_data_frames_task=ifood.PrepareIfoodSessionsDataFrames,
        dataset_class=ContextualBanditsDataset,
        input_columns=[Column("account_idx", IOType.INDEX), Column("merchant_idx", IOType.INDEX),
                        Column("shift_idx", IOType.NUMBER),
                        Column("hist_visits", IOType.NUMBER), Column("hist_buys", IOType.NUMBER)],
        metadata_columns=[Column("trading_name", IOType.ARRAY), Column("description", IOType.ARRAY),
                          Column("category_names", IOType.ARRAY), Column("restaurant_complete_info", IOType.ARRAY)],
        output_column=Column("buy", IOType.NUMBER),
        auxiliar_output_columns=[Column("ps", IOType.NUMBER)],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "ifood_contextual_bandit_direct_estimator": ProjectConfig(
        base_dir=ifood.BASE_DIR,
        prepare_data_frames_task=ifood.PrepareIfoodSessionsDataFrames,
        dataset_class=DirectEstimatorDataset,
        input_columns=[Column("account_idx", IOType.INDEX), Column("merchant_idx", IOType.INDEX),
                        Column("shift_idx", IOType.INDEX),
                        Column("hist_visits", IOType.NUMBER), Column("hist_buys", IOType.NUMBER)],
        metadata_columns=[Column("category_names", IOType.ARRAY), Column("restaurant_complete_info", IOType.ARRAY)],
        output_column=Column("buy", IOType.NUMBER),
        auxiliar_output_columns=[Column("ps", IOType.NUMBER)],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),    
    "ifood_offpolicy_direct_estimator": ProjectConfig(
        base_dir=ifood.BASE_DIR,
        prepare_data_frames_task=ifood.PrepareIfoodSessionsDataFrames,
        dataset_class=DirectEstimatorDataset,
        input_columns=[Column("account_idx", IOType.INDEX), Column("merchant_idx", IOType.INDEX),
                        Column("shift_idx", IOType.INDEX),
                        Column("hist_visits", IOType.NUMBER), Column("hist_buys", IOType.NUMBER)],
        metadata_columns=[Column("category_names", IOType.ARRAY), Column("restaurant_complete_info", IOType.ARRAY)],
        output_column=Column("buy", IOType.NUMBER),
        auxiliar_output_columns=[],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
}
