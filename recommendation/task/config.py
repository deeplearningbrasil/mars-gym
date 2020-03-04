from recommendation.data import InteractionsDataset
from recommendation.task.data_preparation import ifood, trivago
from recommendation.task.meta_config import *

PROJECTS: Dict[str, ProjectConfig] = {
    "ifood_contextual_bandit": ProjectConfig(
        base_dir=ifood.BASE_DIR,
        prepare_data_frames_task=ifood.PrepareIfoodSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("account_idx", IOType.INDEX),
        item_column=Column("merchant_idx", IOType.INDEX),
        other_input_columns=[
            Column("shift_idx", IOType.INDEX), Column("hist_visits", IOType.NUMBER), Column("hist_buys", IOType.NUMBER),
        ],
        metadata_columns=[
            Column("trading_name", IOType.INT_ARRAY), Column("description", IOType.INT_ARRAY),
            Column("category_names", IOType.INT_ARRAY), Column("restaurant_complete_info", IOType.FLOAT_ARRAY),
        ],
        output_column=Column("buy", IOType.NUMBER),
        hist_view_column_name="hist_visits",
        hist_output_column_name="hist_buys",
        auxiliar_output_columns=[Column("ps", IOType.NUMBER)],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "trivago_contextual_bandit": ProjectConfig(
        base_dir=trivago.BASE_DIR,
        prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("user_idx", IOType.INDEX),
        item_column=Column("item_idx", IOType.INDEX),
        other_input_columns=[
            Column("price", IOType.NUMBER),
            Column("platform_idx", IOType.NUMBER),
            Column("device_idx", IOType.NUMBER),
            Column("pos_item_idx", IOType.NUMBER),

            Column("list_action_type_idx", IOType.INT_ARRAY),
            Column("list_reference_clickout_item_idx", IOType.INT_ARRAY),
            Column("list_reference_interaction_item_image_idx", IOType.INT_ARRAY),
            Column("list_reference_interaction_item_info_idx", IOType.INT_ARRAY),
            Column("list_reference_interaction_item_rating_idx", IOType.INT_ARRAY),
            Column("list_reference_interaction_item_deals_idx", IOType.INT_ARRAY),
            Column("list_reference_search_for_item_idx", IOType.INT_ARRAY),

            Column("list_reference_search_for_poi", IOType.INT_ARRAY),
            Column("list_reference_change_of_sort_order", IOType.INT_ARRAY),
            Column("list_reference_search_for_destination", IOType.INT_ARRAY),
            Column("list_reference_filter_selection", IOType.INT_ARRAY),
            Column("list_current_filters", IOType.INT_ARRAY),
        ],
        metadata_columns=[
            Column("list_metadata", IOType.INT_ARRAY),
        ],
        output_column=Column("clicked", IOType.NUMBER),
        hist_view_column_name="hist_views",
        hist_output_column_name="hist_clicked",
        auxiliar_output_columns=[Column("ps", IOType.NUMBER)],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),    
}
