from recommendation.data import InteractionsDataset
from recommendation.task.data_preparation import ifood, trivago, new_ifood
from recommendation.task.meta_config import *

PROJECTS: Dict[str, ProjectConfig] = {
    "ifood_ensamble_mab": ProjectConfig(
        base_dir=ifood.BASE_DIR,
        prepare_data_frames_task=new_ifood.PrepareNewIfoodInteractionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("account_idx", IOType.INDEX),
        item_column=Column("merchant_idx", IOType.INDEX),
        #,,weekday breakfast,weekday dawn,weekday dinner,weekday lunch,weekday snack,weekend dawn,weekend dinner,weekend lunch,,merchant_buys_cum,
        # account_buys_cum,avg_merc_score,avg_delivery_fee,avg_distance
        # ,
        #     Column("avg_merc_score", IOType.NUMBER), Column("avg_delivery_fee", IOType.NUMBER), Column("avg_distance", IOType.NUMBER)
        other_input_columns=[
            Column("weekday breakfast", IOType.NUMBER), Column("weekday dawn", IOType.NUMBER),
            Column("weekday dinner", IOType.NUMBER), Column("weekday lunch", IOType.NUMBER), Column("weekday snack", IOType.NUMBER),
            Column("weekend dawn", IOType.NUMBER), Column("weekend dinner", IOType.NUMBER), Column("weekend lunch", IOType.NUMBER),
            Column("avg_merc_score", IOType.NUMBER), Column("avg_delivery_fee", IOType.NUMBER), Column("avg_distance", IOType.NUMBER)
        ],
        metadata_columns=[
        ],
        output_column=Column("buys", IOType.NUMBER),
        timestamp_column_name="order_date_local",
        available_arms_column_name="merc_list_idx",
        auxiliar_output_columns=[],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
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
        timestamp_column_name="click_timestamp",
        available_arms_column_name="available_merchant_idx_list",
        hist_view_column_name="hist_visits",
        hist_output_column_name="hist_buys",
        auxiliar_output_columns=[],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "trivago_contextual_bandit": ProjectConfig(
        base_dir=trivago.BASE_DIR,
        prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("user_idx", IOType.INDEX),
        item_column=Column("item_idx", IOType.INDEX),
        available_arms_column_name="impressions",
        other_input_columns=[
            Column("pos_item_idx", IOType.NUMBER),

            Column("diff_price", IOType.NUMBER), #price #action_type_item_idx
            Column("platform_idx", IOType.INDEX),
            Column("device_idx", IOType.NUMBER),
            Column("sum_action_item_before", IOType.NUMBER),
            Column("is_first_in_impression", IOType.NUMBER),

            Column("list_action_type_idx", IOType.INDEX_ARRAY),
            Column("list_reference_clickout_item_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_image_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_info_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_rating_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_deals_idx", IOType.INDEX_ARRAY),
            Column("list_reference_search_for_item_idx", IOType.INDEX_ARRAY),

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
        auxiliar_output_columns=[],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "trivago_policy_estimator": ProjectConfig(
        base_dir=trivago.BASE_DIR,
        prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("user_idx", IOType.INDEX),
        item_column=Column("item_idx", IOType.INDEX),
        item_is_input=False,
        available_arms_column_name="impressions",
        other_input_columns=[
            Column("pos_item_idx", IOType.NUMBER),

            Column("diff_price", IOType.NUMBER), #price #action_type_item_idx
            Column("platform_idx", IOType.INDEX),
            Column("device_idx", IOType.NUMBER),
            Column("sum_action_item_before", IOType.NUMBER),
            Column("is_first_in_impression", IOType.NUMBER),

            Column("list_action_type_idx", IOType.INDEX_ARRAY),
            Column("list_reference_clickout_item_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_image_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_info_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_rating_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_deals_idx", IOType.INDEX_ARRAY),
            Column("list_reference_search_for_item_idx", IOType.INDEX_ARRAY),

            Column("list_reference_search_for_poi", IOType.INT_ARRAY),
            Column("list_reference_change_of_sort_order", IOType.INT_ARRAY),
            Column("list_reference_search_for_destination", IOType.INT_ARRAY),
            Column("list_reference_filter_selection", IOType.INT_ARRAY),
            Column("list_current_filters", IOType.INT_ARRAY),
        ],
        output_column=Column("item_idx", IOType.INDEX),
        hist_view_column_name="hist_views",
        hist_output_column_name="hist_clicked",
        auxiliar_output_columns=[],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),
    "trivago_contextual_bandit_available_all": ProjectConfig(
        base_dir=trivago.BASE_DIR,
        prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("user_idx", IOType.INDEX),
        item_column=Column("item_idx", IOType.INDEX),
        other_input_columns=[
            Column("diff_price", IOType.NUMBER), #price #action_type_item_idx
            Column("platform_idx", IOType.INDEX),
            Column("device_idx", IOType.NUMBER),
            Column("pos_item_idx", IOType.NUMBER),
            Column("sum_action_item_before", IOType.NUMBER),
            Column("is_first_in_impression", IOType.NUMBER),

            Column("list_action_type_idx", IOType.INDEX_ARRAY),
            Column("list_reference_clickout_item_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_image_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_info_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_rating_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_deals_idx", IOType.INDEX_ARRAY),
            Column("list_reference_search_for_item_idx", IOType.INDEX_ARRAY),

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
        auxiliar_output_columns=[],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),    
    "test_fixed_trivago_contextual_bandit": ProjectConfig(
        base_dir=trivago.BASE_DIR,
        prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("user_idx", IOType.INDEX),
        item_column=Column("item_idx", IOType.INDEX),
        available_arms_column_name="impressions",
        other_input_columns=[
            Column("first_item_idx", IOType.INDEX),
            Column("popularity_item_idx", IOType.INDEX),            
            Column("action_type_item_idx", IOType.INDEX)
        ],
        metadata_columns=[
        ],
        output_column=Column("clicked", IOType.NUMBER),
        hist_view_column_name="hist_views",
        hist_output_column_name="hist_clicked",
        auxiliar_output_columns=[],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),            
    "trivago_contextual_bandit_bce": ProjectConfig(
        base_dir=trivago.BASE_DIR,
        prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("user_idx", IOType.INDEX),
        item_column=Column("item_idx", IOType.INDEX),
        available_arms_column_name="impressions",
        other_input_columns=[
            Column("diff_price", IOType.NUMBER), #price #action_type_item_idx
            Column("platform_idx", IOType.INDEX),
            Column("device_idx", IOType.NUMBER),
            Column("pos_item_idx", IOType.NUMBER),
            Column("sum_action_item_before", IOType.NUMBER),
            Column("is_first_in_impression", IOType.NUMBER),

            Column("list_action_type_idx", IOType.INDEX_ARRAY),
            Column("list_reference_clickout_item_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_image_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_info_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_rating_idx", IOType.INDEX_ARRAY),
            Column("list_reference_interaction_item_deals_idx", IOType.INDEX_ARRAY),
            Column("list_reference_search_for_item_idx", IOType.INDEX_ARRAY),

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
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),        
    "test_trivago_contextual_bandit": ProjectConfig(
        base_dir=trivago.BASE_DIR,
        prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("user_idx", IOType.INDEX),
        item_column=Column("item_idx", IOType.INDEX),
        available_arms_column_name="impressions",
        other_input_columns=[
            Column("session_idx", IOType.NUMBER),
            Column("action_type_item_idx", IOType.INDEX)
        ],
        metadata_columns=[
        ],
        output_column=Column("clicked", IOType.NUMBER),
        hist_view_column_name="hist_views",
        hist_output_column_name="hist_clicked",
        auxiliar_output_columns=[],
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),        
    "test_trivago_contextual_bandit_bce": ProjectConfig(
        base_dir=trivago.BASE_DIR,
        prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
        dataset_class=InteractionsDataset,
        user_column=Column("user_idx", IOType.INDEX),
        item_column=Column("item_idx", IOType.INDEX),
        available_arms_column_name="impressions",
        other_input_columns=[
            Column("session_idx", IOType.NUMBER),
            Column("action_type_item_idx", IOType.INDEX)
        ],
        metadata_columns=[
        ],
        output_column=Column("clicked", IOType.NUMBER),
        hist_view_column_name="hist_views",
        hist_output_column_name="hist_clicked",
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    ),        
}
