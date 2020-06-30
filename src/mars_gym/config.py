import os
from mars_gym.data.dataset import (
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
)
from mars_gym.data import trivago
from mars_gym.meta_config import *
from mars_gym.data import trivago as exp_trivago_data


# trivago_rio = ProjectConfig(
#     base_dir=exp_trivago_data.BASE_DIR,
#     prepare_data_frames_task=exp_trivago_data.PrepareTrivagoDataFrame,
#     dataset_class=InteractionsDataset,
#     user_column=Column("user_id", IOType.INDEXABLE),
#     item_column=Column("item_id", IOType.INDEXABLE),
#     available_arms_column_name="impressions",
#     other_input_columns=[
#         Column("pos_item_id", IOType.NUMBER),
#         Column("list_reference_item", IOType.INDEXABLE_ARRAY, same_index_as="item_id"),
#     ],
#     metadata_columns=[Column("list_metadata", IOType.INT_ARRAY),],
#     output_column=Column("clicked", IOType.NUMBER),
#     auxiliar_output_columns=[],
#     recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
# )

trivago_contextual_bandit = ProjectConfig(
    base_dir=trivago.BASE_DIR,
    prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("pos_item_idx", IOType.NUMBER),
        Column("diff_price", IOType.NUMBER),  # price #action_type_item_idx
        Column("platform_idx", IOType.INDEXABLE),
        Column("device_idx", IOType.NUMBER),
        Column("sum_action_item_before", IOType.NUMBER),
        Column("is_first_in_impression", IOType.NUMBER),
        Column("list_action_type_idx", IOType.INDEXABLE_ARRAY),
        Column(
            "list_reference_clickout_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_image_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_info_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_rating_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_deals_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_search_for_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column("list_reference_search_for_poi", IOType.INT_ARRAY),
        Column("list_reference_change_of_sort_order", IOType.INT_ARRAY),
        Column("list_reference_search_for_destination", IOType.INT_ARRAY),
        Column("list_reference_filter_selection", IOType.INT_ARRAY),
        Column("list_current_filters", IOType.INT_ARRAY),
    ],
    metadata_columns=[Column("list_metadata", IOType.INT_ARRAY),],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    auxiliar_output_columns=[],
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

trivago_contextual_bandit_with_mostpopular = ProjectConfig(
    base_dir=trivago.BASE_DIR,
    prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("pos_item_idx", IOType.NUMBER),
        Column("popularity_item_idx", IOType.INDEXABLE),
        Column("diff_price", IOType.NUMBER),  # price #action_type_item_idx
        Column("platform_idx", IOType.INDEXABLE),
        Column("device_idx", IOType.NUMBER),
        Column("sum_action_item_before", IOType.NUMBER),
        Column("is_first_in_impression", IOType.NUMBER),
        Column("list_action_type_idx", IOType.INDEXABLE_ARRAY),
        Column(
            "list_reference_clickout_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_image_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_info_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_rating_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_deals_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_search_for_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column("list_reference_search_for_poi", IOType.INT_ARRAY),
        Column("list_reference_change_of_sort_order", IOType.INT_ARRAY),
        Column("list_reference_search_for_destination", IOType.INT_ARRAY),
        Column("list_reference_filter_selection", IOType.INT_ARRAY),
        Column("list_current_filters", IOType.INT_ARRAY),
    ],
    metadata_columns=[Column("list_metadata", IOType.INT_ARRAY),],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    auxiliar_output_columns=[],
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

trivago_contextual_bandit_with_negative_item_generation = ProjectConfig(
    base_dir=trivago.BASE_DIR,
    prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsWithNegativeItemGenerationDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("pos_item_idx", IOType.NUMBER),
        Column("diff_price", IOType.NUMBER),  # price #action_type_item_idx
        Column("platform_idx", IOType.INDEXABLE),
        Column("device_idx", IOType.NUMBER),
        Column("sum_action_item_before", IOType.NUMBER),
        Column("is_first_in_impression", IOType.NUMBER),
        Column("list_action_type_idx", IOType.INDEXABLE_ARRAY),
        Column(
            "list_reference_clickout_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_image_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_info_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_rating_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_deals_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_search_for_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column("list_reference_search_for_poi", IOType.INT_ARRAY),
        Column("list_reference_change_of_sort_order", IOType.INT_ARRAY),
        Column("list_reference_search_for_destination", IOType.INT_ARRAY),
        Column("list_reference_filter_selection", IOType.INT_ARRAY),
        Column("list_current_filters", IOType.INT_ARRAY),
    ],
    metadata_columns=[Column("list_metadata", IOType.INT_ARRAY),],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    auxiliar_output_columns=[],
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

trivago_policy_estimator = ProjectConfig(
    base_dir=trivago.BASE_DIR,
    prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    item_is_input=False,
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("pos_item_idx", IOType.NUMBER),
        Column("diff_price", IOType.NUMBER),  # price #action_type_item_idx
        Column("platform_idx", IOType.INDEXABLE),
        Column("device_idx", IOType.NUMBER),
        Column("sum_action_item_before", IOType.NUMBER),
        Column("is_first_in_impression", IOType.NUMBER),
        Column("list_action_type_idx", IOType.INDEXABLE_ARRAY),
        Column(
            "list_reference_clickout_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_image_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_info_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_rating_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_deals_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_search_for_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column("list_reference_search_for_poi", IOType.INT_ARRAY),
        Column("list_reference_change_of_sort_order", IOType.INT_ARRAY),
        Column("list_reference_search_for_destination", IOType.INT_ARRAY),
        Column("list_reference_filter_selection", IOType.INT_ARRAY),
        Column("list_current_filters", IOType.INT_ARRAY),
    ],
    output_column=Column("item_idx", IOType.INDEXABLE),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    auxiliar_output_columns=[],
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

trivago_contextual_bandit_available_all = ProjectConfig(
    base_dir=trivago.BASE_DIR,
    prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    other_input_columns=[
        Column("diff_price", IOType.NUMBER),  # price #action_type_item_idx
        Column("platform_idx", IOType.INDEXABLE),
        Column("device_idx", IOType.NUMBER),
        Column("pos_item_idx", IOType.NUMBER),
        Column("sum_action_item_before", IOType.NUMBER),
        Column("is_first_in_impression", IOType.NUMBER),
        Column("list_action_type_idx", IOType.INDEXABLE_ARRAY),
        Column(
            "list_reference_clickout_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_image_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_info_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_rating_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_deals_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_search_for_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column("list_reference_search_for_poi", IOType.INT_ARRAY),
        Column("list_reference_change_of_sort_order", IOType.INT_ARRAY),
        Column("list_reference_search_for_destination", IOType.INT_ARRAY),
        Column("list_reference_filter_selection", IOType.INT_ARRAY),
        Column("list_current_filters", IOType.INT_ARRAY),
    ],
    metadata_columns=[Column("list_metadata", IOType.INT_ARRAY),],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    auxiliar_output_columns=[],
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

test_fixed_trivago_contextual_bandit = ProjectConfig(
    base_dir=trivago.BASE_DIR,
    prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("first_item_idx", IOType.INDEXABLE),
        Column("popularity_item_idx", IOType.INDEXABLE),
        Column("action_type_item_idx", IOType.INDEXABLE),
    ],
    metadata_columns=[],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    auxiliar_output_columns=[],
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

trivago_contextual_bandit_bce = ProjectConfig(
    base_dir=trivago.BASE_DIR,
    prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("diff_price", IOType.NUMBER),  # price #action_type_item_idx
        Column("platform_idx", IOType.INDEXABLE),
        Column("device_idx", IOType.NUMBER),
        Column("pos_item_idx", IOType.NUMBER),
        Column("sum_action_item_before", IOType.NUMBER),
        Column("is_first_in_impression", IOType.NUMBER),
        Column("list_action_type_idx", IOType.INDEXABLE_ARRAY),
        Column(
            "list_reference_clickout_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_image_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_info_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_rating_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_interaction_item_deals_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column(
            "list_reference_search_for_item_idx",
            IOType.INDEXABLE_ARRAY,
            same_index_as="item_idx",
        ),
        Column("list_reference_search_for_poi", IOType.INT_ARRAY),
        Column("list_reference_change_of_sort_order", IOType.INT_ARRAY),
        Column("list_reference_search_for_destination", IOType.INT_ARRAY),
        Column("list_reference_filter_selection", IOType.INT_ARRAY),
        Column("list_current_filters", IOType.INT_ARRAY),
    ],
    metadata_columns=[Column("list_metadata", IOType.INT_ARRAY),],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

test_trivago_contextual_bandit = ProjectConfig(
    base_dir=trivago.BASE_DIR,
    prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("session_idx", IOType.NUMBER),
        Column("action_type_item_idx", IOType.INDEXABLE),
    ],
    metadata_columns=[],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    auxiliar_output_columns=[],
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

test_trivago_contextual_bandit_bce = ProjectConfig(
    base_dir=trivago.BASE_DIR,
    prepare_data_frames_task=trivago.PrepareTrivagoSessionsDataFrames,
    dataset_class=InteractionsDataset,
    user_column=Column("user_idx", IOType.INDEXABLE),
    item_column=Column("item_idx", IOType.INDEXABLE),
    available_arms_column_name="impressions",
    other_input_columns=[
        Column("session_idx", IOType.NUMBER),
        Column("action_type_item_idx", IOType.INDEXABLE),
    ],
    metadata_columns=[],
    output_column=Column("clicked", IOType.NUMBER),
    hist_view_column_name="hist_views",
    hist_output_column_name="hist_clicked",
    recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
)

