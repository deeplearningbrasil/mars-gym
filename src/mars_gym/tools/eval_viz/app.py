import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import os
from mars_gym.tools.eval_viz.plot import *
from mars_gym.tools.eval_viz.util import *
import random
import os

pd.set_option("display.max_colwidth", -1)

PATH_EVALUATION = os.getenv('PATH_EVALUATION') or "output/evaluation/"
PATH_EVAL_REINFORCEMENT = os.getenv('PATH_EVAL_REINFORCEMENT') or "output/interaction/"
PATH_TRAIN = os.getenv('PATH_TRAIN') or "output/models/"


PAGES = {"Home": "pages.home", "Model": "pages.model"}

GRAPH_METRIC = {"Bar": plot_bar, "Line": plot_line, "Radar": plot_radar}

GRAPH_METRIC_MODEL = {"Hist": plot_hist, "Box": plot_box}

RECSYS_METRICS = [
    "count",
    "mean_average_precision",
    "precision_at_1",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_15",
    "ndcg_at_20",
    "ndcg_at_50",
    "coverage_at_5",
    "coverage_at_10",
    "coverage_at_15",
    "coverage_at_20",
    "coverage_at_50",
    "personalization_at_5",
    "personalization_at_10",
    "personalization_at_15",
    "personalization_at_20",
    "personalization_at_50",
    "IPS",
    "CIPS",
    "SNIPS",
    "DirectEstimator",
    "DoublyRobust",
]

MISTREATMENT_METRICS = [
    "total_class",
    "false_positive_rate",
    "false_negative_rate",
    "true_positive_rate",
    "true_negative_rate",
    "positive_rate",
    "negative_rate",
    "accuracy",
    "balance_accuracy",
    "total_positives",
    "total_negatives",
    "total_individuals",
]

# @st.cache
def fetch_training_path():
    paths = []
    models = []
    for root, dirs, files in os.walk(PATH_TRAIN):
        if "/results" in root:
            for d in dirs:
                paths.append(os.path.join(root, d))
                models.append(d)

    return dict(zip(models, paths))


# @st.cache
def fetch_results_path():
    paths = []
    models = []
    for root, dirs, files in os.walk(PATH_EVALUATION):
        if "/results" in root and "Evaluate" in root:
            for d in dirs:
                paths.append(os.path.join(root, d))
                models.append(d)  # .replace("_"+d.split("_")[-1], "")

    return dict(zip(models, paths))


# @st.cache
def fetch_iteraction_results_path():
    paths = []
    models = []
    for root, dirs, files in os.walk(PATH_EVAL_REINFORCEMENT, followlinks=True):
        if "/results" in root and "Interaction" in root:
            for d in dirs:
                paths.append(os.path.join(root, d))
                models.append(d)  # .replace("_"+d.split("_")[-1], "")

    return dict(zip(models, paths))


# @st.cache
def load_data_metrics():
    return json2df(fetch_results_path(), "metrics.json", "path")


# @st.cache
def load_fairness_metrics():
    df = csv2df(fetch_results_path(), "fairness_metrics.csv", "path")

    df["sub"] = df["sub"].astype(str)
    df["feature"] = df["sub_key"] + "." + df["sub"]

    return df


# @st.cache
def load_fairness_df():
    df = csv2df(fetch_results_path(), "fairness_df.csv", "path")
    return df


# @st.cache
def load_eval_params():
    return json2df(fetch_results_path(), "params.json", "path")


# @st.cache
def load_train_params():
    return json2df(fetch_training_path(), "params.json", "path")


# @st.cache(allow_output_mutation=True)
def load_iteractions_params(iteractions):
    if len(iteractions) == 0:
        return pd.DataFrame()

    dfs = []

    for model in iteractions:

        file_path = os.path.join(fetch_iteraction_results_path()[model], "params.json")
        data = []

        try:
            with open(file_path) as json_file:
                d = json.load(json_file)
                data.append(d)

            df = pd.DataFrame.from_dict(json_normalize(data), orient="columns")

        except:
            df = pd.DataFrame()

        df["iteraction"] = model
        dfs.append(df)

    return pd.concat(dfs)


# @st.cache(allow_output_mutation=True)
def load_item_most_popular(model):
    random.seed(42)
    file = os.path.join(fetch_iteraction_results_path()[model], "gt-datalog.csv")

    df = pd.read_csv(file, columns=["item_idx"])

    return df


# @st.cache(allow_output_mutation=True)
def load_data_iteractions_metrics(model, sample_size=10000):
    random.seed(42)
    file = os.path.join(fetch_iteraction_results_path()[model], "sim-datalog.csv")

    # Count the lines
    num_lines = sum(1 for l in open(file)) - 1

    # Sample size - in this case ~10%
    size = np.min([sample_size, num_lines])  # int(num_lines / 10)

    # The row indices to skip - make sure 0 is not included to keep the header!
    skip_idx = sorted(random.sample(range(1, num_lines), num_lines - size))
    idx = list(set(list(range(num_lines))) - set(skip_idx))

    df = pd.read_csv(file, skiprows=skip_idx)
    # df        = pd.read_csv(file)#.reset_index()
    # idx       = list(range(len(df)))

    df["idx"] = sorted(idx)
    df = df.sort_values("idx")
    return df


# @st.cache(allow_output_mutation=True)
def load_data_orders_metrics(model):
    return pd.read_csv(
        os.path.join(fetch_results_path()[model], "orders_with_metrics.csv")
    )


# @st.cache(allow_output_mutation=True)
def load_history_train(model):
    return pd.read_csv(
        os.path.join(fetch_training_path()[model], "history.csv")
    ).set_index("epoch")


# @st.cache(allow_output_mutation=True)
def load_all_iteraction_metrics(iteractions, sample_size):
    if len(iteractions) == 0:
        return pd.DataFrame()

    metrics = []

    for iteraction in iteractions:
        # try:
        metric = load_data_iteractions_metrics(iteraction, sample_size)
        # except:
        #  continue

        metric["iteraction"] = iteraction
        metrics.append(metric)

    return pd.concat(metrics)


def display_compare_results():
    st.title("[RecSys Metrics]")

    st.sidebar.markdown("## Filter Options")
    input_models_eval = st.sidebar.multiselect(
        "Results", sorted(fetch_results_path().keys())
    )

    if len(fetch_results_path().keys()) > 0:
        data_metrics = load_data_metrics()
        data_params = load_eval_params()

        input_metrics = st.sidebar.multiselect(
            "Metrics",
            sorted(RECSYS_METRICS),
            default=["ndcg_at_5", "mean_average_precision"],
        )
        input_params = st.sidebar.multiselect("Parameters", sorted(data_params.columns))

        confidence_metrics = data_metrics[
            [c for c in data_metrics.columns if "_C" in c]
        ]
        for c in input_metrics:
            c_column = c + "_C"
            confidence_metrics[c_column] = (
                data_metrics[c_column] if c_column in data_metrics else None
            )
        confidence_metrics = confidence_metrics[[c + "_C" for c in input_metrics]]

        st.sidebar.markdown("## Graph Options")

        input_graph = st.sidebar.radio("Graph", list(GRAPH_METRIC.keys()))
        input_df_trans = st.sidebar.checkbox("Transpose Data?")
        input_sorted = st.sidebar.selectbox(
            "Sort", [""] + sorted(data_metrics.columns), index=0
        )

        df_metrics = filter_df(
            data_metrics, input_models_eval, input_metrics, input_sorted
        )
        df_eval_params = filter_df(
            data_params, input_models_eval, input_params
        ).transpose()

        try:
            df_train_params = filter_df(
                load_train_params(), cut_name(input_models_eval)
            ).transpose()
        except:
            df_train_params = df_hist = None

        GRAPH_METRIC[input_graph](
            df_metrics.transpose() if input_df_trans else df_metrics,
            confidence=confidence_metrics,
            title="Comparison of Recsys Metrics",
        )

        st.markdown("## Metrics")
        st.dataframe(df_metrics)
        st.dataframe(confidence_metrics)
        st.dataframe(df_metrics)
        # df_metrics.to_csv("df_metric.csv")
        # df_metrics.to_csv("confidence_metrics.csv")

        if df_train_params is not None:
            st.markdown("## Params (Train)")
            st.dataframe(df_train_params)

        st.markdown("## Params (Eval)")
        st.dataframe(df_eval_params)


def display_iteraction_result():
    st.sidebar.markdown("## Filter Options")

    st.title("[Iteraction Results]")
    # st.write(input_iteraction)

    # df_metrics       = filter_df(load_data_metrics(), input_models_eval, input_metrics, input_sorted)
    sample_size = st.sidebar.slider(
        "Sample", min_value=1000, max_value=10000, value=5000, step=1000
    )

    input_iteraction = st.sidebar.multiselect(
        "Results", sorted(fetch_iteraction_results_path().keys())
    )
    metrics = load_all_iteraction_metrics(input_iteraction, sample_size)
    params = load_iteractions_params(input_iteraction)
    input_metrics = st.sidebar.selectbox(
        "Metrics",
        [
            "Cumulative Reward",
            "Cumulative Mean Reward",
            "Cumulative Window Mean Reward",
        ],
        index=0,
    )

    st.sidebar.markdown("## Graph Options")

    if len(input_iteraction) > 0 and input_metrics:
        # Add a slider to the sidebar:
        add_slider = None
        if input_metrics == "Cumulative Window Mean Reward":
            add_slider = st.sidebar.slider(
                "Window", min_value=1, max_value=1000, value=500, step=1
            )

        df = (
            metrics.merge(params, on=["iteraction"], how="left")
            .merge(
                metrics.groupby("iteraction")
                .agg({"reward": "mean"})
                .rename(columns={"reward": "sum_reward"})
                .reset_index(),
                on=["iteraction"],
                how="left",
            )
            .reset_index()
            .sort_values(["sum_reward", "idx"], ascending=[False, True])
        )

        input_legend = st.sidebar.multiselect(
            "Legend", list(params.columns), default=["iteraction"]
        )

        plot_line_iteraction(
            df,
            "reward",
            title=input_metrics,
            legend=input_legend,
            yrange=None,
            line_dict=get_colors(input_iteraction),
            window=add_slider,
            cum=(input_metrics == "Cumulative Reward"),
            mean=(input_metrics == "Cumulative Mean Reward"),
            roll=(input_metrics == "Cumulative Window Mean Reward"),
        )

        st.dataframe(df.head())

        st.markdown("## Models")
        input_explorate = {}
        for group, rows in df.groupby("iteraction", sort=False):
            st.markdown("### " + group)
            input_explorate[group] = st.checkbox(
                "Show Explorate Viz", key="input_explorate_" + group
            )

            if input_explorate[group]:
                plot_exploration_arm(df[df.iteraction == group], title=group)

            st.markdown("### Params")
            st.dataframe(params[params.iteraction == group].transpose())

    # st.markdown('## Params')
    # st.dataframe(load_iteractions_params(input_iteraction).transpose())


def display_fairness_metrics():
    st.title("[Fairness Results]")

    st.sidebar.markdown("## Filter Options")

    input_models_eval = st.sidebar.selectbox(
        "Results", [""] + sorted(fetch_results_path().keys()), index=0
    )
    st.write(input_models_eval)
    if input_models_eval and len(fetch_results_path().keys()) > 0:
        df_all_metrics = load_fairness_metrics().loc[input_models_eval]
        df_instances = load_fairness_df().loc[input_models_eval]
        input_features = st.sidebar.selectbox(
            "Features", sorted(df_all_metrics["sub_key"].unique())
        )

        #####################################################################
        st.sidebar.markdown("### Disparate Mistreatment")

        input_metrics = st.sidebar.selectbox("Metrics", sorted(MISTREATMENT_METRICS))
        df_all_metric_filter = df_all_metrics[
            df_all_metrics.sub_key.isin([input_features])
        ]

        st.markdown("### Disparate Mistreatment")

        columns = list(
            np.unique(
                ["sub_key", "sub", "feature", "total_class", "total_individuals"]
                + [input_metrics]
            )
        )
        if input_metrics + "_C" in df_all_metrics.columns:
            columns.append(input_metrics + "_C")

        df_metrics = filter_df(df_all_metrics, input_models_eval, columns, "sub")

        df_metrics = df_metrics[df_metrics.sub_key.isin([input_features])]

        df_metrics = df_metrics.sort_values("feature").set_index("feature")

        plot_fairness_mistreatment(
            df_metrics,
            input_metrics,
            title="Disparate Mistreatment - Feature: "
            + input_features
            + " | "
            + input_metrics,
        )

        df_total = df_metrics[["total_class", "total_individuals"]]
        df_total_sum = df_total.sum(numeric_only=True)
        df_percent = df_total / df_total_sum
        df_total = df_total.apply(
            lambda row: [
                "{} ({:.2f} %)".format(i, p * 100)
                for i, p in zip(row, df_percent[row.name])
            ]
        )
        df_total.loc["total"] = df_total_sum

        # df_percent = df_total/x

        st.dataframe(df_total)

        #####################################################################
        st.sidebar.markdown("### Disparate Treatment and Impact")

        df_mean_action = (
            df_instances.groupby(["action", input_features])
            .agg({"rewards": "count", "rhat_scores": "mean"})
            .reset_index()
        )
        input_items = st.sidebar.multiselect(
            "Items", sorted(df_mean_action["action"].unique())
        )
        input_items_top = st.sidebar.checkbox("Only Top 5 Items")

        st.markdown("### Disparate Treatment")

        plot_fairness_treatment(
            df_instances,
            input_features,
            input_items,
            top=input_items_top,
            title="Disparate Treatment - Feature: " + input_features,
        )

        #####################################################################
        st.sidebar.markdown("### Disparate Impact")
        # input_features_1   = st.sidebar.selectbox("Features (Treatment)", sorted(df_all_metrics['sub_key'].unique()))
        df_mean_action = (
            df_instances.groupby(["action", input_features])
            .agg({"rewards": "count", "rhat_scores": "mean"})
            .reset_index()
        )

        st.markdown("### Disparate Impact")
        # st.dataframe(df_mean_action)
        plot_fairness_impact(
            df_instances,
            input_features,
            input_items,
            top=input_items_top,
            title="Disparate Impact - Feature: " + input_features,
        )

        #######################################################################

        # st.markdown('### Metrics')
        # st.dataframe(df_all_metric_filter)

        # st.markdown('### Individuos')
        # st.dataframe(df_instances.groupby(input_features).count())


def main():
    """Main function of the App"""
    st.sidebar.title("MARS - DataViz Evaluation ")
    st.sidebar.markdown(
        """
    MARS Evaluation Analysis
    """
    )

    st.sidebar.markdown("## Navigation")

    input_page = st.sidebar.radio(
        "Choose a page",
        ["[Iteraction Results]", "[RecSys Metrics]", "[Fairness Metrics]"],
    )  # "[Model Result]",

    if input_page == "[Iteraction Results]":
        display_iteraction_result()
    elif input_page == "[RecSys Metrics]":
        display_compare_results()
    else:
        display_fairness_metrics()

    # input_page        = st.sidebar.radio("Choose a page", ["[Iteraction Results]"])

    # display_iteraction_result()

    st.sidebar.title("About")
    st.sidebar.info(
        """

        """
    )


if __name__ == "__main__":
    main()
