import streamlit as st
import pandas as pd
import numpy as np
import os
from mars_gym.tools.eval_viz.util import *

PATH_EVALUATION = "output/evaluation/"


@st.cache
def list_result_paths():
    paths = []
    models = []
    for root, dirs, files in os.walk(PATH_EVALUATION):
        if "/results" in root and "Evaluate" in root:
            for d in dirs:
                paths.append(os.path.join(root, d))
                models.append(d)

    return dict(zip(models, paths))


@st.cache
def load_data_metrics():
    return json2df(list_result_paths(), "metrics.json")


@st.cache
def load_data_params():
    return json2df(list_result_paths(), "params.json")


def visualize_data(df, x_axis, y_axis):
    graph = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=x_axis,
            y=y_axis,
            color="Origin",
            tooltip=["Name", "Origin", "Horsepower", "Miles_per_Gallon"],
        )
        .interactive()
    )

    st.write(graph)


st.sidebar.title("MARS - DataViz Evaluation ")
st.sidebar.markdown(
    """
...
"""
)

input_page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration"])

input_model_eval = st.sidebar.selectbox(
    "Choose your age: ", sorted(list_result_paths().keys())
)
input_models_eval = st.sidebar.multiselect(
    "Who are your favorite artists?", sorted(list_result_paths().keys())
)
input_metrics = st.sidebar.multiselect("Métrics", sorted(load_data_metrics().columns))
input_params = st.sidebar.multiselect("Params", sorted(load_data_params().columns))

st.line_chart(np.random.randn(20, 2))

st.line_chart(load_data_metrics().loc[input_models_eval][input_metrics])

st.bar_chart(load_data_metrics().loc[input_models_eval][input_metrics])

st.markdown("## Métrics")
st.dataframe(load_data_metrics().loc[input_models_eval][input_metrics], width=1800)


st.markdown("## Params")
st.dataframe(load_data_params().loc[input_models_eval][input_params], width=1800)
