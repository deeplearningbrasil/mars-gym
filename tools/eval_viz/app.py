import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import os
from plot import *
from util import *

PATH_EVALUATION = '../../output/evaluation/'

PAGES = {
  "Home": "pages.home",
  "Model": "pages.model"
}

GRAPH_METRIC = {
  "Line": plot_line,
  "Bar": plot_bar
}

GRAPH_METRIC_MODEL = {
  "Hist": plot_hist,
  "Box": plot_box
}

@st.cache
def fetch_results_path():
    paths  = []
    models = []
    for root, dirs, files in os.walk(PATH_EVALUATION):
      if '/results' in root and 'Evaluate' in root:
        for d in dirs:
          paths.append(os.path.join(root, d))
          models.append(d)
              
    return dict(zip(models, paths))

@st.cache
def load_data_metrics():
  return json2df(fetch_results_path(), 'metrics.json')

@st.cache
def load_data_params():
  return json2df(fetch_results_path(), 'params.json')

def load_data_orders_metrics(model):
  return pd.read_csv(os.path.join(fetch_results_path()[model],'orders_with_metrics.csv'))


def display_compare_results():
  st.title("Compare Results")

  st.sidebar.markdown("## Filter Options")
  input_models_eval = st.sidebar.multiselect("Results", sorted(fetch_results_path().keys()))

  input_metrics     = st.sidebar.multiselect("Metrics", sorted(load_data_metrics().columns))
  input_params      = st.sidebar.multiselect("Parameters", sorted(load_data_params().columns))

  st.sidebar.markdown("## Graph Options")

  input_graph       = st.sidebar.radio("Graph", list(GRAPH_METRIC.keys()))
  input_df_trans    = st.sidebar.checkbox("Transpose Data?")
  input_sorted      = st.sidebar.selectbox("Sort", sorted(load_data_metrics().columns), index=11)

  df_metrics = filter_df(load_data_metrics(), input_models_eval, input_metrics, input_sorted)
  df_params  = filter_df(load_data_params(), input_models_eval, input_params)

  st.markdown('## Metrics')
  st.dataframe(df_metrics)

  st.markdown('## DataViz')
  GRAPH_METRIC[input_graph](df_metrics.transpose() if input_df_trans else df_metrics)

  st.markdown('## Params')
  st.dataframe(df_params)

def display_one_result():
  st.sidebar.markdown("## Filter Options")  
  input_model_eval  = st.sidebar.selectbox("Result", sorted(fetch_results_path().keys()))
  st.title(input_model_eval)

  df_metrics = filter_df(load_data_metrics(), [input_model_eval]).transpose()
  df_params  = filter_df(load_data_params(), [input_model_eval]).transpose()
  df_orders  = load_data_orders_metrics(input_model_eval)

  st.sidebar.markdown("## Graph Options")
  input_column = st.sidebar.multiselect("Column", sorted(df_orders.columns))
  input_graph  = st.sidebar.radio("Graph", list(GRAPH_METRIC_MODEL.keys()))

  st.markdown('## Orders with Metrics')
  st.dataframe(df_orders.head())

  st.markdown('## DataViz')
  if len(input_column) > 0:
    GRAPH_METRIC_MODEL[input_graph](df_orders[input_column])

  st.markdown('## Metrics')
  st.dataframe(df_metrics)

  st.markdown('## Params')
  st.dataframe(df_params)

def main():
    """Main function of the App"""
    st.sidebar.title("DeepFood - DataViz Evaluation ")
    st.sidebar.markdown(
        """
    DeepFood Evaluation Analysis
    """
    )

    st.sidebar.markdown("## Navigation")
    
    input_page        = st.sidebar.radio("Choose a page", ["[Compare Results]", "[Only One Result]"])

    if input_page == "[Compare Results]":
      display_compare_results()
    else:
      display_one_result()

    st.sidebar.title("About")
    st.sidebar.info(
        """
        DeepFood Evaluation Analysis - @UFG
        """
    )

if __name__ == "__main__":
    main()

