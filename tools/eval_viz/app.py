import streamlit as st
import pandas as pd
import dask.dataframe as dd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import os
from plot import *
from util import *
import random
pd.set_option('display.max_colwidth', -1)

PATH_EVALUATION = 'output/evaluation/'
PATH_EVAL_REINFORCEMENT = 'output/interaction/'
PATH_TRAIN      = 'output/models/'


PAGES = {
  "Home": "pages.home",
  "Model": "pages.model"
}

GRAPH_METRIC = {
  "Bar": plot_bar,
  "Line": plot_line,
  "Radar": plot_radar
}

GRAPH_METRIC_MODEL = {
  "Hist": plot_hist,
  "Box": plot_box
}

#@st.cache
def fetch_training_path():
    paths  = []
    models = []
    for root, dirs, files in os.walk(PATH_TRAIN):
      if '/results' in root:
        for d in dirs:
          paths.append(os.path.join(root, d))
          models.append(d)
              
    return dict(zip(models, paths))

#@st.cache
def fetch_results_path():
    paths  = []
    models = []
    for root, dirs, files in os.walk(PATH_EVALUATION):
      if '/results' in root and 'Evaluate' in root:
        for d in dirs:
          paths.append(os.path.join(root, d))
          models.append(d) #.replace("_"+d.split("_")[-1], "")
              
    return dict(zip(models, paths))

#@st.cache
def fetch_iteraction_results_path():
    paths  = []
    models = []
    for root, dirs, files in os.walk(PATH_EVAL_REINFORCEMENT):
      if '/results' in root and 'Interaction' in root:
        for d in dirs:
          paths.append(os.path.join(root, d))
          models.append(d) #.replace("_"+d.split("_")[-1], "")
              
    return dict(zip(models, paths))  

#@st.cache
def load_data_metrics():
  return json2df(fetch_results_path(), 'metrics.json', 'path')

#@st.cache
def load_eval_params():
  return json2df(fetch_results_path(), 'params.json', 'path')

#@st.cache
def load_train_params():
  return json2df(fetch_training_path(), 'params.json', 'path')

#@st.cache(allow_output_mutation=True)
def load_iteractions_params(iteractions):
  if len(iteractions) == 0:
    return pd.DataFrame()

  dfs = []

  for model in iteractions:

    file_path = os.path.join(fetch_iteraction_results_path()[model], 'params.json')
    data      = []

    try:
      with open(file_path) as json_file:
        d = json.load(json_file)
        data.append(d)

      df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
      
    except:
      df = pd.DataFrame()

    df['iteraction'] = model
    dfs.append(df)
  
  return pd.concat(dfs)

#@st.cache(allow_output_mutation=True)
def load_data_iteractions_metrics(model, sample_size = 10000):
  random.seed(42)
  file      = os.path.join(fetch_iteraction_results_path()[model],'sim-datalog.csv')

  # Count the lines
  num_lines = sum(1 for l in open(file)) - 1

  # Sample size - in this case ~10%
  size = np.min([sample_size, num_lines])#int(num_lines / 10)

  # The row indices to skip - make sure 0 is not included to keep the header!
  skip_idx  = sorted(random.sample(range(1, num_lines), num_lines - size))
  idx       = list(set(list(range(num_lines))) - set(skip_idx))
  
  df        = pd.read_csv(file, skiprows=skip_idx)
  #df        = pd.read_csv(file)#.reset_index()
  #idx       = list(range(len(df)))

  df['idx'] = sorted(idx)
  df        = df.sort_values("idx")
  return df

#@st.cache(allow_output_mutation=True)
def load_data_orders_metrics(model):
  return pd.read_csv(os.path.join(fetch_results_path()[model],'orders_with_metrics.csv'))

#@st.cache(allow_output_mutation=True)
def load_history_train(model):
  return pd.read_csv(os.path.join(fetch_training_path()[model],'history.csv')).set_index('epoch')

#@st.cache(allow_output_mutation=True)
def load_all_iteraction_metrics(iteractions, sample_size):
  if len(iteractions) == 0:
    return pd.DataFrame()

  metrics = []

  for iteraction in iteractions:
    #try:
    metric = load_data_iteractions_metrics(iteraction, sample_size)
    #except:
    #  continue

    metric['iteraction'] = iteraction
    metrics.append(metric)

  return pd.concat(metrics)

def display_compare_results():
  st.title("[Compare Results]")

  st.sidebar.markdown("## Filter Options")
  input_models_eval = st.sidebar.multiselect("Results", sorted(fetch_results_path().keys()))

  if len(fetch_results_path().keys()) > 0:

    input_metrics     = st.sidebar.multiselect("Metrics", sorted(load_data_metrics().columns), default=['ndcg_at_5', 'mean_average_precision'])
    input_params      = st.sidebar.multiselect("Parameters", sorted(load_eval_params().columns))

    st.sidebar.markdown("## Graph Options")

    input_graph       = st.sidebar.radio("Graph", list(GRAPH_METRIC.keys()))
    input_df_trans    = st.sidebar.checkbox("Transpose Data?")
    input_sorted      = st.sidebar.selectbox("Sort", [""] + sorted(load_data_metrics().columns), index=0)

    df_metrics      = filter_df(load_data_metrics(), input_models_eval, input_metrics, input_sorted)
    df_eval_params  = filter_df(load_eval_params(), input_models_eval, input_params).transpose()
    
    try:
      df_train_params   = filter_df(load_train_params(), cut_name(input_models_eval)).transpose()
    except:
      df_train_params = df_hist = None

    st.markdown('## Metrics')
    st.dataframe(df_metrics)
    GRAPH_METRIC[input_graph](df_metrics.transpose() if input_df_trans else df_metrics)

    if df_train_params is not None:
      st.markdown('## Params (Train)')
      st.dataframe(df_train_params)

    st.markdown('## Params (Eval)')
    st.dataframe(df_eval_params)
  

def display_one_result():
  st.sidebar.markdown("## Filter Options")  
  input_model_eval  = st.sidebar.selectbox("Result", fetch_results_path().keys(), index=0)
  st.title("[Model Result]")
  st.write(input_model_eval)

  if len(fetch_results_path().keys()) > 0:
    df_metrics        = filter_df(load_data_metrics(), [input_model_eval]).transpose()
    df_eval_params    = filter_df(load_eval_params(), [input_model_eval]).transpose()
    df_orders         = load_data_orders_metrics(input_model_eval)

    try:
      df_train_params   = filter_df(load_train_params(), cut_name([input_model_eval])).transpose()
      df_hist           = load_history_train(cut_name([input_model_eval])[0])
    except:
      df_train_params = df_hist = None

    st.sidebar.markdown("## Graph Options")
    if df_hist is not None:
      input_coluns_tl = st.sidebar.multiselect("Columns Train Log", sorted(df_hist.columns), default=['loss', 'val_loss'])

    input_column    = st.sidebar.multiselect("Column (orders_with_metrics.csv)", sorted(df_orders.columns), default=['rhat_scores'])
    input_graph     = st.sidebar.radio("Graph", list(GRAPH_METRIC_MODEL.keys()))

    if df_hist is not None:
      st.markdown('## Train Logs')
      plot_history(df_hist[input_coluns_tl], "History")

    st.markdown('## Metrics')
    plot_metrics(df_metrics.loc[~df_metrics.index.isin(['count', 'model','model_task'])].transpose(), "Metrics")
    st.dataframe(df_metrics)

    st.markdown('''
      ## Orders with Metrics
      ### orders_with_metrics.csv
    ''')
    
    st.dataframe(df_orders.head())
    if len(input_column) > 0:
      GRAPH_METRIC_MODEL[input_graph](df_orders[input_column], "Distribution Variables")

    if df_train_params is not None:
      st.markdown('## Params (Train)')
      st.dataframe(df_train_params)

    st.markdown('## Params (Eval)')
    st.dataframe(df_eval_params)


def display_iteraction_result():
  st.sidebar.markdown("## Filter Options")


  st.title("[Iteraction Results]")
  #st.write(input_iteraction)

  #df_metrics       = filter_df(load_data_metrics(), input_models_eval, input_metrics, input_sorted)
  sample_size       = st.sidebar.slider('Sample', min_value=1000, max_value=10000, value=5000, step=1000)

  input_iteraction  = st.sidebar.multiselect("Results", sorted(fetch_iteraction_results_path().keys()))
  metrics           = load_all_iteraction_metrics(input_iteraction, sample_size)
  params            = load_iteractions_params(input_iteraction)
  input_metrics     = st.sidebar.selectbox("Metrics", ['Cumulative Reward', 'Cumulative Mean Reward', 'Cumulative Window Mean Reward'], index=0)
  
  st.sidebar.markdown("## Graph Options")

  if len(input_iteraction) > 0 and input_metrics:
    # Add a slider to the sidebar:
    add_slider = None
    if input_metrics == 'Cumulative Window Mean Reward':
      add_slider = st.sidebar.slider('Window', min_value=1, max_value=1000, value=500, step=1)

    df         = metrics.merge(params, on=['iteraction'], how='left')\
                .merge(metrics.groupby("iteraction").agg({'reward': 'mean'}).rename(columns={'reward': 'sum_reward'}).reset_index(), 
                        on=['iteraction'], how='left')\
                .reset_index().sort_values(['sum_reward', 'idx'], ascending=[False, True])
        

    input_legend       = st.sidebar.multiselect("Legend", list(params.columns), default=['iteraction'])

    plot_line_iteraction(df, 'reward', 
                          title=input_metrics, 
                          legend=input_legend,
                          yrange=None, 
                          line_dict=get_colors(input_iteraction),
                          window=add_slider,
                          cum=(input_metrics == 'Cumulative Reward'), 
                          mean=(input_metrics == 'Cumulative Mean Reward'),
                          roll=(input_metrics == 'Cumulative Window Mean Reward'))

    st.dataframe(df.head())

    st.markdown('## Models')
    input_explorate = {}
    for group, rows in df.groupby("iteraction", sort=False):
      st.markdown("### "+group)
      input_explorate[group]    = st.checkbox("Show Explorate Viz", key="input_explorate_"+group)
      
      if input_explorate[group]:
        plot_exploration_arm(df[df.iteraction == group], title=group)

      st.markdown('### Params')
      st.dataframe(params[params.iteraction == group].transpose())


    
  #st.markdown('## Params')
  #st.dataframe(load_iteractions_params(input_iteraction).transpose())



def main():
    """Main function of the App"""
    st.sidebar.title("DeepFood - DataViz Evaluation ")
    st.sidebar.markdown(
        """
    DeepFood Evaluation Analysis
    """
    )

    st.sidebar.markdown("## Navigation")
    
    # input_page        = st.sidebar.radio("Choose a page", ["[Model Result]", "[Compare Results]", "[Iteraction Results]"])

    # if input_page == "[Compare Results]":
    #   display_compare_results()
    # elif input_page == "[Model Result]":
    #   display_one_result()
    # else:
    #   display_iteraction_result()

    input_page        = st.sidebar.radio("Choose a page", ["[Iteraction Results]"])

    display_iteraction_result()

    st.sidebar.title("About")
    st.sidebar.info(
        """
        DeepFood Evaluation Analysis - @UFG
        """
    )

if __name__ == "__main__":
    main()

