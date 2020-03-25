import streamlit as st

from app import fetch_iteraction_results_path, load_all_iteraction_metrics, load_iteractions_params
from plot import plot_line_iteraction, plot_exploration_arm, get_colors
import pandas as pd
import argparse
import copy
import colorlover as cl


def export(args):
  input_legend      = args.legend.split(",")

  results_path      = fetch_iteraction_results_path()
  
  if args.models == "":
    input_iteraction  = results_path.keys()
  else:
    input_iteraction  = args.models.split(",")

  st.set_option("client.displayEnabled", False)

  metrics           = load_all_iteraction_metrics(input_iteraction, args.sample_size)
  params            = load_iteractions_params(input_iteraction)

  df_metrics_reward = metrics.groupby("iteraction").agg({'reward': ['mean', 'sum']}).reset_index().sort_values([('reward', 'sum')], ascending=False)

  df  = metrics.merge(params, on=['iteraction'], how='left')\
                .merge(metrics.groupby("iteraction")\
                        .agg({'reward': 'mean'})\
                        .rename(columns={'reward': 'sum_reward'})\
                        .reset_index(), 
                  on=['iteraction'], how='left')\
                .reset_index()\
                .sort_values(['sum_reward', 'idx'], ascending=[False, True])
      
  print(df.head())
  print(params.head())
  print(df_metrics_reward)

  # GERAL
  for input_metrics in ['Cumulative Reward', 'Cumulative Mean Reward', 'Cumulative Window Mean Reward']:
    fig = plot_line_iteraction(df, 'reward', 
                          title=input_metrics, 
                          legend=input_legend,
                          line_dict=get_colors(input_iteraction),
                          yrange=[0,1], 
                          window=args.window_size,
                          cum=(input_metrics == 'Cumulative Reward'), 
                          mean=(input_metrics == 'Cumulative Mean Reward'),
                          roll=(input_metrics == 'Cumulative Window Mean Reward'))

    fig.write_image(args.output+"/all_{}.png".format(input_metrics.replace(" ", ""))) #, width=1024, height=600, scale=2

  # PEr Parameter
  for param in input_legend:

    for group, rows in df.groupby(param, sort=False):
      print("Extract... ", param, group)
      df_group = df[df[param] == group]

      for input_metrics in ['Cumulative Reward', 'Cumulative Mean Reward', 'Cumulative Window Mean Reward']:
        fig = plot_line_iteraction(df_group, 'reward', 
                              title=input_metrics, 
                              legend=input_legend,
                              line_dict=get_colors(input_iteraction),
                              yrange=[0,1], 
                              window=args.window_size,
                              cum=(input_metrics == 'Cumulative Reward'), 
                              mean=(input_metrics == 'Cumulative Mean Reward'),
                              roll=(input_metrics == 'Cumulative Window Mean Reward'))
        fig.write_image(args.output+"/{}={}-{}.png".format(param, group, input_metrics.replace(" ", "")))

  # Per Model
  for group, rows in df.groupby("iteraction", sort=False):
    print("Extract... ", group)
    df_group = df[df.iteraction == group]

    for input_metrics in ['Cumulative Reward', 'Cumulative Mean Reward', 'Cumulative Window Mean Reward']:
      fig = plot_line_iteraction(df_group, 'reward', 
                            title=input_metrics, 
                            legend=input_legend,
                            line_dict=get_colors(input_iteraction),
                            yrange=[0,1], 
                            window=args.window_size,
                            cum=(input_metrics == 'Cumulative Reward'), 
                            mean=(input_metrics == 'Cumulative Mean Reward'),
                            roll=(input_metrics == 'Cumulative Window Mean Reward'))

      fig.write_image(args.output+"/{}_{}.png".format(group, input_metrics.replace(" ", "")))


    fig = plot_exploration_arm(df_group, title=group, window=args.window_size, roll=False)
    fig.write_image(args.output+"/{}_explorer.png".format(group))
    
  metrics.to_csv(args.output+"/metrics.csv", index=False)
  params.to_csv(args.output+"/params.csv", index=False)
  df_metrics_reward.merge(params.set_index("iteraction")[input_legend], on='iteraction')\
    [["iteraction"]+input_legend+[('reward', 'mean'), ('reward', 'sum')]].to_csv(args.output+"/params_with_metrics_reward.csv", index=False)
# Params
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model Params
    parser.add_argument('--sample_size', default=5000,  type=int)
    parser.add_argument('--window_size', default=500,  type=int)
    parser.add_argument('--models', default='',  type=str)
    parser.add_argument('--output', default='tools/eval_viz/export',  type=str)
    parser.add_argument('--legend', default='bandit_policy,bandit_policy_params.alpha',  type=str)


    export(parser.parse_args())
