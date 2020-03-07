import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import os

TEMPLATE = 'plotly_white' #simple_white

def plot_bar(df, title=""):
  data = []
  for i, row in df.iterrows():
    data.append(go.Bar(name=row.name, x=row.keys(), y=row.values))
  
  fig = go.Figure(data=data)
  # Change the bar mode
  fig.update_layout( template=TEMPLATE, legend_orientation="h", legend=dict(x=-.0, y=1.5), title=title)
  st.plotly_chart(fig)

def plot_line(df, title="", yrange=[0, 1], cum=False):
  data = []
  ymax = yrange[1] if yrange else 1
  
  for i, row in df.iterrows():
    values = np.cumsum(row.values) if cum else row.values
    ymax   = np.max([np.max(values), ymax])
    data.append(go.Scatter(name=row.name, x=row.keys(), y=values))
  
  fig = go.Figure(data=data)
  # Change the bar mode
  fig.update_layout(template=TEMPLATE, legend_orientation="h", legend=dict(x=-.0, y=1.5), title=title)
  if yrange is not None:
    fig.update_yaxes(range=[yrange[0], ymax+(ymax*0.1)])

  st.plotly_chart(fig)

def plot_line_iteraction(df, metric, legend=['iteraction'], title="", yrange=[0, 1], 
                        cum=False, mean=False):
  data = []
  ymax = yrange[1] if yrange else 1
  
  for group, rows in df.groupby("iteraction", sort=False):
    x    = [i+1 for i in range(len(rows))]

    values = rows[metric].values
    if cum:
      values = np.cumsum(values)

    if mean:
      values = np.cumsum(values)/x

    ymax   = np.max([np.max(values), ymax])

    try:
      name   = " - ".join(list(rows.iloc[0][legend].astype(str)))
    except:
      name   = group

    data.append(go.Scatter(name=name, x=x, y=values))
    
  fig = go.Figure(data=data)
  # Change the bar mode
  fig.update_layout(template=TEMPLATE, legend_orientation="v", title=title)
  if yrange is not None:
    fig.update_yaxes(range=[yrange[0], ymax+(ymax*0.1)])

  st.plotly_chart(fig)

def plot_exploration_arm(df, title=""):
    rounds = len(df)
    arms   = np.unique(df['item'].values)
    arms_rewards = df['item'].values

    count_per_arms = {}
    
    for a in arms:
        count_per_arms[a] = np.zeros(rounds)

    for r in range(rounds):
        count_per_arms[arms_rewards[r]][r] = 1
    
    fig = go.Figure()
    x   = (np.array(range(rounds)) + 1)

    for arm, values in count_per_arms.items():    
        fig.add_trace(go.Scatter(
            name="Arm "+str(arm),
            x=x, y=np.cumsum(values),
            hoverinfo='x+y',
            mode='lines',
            line=dict(width=0.5),
            stackgroup='one',
            groupnorm='percent' # define stack group
        ))

    fig.update_layout(template=TEMPLATE, 
                  xaxis_title_text='Time Step', 
                  yaxis_title_text="Cummulative Exploration Arm",
                  title="Cumulative Exploration Arms over time - "+title,
                  yaxis_range=(0, 100))

    st.plotly_chart(fig)

    return fig  

def plot_radar(df, title=""):
  data = []
  for i, row in df.iterrows():
    data.append(go.Scatterpolar(
      r=row.values,
      theta=row.keys(),
      fill='toself',
      name=row.name
    ))
  
  fig = go.Figure(data=data)
  # Change the bar mode
  fig.update_layout( template=TEMPLATE, legend_orientation="h", legend=dict(x=-.0, y=1.5), title=title)

  st.plotly_chart(fig)

def plot_hist(df, title=""):
  data = []

  fig = go.Figure()

  for c in df.columns:
    fig.add_trace(go.Histogram(x=df[c], name=c))

  # Add title
  fig.update_layout(template=TEMPLATE, legend_orientation="h",  barmode='stack', title=title)

  st.plotly_chart(fig)

def plot_box(df, title=""):
  data = []

  fig = go.Figure()

  for c in df.columns:
    fig.add_trace(go.Box(y=df[c], name=c))

  # Add title
  fig.update_layout(template=TEMPLATE, legend_orientation="h",  title=title)

  st.plotly_chart(fig)

def plot_history(df, title=""):
  data = []
  for c in df.columns:
    data.append(go.Scatter(name=c,  y=df[c]))
  
  fig = go.Figure(data=data)
  # Change the bar mode
  fig.update_layout(template=TEMPLATE, legend_orientation="h",  title=title)

  st.plotly_chart(fig)

def plot_metrics(df, title=""):
  data   = []

  for i, row in df.iterrows():
    data.append(go.Bar(name=row.name, x=row.keys(), y=row.values, 
                    marker_color=[_color_by_metric(m) for m in row.keys()]))
  fig = go.Figure(data=data)
  # Change the bar mode
  fig.update_layout(template=TEMPLATE, legend_orientation="h", legend=dict(x=-.0, y=1.5), title=title)

  st.plotly_chart(fig)

def _color_by_metric(metric):
  if "ndcg" in metric:
    return '#DD8452'
  elif "coverage" in metric:
    return '#55A868'
  elif "personalization" in metric:
    return '#C44E51'
  elif "count" in metric:    
    return '#8C8C8C'
  else:    
    return '#CCB974'