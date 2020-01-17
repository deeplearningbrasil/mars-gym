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

def plot_line(df, title="", yrange=None):
  data = []
  for i, row in df.iterrows():
    data.append(go.Scatter(name=row.name, x=row.keys(), y=row.values))
  
  fig = go.Figure(data=data)
  # Change the bar mode
  fig.update_layout(template=TEMPLATE, legend_orientation="h", legend=dict(x=-.0, y=1.5), title=title)
  if yrange is not None:
    fig.update_yaxes(range=yrange)

  st.plotly_chart(fig)

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