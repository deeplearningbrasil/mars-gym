import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import os

def plot_bar(df):
  data = []
  for i, row in df.iterrows():
    data.append(go.Bar(name=row.name, x=row.keys(), y=row.values))
  
  fig = go.Figure(data=data)
  # Change the bar mode
  fig.update_layout( template="simple_white", legend_orientation="h", legend=dict(x=-.0, y=1.5))
  st.plotly_chart(fig)

def plot_line(df):
  data = []
  for i, row in df.iterrows():
    data.append(go.Scatter(name=row.name, x=row.keys(), y=row.values))
  
  fig = go.Figure(data=data)
  # Change the bar mode
  fig.update_layout( template="simple_white", legend_orientation="h", legend=dict(x=-.0, y=1.5))

  st.plotly_chart(fig)

def plot_hist(df):
  data = []

  fig = go.Figure()

  for c in df.columns:
    fig.add_trace(go.Histogram(x=df[c], name=c))

  # Add title
  fig.update_layout()
  fig.update_layout(template="simple_white", legend_orientation="h", legend=dict(x=-.0, y=1.5), barmode='stack')

  st.plotly_chart(fig)

def plot_box(df):
  data = []

  fig = go.Figure()

  for c in df.columns:
    fig.add_trace(go.Box(y=df[c], name=c))

  # Add title
  fig.update_layout( template="simple_white", legend_orientation="h", legend=dict(x=-.0, y=1.5))

  st.plotly_chart(fig)