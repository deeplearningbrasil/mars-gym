import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import os
import seaborn as sns
import plotly.express as px
from mars_gym.tools.eval_viz.util import mean_confidence_interval

TEMPLATE = "plotly_white"  # simple_white
# https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
# https://plot.ly/python/v3/ipython-notebooks/color-scales/#diverging
# sns.color_palette("colorblind", n_colors=15).as_hex()
def get_colors(models, color=px.colors.qualitative.Plotly):
    line_dict = {}
    # dash = ['dash', 'dot',  'dashdot']
    for i, model in enumerate(models):
        line_dict[model] = dict(width=2, color=color[int(i % 10)])
    return line_dict


def plot_bar(df, confidence=None, title=""):
    data = []
    for i, row in df.iterrows():
        data.append(
            go.Bar(
                name=row.name,
                x=row.keys(),
                y=row.values,
                error_y=dict(type="data", array=[] if confidence is None else confidence.loc[row.name].values),
            )
        )

    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        xaxis_title="Metric",
        yaxis_title="Value",
        legend=dict(y=-0.2),
        title=title,
    )
    st.plotly_chart(fig)
    return fig


def plot_line(df, confidence=None, title="", yrange=[0, 1], cum=False):
    data = []
    ymax = yrange[1] if yrange else 1

    for i, row in df.iterrows():
        values = np.cumsum(row.values) if cum else row.values
        ymax = np.max([np.max(values), ymax])
        data.append(go.Scatter(name=row.name, x=row.keys(), y=values))

    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        legend=dict(x=-0.0, y=1.5),
        title=title,
    )
    if yrange is not None:
        fig.update_yaxes(range=[yrange[0], ymax + (ymax * 0.1)])

    st.plotly_chart(fig)


def plot_radar(df, confidence=None, title=""):
    data = []
    for i, row in df.iterrows():
        data.append(
            go.Scatterpolar(
                r=row.values, theta=row.keys(), fill="toself", name=row.name
            )
        )

    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        legend=dict(x=-0.0, y=1.5),
        title=title,
    )

    st.plotly_chart(fig)


def plot_line_iteraction(
    df,
    metric,
    legend=["iteraction"],
    window=20,
    title="",
    yrange=[0, 1],
    cum=False,
    mean=False,
    roll=False,
    line_dict={},
):
    data = []
    ymax = yrange[1] if yrange else 1

    for group, rows in df.groupby("iteraction", sort=False):
        _x = [i + 1 for i in range(len(rows))]
        x = sorted(rows["idx"].values)

        values = rows[metric].values
        if cum:
            values = np.cumsum(values)

        if mean:
            values = np.cumsum(values) / _x

        if roll:
            values = rows[metric].rolling(window=window, min_periods=1).mean()

        x = x[10:-1]
        values = values[10:-1]

        ymax = np.max([np.max(values), ymax])

        try:
            first_len = rows.iloc[0][legend[0]]  # .astype(str)
            v = list(rows.iloc[0][legend[1:]].astype(str))
            f = (
                "(" + ", ".join(["{}".format(v) for k, v in zip(legend[1:], v)]) + ")"
                if ", ".join(["{}".format(v) for k, v in zip(legend[1:], v)]) != ""
                else ""
            )
            name = "<b>" + first_len + "</b>" + f
        except:
            name = group

        data.append(
            go.Scatter(
                name=name,
                x=x,
                y=values,
                line=(line_dict[group] if group in line_dict else {}),
            )
        )

    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        legend=dict(y=-0.2),
        title="Comparison of Contextual Bandit Policies",
        xaxis_title="Interactions",
        yaxis_title=title,
        showlegend=True,
    )
    if yrange is not None:
        fig.update_yaxes(range=[yrange[0], ymax + (ymax * 0.1)])

    st.plotly_chart(fig)

    return fig


def plot_exploration_arm(df, title="", window=20, roll=False, all_items=[]):
    rounds = len(df)
    arms = np.unique(df["item"].values)
    arms_rewards = df["item"].values

    arms_idx = {}
    if len(all_items) == 0:
        all_items = arms
    for i, a in enumerate(all_items):
        arms_idx[a] = i

    count_per_arms = {}

    for a in arms:
        count_per_arms[a] = np.zeros(rounds)

    for r in range(rounds):
        count_per_arms[arms_rewards[r]][r] = 1

    fig = go.Figure()
    x = sorted(df["idx"].values)

    for arm, values in count_per_arms.items():

        if roll:
            y = pd.Series(values).rolling(window=window, min_periods=1).mean()
        else:
            y = np.cumsum(values)

        fig.add_trace(
            go.Scatter(
                name="Arm " + str(arm),
                x=x,
                y=y,
                hoverinfo="x+y",
                mode="lines",
                line=dict(
                    width=0.5,
                    color=px.colors.qualitative.Plotly[int(arms_idx[arm] % 10)],
                ),
                stackgroup="one",
                groupnorm="percent",  # define stack group
            )
        )

    fig.update_layout(
        template=TEMPLATE,
        xaxis_title_text="Iteractions",
        yaxis_title_text="Cummulative Exploration Arm",
        title="Cumulative Exploration Arms over time",  # +title
        yaxis_range=(0, 100),
        showlegend=True,
    )

    st.plotly_chart(fig)

    return fig


def plot_hist(df, title=""):
    data = []

    fig = go.Figure()

    for c in df.columns:
        fig.add_trace(go.Histogram(x=df[c], name=c))

    # Add title
    fig.update_layout(
        template=TEMPLATE, legend_orientation="h", barmode="stack", title=title
    )

    st.plotly_chart(fig)


def plot_box(df, title=""):
    data = []

    fig = go.Figure()

    for c in df.columns:
        fig.add_trace(go.Box(y=df[c], name=c))

    # Add title
    fig.update_layout(template=TEMPLATE, legend_orientation="h", title=title)

    st.plotly_chart(fig)


def plot_history(df, title=""):
    data = []
    for c in df.columns:
        data.append(go.Scatter(name=c, y=df[c]))

    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(template=TEMPLATE, legend_orientation="h", title=title)

    st.plotly_chart(fig)


def plot_metrics(df, title=""):
    data = []

    for i, row in df.iterrows():
        data.append(
            go.Bar(
                name=row.name,
                x=row.keys(),
                y=row.values,
                marker_color=[_color_by_metric(m) for m in row.keys()],
            )
        )
    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        legend=dict(x=-0.0, y=1.5),
        title=title,
    )

    st.plotly_chart(fig)


def plot_fairness_mistreatment(df, metric, title=""):
    data = []

    data.append(
        go.Bar(
            y=df.index,
            x=df[metric],
            orientation="h",
            error_x=dict(type="data", array=df[metric + "_C"])
            if metric + "_C" in df.columns
            else {},
            marker={"color": list(range(len(df.index))), "colorscale": "Tealgrn"},
        )
    )  # Plotly3

    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        xaxis_title=metric,
        xaxis_range=(0, np.max([1, df[metric].max()])),
        legend=dict(y=-0.2),
        title=title,
    )
    # fig.update_traces(texttemplate='%{x:.2f}', textposition='outside')

    fig.update_layout(
        shapes=[
            dict(
                type="line",
                line=dict(width=1, dash="dot",),
                yref="paper",
                y0=0,
                y1=1,
                xref="x",
                x0=df[metric].mean(),
                x1=df[metric].mean(),
            )
        ]
    )

    st.plotly_chart(fig)

    return fig


def plot_fairness_treatment(df, metric, items, min_count=10, top=False, title=""):
    data = []
    i = 0
    score = "rhat_scores"

    if top:
        # Diff min max score
        df_diff = df.groupby("action").agg(total=(score, "count"))
        # df_diff['diff'] = df_diff['max_score']-df_diff['min_score']
        items = df_diff.sort_values("total", ascending=False)#.index[:5]

    df = (
        df.groupby(["action", metric])
        .agg(
            rewards=("rewards", "count"),
            metric=(score, "mean"),
            confidence=(score, confidence),
        )
        .reset_index()
    )  # .sort_values("rhat_scores")

    df = df[df.rewards > min_count]  # filter min interactions

    #------------------
    df_group = df[df['rewards'] > min_count].groupby(
        'action').agg({metric: 'count'}).reset_index()
    df_all = df_group[df_group[metric] >= len(df[metric].unique())]['action'].values

    df = df[df.action.isin(df_all)].iloc[0:int(3*5)]

    for group, rows in df.groupby(metric):
        data.append(
            go.Bar(
                name=metric + "." + str(group),
                x=["Item" + ":" + str(a) for a in rows["action"]],
                y=rows["metric"],
                error_y=dict(type="data", array=rows["confidence"]),
            )
        )  # px.colors.sequential.Purp [i for i in range(len(rows))]

        i += 1
    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        yaxis_title=score,
        legend=dict(y=-0.2),
        title=title,
    )
    # fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    # fig.update_layout(coloraxis = {'colorscale':'Purp'})

    fig.update_layout(
        shapes=[
            dict(
                type="line",
                line=dict(width=1, dash="dot",),
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=df["metric"].mean(),
                y1=df["metric"].mean(),
            )
        ]
    )

    st.plotly_chart(fig)
    st.dataframe(df)

    return fig


def plot_fairness_impact(df, metric, items, min_count=10, top=False, title=""):

    score = "action_percent"

    if top:
        # Diff min max score
        df_diff = df.groupby("action").agg(total=(metric, "count"))
        # df_diff['diff'] = df_diff['max_score']-df_diff['min_score']
        items = df_diff.sort_values("total", ascending=False).index[:5]

    df = (
        df.groupby(["action", metric])
        .agg(
            rewards=("rewards", "count"),
            rhat_scores=("rhat_scores", "mean"),
            confidence=("rhat_scores", confidence),
        )
        .reset_index()
    )  # .sort_values("rhat_scores")

    df = df.merge(
        df.groupby(metric).agg(total_rewards=("rewards", "sum")).reset_index(),
        on=metric,
    )
    df[score] = df["rewards"] / df["total_rewards"]  # .sum()

    df = df[df.action.isin(items)]
    df = df[df.rewards > min_count]  # filter min interactions

    data = []
    i = 0

    for group, rows in df.groupby("action"):
        data.append(
            go.Bar(
                name="Item:" + str(group),
                x=[metric + "." + str(a) for a in rows[metric]],
                y=rows[score],
                text=rows[score],
                textposition="auto",
                marker={"color": px.colors.qualitative.Pastel[int(i % 10)]},
            )
        )
        i += 1
    fig = go.Figure(data=data)
    # Change the bar mode
    # fig.update_traces(overwrite=True, marker={'color': px.colors.qualitative.Plotly[int(arms_idx[arm]%10)]})

    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        yaxis_title=score,
        legend=dict(y=-0.2),
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    fig.update_layout(
        shapes=[
            dict(
                type="line",
                line=dict(width=1, dash="dot",),
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=df[score].mean(),
                y1=df[score].mean(),
            )
        ]
    )

    st.plotly_chart(fig)
    st.dataframe(df)

    return fig


def confidence(x):
    return mean_confidence_interval(x)[1]


def _color_by_metric(metric):
    if "ndcg" in metric:
        return "#DD8452"
    elif "coverage" in metric:
        return "#55A868"
    elif "personalization" in metric:
        return "#C44E51"
    elif "count" in metric:
        return "#8C8C8C"
    else:
        return "#CCB974"
