from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
import numpy as np

sns.set()
plt.style.use("default")


def plot_history(history_df: pd.DataFrame) -> Figure:
    metrics = [
        column
        for column in history_df.columns
        if column != "epoch" and "val_" not in column and "running_" not in column
    ]

    fig = plt.figure(figsize=(8 * len(metrics), 5))

    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics), i + 1)
        if metric == "loss":
            ax.set_yscale("log")
        ax.plot(history_df[metric], label="train")
        if f"val_{metric}" in history_df:
            ax.plot(history_df[f"val_{metric}"], label="validation")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.legend()

    fig.tight_layout()

    return fig


def plot_scores(scores: np.array) -> Figure:
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    ax.hist(scores)

    fig.tight_layout()

    return fig
