import pandas as pd
import json
import os
from pandas.io.json import json_normalize
import numpy as np
import scipy


def csv2df(paths, file, idx):
    data = []
    for model, path in paths.items():
        file_path = os.path.join(path, file)
        try:
            d = pd.read_csv(file_path)
            d["path"] = path.split("/")[-1]
            d["model"] = path.split("/")[-1].replace(
                "_" + path.split("/")[-1].split("_")[-1], ""
            )

            data.append(d)
        except:
            pass

    df = pd.concat(data)
    df = df.set_index(idx)

    return df


def json2df(paths, file, idx):
    data = []
    for model, path in paths.items():
        file_path = os.path.join(path, file)
        try:
            with open(file_path) as json_file:
                d = json.load(json_file)
                d["path"] = path.split("/")[-1]
                d["model"] = d["path"].replace("_" + d["path"].split("_")[-1], "")

                data.append(d)
        except:
            data.append({"path": path.split("/")[-1]})

    df = pd.DataFrame.from_dict(json_normalize(data), orient="columns")

    df = df.set_index(idx)

    return df


def filter_df(df, lines, columns=None, sort=None):
    df = df.loc[lines]

    if sort:
        df = df.sort_values(sort, ascending=False)

    if columns:
        df = df[columns]

    return df


def cut_name(names=[]):
    return [n.replace("_" + n.split("_")[-1], "") for n in names]


def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    data = data[~np.isnan(data)]
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h
