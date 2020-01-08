import pandas as pd
import json
import os
from pandas.io.json import json_normalize

def json2df(paths, file):
  data = []
  for model, path in paths.items():
    file_path = os.path.join(path, file)
    try:
      with open(file_path) as json_file:
        d = json.load(json_file)
        d['path'] = path.split("/")[-1]
        data.append(d)
    except IsADirectoryError:
      data.append({'path': path.split("/")[-1]})

  df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
  df = df.set_index('path')
  return df

def filter_df(df, lines, columns = None, sort = None):
  df = df.loc[lines]
  
  if sort:
    df = df.sort_values(sort, ascending=False)

  if columns:
    df = df[columns]

  return df