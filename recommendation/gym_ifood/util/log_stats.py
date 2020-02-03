import numpy as np
import pandas as pd


class LogStats(object):
  def __init__(self):
    self._log_data = []
    self._rewards  = []

  def logger(self, context, arms, rewards):
    data = list(zip(context, arms, rewards))
    
    self._log_data.extend(data)
    self._rewards.extend(rewards)

  def save_history(self, path):
    d = pd.DataFrame(self._log_data, 
                    columns=['context', 'arm', 'reward']).reset_index()

    d.to_csv(path, index=False)

  @property
  def rewards_stats(self):
    return {
      'sum':  np.sum(self._rewards),
      'mean': np.mean(self._rewards),
      'std':  np.std(self._rewards),
    }