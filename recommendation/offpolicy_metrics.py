"""
OffPolicy Metrics
"""
from typing import List

import numpy as np
import pandas as pd
import scipy

def _calc_sample_weigths(rewards, t_props, l_props):
  # Compute the sample weights - propensity ratios
  p_ratio = t_props / l_props

  # Effective sample size for E_t estimate (from A. Owen)
  n_e = len(rewards) * (np.mean(p_ratio) ** 2) / (p_ratio ** 2).mean()

  # Critical value from t-distribution as we have unknown variance
  alpha = .00125
  cv = scipy.stats.t.ppf(1 - alpha, df=int(n_e) - 1)

  return p_ratio, n_e, cv

def eval_IPS(rewards, t_props, l_props):
  # Calculate Sample Weigths
  p_ratio, n_e, cv =  _calc_sample_weigths(rewards, t_props, l_props)

  ###############
  # VANILLA IPS #
  ###############
  # Expected reward for pi_t
  E_t = np.mean(rewards * p_ratio)

  # Variance of the estimate
  var = ((rewards * p_ratio - E_t) ** 2).mean()
  stddev = np.sqrt(var)

  # C.I. assuming unknown variance - use t-distribution and effective sample size
  min_bound = E_t - cv * stddev / np.sqrt(int(n_e))
  max_bound = E_t + cv * stddev / np.sqrt(int(n_e))

  result = (min_bound, E_t, max_bound) # 0.025, 0.500, 0.975
  return result

def eval_CIPS(rewards, t_props, l_props, cap=15):
  # Calculate Sample Weigths
  p_ratio, n_e, cv =  _calc_sample_weigths(rewards, t_props, l_props)


  ############## 
  # CAPPED IPS #
  ##############
  # Cap ratios
  p_ratio_capped = np.clip(p_ratio, a_min=None, a_max=cap)

  # Expected reward for pi_t
  E_t_capped = np.mean(rewards * p_ratio_capped)

  # Variance of the estimate
  var_capped = ((rewards * p_ratio_capped - E_t_capped) ** 2).mean()
  stddev_capped = np.sqrt(var_capped)

  # C.I. assuming unknown variance - use t-distribution and effective sample size
  min_bound_capped = E_t_capped - cv * stddev_capped / np.sqrt(int(n_e))
  max_bound_capped = E_t_capped + cv * stddev_capped / np.sqrt(int(n_e))

  result = (min_bound_capped, E_t_capped, max_bound_capped) # 0.025, 0.500, 0.975

  return result

def eval_SNIPS(rewards, t_props, l_props):
  # Calculate Sample Weigths
  p_ratio, n_e, cv =  _calc_sample_weigths(rewards, t_props, l_props)


  ##############
  # NORMED IPS #
  ##############
  # Expected reward for pi_t
  E_t_normed = np.sum(rewards * p_ratio) / np.sum(p_ratio)

  # Variance of the estimate
  var_normed = np.sum(((rewards - E_t_normed) ** 2) * (p_ratio ** 2)) / (p_ratio.sum() ** 2)
  stddev_normed = np.sqrt(var_normed)

  # C.I. assuming unknown variance - use t-distribution and effective sample size
  min_bound_normed = E_t_normed - cv * stddev_normed / np.sqrt(int(n_e))
  max_bound_normed = E_t_normed + cv * stddev_normed / np.sqrt(int(n_e))

  # Store result
  result = (min_bound_normed, E_t_normed, max_bound_normed) # 0.025, 0.500, 0.975

  return result


