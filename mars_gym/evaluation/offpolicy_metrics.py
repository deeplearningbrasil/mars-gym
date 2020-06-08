"""
OffPolicy Metrics
"""
from typing import List

import numpy as np
import pandas as pd
import scipy
from typing import Dict, Tuple, List, Any, Type, Union
from torch.utils.data.dataset import Dataset


def _calc_sample_weigths(rewards, t_props, l_props):
    # l_props: Coleta
    # t_props: Avaliação
    #
    # Compute the sample weights - propensity ratios
    p_ratio = t_props / l_props

    # Effective sample size for E_t estimate (from A. Owen)
    n_e = len(rewards) * (np.mean(p_ratio) ** 2) / (p_ratio ** 2).mean()

    # Critical value from t-distribution as we have unknown variance
    alpha = 0.00125
    cv = scipy.stats.t.ppf(1 - alpha, df=int(n_e) - 1)

    return p_ratio, n_e, cv


def eval_IPS(rewards, t_props, l_props):
    # Calculate Sample Weigths
    p_ratio, n_e, cv = _calc_sample_weigths(rewards, t_props, l_props)

    ###############
    # VANILLA IPS #
    ###############
    # Expected reward for pi_t
    E_t = np.mean(rewards * p_ratio)

    # Variance of the estimate
    var = ((rewards * p_ratio - E_t) ** 2).mean()
    stddev = np.sqrt(var)

    # C.I. assuming unknown variance - use t-distribution and effective sample size
    c = cv * stddev / np.sqrt(int(n_e))
    min_bound = E_t - c
    max_bound = E_t + c

    result = (E_t, c)  # 0.025, 0.500, 0.975
    return result


def eval_CIPS(rewards, t_props, l_props, cap=15):
    # Calculate Sample Weigths
    p_ratio, n_e, cv = _calc_sample_weigths(rewards, t_props, l_props)

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
    c = cv * stddev_capped / np.sqrt(int(n_e))

    min_bound_capped = E_t_capped - c
    max_bound_capped = E_t_capped + c

    result = (E_t_capped, c)  # 0.025, 0.500, 0.975

    return result


def eval_SNIPS(rewards, t_props, l_props):
    # Calculate Sample Weigths
    p_ratio, n_e, cv = _calc_sample_weigths(rewards, t_props, l_props)

    ##############
    # NORMED IPS #
    ##############
    # Expected reward for pi_t
    E_t_normed = np.sum(rewards * p_ratio) / np.sum(p_ratio)

    # Variance of the estimate
    var_normed = np.sum(((rewards - E_t_normed) ** 2) * (p_ratio ** 2)) / (
        p_ratio.sum() ** 2
    )
    stddev_normed = np.sqrt(var_normed)

    # C.I. assuming unknown variance - use t-distribution and effective sample size
    c = cv * stddev_normed / np.sqrt(int(n_e))

    min_bound_normed = E_t_normed - c
    max_bound_normed = E_t_normed + c

    # Store result
    result = (E_t_normed, c)  # 0.025, 0.500, 0.975

    return result


def eval_doubly_robust(
    action_rhat_rewards, item_idx_rhat_rewards, rewards, t_props, l_props, cap=None
):
    # Calculate Sample Weigths
    p_ratio, n_e, cv = _calc_sample_weigths(rewards, t_props, l_props)

    # Cap ratios
    if cap is not None:
        p_ratio = np.clip(p_ratio, a_min=None, a_max=cap)

    #################
    # Roubly Robust #
    #################
    dr = action_rhat_rewards + (p_ratio * (rewards - item_idx_rhat_rewards))

    confidence = 0.95
    n = len(dr)
    m, se = np.mean(dr), scipy.stats.sem(dr)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return m, h


def DirectEstimator(object):
    def __init__(estimator):
        self.estimator = estimator

    def predict_proba(self, X):
        pass

    def _flatten_input_and_extract_arm(
        self, input_: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, int]:
        flattened_input = np.concatenate([el.reshape(1, -1) for el in input_], axis=1)[
            0
        ]

        return (
            np.delete(flattened_input, self._arm_index),
            int(flattened_input[self._arm_index]),
        )

    def fit(self, dataset: Dataset) -> None:
        n = len(dataset)

        for i in tqdm(range(n), total=n):
            input_: Tuple[np.ndarray, ...] = dataset[i][0]
            x, arm = self._flatten_input_and_extract_arm(input_)

            raise (Exception("==>", x, arm))
