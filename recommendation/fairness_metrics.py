from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix


def calculate_fairness_metrics(df: pd.DataFrame, sub_keys: List[str], ground_truth_key: str,
                               prediction_key: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for sub_key in sub_keys:
        subs = df[sub_key].unique()

        for sub in subs:
            sub_df = df[df[sub_key] == sub]
            mcm = multilabel_confusion_matrix(sub_df[ground_truth_key], sub_df[prediction_key])
            tn = np.mean(mcm[:, 0, 0])
            fn = np.mean(mcm[:, 1, 0])
            tp = np.mean(mcm[:, 1, 1])
            fp = np.mean(mcm[:, 0, 1])

            cm = confusion_matrix(sub_df[ground_truth_key], sub_df[prediction_key])
            num_positives = np.sum(np.diag(cm))
            num_negatives = np.sum(cm) - num_positives

            tpr = tp / (num_positives or float('nan'))
            tnr = tn / (num_negatives or float('nan'))
            fpr = fp / (num_negatives or float('nan'))
            fnr = fn / (num_positives or float('nan'))
            pr = (tp + fp) / ((num_positives + num_negatives) or float('nan'))
            nr = (tn + fn) / ((num_positives + num_negatives) or float('nan'))

            rows.append({
                "sub_key": sub_key,
                "sub": sub,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "true_positive_rate": tpr,
                "true_negative_rate": tnr,
                "positive_rate": pr,
                "negative_rate": nr,
                "num_positives": num_positives,
                "num_negatives": num_negatives,
                "num_total": num_positives+ num_negatives,
            })

    return pd.DataFrame(data=rows)



