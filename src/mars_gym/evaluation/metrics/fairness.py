from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    classification_report,
)
from mars_gym.utils.utils import mean_confidence_interval
np.seterr(divide='ignore', invalid='ignore')

# https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
# https://en.wikipedia.org/wiki/Sensitivity_and_specificity
def calculate_fairness_metrics(
    df: pd.DataFrame, sub_keys: List[str], ground_truth_key: str, prediction_key: str
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for sub_key in sub_keys:
        subs = df[sub_key].unique()

        for sub in subs:
            sub_df = df[df[sub_key] == sub]
            y_true, y_pred = (
                sub_df[ground_truth_key].astype(str),
                sub_df[prediction_key].astype(str),
            )
            #from IPython import embed
            #embed()
            cnf_matrix = confusion_matrix(y_true, y_pred)

            num_positives = np.sum(np.diag(cnf_matrix))
            num_negatives = np.sum(cnf_matrix) - num_positives

            fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            tp = np.diag(cnf_matrix)
            tn = cnf_matrix.sum() - (fp + fn + tp)

            fp = fp.astype(float)
            fn = fn.astype(float)
            tp = tp.astype(float)
            tn = tn.astype(float)

            # Sensitivity, hit rate, recall, or true positive rate
            tpr = tp / (tp + fn)
            # Specificity or true negative rate
            tnr = tn / (tn + fp)
            # Precision or positive predictive value
            ppv = tp / (tp + fp)
            # Negative predictive value
            npv = tn / (tn + fn)
            # Fall out or false positive rate
            fpr = fp / (fp + tn)
            # False negative rate
            fnr = fn / (tp + fn)
            # False discovery rate
            fdr = fp / (tp + fp)
            # positive rate
            pr = (tp + fp) / (tp + fp + fn + tn)
            # positive rate
            nr = (tn + fn) / (tp + fp + fn + tn)

            # Overall accuracy
            acc = (tp + tn) / (tp + fp + fn + tn)

            # Balanced Accuracy (BA)
            bacc = (tpr + tnr) / 2

            # print(classification_report(y_true,y_pred))
            fpr, fpr_c = mean_confidence_interval(fpr)
            fnr, fnr_c = mean_confidence_interval(fnr)
            tpr, tpr_c = mean_confidence_interval(tpr)
            tnr, tnr_c = mean_confidence_interval(tnr)
            pr, pr_c = mean_confidence_interval(pr)
            nr, nr_c = mean_confidence_interval(nr)
            acc, acc_c = mean_confidence_interval(acc)
            bacc, bacc_c = mean_confidence_interval(bacc)

            rows.append(
                {
                    "sub_key": sub_key,
                    "sub": sub,
                    "total_class": len(tp),
                    "false_positive_rate": fpr,
                    "false_positive_rate_C": fpr_c,
                    "false_negative_rate": fnr,
                    "false_negative_rate_C": fnr_c,
                    "true_positive_rate": tpr,
                    "true_positive_rate_C": tpr_c,
                    "true_negative_rate": tnr,
                    "true_negative_rate_C": tnr_c,
                    "positive_rate": pr,
                    "positive_rate_C": pr_c,
                    "negative_rate": nr,
                    "negative_rate_C": nr_c,
                    "accuracy": acc,
                    "accuracy_C": acc_c,
                    "balance_accuracy": bacc,
                    "balance_accuracy_C": bacc_c,
                    "total_positives": num_positives,
                    "total_negatives": num_negatives,
                    "total_individuals": num_positives + num_negatives,
                }
            )

    return pd.DataFrame(data=rows).sort_values(["sub_key", "sub"])
