from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report

# https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
# https://en.wikipedia.org/wiki/Sensitivity_and_specificity
def calculate_fairness_metrics(df: pd.DataFrame, sub_keys: List[str], ground_truth_key: str,
                               prediction_key: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for sub_key in sub_keys:
        subs = df[sub_key].unique()

        for sub in subs:
            sub_df = df[df[sub_key] == sub]

            y_true, y_pred = sub_df[ground_truth_key], sub_df[prediction_key]

            cnf_matrix     = confusion_matrix(y_true, y_pred)

            num_positives  = np.sum(np.diag(cnf_matrix))
            num_negatives  = np.sum(cnf_matrix) - num_positives

            fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
            fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            tp = np.diag(cnf_matrix)
            tn = cnf_matrix.sum() - (fp + fn + tp)

            fp = fp.astype(float)
            fn = fn.astype(float)
            tp = tp.astype(float)
            tn = tn.astype(float)

            # Sensitivity, hit rate, recall, or true positive rate
            tpr = tp/(tp+fn)
            # Specificity or true negative rate
            tnr = tn/(tn+fp) 
            # Precision or positive predictive value
            ppv = tp/(tp+fp)
            # Negative predictive value
            npv = tn/(tn+fn)
            # Fall out or false positive rate
            fpr = fp/(fp+tn)
            # False negative rate
            fnr = fn/(tp+fn)
            # False discovery rate
            fdr = fp/(tp+fp)
            # positive rate
            pr  = (tp+fp)/(tp+fp+fn+tn)
            # positive rate
            nr  = (tn+fn)/(tp+fp+fn+tn)

            # Overall accuracy
            acc = (tp+tn)/(tp+fp+fn+tn)

            # Balanced Accuracy (BA)
            bacc =  (tpr+tnr)/2

            #print(classification_report(y_true,y_pred))

            rows.append({
                "sub_key": sub_key,
                "sub": sub,
                "total_class": len(acc),
                "false_positive_rate": np.nanmean(fpr),
                "false_negative_rate": np.nanmean(fnr),
                "true_positive_rate": np.nanmean(tpr),
                "true_negative_rate": np.nanmean(tnr),
                "positive_rate": np.nanmean(pr),
                "negative_rate": np.nanmean(nr),
                "accuracy": np.nanmean(acc),
                "balance_accuracy": np.nanmean(bacc),
                "num_positives": num_positives,
                "num_negatives": num_negatives,
                "num_total": num_positives+ num_negatives,
            })

    return pd.DataFrame(data=rows).sort_values(["sub_key", "sub"])





# def calculate_fairness_metrics(df: pd.DataFrame, sub_keys: List[str], ground_truth_key: str,
#                                prediction_key: str) -> pd.DataFrame:
#     rows: List[Dict[str, Any]] = []

#     for sub_key in sub_keys:
#         subs = df[sub_key].unique()

#         for sub in subs:
#             sub_df = df[df[sub_key] == sub]
#             y_true, y_pred = sub_df[ground_truth_key], sub_df[prediction_key]
#             mcm = multilabel_confusion_matrix(y_true, y_pred)
#             tn = np.mean(mcm[:, 0, 0])
#             fn = np.mean(mcm[:, 1, 0])
#             tp = np.mean(mcm[:, 1, 1])
#             fp = np.mean(mcm[:, 0, 1])

#             cm = confusion_matrix(y_true, y_pred)
#             print(cm)
#             num_positives = np.sum(np.diag(cm))
#             num_negatives = np.sum(cm) - num_positives

#             tpr = tp / (num_positives or float('nan'))
#             tnr = tn / (num_negatives or float('nan'))
#             fpr = fp / (num_negatives or float('nan'))
#             fnr = fn / (num_positives or float('nan'))
#             pr = (tp + fp) / ((num_positives + num_negatives) or float('nan'))
#             nr = (tn + fn) / ((num_positives + num_negatives) or float('nan'))


#             print(classification_report(y_true,y_pred))


#             rows.append({
#                 "sub_key": sub_key,
#                 "sub": sub,
#                 "false_positive_rate": fpr,
#                 "false_negative_rate": fnr,
#                 "true_positive_rate": tpr,
#                 "true_negative_rate": tnr,
#                 "positive_rate": pr,
#                 "negative_rate": nr,
#                 "num_positives": num_positives,
#                 "num_negatives": num_negatives,
#                 "num_total": num_positives+ num_negatives,
#             })

#     return pd.DataFrame(data=rows)



