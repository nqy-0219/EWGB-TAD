
from collections import  defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import  average_precision_score, accuracy_score, f1_score, precision_score, recall_score, auc, precision_recall_curve


def get_metrics(df, metric_column, non_outlier="non outlier"):
    """get metrics for the porto dataset
    inputs: data -> pd.DataFrame with with 3 columns: rec_loss (log_perplexity), outlier, seq_length
    
    """

    df = df.copy().reset_index()

    all_non_outlier = df[(df.outlier == non_outlier)]
    std_above  = 3
    std = all_non_outlier[metric_column].std() * std_above
    treshold = all_non_outlier[metric_column].mean() + std 
    
    df["detected_outlier"] =   np.where(df[metric_column].round(1) > treshold, "outlier", non_outlier)
    df["detected_outlier_label"] =   np.where(df[metric_column].round(1) > treshold, 1, 0)
    df["outlier_label"] =   np.where(df["outlier"] == non_outlier, 0, 1)
    
    accuracy = accuracy_score(df["outlier_label"], df["detected_outlier_label"])
    precision = precision_score(df["outlier_label"], df["detected_outlier_label"])
    recall = recall_score(df["outlier_label"], df["detected_outlier_label"])
    f1 = f1_score(df["outlier_label"], df["detected_outlier_label"])
    average_precision = average_precision_score(df["outlier_label"], df["detected_outlier_label"])
    
    precisions, recalls, _  = precision_recall_curve(df["outlier_label"], df[metric_column])
    pr_auc = auc(recalls, precisions)
    
    
    return (accuracy, precision, recall, average_precision, f1, pr_auc), treshold

def get_pattern_of_life_metrics(df, metric_column, remove_outlier_agents=["greeen", "yellow"]):
    """get metrics for the porto dataset
    inputs: data -> pd.DataFrame with with 3 columns: rec_loss (log_perplexity), outlier, seq_length
    
    """

    df = df.copy().reset_index()

    all_non_outlier = df[(df.outlier == "non outlier")]
    std_above  = 3
    std = all_non_outlier[metric_column].std() * std_above
    treshold = all_non_outlier[metric_column].mean() + std 
    
    df["detected_outlier"] =   np.where(df[metric_column].round(1) > treshold, "outlier", "non outlier")
    df["detected_outlier_label"] =   np.where(df[metric_column].round(1) > treshold, 1, 0)
    df["outlier_label"] =   np.where(df["outlier"] == "non outlier", 0, 1)
    
    accuracy = accuracy_score(df["outlier_label"], df["detected_outlier_label"])
    precision = precision_score(df["outlier_label"], df["detected_outlier_label"])
    recall = recall_score(df["outlier_label"], df["detected_outlier_label"])
    f1 = f1_score(df["outlier_label"], df["detected_outlier_label"])
    average_precision = average_precision_score(df["outlier_label"], df["detected_outlier_label"])
    
    precisions, recalls, _  = precision_recall_curve(df["outlier_label"], df[metric_column])
    pr_auc = auc(recalls, precisions)
    
    
    return (accuracy, precision, recall, average_precision, f1, pr_auc), treshold




def get_per_user_metrics(df, outliers, metric_column):
    """get all the metrics at the indivisula user's level in the pattern of life dataset"""

    results = defaultdict(list)
    
    # ipdb.set_trace()

    for agent_id in outliers:
        
        current_df = df[df["user_id"] == f"user_{agent_id}"].copy()
        all_non_outlier = current_df[(current_df.outlier == "non outlier")]

        std_above  = 3
        std = all_non_outlier[metric_column].std() * std_above
        treshold = all_non_outlier[metric_column].mean() + std 

        current_df["detected_outlier"] =   np.where(current_df[metric_column].round(1) > treshold, "outlier", "non outlier")
        current_df["detected_outlier_label"] =   np.where(current_df[metric_column].round(1) > treshold, 1, 0)

        (accuracy, precision, recall, average_precision, f1, pr_auc), treshold = get_metrics(current_df, metric_column) #generic method. TODO later, change this name everywhere

        results["agent"].append(agent_id)
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["average_precision"].append(average_precision)
        results["f1"].append(f1)
        results["pr_auc"].append(pr_auc)

    df_metrics = pd.DataFrame(results)

    return df_metrics