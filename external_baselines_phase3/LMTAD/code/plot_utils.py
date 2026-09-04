
from pathlib import Path
from glob import glob
import random
from collections import  defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



from metrics import get_metrics


RED_OUTLIERS = [546, 644, 347, 62, 551, 992, 554, 949, 900, 57]
YELLOW_OUTLIERS = [83, 478, 1, 244, 379, 161, 147, 353, 517, 364]
GREEN_OUTLIERS = [66, 809, 976, 4, 84, 268, 858, 416, 307, 956]

ALL_OUTLIERS = RED_OUTLIERS + YELLOW_OUTLIERS + GREEN_OUTLIERS

outlier_dict = {}
for type, outliers in zip(["red", "yellow", "green"], [RED_OUTLIERS, YELLOW_OUTLIERS, GREEN_OUTLIERS]):
    outlier_dict.update(dict((k, type) for k in outliers))


OUTLIER_DICT = dict((k, "green") for k in GREEN_OUTLIERS)
for item in [66, 809, 976, 4, 84, 268, 858, 416, 307, 956]:
    outlier_dict[item] = "green"
    
for item in [83, 478, 1, 244, 379, 161, 147, 353, 517, 364]:
    outlier_dict[item] = "yellow"
    
for item in [546, 644, 347, 62, 551, 992, 554, 949, 900, 57]:
    outlier_dict[item] = "red"


def load_tsvs(root_dir):
    """laod tsv files in the rood_dir"""
    files = glob(f"{root_dir}/*.tsv")
    dfs = {}
    for file in files:
        data = pd.read_csv(file, delimiter="\t")
        dfs[Path(file).stem] = data

    return dfs

def plot_metrics(dfs, eval_metric="rec_loss", out_dir=""):
    """plot metrics
        inputs: dfs -> dictionry of panda dataframes, it should have at most 3 dataframes
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 3))
    axes = axes.ravel()

    for idx, (key, df) in enumerate(dfs.items()):
        
        
        
        (accuracy, precision, recall, average_precision, f1, pr_auc), treshold = get_metrics(df[df["outlier"] != "route switch outlier"], eval_metric)
        detour_out_metrics = [precision, recall, f1, pr_auc]
        (accuracy, precision, recall, average_precision, f1, pr_auc), treshold = get_metrics(df[df["outlier"] != "detour outlier" ], eval_metric)
        route_out_metrics = [precision, recall, f1, pr_auc]
        #  ["accuracy", "precision", "recall", "average precision", "f1", "pr-auc"],
        metrics = ["precision", "recall", "f1", "pr-auc"]
        
        dict_metrics = {
            "metrics": metrics,
            "detour outliers": detour_out_metrics,
            "random shift outliers": route_out_metrics,

        }

        df_metrics = pd.DataFrame(dict_metrics)
        df_metrics = df_metrics.set_index("metrics")
        df_metrics
        
        font_size_labels = 26
        if idx == 0:
            sns.heatmap(df_metrics, annot=True, annot_kws={'size': font_size_labels}, cmap="crest", ax=axes[idx], cbar=False, vmin=0.10, vmax=1.0, fmt='.3f')            
        elif idx  == len(dfs) - 1:
           sns.heatmap(df_metrics, annot=True, annot_kws={'size': font_size_labels}, cmap="crest", ax=axes[idx], cbar=True, yticklabels=False, vmin=0.10, vmax=1.0, fmt='.3f')
        else:
            sns.heatmap(df_metrics, annot=True, annot_kws={'size': font_size_labels}, cmap="crest", ax=axes[idx], cbar=False, yticklabels=False, vmin=0.10, vmax=1.0, fmt='.3f')
            
            
        
    #     plt.xlabel("Outliers comparisons", fontsize=25)
    #     plt.ylabel("Metrics", fontsize=22)
    #     plt.xticks(fontsize=18)
    #     plt.yticks(fontsize=18)
        
        
        axes[idx].set_title(f"{key}", fontsize=10)
        # axes[idx].set_xlabel("\noutlier configuration", fontsize=18)
        
        # if idx < 1:
        #     axes[idx].set_ylabel("metrics", fontsize=18)
        # else:
        axes[idx].set_ylabel("", fontsize=1)

        axes[idx].tick_params(labelsize=18, rotation=0)
        
    plt.tight_layout()
    plt.savefig(f'{out_dir}/metrics.png')


def plot_all_outliers(dfs, eval_metric="rec_loss"):
    
    fig, axes = plt.subplots(1, 3, figsize=(30, 7))
    axes = axes.ravel()
    
    colors = ['#E37346', '#8EE35D', '#747FE3']
    for idx, (key, df) in enumerate(dfs.items()):
        
        df = df.sort_values("outlier", ascending=False)
        sns.scatterplot(data=df, x="seq_length", y=eval_metric, hue="outlier", ax=axes[idx], palette=colors)
        axes[idx].set_title(f"{key}\n", fontsize=18)
        axes[idx].set_xlabel("trajectory length", fontsize=16)

        ylabel = "reconstruction loss" if eval_metric == "rec_loss" else "log perplexity"
        axes[idx].set_ylabel(ylabel, fontsize=16)
        axes[idx].tick_params(labelsize=14)
        plt.setp(axes[idx].get_legend().get_texts(), fontsize="14")


def plot_route_switch_outliers(dfs, eval_metric="rec_loss"):
    fig, axes = plt.subplots(1, 3, figsize=(30, 7))
    axes = axes.ravel()
    
    colors = ['#E37346', '#8EE35D']
    for idx, (key, df) in enumerate(dfs.items()):
        
        df = df[df["outlier"] != "detour outlier"].sort_values("outlier", ascending=False)
        sns.scatterplot(data=df, x="seq_length", y="rec_loss", hue="outlier", ax=axes[idx], palette=colors)
        axes[idx].set_title(f"{key}\n", fontsize=18)
        axes[idx].set_xlabel("trajectory length", fontsize=16)

        ylabel = "reconstruction loss" if eval_metric == "rec_loss" else "log perplexity"
        axes[idx].set_ylabel("reconstruction loss", fontsize=16)
        axes[idx].tick_params(labelsize=14)
        plt.setp(axes[idx].get_legend().get_texts(), fontsize="14")
        
        
def plot_detour_outliers(dfs, eval_metric="rec_loss"):
    fig, axes = plt.subplots(1, 3, figsize=(30, 7))
    axes = axes.ravel()
    
    colors = ['#8EE35D', '#747FE3']
    for idx, (key, df) in enumerate(dfs.items()):
        
        df = df[df["outlier"] != "route switch outlier"].sort_values("outlier", ascending=False)
        sns.scatterplot(data=df, x="seq_length", y="rec_loss", hue="outlier", ax=axes[idx], palette=colors)
        axes[idx].set_title(f"{key}\n", fontsize=18)
        axes[idx].set_xlabel("trajectory length", fontsize=16)

        ylabel = "reconstruction loss" if eval_metric == "rec_loss" else "log perplexity"
        axes[idx].set_ylabel("reconstruction loss", fontsize=16)
        axes[idx].tick_params(labelsize=14)
        plt.setp(axes[idx].get_legend().get_texts(), fontsize="14")


def plot_agent_surprisal_rate(data, rows, cols, out_dir):
    """plot the surprisal rate of agents in the pattern of life dataset"""
    
    random.seed(123)
    agents = RED_OUTLIERS.copy()
    non_outliers = list(set(np.arange(0, 1000)) - set(ALL_OUTLIERS))
    subset_list = random.sample(non_outliers, 20)
    agents.extend(subset_list)

    index = 1

    fig, axes = plt.subplots(rows, cols, figsize=(35, 20))
    axes = axes.ravel()

    for idx, agent_id in  enumerate(agents):
        agent_df = data[data["id"] == agent_id]
        surprisal_dict = {}
        for i, row in agent_df.iterrows():

            probs = np.array(row["raw_probs"])
            surp_rates = -np.log2(probs)
            delta = surp_rates[:-1] - surp_rates[1:]
            surprisal_dict[f"{row.id}_{index}"] = (np.arange(surp_rates.shape[0]) + 1, surp_rates, delta, row.outlier)
            index += 1
            # print(probs)
            # print(surp_rates)
            # print(np.arange(surp_rates.shape[0]) + 1)
            # print(row["id"])
            # break
        
        max_size = 0
        treshold = 11
        for key, (x, y, delta, outlier) in surprisal_dict.items():
            
            max_size = max(x.shape[0], max_size)
            color = "green" if y.max() <= treshold else "red"
            axes[idx].plot(x, y, color=color)
            axes[idx].set_ylim([0, 20])

            outlier_label = "outlier" if outlier_dict.get(agent_id, None) else "non outlier"
            title =f"\nagent: {agent_id} - " + outlier_label
            axes[idx].set_title(title, fontsize=28)
            axes[idx].tick_params(axis='both', labelsize=20)
            


        axes[idx].plot(np.arange(1, max_size+1), np.ones(max_size) +treshold-1, color="black", linewidth=8)
        # plt.title(title, fontsize=20)

    
    fig.text(0.5, -0.04, 'Token Position', ha='center', fontsize=46)
    fig.text(-0.02, 0.5, 'Surprisal Rate', va='center', rotation='vertical', fontsize=46)
    plt.tight_layout()
    
    plt.savefig(f'{out_dir}/surprisal_rates.png')

def plot_metrics_pattern_of_life(data, eval_metric, out_dir):

    # filtered_data = data[~data["id"].isin(YELLOW_OUTLIERS + GREEN_OUTLIERS)].reset_index()

    idx = 0
    fig = plt.figure(figsize=(20, 8))
    # ax = fig.add_subplot(231)
    # axes = axes.ravel()
    # all_metrics = []

    dict_metrics = {
        "metrics": ["precision", "recall", "f1", "pr-auc"]
    }
    
    results = defaultdict(list)

    for agent_id in sorted(RED_OUTLIERS):
        
        current_df = data[data["id"] == agent_id].copy()
        
        all_non_outlier = current_df[(current_df.outlier == "non outlier")]

        std_above  = 3
        std = all_non_outlier[eval_metric].std() * std_above
        treshold = all_non_outlier[eval_metric].mean() + std 

        current_df["detected_outlier"] =   np.where(current_df[eval_metric].round(1) > treshold, "outlier", "non outlier")
        current_df["detected_outlier_label"] =   np.where(current_df[eval_metric].round(1) > treshold, 1, 0)

        # current_metrics = print_metrics(current_df)
        
        (_, precision, recall, _, f1, pr_auc), _ = get_metrics(current_df, eval_metric)
        
        dict_metrics[f"{agent_id}"] = [precision, recall, f1, pr_auc]
        
        results["agent"].append(agent_id)
        # results["accuracy"].append(current_metrics[0])
        results["precision"].append(precision)
        results["recall"].append(recall)
        # results["average_precision"].append(current_metrics[3])
        results["f1"].append(f1)
        results["pr_auc"].append(pr_auc)
    

    df_metrics_results = pd.DataFrame(dict_metrics)
    df_metrics = df_metrics_results.set_index("metrics")
    # all_metrics.append(df_metrics)

    # if idx < 2:
    #     sns.heatmap(df_metrics, annot=True, annot_kws={'size': 20}, cmap="crest", ax=axes[idx], cbar=False, vmin=0.10, vmax=1.0, fmt='.3f')
    # else:
    ax = sns.heatmap(df_metrics, annot=True, annot_kws={'size': 12}, cmap="crest", cbar=True, vmin=0.10, vmax=1.0, fmt='.3f')

    ax.set_title(f"All Agents\n", fontsize=14)
    ax.set_xlabel("\nagent", fontsize=12)

    # if idx < 1:
    #     axes[idx].set_ylabel("metrics", fontsize=22)
    # else:
    ax.set_ylabel("", fontsize=1)
        
    ax.tick_params(labelsize=10, rotation=0)
        
    plt.tight_layout()
    fig.savefig(f'{out_dir}/metrics.png')
    return pd.DataFrame(results)

def plot_agent_perlexity_over_date(data, eval_metric, out_dir):
    figure = plt.figure(figsize=(20, 10))
    filtered_data = data[data["id"].isin(RED_OUTLIERS)].reset_index()

    ax = sns.scatterplot(data=filtered_data[filtered_data.outlier != "outlier"], x="date", y=eval_metric,s=300, color='grey', label="normal")
    ax = sns.scatterplot(data=filtered_data[filtered_data.outlier == "outlier"], x="date", y=eval_metric, s=300, color='r', label="abnormal")

    # ax.yaxis.offsetText.set_fontsize(30)
    ax.yaxis.get_offset_text().set_fontsize(30)

    plt.xlabel("date", fontsize=50)

    y_label =  "Perplexity (in log scale)" if eval_metric == "log_perplexity" else "Reconstruction loss"
    plt.ylabel(y_label, fontsize=45)
    plt.setp(ax.get_legend().get_texts(), fontsize='30') # for legend text
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=30)
    plt.savefig(f'{out_dir}/perplexity_over_dates.png')

    return filtered_data