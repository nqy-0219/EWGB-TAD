"""Build Phase 5 manuscript tables and figures from the fixed Phase 4 outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = Path(
    os.environ.get(
        "EWGB_PHASE4_SUMMARY_ROOT",
        str(ROOT / "paper_work" / "final_neurocomputing_results" / "summary"),
    )
)
OVERLEAF = Path(
    os.environ.get(
        "EWGB_OVERLEAF_ROOT",
        str(ROOT.parent / "neurocomputing_submission" / "overleaf_neurocomputing"),
    )
)
TABLE_DIR = OVERLEAF / "tables"
FIGURE_DIR = OVERLEAF / "figures"

DATASETS = ("Synthetic", "Grid-Network", "Porto-derived", "GeoLife")
DATASET_LABELS = {
    "Synthetic": "Synthetic",
    "Grid-Network": "Grid",
    "Porto-derived": "Porto",
    "GeoLife": "GeoLife",
}
METHODS = (
    "EWGB-TAD",
    "IForest",
    "ECOD",
    "iBoost-ODE",
    "CoMadOut",
    "Shape-KNN",
    "SegmentOD",
    "TADS",
    "Profile-TAD",
    "LSTM-AE",
    "USAD",
    "LM-TAD",
    "MST-OATD",
)
GPU_METHODS = {"LSTM-AE", "USAD", "LM-TAD", "MST-OATD"}


def load_json(name: str) -> dict:
    return json.loads((PHASE4 / name).read_text(encoding="utf-8"))


MAIN = load_json("main_aggregate.json")
MAIN_STATS = load_json("main_statistics.json")
ANALYSIS = load_json("analysis_aggregate.json")
ANALYSIS_STATS = load_json("analysis_statistics.json")
LOCAL_ENTROPY_PATH = PHASE4.parent / "sensitivity_stability" / "local_entropy_bins" / "local_entropy_bins_summary.json"
LOCAL_ENTROPY = (
    json.loads(LOCAL_ENTROPY_PATH.read_text(encoding="utf-8"))
    if LOCAL_ENTROPY_PATH.exists()
    else None
)


def macro_mean(method: str, metric: str) -> float:
    return float(np.mean([MAIN[dataset][method][metric]["mean"] for dataset in DATASETS]))


def macro_summary(method: str, metric: str) -> tuple[float, float]:
    values = np.asarray(
        [MAIN[dataset][method][metric]["mean"] for dataset in DATASETS],
        dtype=float,
    )
    return float(np.mean(values)), float(np.std(values, ddof=1))


def analysis_macro(analysis: str, variant: str, metric: str) -> float:
    return float(
        np.mean([ANALYSIS[analysis][variant][dataset][metric]["mean"] for dataset in DATASETS])
    )


def analysis_macro_summary(analysis: str, variant: str, metric: str) -> tuple[float, float]:
    values = np.asarray(
        [ANALYSIS[analysis][variant][dataset][metric]["mean"] for dataset in DATASETS],
        dtype=float,
    )
    return float(np.mean(values)), float(np.std(values, ddof=1))


def analysis_macro_cell(analysis: str, variant: str, metric: str) -> str:
    mean, std = analysis_macro_summary(analysis, variant, metric)
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def metric_cell(mean: float, std: float, bold: bool = False) -> str:
    value = f"{mean:.3f} $\\pm$ {std:.3f}"
    return f"\\textbf{{{value}}}" if bold else value


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def latex_p_value(value: float) -> str:
    if np.isclose(value, 1.0):
        return "1.000"
    if value >= 0.001:
        return f"{value:.3g}"
    exponent = int(np.floor(np.log10(value)))
    coefficient = value / (10 ** exponent)
    return f"${coefficient:.2f}\\times10^{{{exponent}}}$"


def build_metric_table(metric: str) -> None:
    label = metric.lower()
    ranks = MAIN_STATS[metric]["paired_seed_blocks"]["average_ranks"]
    best_by_dataset = {
        dataset: max(MAIN[dataset][method][metric]["mean"] for method in METHODS)
        for dataset in DATASETS
    }
    macro_values = {method: macro_mean(method, metric) for method in METHODS}
    best_macro = max(macro_values.values())

    lines = [
        "\\begin{center}",
        f"\\castablecaption[\\textwidth]{{tab:main_{label}}}{{{metric} performance under the predefined 10\\% controlled-injection protocol. Dataset cells are mean $\\pm$ standard deviation across 10/10/5/5 seeds for Synthetic/Grid/Porto/GeoLife; Macro is the unweighted mean $\\pm$ standard deviation across the four dataset means.}}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{2.6pt}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Synthetic} & \\textbf{Grid} & \\textbf{Porto} & \\textbf{GeoLife} & \\textbf{Macro} & \\textbf{Avg. rank} \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        cells = []
        for dataset in DATASETS:
            summary = MAIN[dataset][method][metric]
            cells.append(
                metric_cell(
                    summary["mean"],
                    summary["std"],
                    np.isclose(summary["mean"], best_by_dataset[dataset]),
                )
            )
        macro_mean_value, macro_std = macro_summary(method, metric)
        macro = f"{macro_mean_value:.3f} $\\pm$ {macro_std:.3f}"
        if np.isclose(macro_values[method], best_macro):
            macro = f"\\textbf{{{macro}}}"
        rank = f"{ranks[method]:.3f}"
        if np.isclose(ranks[method], min(ranks.values())):
            rank = f"\\textbf{{{rank}}}"
        lines.append(f"{method} & " + " & ".join(cells + [macro, rank]) + " \\\\")
    lines += [
        "\\botrule",
        "\\end{tabular*}",
        "\\endgroup",
        "\\end{center}",
    ]
    write_text(TABLE_DIR / f"main_{label}_table.tex", "\n".join(lines))


def build_auc_pairwise_table() -> None:
    pairwise = MAIN_STATS["AUC"]["paired_seed_blocks"]["ewgb_pairwise_wilcoxon_holm"]
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:auc_pairwise}{Paired AUC comparisons over the 30 matched dataset--seed blocks. Positive differences favor EWGB-TAD; $p$-values are Holm-adjusted. Complete win, tie, and loss counts are reported in Supplementary Table~S2.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{4.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular*}{0.78\\textwidth}{@{\\extracolsep{\\fill}}lrr@{}}",
        "\\toprule",
        "\\textbf{Comparator} & \\textbf{Mean $\\Delta$AUC} & \\textbf{Wins} & \\textbf{Losses} & \\textbf{Adjusted $p$} \\\\",
        "\\midrule",
    ]
    for method in METHODS[1:]:
        row = pairwise[method]
        p = row["p_holm"]
        p_text = latex_p_value(p)
        lines.append(
            f"{method} & {row['mean_difference']:.4f} & {row['wins']} & {row['losses']} & {p_text} \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    compact_lines = [
        r"\begin{center}",
        r"\castablecaption[\textwidth]{tab:auc_pairwise}{Paired AUC comparisons over the 30 matched dataset--seed blocks. Positive differences favor EWGB-TAD; $p$-values are Holm-adjusted. Complete win, tie, and loss counts are reported in Supplementary Table~S2.}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular*}{0.78\textwidth}{@{\extracolsep{\fill}}lrr@{}}",
        r"\toprule",
        r"\textbf{Comparator} & \textbf{Mean $\Delta$AUC} & \textbf{Adjusted $p$} \\",
        r"\midrule",
    ]
    supplement_lines = [
        r"\begin{center}",
        r"\supplementtablecaption[\textwidth]{tab:supp_auc_pairwise}{Complete paired AUC comparisons between EWGB-TAD and each baseline over the 30 matched dataset--seed blocks. Positive differences favor EWGB-TAD; $p$-values are Holm-adjusted.}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Comparator} & \textbf{Mean $\Delta$AUC} & \textbf{Wins} & \textbf{Ties} & \textbf{Losses} & \textbf{Adjusted $p$} \\",
        r"\midrule",
    ]
    for method in METHODS[1:]:
        row = pairwise[method]
        p = row["p_holm"]
        p_text = latex_p_value(p)
        compact_lines.append(f"{method} & {row['mean_difference']:.4f} & {p_text} \\\\")
        supplement_lines.append(
            f"{method} & {row['mean_difference']:.4f} & {row['wins']} & {row['ties']} & "
            f"{row['losses']} & {p_text} \\\\"
        )
    compact_lines += [r"\botrule", r"\end{tabular*}", r"\endgroup", r"\end{center}"]
    supplement_lines += [r"\botrule", r"\end{tabular*}", r"\endgroup", r"\end{center}"]
    write_text(TABLE_DIR / "main_auc_pairwise_table.tex", "\n".join(compact_lines))
    write_text(TABLE_DIR / "supp_main_auc_pairwise_full_table.tex", "\n".join(supplement_lines))


def build_component_tables() -> None:
    variants = tuple(ANALYSIS["component"])
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:component_factorial}{Full $2\\times2\\times2$ component analysis. G, L, and F denote adaptive granular partitioning, the local entropy-adaptive metric, and entropy-based view fusion. The three direct reference variants are labelled KMeans-Prototype, global entropy weighting, and average fusion. All rows use the same PCA-based shape representation and predefined dataset--seed protocol. Values are macro means $\\pm$ standard deviations across the four dataset means.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{5.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Variant} & \\textbf{G} & \\textbf{L} & \\textbf{F} & \\textbf{AUC} & \\textbf{AUPRC} & \\textbf{F1} \\\\",
        "\\midrule",
    ]
    for variant in variants:
        g, l, f = variant.split("_")
        flags = ["$\\checkmark$" if token.endswith("1") else "--" for token in (g, l, f)]
        named_variants = {
            "g0_l1_f1": "KMeans-Prototype",
            "g1_l0_f1": "Global entropy weighting",
            "g1_l1_f0": "Average fusion",
            "g1_l1_f1": "EWGB-TAD",
        }
        name = named_variants.get(variant, variant.replace("_", "-"))
        lines.append(
            f"{name} & " + " & ".join(
                flags + [
                    analysis_macro_cell("component", variant, metric)
                    for metric in ("AUC", "AUPRC", "F1")
                ]
            ) + " \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    write_text(TABLE_DIR / "component_factorial_table.tex", "\n".join(lines))

    factor_names = {
        "granular_partition": "Adaptive granular partition",
        "local_metric": "Local entropy-adaptive metric",
        "entropy_fusion": "Entropy view fusion",
        "granular_x_local": "$G\\times L$",
        "granular_x_fusion": "$G\\times F$",
        "local_x_fusion": "$L\\times F$",
        "granular_x_local_x_fusion": "$G\\times L\\times F$",
    }
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:component_effects}{Factorial main-effect and interaction contrasts across 30 matched dataset--seed blocks. Each cell reports the mean paired contrast and the Holm-adjusted $p$-value from a two-sided Wilcoxon test.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{4.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccc@{}}",
        "\\toprule",
        "\\textbf{Factor} & \\textbf{$\\Delta$AUC / $p$} & \\textbf{$\\Delta$AUPRC / $p$} & \\textbf{$\\Delta$F1 / $p$} \\\\",
        "\\midrule",
    ]
    for factor, display in factor_names.items():
        cells = []
        for metric in ("AUC", "AUPRC", "F1"):
            effect = ANALYSIS_STATS["component"][metric]["factorial_contrasts"]["effects"][factor]
            cells.append(f"{effect['mean_paired_effect']:+.4f} / {latex_p_value(effect['p_holm'])}")
        lines.append(f"{display} & " + " & ".join(cells) + " \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    write_text(TABLE_DIR / "component_effects_table.tex", "\n".join(lines))


def build_view_table() -> None:
    display = {
        "spatial_path": "Spatial-path",
        "kinematic": "Kinematic",
        "trajectory_shape": "Trajectory-shape",
        "spatial_path+kinematic": "Spatial-path + kinematic",
        "spatial_path+trajectory_shape": "Spatial-path + trajectory-shape",
        "kinematic+trajectory_shape": "Kinematic + trajectory-shape",
        "spatial_path+kinematic+trajectory_shape": "All three views",
    }
    ranks = ANALYSIS_STATS["view"]["AUC"]["average_ranks"]
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:view_complementarity}{Single-, pair-, and three-view comparison under the same detector and fusion protocol. Metric values are macro means $\\pm$ standard deviations across the four dataset means; AUC rank is descriptive across the 30 matched blocks.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{5.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc@{}}",
        "\\toprule",
        "\\textbf{View set} & \\textbf{AUC} & \\textbf{AUPRC} & \\textbf{F1} & \\textbf{AUC rank} \\\\",
        "\\midrule",
    ]
    for variant in ANALYSIS["view"]:
        lines.append(
            f"{display[variant]} & "
            + " & ".join(
                analysis_macro_cell("view", variant, metric)
                for metric in ("AUC", "AUPRC", "F1")
            )
            + f" & {ranks[variant]:.3f} \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    write_text(TABLE_DIR / "view_complementarity_table.tex", "\n".join(lines))


def build_sensitivity_table() -> None:
    display = {
        "min4": "$n_{min}=4$",
        "min8": "$n_{min}=8$ (default)",
        "min16": "$n_{min}=16$",
        "min32": "$n_{min}=32$",
    }
    variants = tuple(variant for variant in ANALYSIS["sensitivity"] if variant in display)
    comparisons = ANALYSIS_STATS["sensitivity"]["AUC"]["reference_pairwise_wilcoxon_holm"]
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:sensitivity_cross_dataset}{Cross-dataset AUC sensitivity to the minimum terminal-ball size under the sample-size-adaptive local entropy rule.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Setting} & \\textbf{Synthetic} & \\textbf{Grid} & \\textbf{Porto} & \\textbf{GeoLife} & \\textbf{Macro} & \\textbf{Adjusted $p$} \\\\",
        "\\midrule",
    ]
    for variant in variants:
        values = [ANALYSIS["sensitivity"][variant][dataset]["AUC"]["mean"] for dataset in DATASETS]
        p_text = "reference" if variant == "min8" else latex_p_value(
            comparisons[variant]["p_holm"]
        )
        lines.append(
            f"{display[variant]} & "
            + " & ".join([f"{value:.4f}" for value in values] + [f"{np.mean(values):.4f}", p_text])
            + " \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    write_text(TABLE_DIR / "sensitivity_cross_dataset_table.tex", "\n".join(lines))


def build_local_entropy_bins_table() -> None:
    """Report fixed-bin diagnostics separately from the adaptive default."""
    if LOCAL_ENTROPY is None:
        return
    summary = LOCAL_ENTROPY["summary"]
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:local_entropy_bins}{Sensitivity to the local histogram bin count. The default uses the sample-size-adaptive rule $B_j=\\max\\{2,\\lceil\\log_2(n_j)+1\\rceil\\}$; fixed-bin rows are independent diagnostics with all other settings fixed.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc@{}}",
        "\\toprule",
        "\\textbf{Setting} & \\textbf{Synthetic} & \\textbf{Grid} & \\textbf{Porto} & \\textbf{GeoLife} & \\textbf{Macro AUC} & \\textbf{Macro AUPRC} & \\textbf{Macro F1} \\\\",
        "\\midrule",
    ]
    default = [MAIN[dataset]["EWGB-TAD"] for dataset in DATASETS]
    lines.append(
        "$B_j$ adaptive (default) & "
        + " & ".join(f"{item['AUC']['mean']:.4f}" for item in default)
        + f" & {macro_mean('EWGB-TAD', 'AUC'):.4f} & {macro_mean('EWGB-TAD', 'AUPRC'):.4f} & {macro_mean('EWGB-TAD', 'F1'):.4f} \\\\")
    for bin_count in LOCAL_ENTROPY["bin_counts"]:
        variant = f"bins_{bin_count}"
        values = summary[variant]
        lines.append(
            f"$B_j={bin_count}$ & "
            + " & ".join(f"{values['dataset'][dataset]['AUC']['mean']:.4f}" for dataset in DATASETS)
            + " & "
            + " & ".join(f"{values['macro'][metric]:.4f}" for metric in ("AUC", "AUPRC", "F1"))
            + " \\\\")
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    write_text(TABLE_DIR / "local_entropy_bins_table.tex", "\n".join(lines))


def build_runtime_tables() -> None:
    runtime = pd.read_csv(PHASE4 / "runtime_by_dataset_method.csv")
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:runtime_summary}{End-to-end wall-clock time under the recorded CPU/GPU execution modes. Values are mean $\\pm$ standard deviation in seconds across seeds.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{2.7pt}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llccccc@{}}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Device} & \\textbf{Synthetic} & \\textbf{Grid} & \\textbf{Porto} & \\textbf{GeoLife} & \\textbf{Macro} \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        cells = []
        means = []
        for dataset in DATASETS:
            row = runtime[(runtime.dataset == dataset) & (runtime.method == method)].iloc[0]
            mean = float(row.end_to_end_mean_seconds)
            std = float(row.end_to_end_std_seconds)
            means.append(mean)
            cells.append(f"{mean:.2f} $\\pm$ {std:.2f}")
        device = "GPU" if method in GPU_METHODS else "CPU"
        lines.append(f"{method} & {device} & " + " & ".join(cells + [f"{np.mean(means):.2f}"]) + " \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    write_text(TABLE_DIR / "runtime_summary_table.tex", "\n".join(lines))

    timing = pd.read_csv(PHASE4 / "deep_phase_timing_seed42.csv")
    lines = [
        "\\begin{center}",
        "\\supplementtablecaption[\\textwidth]{tab:supp_deep_timing}{Representative phase timing for all four deep baselines at seed 42. Ranges are minima--maxima over the four datasets; optimization includes validation passes where reported and pretraining for MST-OATD.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{4.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccc@{}}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Prepare (s)} & \\textbf{Optimization (s)} & \\textbf{Score (s)} & \\textbf{End-to-end (s)} & \\textbf{Peak GPU (GiB)} \\\\",
        "\\midrule",
    ]
    for method in ("LSTM-AE", "USAD", "LM-TAD", "MST-OATD"):
        rows = timing[timing.method == method]

        def range_text(column: str) -> str:
            values = rows[column].dropna().astype(float)
            if values.empty:
                return "--"
            return f"{values.min():.2f}--{values.max():.2f}"

        peak = rows.peak_gpu_memory_bytes.astype(float).max() / (1024**3)
        lines.append(
            f"{method} & {range_text('data_preparation_seconds')} & {range_text('optimization_seconds')} & "
            f"{range_text('scoring_seconds')} & {range_text('end_to_end_seconds')} & {peak:.2f} \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    write_text(TABLE_DIR / "supp_deep_timing_table.tex", "\n".join(lines))


def build_deep_protocol_tables() -> None:
    baselines_dl_hash = hashlib.sha256(
        (ROOT / "src" / "baselines_dl.py").read_bytes()
    ).hexdigest().upper()[:12]
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:deep_protocol}{Deep-baseline implementation and model-selection protocols. All normalization statistics and validation criteria are fitted without anomaly labels; larger final scores indicate greater anomaly likelihood.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lp{0.15\\textwidth}p{0.18\\textwidth}p{0.21\\textwidth}p{0.29\\textwidth}@{}}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Code source/version} & \\textbf{Input and normalization} & \\textbf{Objective and initialization} & \\textbf{Search, selection, and stopping} \\\\",
        "\\midrule",
        "LSTM-AE & Local reproducible implementation; \\texttt{baselines\\_dl.py}, SHA-256 \\texttt{CURRENTCODE} & $(N,32,2)$ coordinates; StandardScaler fitted on the 90\\% inner training split & Sequence reconstruction MSE; seeded PyTorch default LSTM/Linear initialization & 12 declared candidates; minimum unlabeled validation MSE on seed 42 per dataset; at most 80 epochs, patience 8, minimum 10 epochs, $\\Delta=10^{-5}$; restore best checkpoint \\\\",
        "USAD & Local reproducible implementation; \\texttt{baselines\\_dl.py}, SHA-256 \\texttt{CURRENTCODE} & $(N,34)$ canonical features; StandardScaler fitted on the 90\\% inner training split & Two-autoencoder USAD reconstruction objective; seeded PyTorch default Linear initialization & 12 declared candidates; minimum unlabeled validation reconstruction score on seed 42 per dataset; same stopping and checkpoint rule as LSTM-AE \\\\",
        "LM-TAD & Official repository, commit \\texttt{80bb89a8} & $\\mathrm{SOT}+32$ tokens$+\\mathrm{EOT}$ from one global $18\\times18$ coordinate grid & Official autoregressive next-token cross-entropy; seeded from-scratch official model initialization & Official architecture and optimizer; deterministic 90/10 unlabeled split; minimum validation cross-entropy checkpoint; fixed 50 epochs, no early stopping \\\\",
        "MST-OATD & Official repository, commit \\texttt{db94b41c} & Global $18\\times18$ grid tokens, four-neighbour adjacency, and deterministic position-derived clock & Official spatial/temporal reconstruction and KL terms; official temporal checkpoint plus seeded initialization & Official defaults; no label-based search or checkpoint selection; 8 pretraining and 10 main-training epochs; no early stopping \\\\",
        "\\botrule",
        "\\end{tabular*}",
        "\\endgroup",
        "\\end{center}",
    ]
    lines = [
        line.replace("CURRENTCODE", baselines_dl_hash)
        for line in lines
    ]
    protocol_lines = [
        r"\begin{center}",
        r"\castablecaption[\textwidth]{tab:deep_protocol}{Deep-baseline protocol summary. Normalization and model selection use no anomaly labels; detailed implementation hashes and selected configurations are provided in Supplementary Table~S1 and the accompanying reproducibility materials.}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lp{0.18\textwidth}p{0.22\textwidth}p{0.44\textwidth}@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Source} & \textbf{Input / normalization} & \textbf{Training and model selection} \\",
        r"\midrule",
        f"LSTM-AE & Local reproducible implementation; SHA-256 \\texttt{{{baselines_dl_hash}}} & $(N,32,2)$ coordinates; inner-split StandardScaler & Reconstruction MSE; seeded PyTorch initialization; 12-candidate dataset-specific search; minimum unlabeled validation MSE; 80-epoch cap, patience 8, best checkpoint restored \\\\",
        f"USAD & Local reproducible implementation; SHA-256 \\texttt{{{baselines_dl_hash}}} & $(N,34)$ trajectory features; inner-split StandardScaler & Two-autoencoder reconstruction objective; seeded PyTorch initialization; equal 12-candidate budget; minimum unlabeled validation loss; same stopping rule \\\\",
        r"LM-TAD & Official repository, commit \texttt{80bb89a8} & $\mathrm{SOT}+32$ grid tokens$+\mathrm{EOT}$ & Official autoregressive objective and initialization; unlabeled validation checkpoint; fixed 50 epochs \\",
        r"MST-OATD & Official repository, commit \texttt{db94b41c} & Grid tokens, adjacency, and position-derived clock & Official reconstruction/KL objectives and initialization; official defaults; 8 pretraining and 10 main-training epochs \\",
        r"\botrule",
        r"\end{tabular*}",
        r"\endgroup",
        r"\end{center}",
    ]
    write_text(TABLE_DIR / "deep_protocol_table.tex", "\n".join(protocol_lines))

    selected_path = PHASE4.parent / "deep_tuning" / "selected_configurations.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    lines = [
        "\\begin{center}",
        "\\supplementtablecaption[\\textwidth]{tab:supp_deep_selected}{Dataset-specific configurations selected for LSTM-AE and USAD by the label-free validation protocol. $h$ and $z$ denote hidden and latent dimensions; LR denotes learning rate; Best/run gives the selected checkpoint epoch and total epochs executed.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llccccccc@{}}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Method} & \\textbf{$h$} & \\textbf{$z$} & \\textbf{Layers} & \\textbf{Dropout} & \\textbf{LR} & \\textbf{Best/run} & \\textbf{Early stop} \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for method in ("LSTM-AE", "USAD"):
            record = selected[dataset][method]
            config = record["configuration"]
            layers = str(config.get("num_layers", "--"))
            dropout = f"{config['dropout']:.1f}" if "dropout" in config else "--"
            lr = f"{config['learning_rate']:.0e}"
            stopped = "yes" if record["early_stopped"] else "no"
            lines.append(
                f"{DATASET_LABELS[dataset]} & {method} & {config['hidden_dim']} & "
                f"{config['latent_dim']} & {layers} & {dropout} & {lr} & "
                f"{record['best_epoch']}/{record['epochs_completed']} & {stopped} \\\\"
            )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    write_text(TABLE_DIR / "supp_deep_selected_configurations_table.tex", "\n".join(lines))


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def build_auc_rank_distribution_figure() -> None:
    configure_plotting()
    seed_level = pd.read_csv(PHASE4 / "main_seed_level.csv")
    auc = seed_level.pivot(index=["dataset", "seed"], columns="method", values="AUC")
    auc = auc.loc[:, list(METHODS)]
    ranks = auc.rank(axis=1, ascending=False, method="average")
    average_ranks = ranks.mean(axis=0).to_dict()
    ordered = sorted(METHODS, key=average_ranks.get)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    values = [ranks[method].to_numpy(dtype=float) for method in ordered]
    positions = np.arange(1, len(ordered) + 1)
    box = ax.boxplot(
        values,
        vert=False,
        positions=positions,
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": "#202020", "linewidth": 1.05},
        whiskerprops={"color": "#777777", "linewidth": 0.75},
        capprops={"color": "#777777", "linewidth": 0.75},
    )
    gpu_methods = {"LSTM-AE", "USAD", "LM-TAD", "MST-OATD"}
    trajectory_cpu = {"Shape-KNN", "SegmentOD", "TADS", "Profile-TAD"}
    family_colors = {
        "EWGB-TAD": "#C43C39",
        "trajectory": "#F2A65A",
        "general": "#77AADD",
        "gpu": "#77B77D",
    }
    for patch, method in zip(box["boxes"], ordered):
        if method == "EWGB-TAD":
            color = family_colors["EWGB-TAD"]
        elif method in gpu_methods:
            color = family_colors["gpu"]
        elif method in trajectory_cpu:
            color = family_colors["trajectory"]
        else:
            color = family_colors["general"]
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.72 if method != "EWGB-TAD" else 0.90)
        patch.set_linewidth(0.8)

    means = np.asarray([average_ranks[method] for method in ordered])
    mean_colors = ["#8F1D1A" if method == "EWGB-TAD" else "#FFFFFF" for method in ordered]
    ax.scatter(
        means,
        positions,
        marker="D",
        s=24,
        c=mean_colors,
        edgecolors="#303030",
        linewidths=0.65,
        zorder=4,
    )
    for y, mean in zip(positions, means):
        ax.text(mean + 0.18, y, f"{mean:.2f}", va="center", ha="left", fontsize=7.0, color="#303030")

    ax.set_yticks(positions)
    ax.set_yticklabels(ordered)
    ax.invert_yaxis()
    ax.set_xlim(0.5, len(METHODS) + 0.85)
    ax.set_xticks(np.arange(1, len(METHODS) + 1, 2))
    ax.set_xlabel("AUC rank within each matched dataset-seed block (1 = best)")
    ax.grid(axis="x", color="#D8D8D8", linestyle=":", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([], [], color=family_colors["EWGB-TAD"], lw=6, label="EWGB-TAD"),
        plt.Line2D([], [], color=family_colors["general"], lw=6, label="General CPU"),
        plt.Line2D([], [], color=family_colors["trajectory"], lw=6, label="Trajectory CPU"),
        plt.Line2D([], [], color=family_colors["gpu"], lw=6, label="Neural GPU"),
        plt.Line2D([], [], color="#303030", marker="D", markerfacecolor="white", linestyle="", label="Mean rank"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        fontsize=7.5,
        columnspacing=1.0,
        handlelength=1.2,
    )
    fig.tight_layout()
    for ext in ("pdf", "png", "svg", "tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 700
        fig.savefig(FIGURE_DIR / f"auc_rank_distribution.{ext}", **kwargs)
    plt.close(fig)


def build_scalability_figure() -> None:
    configure_plotting()
    runtime = pd.read_csv(PHASE4 / "runtime_by_dataset_method.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.35))
    colors = {
        "EWGB-TAD": "#C43C39",
        "general": "#77AADD",
        "trajectory": "#F2A65A",
        "gpu": "#77B77D",
    }
    trajectory_cpu = {"Shape-KNN", "SegmentOD", "TADS", "Profile-TAD"}
    label_offsets = {
        "EWGB-TAD": (7, 5),
        "IForest": (5, 5),
        "ECOD": (5, 5),
        "iBoost-ODE": (5, 5),
        "CoMadOut": (5, 5),
        "Shape-KNN": (5, 5),
        "SegmentOD": (5, 5),
        "TADS": (5, 5),
        "Profile-TAD": (5, 5),
        "LSTM-AE": (5, 5),
        "USAD": (5, 5),
        "LM-TAD": (-5, 5),
        "MST-OATD": (5, 5),
    }
    for method in METHODS:
        rows = runtime[runtime.method == method]
        x = float(rows.end_to_end_mean_seconds.mean())
        y = macro_mean(method, "AUC")
        if method == "EWGB-TAD":
            color, marker, size = colors["EWGB-TAD"], "D", 68
        elif method in GPU_METHODS:
            color, marker, size = colors["gpu"], "s", 48
        elif method in trajectory_cpu:
            color, marker, size = colors["trajectory"], "^", 48
        else:
            color, marker, size = colors["general"], "o", 46
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=size,
            edgecolor="#FFFFFF",
            linewidth=0.7,
            zorder=3,
        )
        dx, dy = label_offsets[method]
        ax.annotate(
            method,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            ha="right" if method == "LM-TAD" else "left",
            va="bottom",
            color="#292929",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Macro mean end-to-end runtime (s, log scale)")
    ax.set_ylabel("Macro AUC")
    ax.set_ylim(0.695, 0.915)
    ax.grid(color="#D8D8D8", linestyle=":", linewidth=0.6, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Line2D([], [], color=colors["EWGB-TAD"], marker="D", linestyle="", label="EWGB-TAD"),
        plt.Line2D([], [], color=colors["general"], marker="o", linestyle="", label="General CPU"),
        plt.Line2D([], [], color=colors["trajectory"], marker="^", linestyle="", label="Trajectory CPU"),
        plt.Line2D([], [], color=colors["gpu"], marker="s", linestyle="", label="Neural GPU"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        handletextpad=0.5,
        columnspacing=1.25,
    )
    fig.tight_layout(pad=0.7)
    for ext in ("pdf", "png", "svg", "tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 700
        fig.savefig(FIGURE_DIR / f"scalability.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    for metric in ("AUC", "AUPRC", "F1"):
        build_metric_table(metric)
    build_auc_pairwise_table()
    build_component_tables()
    build_view_table()
    build_sensitivity_table()
    build_local_entropy_bins_table()
    build_runtime_tables()
    build_deep_protocol_tables()
    build_auc_rank_distribution_figure()
    build_scalability_figure()
    print(f"Phase 5 assets written to {OVERLEAF}")


if __name__ == "__main__":
    main()
