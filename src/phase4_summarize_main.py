"""Aggregate immutable Phase 4 main-benchmark outputs and statistics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from phase4_common import (
    DATASET_SEEDS,
    MAIN_METHODS,
    SUMMARY_ROOT,
    atomic_write_json,
    job_directory,
    resolve_marker_artifact,
)


def holm_adjust(pairs: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pairs, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - index) * p_value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def load_complete(dataset: str, method: str, seed: int) -> dict | None:
    complete_path = job_directory(dataset, method, seed) / "COMPLETE.json"
    if not complete_path.exists():
        return None
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    result_path = resolve_marker_artifact(complete_path, complete, "result")
    if result_path is None:
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def paired_statistics(matrix: np.ndarray, metric: str, block_label: str) -> dict:
    friedman = friedmanchisquare(*[matrix[:, index] for index in range(matrix.shape[1])])
    ranks = np.vstack([rankdata(-row, method="average") for row in matrix])
    ewgb = matrix[:, MAIN_METHODS.index("EWGB-TAD")]
    pairwise: list[tuple[str, float]] = []
    pair_details: dict[str, dict] = {}
    for method in MAIN_METHODS:
        if method == "EWGB-TAD":
            continue
        values = matrix[:, MAIN_METHODS.index(method)]
        differences = ewgb - values
        try:
            test = wilcoxon(ewgb, values, alternative="two-sided", zero_method="wilcox")
            p_value = float(test.pvalue)
            statistic = float(test.statistic)
        except ValueError:
            p_value = 1.0
            statistic = 0.0
        pairwise.append((method, p_value))
        tolerance = 1e-12
        pair_details[method] = {
            "wilcoxon_statistic": statistic,
            "p_raw": p_value,
            "mean_difference": float(np.mean(differences)),
            "median_difference": float(np.median(differences)),
            "wins": int(np.sum(differences > tolerance)),
            "ties": int(np.sum(np.abs(differences) <= tolerance)),
            "losses": int(np.sum(differences < -tolerance)),
        }
    adjusted = holm_adjust(pairwise)
    for method, value in adjusted.items():
        pair_details[method]["p_holm"] = value
    return {
        "metric": metric,
        "block_definition": block_label,
        "n_blocks": int(matrix.shape[0]),
        "friedman_statistic": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "degrees_of_freedom": len(MAIN_METHODS) - 1,
        "average_ranks": {
            method: float(ranks[:, index].mean())
            for index, method in enumerate(MAIN_METHODS)
        },
        "ewgb_pairwise_wilcoxon_holm": pair_details,
    }


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    missing: list[dict] = []
    for dataset, seeds in DATASET_SEEDS.items():
        for method in MAIN_METHODS:
            for seed in seeds:
                result = load_complete(dataset, method, seed)
                if result is None:
                    missing.append({"dataset": dataset, "method": method, "seed": seed})
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        **result["metrics"],
                        "Runtime": result["runtime_seconds"],
                    }
                )

    seed_csv = SUMMARY_ROOT / "main_seed_level.csv"
    with seed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "method",
                "seed",
                "AUC",
                "AUPRC",
                "F1",
                "Precision",
                "Recall",
                "Runtime",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        for metric in ("AUC", "AUPRC", "F1", "Precision", "Recall", "Runtime"):
            grouped[row["dataset"]][row["method"]][metric].append(float(row[metric]))

    aggregate: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for dataset, methods in grouped.items():
        aggregate[dataset] = {}
        for method, metrics in methods.items():
            aggregate[dataset][method] = {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
                for metric, values in metrics.items()
            }

    statistics: dict[str, dict] = {}
    if not missing:
        row_lookup = {
            (row["dataset"], int(row["seed"]), row["method"]): row
            for row in rows
        }
        seed_blocks = [
            (dataset, seed)
            for dataset, seeds in DATASET_SEEDS.items()
            for seed in seeds
        ]
        for metric in ("AUC", "AUPRC", "F1"):
            seed_matrix = np.array(
                [
                    [row_lookup[(dataset, seed, method)][metric] for method in MAIN_METHODS]
                    for dataset, seed in seed_blocks
                ],
                dtype=float,
            )
            dataset_mean_matrix = np.array(
                [
                    [aggregate[dataset][method][metric]["mean"] for method in MAIN_METHODS]
                    for dataset in DATASET_SEEDS
                ],
                dtype=float,
            )
            statistics[metric] = {
                "paired_seed_blocks": paired_statistics(
                    seed_matrix,
                    metric,
                    "30 matched dataset-seed runs; method scores share each predefined cache",
                ),
                "dataset_mean_blocks": paired_statistics(
                    dataset_mean_matrix,
                    metric,
                    "4 dataset-level means; included as a descriptive dataset-generalization check",
                ),
                "per_dataset_seed_blocks": {
                    dataset: paired_statistics(
                        np.array(
                            [
                                [row_lookup[(dataset, seed, method)][metric] for method in MAIN_METHODS]
                                for seed in seeds
                            ],
                            dtype=float,
                        ),
                        metric,
                        f"{dataset} matched seeds",
                    )
                    for dataset, seeds in DATASET_SEEDS.items()
                },
            }

    atomic_write_json(SUMMARY_ROOT / "main_aggregate.json", aggregate)
    atomic_write_json(SUMMARY_ROOT / "main_statistics.json", statistics)
    atomic_write_json(SUMMARY_ROOT / "main_missing_jobs.json", missing)
    atomic_write_json(
        SUMMARY_ROOT / "main_summary_status.json",
        {
            "completed_jobs": len(rows),
            "expected_jobs": sum(len(seeds) for seeds in DATASET_SEEDS.values()) * len(MAIN_METHODS),
            "missing_jobs": len(missing),
            "statistics_ready": not missing,
        },
    )
    print(f"completed={len(rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
