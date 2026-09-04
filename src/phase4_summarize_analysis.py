"""Aggregate Phase 4 component, view-complementarity, and sensitivity jobs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from phase4_build_analysis_manifests import (
    COMPONENT_VARIANTS,
    SENSITIVITY_VARIANTS,
    VIEW_VARIANTS,
)
from phase4_common import (
    DATASET_SEEDS,
    SUMMARY_ROOT,
    atomic_write_json,
    resolve_marker_artifact,
)
from phase4_run_analysis_job import analysis_job_dir


ANALYSIS_VARIANTS = {
    "component": COMPONENT_VARIANTS,
    "view": VIEW_VARIANTS,
    "sensitivity": SENSITIVITY_VARIANTS,
}


def holm_adjust(pairs: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pairs, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * p_value))
        adjusted[name] = running
    return adjusted


def load_complete(analysis: str, variant: str, dataset: str, seed: int) -> dict | None:
    marker_path = analysis_job_dir(analysis, variant, dataset, seed) / "COMPLETE.json"
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    result_path = resolve_marker_artifact(marker_path, marker, "result")
    if result_path is None:
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def paired_variant_statistics(
    matrix: np.ndarray,
    variants: list[str],
    reference: str,
    metric: str,
) -> dict:
    friedman = friedmanchisquare(*[matrix[:, index] for index in range(matrix.shape[1])])
    ranks = np.vstack([rankdata(-row, method="average") for row in matrix])
    reference_values = matrix[:, variants.index(reference)]
    pairwise: list[tuple[str, float]] = []
    details: dict[str, dict] = {}
    for variant in variants:
        if variant == reference:
            continue
        values = matrix[:, variants.index(variant)]
        differences = reference_values - values
        try:
            test = wilcoxon(reference_values, values, alternative="two-sided", zero_method="wilcox")
            p_value = float(test.pvalue)
            statistic = float(test.statistic)
        except ValueError:
            p_value = 1.0
            statistic = 0.0
        pairwise.append((variant, p_value))
        details[variant] = {
            "wilcoxon_statistic": statistic,
            "p_raw": p_value,
            "mean_difference": float(np.mean(differences)),
            "median_difference": float(np.median(differences)),
            "wins": int(np.sum(differences > 1e-12)),
            "ties": int(np.sum(np.abs(differences) <= 1e-12)),
            "losses": int(np.sum(differences < -1e-12)),
        }
    for variant, p_value in holm_adjust(pairwise).items():
        details[variant]["p_holm"] = p_value
    return {
        "metric": metric,
        "n_blocks": int(matrix.shape[0]),
        "reference": reference,
        "friedman_statistic": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "average_ranks": {
            variant: float(ranks[:, index].mean())
            for index, variant in enumerate(variants)
        },
        "reference_pairwise_wilcoxon_holm": details,
    }


def component_contrasts(lookup: dict, metric: str, blocks: list[tuple[str, int]]) -> dict:
    """Estimate all seven orthogonal contrasts of the 2x2x2 design."""
    effects: dict[str, list[float]] = {
        "granular_partition": [],
        "local_metric": [],
        "entropy_fusion": [],
        "granular_x_local": [],
        "granular_x_fusion": [],
        "local_x_fusion": [],
        "granular_x_local_x_fusion": [],
    }
    for dataset, seed in blocks:
        def value(granular: int, local: int, fusion: int) -> float:
            return float(
                lookup[
                    ("component", f"g{granular}_l{local}_f{fusion}", dataset, seed)
                ][metric]
            )

        effects["granular_partition"].append(
            float(np.mean([value(1, local, fusion) - value(0, local, fusion)
                           for local in (0, 1) for fusion in (0, 1)]))
        )
        effects["local_metric"].append(
            float(np.mean([value(granular, 1, fusion) - value(granular, 0, fusion)
                           for granular in (0, 1) for fusion in (0, 1)]))
        )
        effects["entropy_fusion"].append(
            float(np.mean([value(granular, local, 1) - value(granular, local, 0)
                           for granular in (0, 1) for local in (0, 1)]))
        )
        effects["granular_x_local"].append(
            float(np.mean([
                (value(1, 1, fusion) - value(1, 0, fusion))
                - (value(0, 1, fusion) - value(0, 0, fusion))
                for fusion in (0, 1)
            ]))
        )
        effects["granular_x_fusion"].append(
            float(np.mean([
                (value(1, local, 1) - value(1, local, 0))
                - (value(0, local, 1) - value(0, local, 0))
                for local in (0, 1)
            ]))
        )
        effects["local_x_fusion"].append(
            float(np.mean([
                (value(granular, 1, 1) - value(granular, 1, 0))
                - (value(granular, 0, 1) - value(granular, 0, 0))
                for granular in (0, 1)
            ]))
        )
        granular_local_at_fusion = []
        for fusion in (0, 1):
            granular_local_at_fusion.append(
                (value(1, 1, fusion) - value(1, 0, fusion))
                - (value(0, 1, fusion) - value(0, 0, fusion))
            )
        effects["granular_x_local_x_fusion"].append(
            float(granular_local_at_fusion[1] - granular_local_at_fusion[0])
        )
    output: dict[str, dict] = {}
    p_values: list[tuple[str, float]] = []
    for name, values in effects.items():
        value_array = np.asarray(values, dtype=float)
        try:
            test = wilcoxon(value_array, alternative="two-sided", zero_method="wilcox")
            p_value = float(test.pvalue)
            statistic = float(test.statistic)
        except ValueError:
            p_value = 1.0
            statistic = 0.0
        p_values.append((name, p_value))
        output[name] = {
            "mean_paired_effect": float(np.mean(values)),
            "median_paired_effect": float(np.median(values)),
            "positive_blocks": int(np.sum(value_array > 1e-12)),
            "zero_blocks": int(np.sum(np.abs(value_array) <= 1e-12)),
            "negative_blocks": int(np.sum(value_array < -1e-12)),
            "n_blocks": len(values),
            "wilcoxon_statistic": statistic,
            "p_raw": p_value,
        }
    for name, p_value in holm_adjust(p_values).items():
        output[name]["p_holm"] = p_value
    return {
        "contrast_coding": "high-minus-low marginal contrasts; interactions are differences of simple effects",
        "holm_family_size": len(effects),
        "effects": output,
    }


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    missing: list[dict] = []
    for analysis, variants in ANALYSIS_VARIANTS.items():
        for dataset, seeds in DATASET_SEEDS.items():
            for seed in seeds:
                for variant in variants:
                    result = load_complete(analysis, variant, dataset, seed)
                    if result is None:
                        missing.append(
                            {"analysis": analysis, "variant": variant, "dataset": dataset, "seed": seed}
                        )
                        continue
                    rows.append(
                        {
                            "analysis": analysis,
                            "variant": variant,
                            "dataset": dataset,
                            "seed": seed,
                            **result["metrics"],
                            "Runtime": result["runtime_seconds"],
                        }
                    )

    with (SUMMARY_ROOT / "analysis_seed_level.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "analysis",
                "variant",
                "dataset",
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

    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for row in rows:
        for metric in ("AUC", "AUPRC", "F1", "Precision", "Recall", "Runtime"):
            grouped[row["analysis"]][row["variant"]][row["dataset"]][metric].append(float(row[metric]))
    aggregate = {
        analysis: {
            variant: {
                dataset: {
                    metric: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "n": len(values),
                    }
                    for metric, values in metrics.items()
                }
                for dataset, metrics in datasets.items()
            }
            for variant, datasets in variants.items()
        }
        for analysis, variants in grouped.items()
    }

    statistics: dict = {}
    if not missing:
        lookup = {
            (row["analysis"], row["variant"], row["dataset"], int(row["seed"])): row
            for row in rows
        }
        blocks = [(dataset, seed) for dataset, seeds in DATASET_SEEDS.items() for seed in seeds]
        references = {
            "component": "g1_l1_f1",
            "view": "spatial_path+kinematic+trajectory_shape",
            "sensitivity": "min8",
        }
        for analysis, variants in ANALYSIS_VARIANTS.items():
            statistics[analysis] = {}
            for metric in ("AUC", "AUPRC", "F1"):
                matrix = np.array(
                    [
                        [lookup[(analysis, variant, dataset, seed)][metric] for variant in variants]
                        for dataset, seed in blocks
                    ],
                    dtype=float,
                )
                statistics[analysis][metric] = paired_variant_statistics(
                    matrix, variants, references[analysis], metric
                )
                if analysis == "component":
                    statistics[analysis][metric]["factorial_contrasts"] = component_contrasts(
                        lookup, metric, blocks
                    )

    atomic_write_json(SUMMARY_ROOT / "analysis_aggregate.json", aggregate)
    atomic_write_json(SUMMARY_ROOT / "analysis_statistics.json", statistics)
    atomic_write_json(SUMMARY_ROOT / "analysis_missing_jobs.json", missing)
    atomic_write_json(
        SUMMARY_ROOT / "analysis_summary_status.json",
        {
            "completed_jobs": len(rows),
            "expected_jobs": sum(
                len(variants) * sum(len(seeds) for seeds in DATASET_SEEDS.values())
                for variants in ANALYSIS_VARIANTS.values()
            ),
            "missing_jobs": len(missing),
            "statistics_ready": not missing,
        },
    )
    print(f"completed={len(rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
