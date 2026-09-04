"""Summarize representative phase timings for all deep baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from phase4_common import DATASET_SEEDS, SUMMARY_ROOT, atomic_write_json, resolve_marker_artifact
from phase4_run_deep_timing import timing_directory


TIMED_METHODS = ("LSTM-AE", "USAD", "LM-TAD", "MST-OATD")


FIELDS = [
    "dataset",
    "method",
    "seed",
    "n_samples",
    "data_preparation_seconds",
    "model_setup_seconds",
    "pretraining_seconds",
    "training_seconds",
    "validation_seconds",
    "training_validation_seconds",
    "optimization_seconds",
    "checkpoint_restore_seconds",
    "scoring_seconds",
    "fit_score_seconds",
    "end_to_end_seconds",
    "epochs",
    "pretrain_epochs",
    "parameter_count",
    "peak_gpu_memory_bytes",
    "score_match_main",
    "max_abs_score_difference",
    "metric_match_main",
]


def load_result(dataset: str, method: str, seed: int) -> dict | None:
    marker_path = timing_directory(dataset, method, seed) / "COMPLETE.json"
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    result_path = resolve_marker_artifact(marker_path, marker, "result")
    if result_path is None:
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def main() -> None:
    rows: list[dict] = []
    missing: list[dict] = []
    seed = 42
    for dataset in DATASET_SEEDS:
        for method in TIMED_METHODS:
            result = load_result(dataset, method, seed)
            if result is None:
                missing.append({"dataset": dataset, "method": method, "seed": seed})
                continue
            model = result["model_metadata"]
            pretraining_seconds = model.get("pretraining_seconds")
            training_seconds = model.get("training_seconds")
            training_validation_seconds = model.get("training_validation_seconds")
            if training_validation_seconds is not None:
                optimization_seconds = training_validation_seconds
            elif pretraining_seconds is not None:
                optimization_seconds = float(pretraining_seconds) + float(training_seconds or 0.0)
            else:
                optimization_seconds = training_seconds
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "n_samples": result["dataset_cache"]["n_total"],
                    "data_preparation_seconds": model.get("data_preparation_seconds"),
                    "model_setup_seconds": model.get("model_setup_seconds"),
                    "pretraining_seconds": pretraining_seconds,
                    "training_seconds": training_seconds,
                    "validation_seconds": model.get("validation_seconds"),
                    "training_validation_seconds": training_validation_seconds,
                    "optimization_seconds": optimization_seconds,
                    "checkpoint_restore_seconds": model.get("checkpoint_restore_seconds"),
                    "scoring_seconds": model.get("scoring_seconds"),
                    "fit_score_seconds": model.get(
                        "total_fit_score_seconds", model.get("fit_score_seconds", model.get("runtime_seconds"))
                    ),
                    "end_to_end_seconds": result["end_to_end_seconds"],
                    "epochs": model.get("epochs_completed", model.get("epochs")),
                    "pretrain_epochs": model.get("pretrain_epochs_completed"),
                    "parameter_count": model.get("parameter_count"),
                    "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"),
                    "score_match_main": result["score_match_main"],
                    "max_abs_score_difference": result["max_abs_score_difference"],
                    "metric_match_main": result["metric_match_main"],
                }
            )

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    with (SUMMARY_ROOT / "deep_phase_timing_seed42.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(SUMMARY_ROOT / "deep_phase_timing_seed42.json", rows)
    atomic_write_json(SUMMARY_ROOT / "deep_phase_timing_missing.json", missing)
    atomic_write_json(
        SUMMARY_ROOT / "deep_phase_timing_status.json",
        {
            "completed_jobs": len(rows),
            "expected_jobs": len(DATASET_SEEDS) * len(TIMED_METHODS),
            "missing_jobs": len(missing),
            "all_scores_match_main": bool(rows) and all(row["score_match_main"] for row in rows),
            "all_metrics_match_main": bool(rows) and all(row["metric_match_main"] for row in rows),
            "timing_seed": seed,
        },
    )

    lines = [
        "# Deep-baseline phase timing (representative seed 42)",
        "",
        "Data preparation, training, and scoring are measured separately for all four deep baselines under their formal dataset-specific configurations.",
        "",
        "| Dataset | Method | Prep (s) | Pretrain (s) | Train (s) | Validation (s) | Optimization (s) | Score (s) | End-to-end (s) | Peak GPU (GiB) | Main score exact |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def value(name: str) -> str:
            item = row[name]
            return "-" if item is None else f"{float(item):.3f}"

        peak_gib = (
            "-"
            if row["peak_gpu_memory_bytes"] is None
            else f"{float(row['peak_gpu_memory_bytes']) / (1024 ** 3):.3f}"
        )
        lines.append(
            f"| {row['dataset']} | {row['method']} | {value('data_preparation_seconds')} | "
            f"{value('pretraining_seconds')} | {value('training_seconds')} | "
            f"{value('validation_seconds')} | {value('optimization_seconds')} | "
            f"{value('scoring_seconds')} | "
            f"{value('end_to_end_seconds')} | {peak_gib} | {row['score_match_main']} |"
        )
    (SUMMARY_ROOT / "deep_phase_timing_seed42.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"deep timing completed={len(rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
