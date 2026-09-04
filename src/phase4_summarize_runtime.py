"""Export seed-level and dataset-level Phase 4 runtime/resource records."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase4_common import (
    DATASET_SEEDS,
    MAIN_METHODS,
    SUMMARY_ROOT,
    atomic_write_json,
    job_directory,
    resolve_marker_artifact,
)


DEEP_METHODS = {"LSTM-AE", "USAD", "LM-TAD", "MST-OATD"}


def load_result(dataset: str, method: str, seed: int) -> dict | None:
    marker_path = job_directory(dataset, method, seed) / "COMPLETE.json"
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    result_path = resolve_marker_artifact(marker_path, marker, "result")
    if result_path is None:
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    missing: list[dict] = []
    for dataset, seeds in DATASET_SEEDS.items():
        for method in MAIN_METHODS:
            for seed in seeds:
                result = load_result(dataset, method, seed)
                if result is None:
                    missing.append({"dataset": dataset, "method": method, "seed": seed})
                    continue
                model = result.get("model_metadata", {})
                cache = result.get("dataset_cache", {})
                adapter_seconds = model.get("runtime_seconds")
                if adapter_seconds is None and method in {"LSTM-AE", "USAD"}:
                    adapter_seconds = result["runtime_seconds"]
                    timing_scope = "fit_score_plus_job_io; pure training and scoring are not separated"
                elif adapter_seconds is not None:
                    timing_scope = "adapter_fit_plus_score; pure training and scoring are not separated"
                else:
                    timing_scope = "end_to_end_job"
                epochs = model.get("epochs_completed", model.get("epochs"))
                pretrain_epochs = model.get("pretrain_epochs_completed")
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "n_samples": cache.get("n_total"),
                        "end_to_end_seconds": result["runtime_seconds"],
                        "fit_score_seconds": adapter_seconds,
                        "timing_scope": timing_scope,
                        "epochs": epochs,
                        "pretrain_epochs": pretrain_epochs,
                        "parameter_count": model.get("parameter_count"),
                        "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"),
                        "device": model.get("device", result.get("environment", {}).get("device_requested")),
                    }
                )

    fields = [
        "dataset",
        "method",
        "seed",
        "n_samples",
        "end_to_end_seconds",
        "fit_score_seconds",
        "timing_scope",
        "epochs",
        "pretrain_epochs",
        "parameter_count",
        "peak_gpu_memory_bytes",
        "device",
    ]
    with (SUMMARY_ROOT / "runtime_seed_level.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    aggregate_rows: list[dict] = []
    aggregate_json: dict = defaultdict(dict)
    for (dataset, method), group in grouped.items():
        total = np.asarray([float(row["end_to_end_seconds"]) for row in group], dtype=float)
        n_samples = np.asarray([float(row["n_samples"]) for row in group], dtype=float)
        throughput = n_samples / total
        fit_score = np.asarray(
            [float(row["fit_score_seconds"]) for row in group if row["fit_score_seconds"] is not None],
            dtype=float,
        )
        gpu_memory = np.asarray(
            [float(row["peak_gpu_memory_bytes"]) for row in group if row["peak_gpu_memory_bytes"] is not None],
            dtype=float,
        )
        record = {
            "dataset": dataset,
            "method": method,
            "n_seeds": len(group),
            "end_to_end_mean_seconds": float(total.mean()),
            "end_to_end_std_seconds": float(total.std(ddof=1)) if len(total) > 1 else 0.0,
            "end_to_end_median_seconds": float(np.median(total)),
            "end_to_end_min_seconds": float(total.min()),
            "end_to_end_max_seconds": float(total.max()),
            "throughput_mean_samples_per_second": float(throughput.mean()),
            "fit_score_mean_seconds": float(fit_score.mean()) if len(fit_score) else None,
            "fit_score_std_seconds": (
                float(fit_score.std(ddof=1)) if len(fit_score) > 1 else 0.0 if len(fit_score) else None
            ),
            "peak_gpu_memory_mean_bytes": float(gpu_memory.mean()) if len(gpu_memory) else None,
            "peak_gpu_memory_max_bytes": float(gpu_memory.max()) if len(gpu_memory) else None,
        }
        aggregate_rows.append(record)
        aggregate_json[dataset][method] = record

    aggregate_fields = list(aggregate_rows[0]) if aggregate_rows else []
    with (SUMMARY_ROOT / "runtime_by_dataset_method.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with (SUMMARY_ROOT / "deep_runtime_seed_level.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row for row in rows if row["method"] in DEEP_METHODS])

    atomic_write_json(SUMMARY_ROOT / "runtime_by_dataset_method.json", aggregate_json)
    atomic_write_json(SUMMARY_ROOT / "runtime_missing_jobs.json", missing)
    deep_timing_status_path = SUMMARY_ROOT / "deep_phase_timing_status.json"
    deep_timing_status = (
        json.loads(deep_timing_status_path.read_text(encoding="utf-8"))
        if deep_timing_status_path.exists()
        else None
    )
    pure_timing_available = bool(
        deep_timing_status
        and deep_timing_status.get("missing_jobs") == 0
        and deep_timing_status.get("all_scores_match_main")
        and deep_timing_status.get("all_metrics_match_main")
    )
    atomic_write_json(
        SUMMARY_ROOT / "runtime_summary_status.json",
        {
            "completed_jobs": len(rows),
            "expected_jobs": sum(len(seeds) for seeds in DATASET_SEEDS.values()) * len(MAIN_METHODS),
            "missing_jobs": len(missing),
            "pure_deep_training_time_available": pure_timing_available,
            "deep_phase_timing_seed": deep_timing_status.get("timing_seed") if deep_timing_status else None,
            "follow_up_required": (
                None
                if pure_timing_available
                else "Run the fixed post-main deep timing profile to separate data preparation, training, and scoring."
            ),
        },
    )
    print(f"runtime rows={len(rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
