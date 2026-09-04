"""Evaluate fixed local-entropy bin counts as a separate sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation_protocol import evaluate_oracle_top_k
from phase4_analysis_models import VIEW_NAMES, fit_score_variant
from phase4_common import (
    DATASET_SEEDS,
    atomic_write_json,
    canonical_shape_view,
    load_cache,
    set_reproducible,
    sha256_file,
    slug,
)


BIN_COUNTS = (4, 8, 16, 32)
OUTPUT_ROOT = Path(
    os.environ.get(
        "EWGB_LOCAL_ENTROPY_SENSITIVITY_ROOT",
        str(ROOT / "paper_work" / "final_neurocomputing_results" / "sensitivity_stability" / "local_entropy_bins"),
    )
)


def job_root(bin_count: int, dataset: str, seed: int) -> Path:
    return OUTPUT_ROOT / "raw" / f"bins_{bin_count}" / slug(dataset) / f"seed_{seed}"


def job_key(bin_count: int, dataset: str, seed: int) -> tuple[int, str, int]:
    return bin_count, dataset, seed


def read_completed(bin_count: int, dataset: str, seed: int) -> dict | None:
    root = job_root(bin_count, dataset, seed)
    marker_path = root / "COMPLETE.json"
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    attempt = marker.get("attempt")
    if not attempt:
        return None
    result_path = root / str(attempt) / "result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def run_job(bin_count: int, dataset: str, seed: int, force: bool = False) -> dict:
    existing = None if force else read_completed(bin_count, dataset, seed)
    if existing is not None:
        return row_from_result(existing)

    root = job_root(bin_count, dataset, seed)
    attempt = root / (
        f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    set_reproducible(seed, torch_threads=1)
    try:
        arrays, cache_metadata = load_cache(dataset, seed)
        views = {
            "spatial_path": np.asarray(arrays["spatial_path"], dtype=np.float32),
            "kinematic": np.asarray(arrays["kinematic"], dtype=np.float32),
            "trajectory_shape": canonical_shape_view(arrays["trajectories"]),
        }
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        scores, model_stats = fit_score_variant(
            views,
            selected_views=VIEW_NAMES,
            seed=seed,
            granular_partition=True,
            local_metric=True,
            fusion="entropy",
            min_samples=8,
            entropy_fixed_bins=bin_count,
        )
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise RuntimeError("local entropy sensitivity scores are invalid")
        metrics = evaluate_oracle_top_k(labels, scores)
        score_path = attempt / "scores.npz"
        with score_path.open("wb") as handle:
            np.savez_compressed(handle, scores=scores)
        result = {
            "status": "complete",
            "analysis": "local_entropy_bins",
            "variant": f"bins_{bin_count}",
            "bin_count": bin_count,
            "dataset": dataset,
            "seed": seed,
            "metrics": metrics,
            "runtime_seconds": time.perf_counter() - started,
            "labels_consumed_during_fit": False,
            "dataset_cache": cache_metadata,
            "score_sha256": sha256_file(score_path),
            "model_stats": model_stats,
        }
        result_path = attempt / "result.json"
        atomic_write_json(result_path, result)
        atomic_write_json(
            root / "COMPLETE.json",
            {
                "status": "complete",
                "attempt": attempt.name,
                "result": f"{attempt.name}/result.json",
                "score": f"{attempt.name}/scores.npz",
                "result_sha256": sha256_file(result_path),
            },
        )
        return row_from_result(result)
    except Exception:
        raise


def row_from_result(result: dict) -> dict:
    return {
        "bin_count": int(result["bin_count"]),
        "dataset": result["dataset"],
        "seed": int(result["seed"]),
        **{metric: float(result["metrics"][metric]) for metric in ("AUC", "AUPRC", "F1", "Precision", "Recall")},
        "Runtime": float(result["runtime_seconds"]),
        "labels_consumed_during_fit": bool(result["labels_consumed_during_fit"]),
    }


def expected_jobs() -> list[tuple[int, str, int]]:
    return [
        job_key(bin_count, dataset, seed)
        for bin_count in BIN_COUNTS
        for dataset, seeds in DATASET_SEEDS.items()
        for seed in seeds
    ]


def write_summary(rows: list[dict], errors: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["bin_count"], row["dataset"], row["seed"]))
    fields = ["bin_count", "dataset", "seed", "AUC", "AUPRC", "F1", "Precision", "Recall", "Runtime", "labels_consumed_during_fit"]
    with (OUTPUT_ROOT / "local_entropy_bins_seed_level.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict] = {}
    for bin_count in BIN_COUNTS:
        variant = f"bins_{bin_count}"
        summary[variant] = {"dataset": {}, "macro": {}}
        for dataset in DATASET_SEEDS:
            subset = [row for row in rows if row["bin_count"] == bin_count and row["dataset"] == dataset]
            summary[variant]["dataset"][dataset] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in subset])),
                    "std": float(np.std([row[metric] for row in subset], ddof=1)),
                    "n": len(subset),
                }
                for metric in ("AUC", "AUPRC", "F1")
            }
        for metric in ("AUC", "AUPRC", "F1"):
            values = [summary[variant]["dataset"][dataset][metric]["mean"] for dataset in DATASET_SEEDS]
            summary[variant]["macro"][metric] = float(np.mean(values))
    atomic_write_json(
        OUTPUT_ROOT / "local_entropy_bins_summary.json",
        {
            "analysis": "local_entropy_bins",
            "estimator": "fixed local histogram bins; all other EWGB-TAD settings fixed",
            "bin_counts": list(BIN_COUNTS),
            "expected_jobs": len(expected_jobs()),
            "completed_jobs": len(rows),
            "missing_jobs": len(errors),
            "errors": errors,
            "summary": summary,
        },
    )
    atomic_write_json(
        OUTPUT_ROOT / "local_entropy_bins_status.json",
        {
            "expected_jobs": len(expected_jobs()),
            "completed_jobs": len(rows),
            "missing_jobs": len(errors),
            "ready_for_reporting": not errors and len(rows) == len(expected_jobs()),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="recompute completed jobs")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    errors: list[dict] = []
    jobs = expected_jobs()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, *job, args.force): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({
                    "bin_count": job[0],
                    "dataset": job[1],
                    "seed": job[2],
                    "error": repr(exc),
                })
                print(f"FAILED bins={job[0]} {job[1]} seed={job[2]}: {exc}", flush=True)
    write_summary(rows, errors)
    print(f"completed={len(rows)} missing={len(errors)}", flush=True)
    return 0 if not errors and len(rows) == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
