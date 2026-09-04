"""Run one representative deep-model timing job without altering main results."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from evaluation_protocol import evaluate_oracle_top_k
from phase4_common import (
    DATASET_SEEDS,
    PHASE4_ROOT,
    atomic_write_json,
    environment_record,
    job_directory,
    load_cache,
    resolve_marker_artifact,
    set_reproducible,
    sha256_file,
    slug,
)
from phase4_run_job import run_method


DEEP_METHODS = ("LSTM-AE", "USAD", "LM-TAD", "MST-OATD")


def timing_directory(dataset: str, method: str, seed: int) -> Path:
    return PHASE4_ROOT / "deep_timing_raw" / slug(dataset) / slug(method) / f"seed_{seed}"


def load_main_scores(dataset: str, method: str, seed: int) -> tuple[np.ndarray, dict]:
    marker_path = job_directory(dataset, method, seed) / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    result_path = resolve_marker_artifact(marker_path, marker, "result")
    score_path = resolve_marker_artifact(marker_path, marker, "score")
    if result_path is None or score_path is None:
        raise FileNotFoundError(f"missing main artifacts for {dataset}/{method}/{seed}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(score_path) as archive:
        scores = np.asarray(archive["scores"], dtype=np.float64)
    return scores, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_SEEDS), required=True)
    parser.add_argument("--method", choices=DEEP_METHODS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed not in DATASET_SEEDS[args.dataset]:
        raise ValueError("timing seed is outside the predefined dataset protocol")

    job_dir = timing_directory(args.dataset, args.method, args.seed)
    complete_path = job_dir / "COMPLETE.json"
    if complete_path.exists() and not args.force:
        print(f"timing complete; skip {args.dataset} {args.method} seed={args.seed}")
        return 0
    attempt = job_dir / f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    set_reproducible(args.seed, torch_threads=2)
    timing_device = torch.device(args.device)
    if torch.cuda.is_available() and timing_device.type == "cuda":
        torch.cuda.set_device(timing_device)
        torch.cuda.reset_peak_memory_stats(timing_device)

    try:
        arrays, cache_metadata = load_cache(args.dataset, args.seed)
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        scores, model_metadata = run_method(
            args.method, arrays, args.seed, args.device, "full", args.dataset
        )
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise RuntimeError("timing rerun returned invalid scores")
        metrics = evaluate_oracle_top_k(labels, scores)
        score_path = attempt / "scores.npz"
        with score_path.open("wb") as handle:
            np.savez_compressed(handle, scores=scores)
        main_scores, main_result = load_main_scores(args.dataset, args.method, args.seed)
        max_abs_difference = float(np.max(np.abs(scores - main_scores)))
        score_match_main = bool(np.array_equal(scores, main_scores))
        result = {
            "status": "complete",
            "purpose": "representative_phase_timing",
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "metrics": metrics,
            "end_to_end_seconds": time.perf_counter() - started,
            "model_metadata": model_metadata,
            "dataset_cache": cache_metadata,
            "score_sha256": sha256_file(score_path),
            "main_score_sha256": main_result["score_sha256"],
            "score_match_main": score_match_main,
            "max_abs_score_difference": max_abs_difference,
            "metric_match_main": metrics == main_result["metrics"],
            "labels_consumed_during_fit": False,
            "environment": environment_record(args.device),
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(timing_device))
                if torch.cuda.is_available() and timing_device.type == "cuda"
                else None
            ),
            "timing_device": str(timing_device),
        }
        result_path = attempt / "result.json"
        atomic_write_json(result_path, result)
        atomic_write_json(
            complete_path,
            {
                "status": "complete",
                "attempt": attempt.name,
                "result": f"{attempt.name}/result.json",
                "score": f"{attempt.name}/scores.npz",
                "result_sha256": sha256_file(result_path),
            },
        )
        print(
            f"TIMING_COMPLETE {args.dataset} {args.method} seed={args.seed} "
            f"time={result['end_to_end_seconds']:.2f}s score_match_main={score_match_main}",
            flush=True,
        )
        return 0
    except Exception as exc:
        atomic_write_json(
            attempt / "ERROR.json",
            {
                "status": "failed",
                "dataset": args.dataset,
                "method": args.method,
                "seed": args.seed,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - started,
            },
        )
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
