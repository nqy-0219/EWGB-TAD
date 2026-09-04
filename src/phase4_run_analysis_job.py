"""Run one immutable Phase 4 component, view, or sensitivity job."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation_protocol import evaluate_oracle_top_k
from phase4_analysis_models import VIEW_NAMES, fit_score_variant
from phase4_common import (
    DATASET_SEEDS,
    PHASE4_ROOT,
    atomic_write_json,
    canonical_shape_view,
    load_cache,
    set_reproducible,
    sha256_file,
    slug,
)


def analysis_job_dir(analysis: str, variant: str, dataset: str, seed: int) -> Path:
    return PHASE4_ROOT / "analysis_raw" / slug(analysis) / slug(variant) / slug(dataset) / f"seed_{seed}"


def parse_variant(analysis: str, variant: str) -> dict:
    if analysis == "component":
        tokens = variant.split("_")
        if len(tokens) != 3:
            raise ValueError(f"invalid component variant: {variant}")
        return {
            "selected_views": VIEW_NAMES,
            "granular_partition": tokens[0] == "g1",
            "local_metric": tokens[1] == "l1",
            "fusion": "entropy" if tokens[2] == "f1" else "equal",
            "min_samples": 8,
        }
    if analysis == "view":
        selected = tuple(variant.split("+"))
        if not selected or any(name not in VIEW_NAMES for name in selected):
            raise ValueError(f"invalid view variant: {variant}")
        return {
            "selected_views": selected,
            "granular_partition": True,
            "local_metric": True,
            "fusion": "entropy",
            "min_samples": 8,
        }
    if analysis == "sensitivity":
        if not variant.startswith("min"):
            raise ValueError(f"invalid sensitivity variant: {variant}")
        return {
            "selected_views": VIEW_NAMES,
            "granular_partition": True,
            "local_metric": True,
            "fusion": "entropy",
            "min_samples": int(variant.removeprefix("min")),
        }
    raise ValueError(f"unknown analysis: {analysis}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=("component", "view", "sensitivity"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset", choices=list(DATASET_SEEDS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed not in DATASET_SEEDS[args.dataset]:
        raise ValueError("seed is outside the predefined protocol")

    job_dir = analysis_job_dir(args.analysis, args.variant, args.dataset, args.seed)
    complete_path = job_dir / "COMPLETE.json"
    if complete_path.exists() and not args.force:
        print(f"complete; skip {args.analysis} {args.variant} {args.dataset} {args.seed}")
        return 0
    attempt = job_dir / f"attempt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    set_reproducible(args.seed, torch_threads=1)
    try:
        arrays, cache_metadata = load_cache(args.dataset, args.seed)
        views = {
            "spatial_path": np.asarray(arrays["spatial_path"], dtype=np.float32),
            "kinematic": np.asarray(arrays["kinematic"], dtype=np.float32),
            "trajectory_shape": canonical_shape_view(arrays["trajectories"]),
        }
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        settings = parse_variant(args.analysis, args.variant)
        scores, model_stats = fit_score_variant(views, seed=args.seed, **settings)
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise RuntimeError("analysis scores are invalid")
        metrics = evaluate_oracle_top_k(labels, scores)
        source_dir = Path(__file__).resolve().parent
        implementation_hashes = {
            name: sha256_file(source_dir / name)
            for name in (
                "granular_ball.py",
                "phase4_analysis_models.py",
                "phase4_common.py",
                "phase4_run_analysis_job.py",
            )
        }
        score_path = attempt / "scores.npz"
        with score_path.open("wb") as handle:
            np.savez_compressed(handle, scores=scores)
        result = {
            "status": "complete",
            "analysis": args.analysis,
            "variant": args.variant,
            "dataset": args.dataset,
            "seed": args.seed,
            "metrics": metrics,
            "runtime_seconds": time.perf_counter() - started,
            "labels_consumed_during_fit": False,
            "dataset_cache": cache_metadata,
            "implementation": {
                "name": "EWGB-TAD current component protocol",
                "source_hashes_sha256": implementation_hashes,
                "shape_view": "canonical_shape_view from phase4_common.py",
            },
            "score_sha256": sha256_file(score_path),
            "model_stats": model_stats,
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
            f"COMPLETE {args.analysis} {args.variant} {args.dataset} seed={args.seed} "
            f"AUC={metrics['AUC']:.6f}",
            flush=True,
        )
        return 0
    except Exception as exc:
        atomic_write_json(
            attempt / "ERROR.json",
            {
                "status": "failed",
                "analysis": args.analysis,
                "variant": args.variant,
                "dataset": args.dataset,
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
