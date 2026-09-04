"""Validate immutable Phase 4 caches and completed experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase4_common import (
    CACHE_ROOT,
    DATASET_SEEDS,
    PHASE4_ROOT,
    cache_metadata_path,
    cache_path,
    resolve_marker_artifact,
    sha256_file,
)


EXPECTED_COUNTS = {
    "Synthetic": (5552, 5000, 552),
    "Grid-Network": (5552, 5000, 552),
    "Porto-derived": (5552, 5000, 552),
    "GeoLife": (3332, 3000, 332),
}


def validate_caches() -> list[str]:
    errors: list[str] = []
    expected_jobs = sum(len(seeds) for seeds in DATASET_SEEDS.values())
    observed_jobs = 0
    for dataset, seeds in DATASET_SEEDS.items():
        expected_total, expected_normal, expected_anomaly = EXPECTED_COUNTS[dataset]
        for seed in seeds:
            npz_path = cache_path(dataset, seed)
            metadata_path = cache_metadata_path(dataset, seed)
            if not npz_path.exists() or not metadata_path.exists():
                errors.append(f"missing cache: {dataset} seed={seed}")
                continue
            observed_jobs += 1
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if sha256_file(npz_path) != metadata.get("sha256"):
                errors.append(f"hash mismatch: {npz_path}")
                continue
            with np.load(npz_path) as archive:
                required = {
                    "trajectories",
                    "labels",
                    "spatial_path",
                    "kinematic",
                    "trajectory_shape",
                    "canonical_features",
                }
                missing = required.difference(archive.files)
                if missing:
                    errors.append(f"missing arrays {sorted(missing)}: {npz_path}")
                    continue
                trajectories = archive["trajectories"]
                labels = archive["labels"]
                shapes = {
                    "spatial_path": archive["spatial_path"].shape,
                    "kinematic": archive["kinematic"].shape,
                    "trajectory_shape": archive["trajectory_shape"].shape,
                    "canonical_features": archive["canonical_features"].shape,
                }
            expected_shapes = {
                "spatial_path": (expected_total, 16),
                "kinematic": (expected_total, 8),
                "trajectory_shape": (expected_total, 10),
                "canonical_features": (expected_total, 34),
            }
            if trajectories.shape != (expected_total, 32, 2):
                errors.append(f"trajectory shape mismatch: {dataset} seed={seed} {trajectories.shape}")
            if labels.shape != (expected_total,):
                errors.append(f"label shape mismatch: {dataset} seed={seed} {labels.shape}")
            elif int(labels.sum()) != expected_anomaly or int((labels == 0).sum()) != expected_normal:
                errors.append(f"label count mismatch: {dataset} seed={seed}")
            if shapes != expected_shapes:
                errors.append(f"feature shape mismatch: {dataset} seed={seed} {shapes}")
            if metadata.get("n_total") != expected_total or metadata.get("n_anomaly") != expected_anomaly:
                errors.append(f"metadata count mismatch: {dataset} seed={seed}")
    if observed_jobs != expected_jobs:
        errors.append(f"cache count mismatch: observed={observed_jobs}, expected={expected_jobs}")
    print(f"CACHE_VALIDATION observed={observed_jobs} expected={expected_jobs} errors={len(errors)}")
    return errors


def validate_results(root: Path) -> list[str]:
    errors: list[str] = []
    complete_files = sorted(root.rglob("COMPLETE.json")) if root.exists() else []
    for complete_path in complete_files:
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        result_path = resolve_marker_artifact(complete_path, marker, "result")
        score_path = resolve_marker_artifact(complete_path, marker, "score")
        if result_path is None or score_path is None:
            errors.append(f"dangling completion marker: {complete_path}")
            continue
        if sha256_file(result_path) != marker.get("result_sha256"):
            errors.append(f"result hash mismatch: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or result.get("labels_consumed_during_fit") is not False:
            errors.append(f"invalid result metadata: {result_path}")
        with np.load(score_path) as archive:
            scores = np.asarray(archive["scores"], dtype=float)
        if scores.ndim != 1 or not np.isfinite(scores).all():
            errors.append(f"invalid scores: {score_path}")
        if sha256_file(score_path) != result.get("score_sha256"):
            errors.append(f"score hash mismatch: {score_path}")
        metrics = result.get("metrics", {})
        for name in ("AUC", "AUPRC", "F1"):
            value = metrics.get(name)
            if value is None or not 0.0 <= float(value) <= 1.0:
                errors.append(f"invalid {name}: {result_path}")
    print(f"RESULT_VALIDATION root={root} complete={len(complete_files)} errors={len(errors)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("cache", "preflight", "main", "analysis", "timing", "all"),
        default="all",
    )
    args = parser.parse_args()
    errors: list[str] = []
    if args.scope in {"cache", "all"}:
        errors.extend(validate_caches())
    roots = {
        "preflight": PHASE4_ROOT / "preflight_raw",
        "main": PHASE4_ROOT / "raw",
        "analysis": PHASE4_ROOT / "analysis_raw",
        "timing": PHASE4_ROOT / "deep_timing_raw",
    }
    selected = roots if args.scope == "all" else {args.scope: roots[args.scope]} if args.scope in roots else {}
    for root in selected.values():
        errors.extend(validate_results(root))
    for error in errors:
        print(f"ERROR {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
