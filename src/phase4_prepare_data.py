"""Build immutable, feature-complete dataset caches for Phase 4."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from feature_extraction_v2 import extract_all_features_v2
from phase4_common import (
    CACHE_ROOT,
    DATASET_SEEDS,
    atomic_write_json,
    cache_metadata_path,
    cache_path,
    sha256_file,
)
from run_cross_dataset_ablation import (
    geolife_dataset,
    grid_dataset,
    porto_dataset,
    prepare_geolife_context,
    prepare_porto_context,
    synthetic_dataset,
)


def canonical_features(trajectories: list[np.ndarray]) -> dict[str, np.ndarray]:
    spatial, kinematic, _, path_features, _ = extract_all_features_v2(trajectories)
    spatial_path = np.hstack((spatial, path_features)).astype(np.float32)
    flat = np.asarray([np.asarray(trajectory).reshape(-1) for trajectory in trajectories], dtype=np.float64)
    shape = PCA(n_components=10, svd_solver="full").fit_transform(flat).astype(np.float32)
    canonical = np.hstack((spatial_path, kinematic, shape)).astype(np.float32)
    if canonical.shape[1] != 34:
        raise RuntimeError(f"canonical feature dimension is {canonical.shape[1]}, expected 34")
    return {
        "spatial_path": spatial_path,
        "kinematic": np.asarray(kinematic, dtype=np.float32),
        "trajectory_shape": shape,
        "canonical_features": canonical,
    }


def save_cache(dataset: str, seed: int, trajectories: list[np.ndarray], labels: np.ndarray, force: bool) -> None:
    destination = cache_path(dataset, seed)
    metadata_path = cache_metadata_path(dataset, seed)
    if destination.exists() and metadata_path.exists() and not force:
        print(f"[{dataset}] seed={seed} cache exists; skip", flush=True)
        return

    started = time.perf_counter()
    trajectory_array = np.asarray(trajectories, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int8)
    if trajectory_array.ndim != 3 or trajectory_array.shape[1:] != (32, 2):
        raise RuntimeError(f"invalid trajectory shape: {trajectory_array.shape}")
    expected_positive = 552 if dataset != "GeoLife" else 332
    if int(label_array.sum()) != expected_positive:
        raise RuntimeError(
            f"{dataset} seed={seed} has {int(label_array.sum())} positives, expected {expected_positive}"
        )

    features = canonical_features(trajectories)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            trajectories=trajectory_array,
            labels=label_array,
            **features,
        )
    os.replace(temporary, destination)
    metadata = {
        "dataset": dataset,
        "seed": int(seed),
        "n_total": int(len(label_array)),
        "n_normal": int((label_array == 0).sum()),
        "n_anomaly": int((label_array == 1).sum()),
        "trajectory_shape": list(trajectory_array.shape),
        "canonical_feature_shape": list(features["canonical_features"].shape),
        "labels_used_for_feature_construction": False,
        "generation_seconds": time.perf_counter() - started,
        "sha256": sha256_file(destination),
    }
    atomic_write_json(metadata_path, metadata)
    print(
        f"[{dataset}] seed={seed} cached {len(label_array)} samples in "
        f"{metadata['generation_seconds']:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SEEDS), choices=list(DATASET_SEEDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = set(args.datasets)
    porto_context = prepare_porto_context() if "Porto-derived" in selected else None
    geolife_context = prepare_geolife_context() if "GeoLife" in selected else None
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for seed in DATASET_SEEDS[dataset]:
            if dataset == "Synthetic":
                trajectories, labels = synthetic_dataset(seed, n_normal=5000)
            elif dataset == "Grid-Network":
                trajectories, labels = grid_dataset(seed, n_normal=5000)
            elif dataset == "Porto-derived":
                trajectories, labels = porto_dataset(seed, porto_context, n_normal=5000)
            else:
                trajectories, labels = geolife_dataset(seed, geolife_context, n_normal=3000)
            save_cache(dataset, seed, trajectories, labels, force=args.force)


if __name__ == "__main__":
    main()

