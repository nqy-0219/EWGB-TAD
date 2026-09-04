"""Run one immutable Phase 4 dataset/method/seed job."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines import CoMadOutDetector, ECODDetector, IBoostODEDetector, IForestDetector
from baselines_dl import TrajLSTMAEDetector, USADDetector
from baselines_v2 import ProfileTADDetector, SegmentOutlierDetector, ShapeKNNDetector, TADSDetector
from evaluation_protocol import evaluate_oracle_top_k
from ewgb_tad_current import ThreeViewEWGBDetector
from phase3_lmtad_adapter import LMTADAdapterConfig, fit_score_lmtad
from phase3_mstoatd_adapter import MSTOATDAdapterConfig, fit_score_mstoatd
from phase4_common import (
    DATASET_SEEDS,
    MAIN_METHODS,
    PHASE4_ROOT,
    SOURCE_ROOT,
    atomic_write_json,
    environment_record,
    job_directory,
    load_cache,
    set_reproducible,
    sha256_file,
    slug,
)


_DEEP_CONFIG_CACHE: dict[str, dict] | None = None


def load_selected_deep_config(dataset: str, method: str) -> dict:
    """Load the fixed dataset-specific configuration selected without labels."""
    global _DEEP_CONFIG_CACHE
    path = Path(
        os.environ.get(
            "EWGB_DEEP_CONFIG_PATH",
            str(SOURCE_ROOT / "configs" / "deep_tuning_selected_latest.json"),
        )
    )
    if _DEEP_CONFIG_CACHE is None:
        if not path.exists():
            raise FileNotFoundError(
                f"missing fixed deep-baseline configuration: {path}"
            )
        _DEEP_CONFIG_CACHE = json.loads(path.read_text(encoding="utf-8"))
    try:
        return dict(_DEEP_CONFIG_CACHE["datasets"][dataset][method])
    except KeyError as exc:
        raise KeyError(f"no selected deep configuration for {dataset}/{method}") from exc


def build_cpu_detector(method: str, seed: int) -> Any:
    if method == "IForest":
        return IForestDetector(n_estimators=200, contamination=0.1, seed=seed)
    if method == "ECOD":
        return ECODDetector()
    if method == "iBoost-ODE":
        return IBoostODEDetector(contamination=0.1, seed=seed, n_rounds=5, subspace_ratio=0.65)
    if method == "CoMadOut":
        return CoMadOutDetector(max_dim=40)
    if method == "Shape-KNN":
        return ShapeKNNDetector(k=5)
    if method == "SegmentOD":
        return SegmentOutlierDetector(n_segments=4)
    if method == "TADS":
        return TADSDetector(grid_size=18, n_prototypes=18, contamination=0.1, seed=seed)
    if method == "Profile-TAD":
        return ProfileTADDetector(n_routes=20, seed=seed)
    raise ValueError(f"unknown CPU method: {method}")


def run_method(
    method: str,
    arrays: dict[str, np.ndarray],
    seed: int,
    device: str,
    profile: str,
    dataset: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    trajectories_array = np.asarray(arrays["trajectories"], dtype=np.float32)
    trajectories = [trajectory for trajectory in trajectories_array]
    features = np.asarray(arrays["canonical_features"], dtype=np.float32)
    spatial_path = np.asarray(arrays["spatial_path"], dtype=np.float32)
    kinematic = np.asarray(arrays["kinematic"], dtype=np.float32)

    if method == "EWGB-TAD":
        detector = ThreeViewEWGBDetector(
            min_samples=8,
            purity_threshold=0.85,
            n_shape_dims=10,
            view_fusion="entropy",
            use_local_entropy=True,
            constant_tol=1e-10,
            fusion_score_bins=20,
            fusion_base_weight=0.1,
        )
        detector.fit(spatial_path, kinematic, trajectories)
        return detector.score(spatial_path, kinematic, trajectories), detector.get_stats()

    if method in {"IForest", "ECOD", "iBoost-ODE", "CoMadOut", "Shape-KNN", "SegmentOD", "TADS", "Profile-TAD"}:
        detector = build_cpu_detector(method, seed)
        if method in {"Shape-KNN", "SegmentOD", "TADS", "Profile-TAD"}:
            scores = detector.fit_score(features, trajectories)
        else:
            scores = detector.fit_score(features)
        return np.asarray(scores), {"detector": method, "canonical_feature_dimension": 34}

    if method == "LSTM-AE":
        selected = load_selected_deep_config(dataset, method)
        max_epochs = 1 if profile == "preflight" else int(selected["max_epochs"])
        patience = None if profile == "preflight" else int(selected["patience"])
        min_epochs = 1 if profile == "preflight" else int(selected["min_epochs"])
        detector = TrajLSTMAEDetector(
            epochs=max_epochs,
            batch_size=int(selected["batch_size"]),
            lr=float(selected["learning_rate"]),
            hidden_dim=int(selected["hidden_dim"]),
            latent_dim=int(selected["latent_dim"]),
            num_layers=int(selected["num_layers"]),
            dropout=float(selected["dropout"]),
            validation_fraction=float(selected["validation_fraction"]),
            patience=patience,
            min_epochs=min_epochs,
            min_delta=float(selected["min_delta"]),
            device=device,
        )
        scores = detector.fit_score(features, trajectories, seed=seed)
        return scores, {
            "dataset_specific_selection": selected,
            **detector.timing_,
        }
    if method == "USAD":
        selected = load_selected_deep_config(dataset, method)
        max_epochs = 1 if profile == "preflight" else int(selected["max_epochs"])
        patience = None if profile == "preflight" else int(selected["patience"])
        min_epochs = 1 if profile == "preflight" else int(selected["min_epochs"])
        detector = USADDetector(
            epochs=max_epochs,
            batch_size=int(selected["batch_size"]),
            lr=float(selected["learning_rate"]),
            hidden_dim=int(selected["hidden_dim"]),
            latent_dim=int(selected["latent_dim"]),
            validation_fraction=float(selected["validation_fraction"]),
            patience=patience,
            min_epochs=min_epochs,
            min_delta=float(selected["min_delta"]),
            device=device,
        )
        scores = detector.fit_score(features, seed=seed)
        return scores, {
            "dataset_specific_selection": selected,
            **detector.timing_,
        }
    if method == "LM-TAD":
        epochs = 1 if profile == "preflight" else 50
        config = LMTADAdapterConfig(
            seed=seed,
            grid_side=18,
            validation_fraction=0.1,
            epochs=epochs,
            patience=None,
            batch_size=32,
            learning_rate=3e-4,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.99,
            grad_clip=1.0,
            use_cosine_decay=profile == "full",
            warmup_steps=5000,
            lr_decay_steps=60000,
            min_learning_rate=3e-5,
            n_layer=8,
            n_head=12,
            n_embd=768,
            dropout=0.2,
            torch_threads=2,
            device=device,
            mixed_precision=True,
            score_batch_size=128,
        )
        return fit_score_lmtad(trajectories_array, config)
    if method == "MST-OATD":
        pretrain_epochs = 1 if profile == "preflight" else 8
        epochs = 1 if profile == "preflight" else 10
        config = MSTOATDAdapterConfig(
            seed=seed,
            grid_side=18,
            pretrain_epochs=pretrain_epochs,
            epochs=epochs,
            batch_size=1600,
            embedding_size=128,
            hidden_size=512,
            n_cluster=20,
            pretrain_lr_s=2e-3,
            pretrain_lr_t=2e-3,
            lr_s=3e-4,
            lr_t=3e-4,
            s1_size=2,
            s2_size=4,
            sampling_interval_seconds=15,
            torch_threads=2,
            device=device,
            runtime_parent=os.environ.get("EWGB_PHASE4_TMP"),
        )
        return fit_score_mstoatd(trajectories_array, config)
    raise ValueError(f"unknown method: {method}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_SEEDS), required=True)
    parser.add_argument("--method", choices=MAIN_METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", choices=("preflight", "full"), default="full")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed not in DATASET_SEEDS[args.dataset]:
        raise ValueError(f"seed {args.seed} is not predefined for {args.dataset}")

    if args.profile == "preflight":
        job_dir = (
            PHASE4_ROOT
            / "preflight_raw"
            / slug(args.dataset)
            / slug(args.method)
            / f"seed_{args.seed}"
        )
    else:
        job_dir = job_directory(args.dataset, args.method, args.seed)
    complete_path = job_dir / "COMPLETE.json"
    if complete_path.exists() and not args.force:
        print(f"complete; skip {args.dataset} {args.method} seed={args.seed}", flush=True)
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = job_dir / f"attempt_{timestamp}_{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    set_reproducible(args.seed, torch_threads=2)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    try:
        arrays, cache_metadata = load_cache(args.dataset, args.seed)
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        scores, model_metadata = run_method(
            args.method,
            arrays,
            args.seed,
            args.device,
            args.profile,
            args.dataset,
        )
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape != labels.shape:
            raise RuntimeError(f"score shape {scores.shape} does not match labels {labels.shape}")
        if not np.isfinite(scores).all():
            raise RuntimeError("scores contain non-finite values")
        metrics = evaluate_oracle_top_k(labels, scores)
        score_path = attempt / "scores.npz"
        with score_path.open("wb") as handle:
            np.savez_compressed(handle, scores=scores)
        runtime = time.perf_counter() - started
        result = {
            "status": "complete",
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "profile": args.profile,
            "metrics": metrics,
            "runtime_seconds": runtime,
            "n_scores": int(len(scores)),
            "score_direction": "higher_is_more_anomalous",
            "labels_consumed_during_fit": False,
            "dataset_cache": cache_metadata,
            "score_sha256": sha256_file(score_path),
            "environment": environment_record(args.device),
            "model_metadata": model_metadata,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() and args.device.startswith("cuda")
                else None
            ),
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
            f"COMPLETE {args.dataset} {args.method} seed={args.seed} "
            f"AUC={metrics['AUC']:.6f} AUPRC={metrics['AUPRC']:.6f} "
            f"F1={metrics['F1']:.6f} time={runtime:.1f}s",
            flush=True,
        )
        return 0
    except Exception as exc:
        error = {
            "status": "failed",
            "dataset": args.dataset,
            "method": args.method,
            "seed": args.seed,
            "profile": args.profile,
            "runtime_seconds": time.perf_counter() - started,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "environment": environment_record(args.device),
        }
        atomic_write_json(attempt / "ERROR.json", error)
        print(error["traceback"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
