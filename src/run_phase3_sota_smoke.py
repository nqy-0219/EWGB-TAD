"""Run isolated smoke tests for the included official SOTA baselines."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data_generator import generate_synthetic_trajectories


SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = SOURCE_ROOT / "results" / "phase3_sota_smoke"


def build_smoke_pool(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectories, labels, anomaly_types, _ = generate_synthetic_trajectories(
        n_normal=128,
        n_anomaly_per_type=4,
        seq_len=32,
        n_routes=6,
        noise_std=0.02,
        seed=seed,
    )
    return (
        np.asarray(trajectories, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(anomaly_types, dtype=np.int64),
    )


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "device": "cpu",
    }


def _run_one(name: str, output_dir: Path) -> dict[str, Any]:
    trajectories, labels, anomaly_types = build_smoke_pool()
    started = time.perf_counter()
    if name == "lmtad":
        from phase3_lmtad_adapter import LMTADAdapterConfig, fit_score_lmtad

        scores, metadata = fit_score_lmtad(trajectories, LMTADAdapterConfig())
    elif name == "mstoatd":
        from phase3_mstoatd_adapter import MSTOATDAdapterConfig, fit_score_mstoatd

        scores, metadata = fit_score_mstoatd(trajectories, MSTOATDAdapterConfig())
    else:
        raise ValueError(f"unknown baseline: {name}")

    elapsed = time.perf_counter() - started
    result = {
        "status": "pass",
        "smoke_test_only": True,
        "baseline": metadata["method"],
        "dataset": {
            "name": "Synthetic-Phase3-Smoke-v1",
            "seed": 42,
            "n_normal": int((labels == 0).sum()),
            "n_anomaly": int((labels == 1).sum()),
            "n_total": int(len(labels)),
            "sequence_length": int(trajectories.shape[1]),
            "anomaly_type_counts": {
                str(int(kind)): int((anomaly_types == kind).sum())
                for kind in np.unique(anomaly_types[labels == 1])
            },
        },
        "checks": {
            "training_completed": True,
            "score_count_matches": bool(len(scores) == len(labels)),
            "scores_finite": bool(np.isfinite(scores).all()),
            "higher_score_is_more_anomalous": metadata["score_direction"] == "higher_is_more_anomalous",
            "labels_consumed_during_fit": bool(metadata["labels_consumed_during_fit"]),
        },
        "diagnostic_metrics_not_used_for_selection": {
            "auc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
            "normal_score_mean": float(scores[labels == 0].mean()),
            "anomaly_score_mean": float(scores[labels == 1].mean()),
        },
        "score_summary": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        },
        "wall_clock_seconds": elapsed,
        "environment": _environment(),
        "adapter_metadata": metadata,
    }
    required = (
        result["checks"]["training_completed"]
        and result["checks"]["score_count_matches"]
        and result["checks"]["scores_finite"]
        and result["checks"]["higher_score_is_more_anomalous"]
        and not result["checks"]["labels_consumed_during_fit"]
    )
    if not required:
        result["status"] = "fail"

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{name}_smoke_scores.npz",
        scores=scores,
        labels=labels,
        anomaly_types=anomaly_types,
    )
    (output_dir / f"{name}_smoke_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def _run_isolated(name: str, output_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--baseline",
        name,
        "--output-dir",
        str(output_dir),
        "--child",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "status": "fail",
            "baseline": name,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    path = output_dir / f"{name}_smoke_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["child_stdout"] = completed.stdout
    result["child_stderr"] = completed.stderr
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=("lmtad", "mstoatd", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if args.child:
        result = _run_one(args.baseline, output_dir)
        print(json.dumps({"baseline": result["baseline"], "status": result["status"]}))
        return 0 if result["status"] == "pass" else 1

    names = ("lmtad", "mstoatd") if args.baseline == "all" else (args.baseline,)
    results = [_run_isolated(name, output_dir) for name in names]
    summary = {
        "phase": 3,
        "status": "pass" if all(item.get("status") == "pass" for item in results) else "fail",
        "scope": "small-data CPU smoke tests only; not manuscript benchmark results",
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase3_sota_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    for result in results:
        print(f"{result.get('baseline')}: {result.get('status')}")
        if result.get("status") == "fail":
            print(result.get("stderr", ""))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
