"""Run representative scalability timing experiments for Figure 4.

The original scalability figure only included EWGB-TAD, IsolationForest, LOF,
and ECOD. This script extends the CPU scalability comparison with stronger and
more task-relevant baselines used elsewhere in the paper.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from data_generator import generate_synthetic_trajectories
from ewgb_tad_current import ThreeViewEWGBDetector
from feature_extraction_v2 import extract_all_features_v2
from baselines import ECODDetector, IForestDetector
from baselines_v2 import SegmentOutlierDetector, ShapeKNNDetector


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("EWGB_AUX_RESULTS_DIR", str(ROOT / "results")))

SIZES = [1000, 2000, 5000, 10000, 20000]
CONTAMINATION = 0.10
SEED = 42


def _anomalies_per_type(n_normal: int) -> int:
    return max(1, int(n_normal * CONTAMINATION / (4 * (1 - CONTAMINATION))))


def _time_call(fn, repeats: int = 1, warmup: int = 0) -> float:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        values.append(time.perf_counter() - t0)
    return float(statistics.median(values))


def run_scalability() -> dict[str, dict[str, float]]:
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results: dict[str, dict[str, float]] = {}

    for n_normal in SIZES:
        print(f"\nN={n_normal}")
        n_anom = _anomalies_per_type(n_normal)
        trajs, labels, _, _ = generate_synthetic_trajectories(
            n_normal=n_normal,
            n_anomaly_per_type=n_anom,
            seed=SEED,
        )
        spatial, kinematic, _, path_feat, all_features = extract_all_features_v2(trajs)
        spatial_path = np.hstack([spatial, path_feat])

        methods: dict[str, float] = {}

        def run_ewgb() -> None:
            det = ThreeViewEWGBDetector(
                min_samples=8,
                purity_threshold=0.85,
                n_shape_dims=10,
            )
            det.fit(spatial_path, kinematic, trajs)
            det.score(spatial_path, kinematic, trajs)

        methods["EWGB-TAD"] = _time_call(run_ewgb)

        def run_shape_knn() -> None:
            det = ShapeKNNDetector(k=5)
            det.fit_score(all_features, trajs)

        methods["Shape-KNN"] = _time_call(run_shape_knn)

        def run_segment_od() -> None:
            det = SegmentOutlierDetector(n_segments=4)
            det.fit_score(all_features, trajs)

        methods["SegmentOD"] = _time_call(run_segment_od, repeats=3, warmup=1)

        def run_iforest() -> None:
            det = IForestDetector(contamination=CONTAMINATION, seed=SEED)
            det.fit_score(all_features)

        methods["IsolationForest"] = _time_call(run_iforest)

        def run_ecod() -> None:
            det = ECODDetector()
            det.fit_score(all_features)

        methods["ECOD"] = _time_call(run_ecod)

        all_results[str(n_normal)] = methods
        for method, runtime in methods.items():
            print(f"  {method:<18} {runtime:.3f}s")

    out = RESULTS_DIR / "scalability_representative_results.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return all_results


if __name__ == "__main__":
    run_scalability()
