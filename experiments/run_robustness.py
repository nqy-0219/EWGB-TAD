"""
Robustness stress tests for EWGB-TAD.
Tests: GPS jitter levels, missing points, variable resampling length,
more route templates, variable anomaly mixtures.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from ewgb_tad.data_generator import generate_synthetic_trajectories
from ewgb_tad.feature_extraction_v2 import extract_all_features_v2
from ewgb_tad.detector import EWGBTAD as EWGBTADv3, evaluate_detector
from ewgb_tad.baselines_v2 import get_all_baselines_v2


def add_gps_jitter(trajectories, jitter_std, rng):
    """Add additional GPS noise to trajectories."""
    return [t + rng.normal(0, jitter_std, t.shape) for t in trajectories]


def drop_points(trajectories, drop_rate, rng):
    """Randomly drop points and linearly interpolate to fill gaps."""
    result = []
    for t in trajectories:
        n = len(t)
        mask = rng.random(n) > drop_rate
        mask[0] = True  # keep first
        mask[-1] = True  # keep last
        kept_idx = np.where(mask)[0]
        # Interpolate back to original length
        new_t = np.zeros_like(t)
        for d in range(t.shape[1]):
            new_t[:, d] = np.interp(np.arange(n), kept_idx, t[kept_idx, d])
        result.append(new_t)
    return result


def resample_trajectories(trajectories, new_len):
    """Resample all trajectories to a new length."""
    result = []
    for t in trajectories:
        old_len = len(t)
        old_t = np.linspace(0, 1, old_len)
        new_t_vals = np.linspace(0, 1, new_len)
        new_traj = np.zeros((new_len, t.shape[1]))
        for d in range(t.shape[1]):
            new_traj[:, d] = np.interp(new_t_vals, old_t, t[:, d])
        result.append(new_traj)
    return result


def run_robustness_jitter(seeds=[42, 123, 456, 789, 1024], n_normal=5000, contamination=0.10):
    """Test robustness to GPS jitter levels."""
    jitter_levels = [0.0, 0.01, 0.02, 0.04, 0.08]
    results = {}

    for jitter in jitter_levels:
        aucs = []
        for seed in seeds:
            rng = np.random.RandomState(seed + 999)
            n_anom = max(1, int(n_normal * contamination / (4 * (1 - contamination))))
            trajs, labels, _, _ = generate_synthetic_trajectories(
                n_normal=n_normal, n_anomaly_per_type=n_anom, seed=seed
            )
            if jitter > 0:
                trajs = add_gps_jitter(trajs, jitter, rng)

            spatial, kinematic, entropy_feat, path_feat, all_features = extract_all_features_v2(trajs)
            spatial_path = np.hstack([spatial, path_feat])

            det = EWGBTADv3(min_samples=8, purity_threshold=0.85, n_shape_dims=10)
            det.fit(spatial_path, kinematic, entropy_feat, trajs)
            scores = det.score(spatial_path, kinematic, entropy_feat, trajs)
            auc = roc_auc_score(labels, scores)
            aucs.append(auc)

        results[f'jitter_{jitter}'] = {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))}
        print(f"  Jitter σ={jitter:.2f}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    return results


def run_robustness_missing(seeds=[42, 123, 456, 789, 1024], n_normal=5000, contamination=0.10):
    """Test robustness to missing points."""
    drop_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    results = {}

    for drop in drop_rates:
        aucs = []
        for seed in seeds:
            rng = np.random.RandomState(seed + 777)
            n_anom = max(1, int(n_normal * contamination / (4 * (1 - contamination))))
            trajs, labels, _, _ = generate_synthetic_trajectories(
                n_normal=n_normal, n_anomaly_per_type=n_anom, seed=seed
            )
            if drop > 0:
                trajs = drop_points(trajs, drop, rng)

            spatial, kinematic, entropy_feat, path_feat, all_features = extract_all_features_v2(trajs)
            spatial_path = np.hstack([spatial, path_feat])

            det = EWGBTADv3(min_samples=8, purity_threshold=0.85, n_shape_dims=10)
            det.fit(spatial_path, kinematic, entropy_feat, trajs)
            scores = det.score(spatial_path, kinematic, entropy_feat, trajs)
            auc = roc_auc_score(labels, scores)
            aucs.append(auc)

        results[f'drop_{drop}'] = {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))}
        print(f"  Drop rate={drop:.0%}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    return results


def run_robustness_resampling(seeds=[42, 123, 456, 789, 1024], n_normal=5000, contamination=0.10):
    """Test robustness to resampling length."""
    resample_lengths = [16, 24, 32, 48, 64]
    results = {}

    for new_len in resample_lengths:
        aucs = []
        for seed in seeds:
            n_anom = max(1, int(n_normal * contamination / (4 * (1 - contamination))))
            trajs, labels, _, _ = generate_synthetic_trajectories(
                n_normal=n_normal, n_anomaly_per_type=n_anom, seed=seed, seq_len=32
            )
            if new_len != 32:
                trajs = resample_trajectories(trajs, new_len)

            spatial, kinematic, entropy_feat, path_feat, all_features = extract_all_features_v2(trajs)
            spatial_path = np.hstack([spatial, path_feat])

            n_shape = min(10, 2 * new_len - 1)
            det = EWGBTADv3(min_samples=8, purity_threshold=0.85, n_shape_dims=n_shape)
            det.fit(spatial_path, kinematic, entropy_feat, trajs)
            scores = det.score(spatial_path, kinematic, entropy_feat, trajs)
            auc = roc_auc_score(labels, scores)
            aucs.append(auc)

        results[f'len_{new_len}'] = {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))}
        print(f"  Resample len={new_len}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    return results


def run_robustness_routes(seeds=[42, 123, 456, 789, 1024], n_normal=5000, contamination=0.10):
    """Test robustness to number of route templates."""
    route_counts = [3, 6, 9, 12]
    results = {}

    for n_routes in route_counts:
        aucs = []
        for seed in seeds:
            n_anom = max(1, int(n_normal * contamination / (4 * (1 - contamination))))
            trajs, labels, _, _ = generate_synthetic_trajectories(
                n_normal=n_normal, n_anomaly_per_type=n_anom, seed=seed, n_routes=n_routes
            )

            spatial, kinematic, entropy_feat, path_feat, all_features = extract_all_features_v2(trajs)
            spatial_path = np.hstack([spatial, path_feat])

            det = EWGBTADv3(min_samples=8, purity_threshold=0.85, n_shape_dims=10)
            det.fit(spatial_path, kinematic, entropy_feat, trajs)
            scores = det.score(spatial_path, kinematic, entropy_feat, trajs)
            auc = roc_auc_score(labels, scores)
            aucs.append(auc)

        results[f'routes_{n_routes}'] = {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))}
        print(f"  Routes={n_routes}: AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    return results


if __name__ == '__main__':
    all_results = {}

    print("=" * 60)
    print("Robustness Test 1: GPS Jitter")
    print("=" * 60)
    all_results['jitter'] = run_robustness_jitter()

    print("\n" + "=" * 60)
    print("Robustness Test 2: Missing Points")
    print("=" * 60)
    all_results['missing'] = run_robustness_missing()

    print("\n" + "=" * 60)
    print("Robustness Test 3: Resampling Length")
    print("=" * 60)
    all_results['resampling'] = run_robustness_resampling()

    print("\n" + "=" * 60)
    print("Robustness Test 4: Route Complexity")
    print("=" * 60)
    all_results['routes'] = run_robustness_routes()

    os.makedirs('../results', exist_ok=True)
    with open('../results/robustness_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to ../results/robustness_results.json")
