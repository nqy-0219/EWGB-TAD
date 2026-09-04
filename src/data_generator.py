"""
Data generation and preprocessing for EWGB-TAD experiments.
Supports both synthetic data and real dataset loading (Porto Taxi, GeoLife).
"""

import numpy as np
import os
import json
from collections import defaultdict

np.random.seed(42)


# =============================================================
# Synthetic Trajectory Generation
# =============================================================
def generate_synthetic_trajectories(n_normal=5000, n_anomaly_per_type=125,
                                     seq_len=32, n_routes=6, noise_std=0.02,
                                     seed=42):
    """
    Generate synthetic GPS trajectories with controlled anomaly injection.

    Args:
        n_normal: Number of normal trajectories
        n_anomaly_per_type: Number of anomalies per type (4 types)
        seq_len: Number of points per trajectory
        n_routes: Number of normal route patterns
        noise_std: Gaussian noise std for normal trajectories
        seed: Random seed

    Returns:
        trajectories: list of (seq_len, 2) arrays
        labels: array of 0/1
        anomaly_types: array of anomaly type labels (-1 for normal)
        contexts: array of context IDs (route index)
    """
    rng = np.random.RandomState(seed)
    trajectories = []
    labels = []
    anomaly_types = []
    contexts = []

    # Define normal route patterns as (start, end, curvature_range)
    route_defs = []
    for i in range(n_routes):
        angle = 2 * np.pi * i / n_routes
        start = np.array([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
        end = np.array([0.5 + 0.4 * np.cos(angle + np.pi), 0.5 + 0.4 * np.sin(angle + np.pi)])
        route_defs.append((start, end))

    def make_normal_traj(route_idx):
        start, end = route_defs[route_idx]
        t = np.linspace(0, 1, seq_len).reshape(-1, 1)
        path = start + t * (end - start)
        # Add natural curvature
        mid_offset = rng.normal(0, 0.04, (1, 2))
        curvature = 4 * t * (1 - t) * mid_offset
        path += curvature
        # Add GPS noise
        path += rng.normal(0, noise_std, path.shape)
        return path

    # Generate normal trajectories
    for i in range(n_normal):
        route_idx = i % n_routes
        traj = make_normal_traj(route_idx)
        trajectories.append(traj)
        labels.append(0)
        anomaly_types.append(-1)
        contexts.append(route_idx)

    # Generate anomalous trajectories (4 types)
    for anom_type in range(4):
        for j in range(n_anomaly_per_type):
            route_idx = rng.randint(0, n_routes)
            base_traj = make_normal_traj(route_idx)

            if anom_type == 0:  # DETOUR
                start_frac = rng.uniform(0.3, 0.4)
                end_frac = rng.uniform(0.6, 0.7)
                s = int(start_frac * seq_len)
                e = int(end_frac * seq_len)
                od_dist = np.linalg.norm(route_defs[route_idx][1] - route_defs[route_idx][0])
                offset_mag = rng.uniform(0.1, 0.25) * od_dist
                # Perpendicular direction
                direction = route_defs[route_idx][1] - route_defs[route_idx][0]
                perp = np.array([-direction[1], direction[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-8)
                sign = rng.choice([-1, 1])
                for k in range(s, e):
                    frac = (k - s) / (e - s + 1e-8)
                    offset = sign * offset_mag * np.sin(np.pi * frac)
                    base_traj[k] += perp * offset

            elif anom_type == 1:  # LOOP
                center_idx = int(rng.uniform(0.4, 0.6) * seq_len)
                od_dist = np.linalg.norm(route_defs[route_idx][1] - route_defs[route_idx][0])
                radius = rng.uniform(0.05, 0.15) * od_dist
                n_loop = min(8, seq_len // 4)
                start_idx = max(0, center_idx - n_loop // 2)
                for k in range(n_loop):
                    idx = start_idx + k
                    if idx < seq_len:
                        angle = 2 * np.pi * k / n_loop
                        base_traj[idx] += np.array([radius * np.cos(angle), radius * np.sin(angle)])

            elif anom_type == 2:  # SPEED ANOMALY (clustering of points)
                anomaly_start = int(rng.uniform(0.2, 0.5) * seq_len)
                n_slow = min(6, seq_len // 5)
                center_point = base_traj[anomaly_start].copy()
                for k in range(n_slow):
                    idx = anomaly_start + k
                    if idx < seq_len:
                        base_traj[idx] = center_point + rng.normal(0, 0.005, 2)

            elif anom_type == 3:  # ROUTE DEVIATION
                deviation_start = int(rng.uniform(0.5, 0.7) * seq_len)
                angle_dev = rng.uniform(30, 60) * np.pi / 180 * rng.choice([-1, 1])
                for k in range(deviation_start, seq_len):
                    if k > 0:
                        diff = base_traj[k] - base_traj[k-1]
                        cos_a, sin_a = np.cos(angle_dev), np.sin(angle_dev)
                        rotated = np.array([cos_a * diff[0] - sin_a * diff[1],
                                           sin_a * diff[0] + cos_a * diff[1]])
                        base_traj[k] = base_traj[k-1] + rotated
                    angle_dev *= 0.95  # decay

            trajectories.append(base_traj)
            labels.append(1)
            anomaly_types.append(anom_type)
            contexts.append(route_idx)

    # Shuffle
    indices = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in indices]
    labels = np.array(labels)[indices]
    anomaly_types = np.array(anomaly_types)[indices]
    contexts = np.array(contexts)[indices]

    return trajectories, labels, anomaly_types, contexts


# =============================================================
# Porto Taxi Data Loading
# =============================================================
def load_porto_taxi(data_dir='data', max_trips=30000, seq_len=32, seed=42):
    """
    Load and preprocess Porto Taxi dataset.
    Expects train.csv in data_dir.

    Returns same format as generate_synthetic_trajectories but without anomaly labels.
    """
    import csv

    csv_path = os.path.join(data_dir, 'train.csv')
    if not os.path.exists(csv_path):
        print(f"  Porto Taxi data not found at {csv_path}")
        print("  Please download from: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory/data")
        print("  Falling back to synthetic data.")
        return None

    rng = np.random.RandomState(seed)
    trajectories = []

    print("  Loading Porto Taxi data...")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if count >= max_trips * 3:  # read extra for filtering
                break
            try:
                polyline = json.loads(row['POLYLINE'])
                if len(polyline) < 20 or len(polyline) > 200:
                    continue
                coords = np.array(polyline)
                if coords.shape[1] != 2:
                    continue
                # Check bounding box (Porto area)
                if (coords[:, 0].min() < -8.7 or coords[:, 0].max() > -8.5 or
                    coords[:, 1].min() < 41.1 or coords[:, 1].max() > 41.2):
                    continue
                trajectories.append(coords)
                count += 1
                if len(trajectories) >= max_trips:
                    break
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    if len(trajectories) == 0:
        print("  No valid trajectories found. Falling back to synthetic data.")
        return None

    print(f"  Loaded {len(trajectories)} trajectories")

    # Resample to fixed length
    resampled = []
    for traj in trajectories:
        resampled.append(resample_trajectory(traj, seq_len))

    return resampled


def resample_trajectory(traj, target_len):
    """Resample trajectory to fixed number of points by arc-length interpolation."""
    diffs = np.diff(traj, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length < 1e-10:
        return np.tile(traj[0], (target_len, 1))

    target_distances = np.linspace(0, total_length, target_len)
    resampled = np.zeros((target_len, 2))

    for i, d in enumerate(target_distances):
        idx = np.searchsorted(cum_lengths, d, side='right') - 1
        idx = min(idx, len(traj) - 2)
        idx = max(idx, 0)
        seg_len = cum_lengths[idx + 1] - cum_lengths[idx]
        if seg_len < 1e-10:
            resampled[i] = traj[idx]
        else:
            t = (d - cum_lengths[idx]) / seg_len
            resampled[i] = traj[idx] * (1 - t) + traj[idx + 1] * t

    return resampled


# =============================================================
# Anomaly Injection for Real Data
# =============================================================
def inject_anomalies(trajectories, contamination=0.1, seed=42):
    """
    Inject anomalies into a clean trajectory set.

    Args:
        trajectories: list of (seq_len, 2) arrays (assumed clean)
        contamination: fraction of anomalies to inject
        seed: random seed

    Returns:
        mixed_trajectories, labels, anomaly_types
    """
    rng = np.random.RandomState(seed)
    n = len(trajectories)
    n_anomaly = int(n * contamination / (1 - contamination))
    n_per_type = max(1, n_anomaly // 4)

    injected = []
    anom_types = []

    for anom_type in range(4):
        for _ in range(n_per_type):
            idx = rng.randint(0, n)
            traj = trajectories[idx].copy()
            seq_len = len(traj)
            od_dist = np.linalg.norm(traj[-1] - traj[0])

            if anom_type == 0:  # DETOUR
                s = int(0.3 * seq_len)
                e = int(0.7 * seq_len)
                direction = traj[-1] - traj[0]
                perp = np.array([-direction[1], direction[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-8)
                offset_mag = rng.uniform(0.1, 0.25) * od_dist
                sign = rng.choice([-1, 1])
                for k in range(s, e):
                    frac = (k - s) / (e - s + 1e-8)
                    traj[k] += sign * perp * offset_mag * np.sin(np.pi * frac)

            elif anom_type == 1:  # LOOP
                center = seq_len // 2
                radius = rng.uniform(0.05, 0.15) * od_dist
                n_loop = min(8, seq_len // 4)
                s = max(0, center - n_loop // 2)
                for k in range(n_loop):
                    idx_k = s + k
                    if idx_k < seq_len:
                        angle = 2 * np.pi * k / n_loop
                        traj[idx_k] += np.array([radius * np.cos(angle), radius * np.sin(angle)])

            elif anom_type == 2:  # SPEED ANOMALY
                start = int(rng.uniform(0.2, 0.5) * seq_len)
                n_slow = min(6, seq_len // 5)
                center_pt = traj[start].copy()
                for k in range(n_slow):
                    if start + k < seq_len:
                        traj[start + k] = center_pt + rng.normal(0, od_dist * 0.005, 2)

            elif anom_type == 3:  # ROUTE DEVIATION
                dev_start = int(rng.uniform(0.5, 0.7) * seq_len)
                angle_dev = rng.uniform(30, 60) * np.pi / 180 * rng.choice([-1, 1])
                for k in range(dev_start, seq_len):
                    if k > 0:
                        diff = traj[k] - traj[k-1]
                        cos_a, sin_a = np.cos(angle_dev), np.sin(angle_dev)
                        rotated = np.array([cos_a * diff[0] - sin_a * diff[1],
                                           sin_a * diff[0] + cos_a * diff[1]])
                        traj[k] = traj[k-1] + rotated
                    angle_dev *= 0.95

            injected.append(traj)
            anom_types.append(anom_type)

    # Combine
    all_trajs = list(trajectories) + injected
    all_labels = np.concatenate([np.zeros(n), np.ones(len(injected))])
    all_anom_types = np.concatenate([np.full(n, -1), np.array(anom_types)])

    # Shuffle
    perm = rng.permutation(len(all_trajs))
    all_trajs = [all_trajs[i] for i in perm]
    all_labels = all_labels[perm]
    all_anom_types = all_anom_types[perm]

    return all_trajs, all_labels, all_anom_types


if __name__ == '__main__':
    # Test synthetic generation
    trajs, labels, atypes, ctxs = generate_synthetic_trajectories(
        n_normal=5000, n_anomaly_per_type=125, seed=42
    )
    n_anom = sum(labels == 1)
    n_norm = sum(labels == 0)
    print(f"Generated: {len(trajs)} trajectories ({n_norm} normal, {n_anom} anomaly)")
    print(f"Anomaly types: {dict(zip(*np.unique(atypes[labels==1], return_counts=True)))}")
    print(f"Contamination: {n_anom/(n_norm+n_anom)*100:.1f}%")
    print(f"Trajectory shape: {trajs[0].shape}")
