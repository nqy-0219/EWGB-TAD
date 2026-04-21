"""
Enhanced feature extraction for EWGB-TAD v2.
Adds path-deviation features to improve detour/route-deviation detection.
"""

import numpy as np
from .feature_extraction import (
    extract_spatial_features, extract_kinematic_features, extract_entropy_features,
    SPATIAL_NAMES, KINEMATIC_NAMES, ENTROPY_NAMES
)


def extract_path_deviation_features(trajectory, reference_paths=None):
    """
    Extract path-deviation features that capture route-level anomalies.

    Features:
    - Hausdorff-like max deviation from nearest reference path
    - Mean segment direction consistency (how aligned each segment is with OD direction)
    - Endpoint deviation (distance from expected destination)
    - Path curvature variance
    - Directional entropy of second-half trajectory (captures late deviations)
    - Cumulative angular deviation from straight path
    """
    n = len(trajectory)
    diffs = np.diff(trajectory, axis=0)

    # OD vector and direction
    od_vec = trajectory[-1] - trajectory[0]
    od_dist = np.linalg.norm(od_vec)
    od_dir = od_vec / (od_dist + 1e-8)

    # 1. Segment direction consistency with OD direction
    seg_dirs = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8)
    direction_consistency = np.dot(seg_dirs, od_dir)
    mean_consistency = np.mean(direction_consistency)
    min_consistency = np.min(direction_consistency)

    # 2. Second-half direction consistency (captures late route deviations)
    mid = n // 2
    if mid < n - 1:
        second_half_diffs = np.diff(trajectory[mid:], axis=0)
        second_half_dirs = second_half_diffs / (np.linalg.norm(second_half_diffs, axis=1, keepdims=True) + 1e-8)
        second_half_consistency = np.mean(np.dot(second_half_dirs, od_dir))
    else:
        second_half_consistency = mean_consistency

    # 3. Cumulative angular deviation from straight path
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    od_angle = np.arctan2(od_vec[1], od_vec[0])
    angle_devs = np.abs(angles - od_angle)
    angle_devs = np.minimum(angle_devs, 2 * np.pi - angle_devs)
    cumulative_angle_dev = np.sum(angle_devs)
    max_angle_dev = np.max(angle_devs)

    # 4. Path curvature variance
    if len(angles) > 1:
        curvatures = np.diff(angles)
        curvatures = np.arctan2(np.sin(curvatures), np.cos(curvatures))
        curvature_var = np.var(curvatures)
        curvature_max = np.max(np.abs(curvatures))
    else:
        curvature_var = 0.0
        curvature_max = 0.0

    # 5. Progress monotonicity (how steadily the trajectory approaches destination)
    # Project each point onto OD line
    progress = np.dot(trajectory - trajectory[0], od_dir) / (od_dist + 1e-8)
    progress_diffs = np.diff(progress)
    backward_ratio = np.sum(progress_diffs < 0) / (len(progress_diffs) + 1e-8)

    # 6. Maximum local detour ratio (sliding window)
    window_size = max(3, n // 4)
    max_local_detour = 0
    for i in range(0, n - window_size):
        seg = trajectory[i:i + window_size]
        seg_length = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
        seg_direct = np.linalg.norm(seg[-1] - seg[0])
        local_detour = seg_length / (seg_direct + 1e-8)
        max_local_detour = max(max_local_detour, local_detour)

    return np.array([
        mean_consistency, min_consistency,
        second_half_consistency,
        cumulative_angle_dev, max_angle_dev,
        curvature_var, curvature_max,
        backward_ratio, max_local_detour
    ])


PATH_DEVIATION_NAMES = [
    'mean_dir_consistency', 'min_dir_consistency',
    'second_half_consistency',
    'cum_angle_dev', 'max_angle_dev',
    'curvature_var', 'curvature_max',
    'backward_ratio', 'max_local_detour'
]


def extract_all_features_v2(trajectories, n_bins=16, grid_size=10):
    """
    Extract all features including path-deviation view.

    Returns:
        spatial_features: (n, 7)
        kinematic_features: (n, 8)
        entropy_features: (n, 6)
        path_features: (n, 9)
        all_features: (n, 30)
    """
    spatial_list = []
    kinematic_list = []
    entropy_list = []
    path_list = []

    for traj in trajectories:
        spatial_list.append(extract_spatial_features(traj))
        kinematic_list.append(extract_kinematic_features(traj))
        entropy_list.append(extract_entropy_features(traj, n_bins, grid_size))
        path_list.append(extract_path_deviation_features(traj))

    spatial_features = np.array(spatial_list)
    kinematic_features = np.array(kinematic_list)
    entropy_features = np.array(entropy_list)
    path_features = np.array(path_list)
    all_features = np.hstack([spatial_features, kinematic_features,
                               entropy_features, path_features])

    return spatial_features, kinematic_features, entropy_features, path_features, all_features


ALL_FEATURE_NAMES_V2 = SPATIAL_NAMES + KINEMATIC_NAMES + ENTROPY_NAMES + PATH_DEVIATION_NAMES


if __name__ == '__main__':
    from data_generator import generate_synthetic_trajectories

    trajs, labels, atypes, _ = generate_synthetic_trajectories(n_normal=100, n_anomaly_per_type=10)
    s, k, e, p, a = extract_all_features_v2(trajs)
    print(f"Spatial: {s.shape}, Kinematic: {k.shape}, Entropy: {e.shape}, Path: {p.shape}, All: {a.shape}")

    # Check if path features discriminate anomaly types
    for atype in range(4):
        type_names = {0: 'detour', 1: 'loop', 2: 'speed', 3: 'route_dev'}
        mask = atypes == atype
        normal_mask = labels == 0
        if mask.sum() > 0:
            anom_mean = np.mean(p[mask], axis=0)
            norm_mean = np.mean(p[normal_mask], axis=0)
            diff = anom_mean - norm_mean
            print(f"\n{type_names[atype]}:")
            for fname, d in zip(PATH_DEVIATION_NAMES, diff):
                print(f"  {fname}: diff={d:+.4f}")
