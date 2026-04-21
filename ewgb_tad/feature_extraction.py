"""
Feature extraction for EWGB-TAD.
Three views: spatial shape, kinematic, entropy.
"""

import numpy as np
from collections import defaultdict


def extract_spatial_features(trajectory):
    """
    View 1: Spatial shape features.
    - total_length, od_distance, detour_ratio
    - mean_lateral_dev, max_lateral_dev
    - bbox_area, radius_of_gyration
    """
    diffs = np.diff(trajectory, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)

    total_length = np.sum(seg_lengths)
    od_vec = trajectory[-1] - trajectory[0]
    od_distance = np.linalg.norm(od_vec)
    detour_ratio = total_length / (od_distance + 1e-8)

    # Lateral deviation from OD line
    if od_distance > 1e-8:
        od_unit = od_vec / od_distance
        od_perp = np.array([-od_unit[1], od_unit[0]])
        lateral_devs = np.abs(np.dot(trajectory - trajectory[0], od_perp))
    else:
        lateral_devs = np.linalg.norm(trajectory - trajectory[0], axis=1)

    mean_lateral_dev = np.mean(lateral_devs)
    max_lateral_dev = np.max(lateral_devs)

    # Bounding box area
    bbox_width = np.ptp(trajectory[:, 0])
    bbox_height = np.ptp(trajectory[:, 1])
    bbox_area = bbox_width * bbox_height

    # Radius of gyration
    centroid = np.mean(trajectory, axis=0)
    rog = np.sqrt(np.mean(np.sum((trajectory - centroid) ** 2, axis=1)))

    return np.array([
        total_length, od_distance, detour_ratio,
        mean_lateral_dev, max_lateral_dev,
        bbox_area, rog
    ])


def extract_kinematic_features(trajectory, dt=1.0):
    """
    View 2: Kinematic features.
    - mean/std/max speed, mean/std acceleration
    - stop_ratio, mean/max turning angle
    """
    diffs = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) / dt

    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    max_speed = np.max(speeds)

    # Acceleration
    if len(speeds) > 1:
        accels = np.abs(np.diff(speeds)) / dt
        mean_accel = np.mean(accels)
        std_accel = np.std(accels)
    else:
        mean_accel = 0.0
        std_accel = 0.0

    # Stop ratio (speed < threshold)
    speed_threshold = mean_speed * 0.1 + 1e-8
    stop_ratio = np.sum(speeds < speed_threshold) / len(speeds)

    # Turning angles
    if len(diffs) > 1:
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        turn_angles = np.abs(np.diff(angles))
        turn_angles = np.minimum(turn_angles, 2 * np.pi - turn_angles)
        mean_turn = np.mean(turn_angles)
        max_turn = np.max(turn_angles)
    else:
        mean_turn = 0.0
        max_turn = 0.0

    return np.array([
        mean_speed, std_speed, max_speed,
        mean_accel, std_accel,
        stop_ratio, mean_turn, max_turn
    ])


def compute_entropy(values, n_bins=16):
    """Compute Shannon entropy of a distribution."""
    if len(values) < 2:
        return 0.0
    hist, _ = np.histogram(values, bins=n_bins, density=False)
    hist = hist / (hist.sum() + 1e-8)
    # Remove zeros
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist + 1e-12))


def extract_entropy_features(trajectory, n_bins=16, grid_size=10):
    """
    View 3: Entropy features.
    - speed_entropy, accel_entropy, heading_entropy
    - turn_entropy, grid_occupancy_entropy, grid_transition_entropy
    """
    diffs = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)

    # Speed entropy
    speed_entropy = compute_entropy(speeds, n_bins)

    # Acceleration entropy
    if len(speeds) > 1:
        accels = np.diff(speeds)
        accel_entropy = compute_entropy(accels, n_bins)
    else:
        accel_entropy = 0.0

    # Heading entropy
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    heading_entropy = compute_entropy(headings, n_bins)

    # Turn angle entropy
    if len(headings) > 1:
        turns = np.diff(headings)
        turns = np.arctan2(np.sin(turns), np.cos(turns))  # normalize to [-pi, pi]
        turn_entropy = compute_entropy(turns, n_bins)
    else:
        turn_entropy = 0.0

    # Grid occupancy entropy
    # Normalize trajectory to [0, 1] based on bounding box
    traj_min = trajectory.min(axis=0)
    traj_range = trajectory.ptp(axis=0) + 1e-8
    norm_traj = (trajectory - traj_min) / traj_range
    grid_indices = np.clip((norm_traj * grid_size).astype(int), 0, grid_size - 1)
    grid_cells = grid_indices[:, 0] * grid_size + grid_indices[:, 1]
    unique_cells, cell_counts = np.unique(grid_cells, return_counts=True)
    cell_probs = cell_counts / cell_counts.sum()
    grid_occ_entropy = -np.sum(cell_probs * np.log2(cell_probs + 1e-12))

    # Grid transition entropy
    if len(grid_cells) > 1:
        transitions = grid_cells[:-1] * grid_size * grid_size + grid_cells[1:]
        unique_trans, trans_counts = np.unique(transitions, return_counts=True)
        trans_probs = trans_counts / trans_counts.sum()
        grid_trans_entropy = -np.sum(trans_probs * np.log2(trans_probs + 1e-12))
    else:
        grid_trans_entropy = 0.0

    return np.array([
        speed_entropy, accel_entropy, heading_entropy,
        turn_entropy, grid_occ_entropy, grid_trans_entropy
    ])


def extract_all_features(trajectories, n_bins=16, grid_size=10):
    """
    Extract all three views of features for a list of trajectories.

    Returns:
        spatial_features: (n, 7)
        kinematic_features: (n, 8)
        entropy_features: (n, 6)
        all_features: (n, 21) concatenated
    """
    spatial_list = []
    kinematic_list = []
    entropy_list = []

    for traj in trajectories:
        spatial_list.append(extract_spatial_features(traj))
        kinematic_list.append(extract_kinematic_features(traj))
        entropy_list.append(extract_entropy_features(traj, n_bins, grid_size))

    spatial_features = np.array(spatial_list)
    kinematic_features = np.array(kinematic_list)
    entropy_features = np.array(entropy_list)
    all_features = np.hstack([spatial_features, kinematic_features, entropy_features])

    return spatial_features, kinematic_features, entropy_features, all_features


# Feature names for reporting
SPATIAL_NAMES = [
    'total_length', 'od_distance', 'detour_ratio',
    'mean_lateral_dev', 'max_lateral_dev',
    'bbox_area', 'radius_of_gyration'
]

KINEMATIC_NAMES = [
    'mean_speed', 'std_speed', 'max_speed',
    'mean_accel', 'std_accel',
    'stop_ratio', 'mean_turn', 'max_turn'
]

ENTROPY_NAMES = [
    'speed_entropy', 'accel_entropy', 'heading_entropy',
    'turn_entropy', 'grid_occ_entropy', 'grid_trans_entropy'
]

ALL_FEATURE_NAMES = SPATIAL_NAMES + KINEMATIC_NAMES + ENTROPY_NAMES


if __name__ == '__main__':
    from data_generator import generate_synthetic_trajectories

    trajs, labels, _, _ = generate_synthetic_trajectories(n_normal=100, n_anomaly_per_type=10)
    s, k, e, a = extract_all_features(trajs)
    print(f"Spatial features: {s.shape}")
    print(f"Kinematic features: {k.shape}")
    print(f"Entropy features: {e.shape}")
    print(f"All features: {a.shape}")
