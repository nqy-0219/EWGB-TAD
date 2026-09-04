"""GeoLife trajectory loading, segmentation, and anomaly injection utilities."""

import glob
import os
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from data_generator import resample_trajectory


GEOLIFE_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "data", "geolife", "Geolife Trajectories 1.3", "Data"
)


def load_geolife_trajectories(
    root=None,
    max_trajs=10000,
    seq_len=32,
    min_points=20,
    max_points=2000,
    seed=42,
    bbox=None,
    min_od_dist_deg=0.005,
):
    if root is None:
        root = GEOLIFE_ROOT
    if bbox is None:
        bbox = (116.1, 39.75, 116.65, 40.15)
    rng = np.random.RandomState(seed)
    trajectories = []
    lon_min, lat_min, lon_max, lat_max = bbox
    user_dirs = sorted(glob.glob(os.path.join(root, "*")))
    print(f"  Found {len(user_dirs)} users in GeoLife")
    print(f"  Filtering to bbox: lon[{lon_min},{lon_max}] lat[{lat_min},{lat_max}]")

    for user_dir in user_dirs:
        trajectory_dir = os.path.join(user_dir, "Trajectory")
        if not os.path.isdir(trajectory_dir):
            continue
        for plt_file in sorted(glob.glob(os.path.join(trajectory_dir, "*.plt"))):
            try:
                points = []
                with open(plt_file, "r") as handle:
                    for line_number, line in enumerate(handle):
                        if line_number < 6:
                            continue
                        parts = line.strip().split(",")
                        if len(parts) >= 2:
                            lat, lon = float(parts[0]), float(parts[1])
                            if lat_min < lat < lat_max and lon_min < lon < lon_max:
                                points.append([lon, lat])
                if len(points) < min_points:
                    continue
                segments = segment_trajectory(
                    np.asarray(points), max_gap_meters=300, min_segment_len=min_points
                )
                for segment in segments:
                    if min_points <= len(segment) <= max_points:
                        od_distance = np.linalg.norm(segment[-1] - segment[0])
                        if od_distance >= min_od_dist_deg:
                            trajectories.append(resample_trajectory(segment, seq_len))
            except (ValueError, IndexError):
                continue

    rng.shuffle(trajectories)
    trajectories = trajectories[:max_trajs]
    print(f"  Loaded {len(trajectories)} trajectory segments from GeoLife")
    return trajectories


def segment_trajectory(points, max_gap_meters=500, min_segment_len=15):
    if len(points) < min_segment_len:
        return []
    distances_m = np.linalg.norm(np.diff(points, axis=0), axis=1) * 111000
    segments = []
    start = 0
    for index, distance in enumerate(distances_m):
        if distance > max_gap_meters:
            if index - start >= min_segment_len:
                segments.append(points[start : index + 1])
            start = index + 1
    if len(points) - start >= min_segment_len:
        segments.append(points[start:])
    return segments


def cluster_geolife_trajectories(trajectories, n_clusters=15, seed=42):
    endpoint_features = np.asarray(
        [[trajectory[0, 0], trajectory[0, 1], trajectory[-1, 0], trajectory[-1, 1]] for trajectory in trajectories]
    )
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(endpoint_features)
    clusters = defaultdict(list)
    for index, cluster_id in enumerate(labels):
        clusters[cluster_id].append(index)
    return clusters, kmeans.cluster_centers_


def inject_anomalies_geolife(trajectories, contamination=0.1, seed=42):
    rng = np.random.RandomState(seed)
    n = len(trajectories)
    n_anomaly = int(n * contamination / (1 - contamination))
    n_per_type = max(1, n_anomaly // 4)
    injected = []
    anomaly_types = []

    for anomaly_type in range(4):
        for _ in range(n_per_type):
            index = rng.randint(0, n)
            trajectory = trajectories[index].copy()
            seq_len = len(trajectory)
            od_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            if od_distance < 1e-6:
                od_distance = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) * seq_len
            if anomaly_type == 0:
                start, end = int(0.2 * seq_len), int(0.8 * seq_len)
                direction = trajectory[-1] - trajectory[0]
                if np.linalg.norm(direction) < 1e-8:
                    direction = trajectory[seq_len // 2] - trajectory[0]
                perpendicular = np.array([-direction[1], direction[0]])
                perpendicular /= np.linalg.norm(perpendicular) + 1e-8
                offset = rng.uniform(0.4, 0.7) * od_distance
                sign = rng.choice([-1, 1])
                for point in range(start, end):
                    fraction = (point - start) / (end - start + 1e-8)
                    trajectory[point] += sign * perpendicular * offset * np.sin(np.pi * fraction)
            elif anomaly_type == 1:
                center = seq_len // 2
                radius = rng.uniform(0.2, 0.4) * od_distance
                n_loop = min(12, seq_len // 3)
                start = max(0, center - n_loop // 2)
                for point in range(n_loop):
                    index_point = start + point
                    if index_point < seq_len:
                        angle = 2 * np.pi * point / n_loop
                        trajectory[index_point] += np.array([radius * np.cos(angle), radius * np.sin(angle)])
            elif anomaly_type == 2:
                start = int(rng.uniform(0.2, 0.5) * seq_len)
                n_slow = min(10, seq_len // 3)
                center_point = trajectory[start].copy()
                for point in range(n_slow):
                    if start + point < seq_len:
                        trajectory[start + point] = center_point + rng.normal(0, od_distance * 0.003, 2)
            else:
                start = int(rng.uniform(0.35, 0.55) * seq_len)
                angle_deviation = rng.uniform(50, 90) * np.pi / 180 * rng.choice([-1, 1])
                for point in range(start, seq_len):
                    if point > 0:
                        difference = trajectory[point] - trajectory[point - 1]
                        if np.linalg.norm(difference) < 1e-10:
                            difference = (trajectory[-1] - trajectory[0]) / seq_len
                        cosine, sine = np.cos(angle_deviation), np.sin(angle_deviation)
                        rotated = np.array(
                            [
                                cosine * difference[0] - sine * difference[1],
                                sine * difference[0] + cosine * difference[1],
                            ]
                        )
                        trajectory[point] = trajectory[point - 1] + rotated * 1.5
                    angle_deviation *= 0.90
            injected.append(trajectory)
            anomaly_types.append(anomaly_type)

    all_trajectories = list(trajectories) + injected
    labels = np.concatenate([np.zeros(n), np.ones(len(injected))])
    all_anomaly_types = np.concatenate([np.full(n, -1), np.array(anomaly_types)])
    permutation = rng.permutation(len(all_trajectories))
    return (
        [all_trajectories[index] for index in permutation],
        labels[permutation],
        all_anomaly_types[permutation],
    )
