"""Porto-derived trajectory construction and anomaly injection utilities."""

import csv
import re
from collections import defaultdict

import numpy as np


def parse_point(value):
    match = re.match(r"POINT\(([-\d.]+)\s+([-\d.]+)\)", value)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def load_porto_od_pairs(csv_path, max_pairs=50000, seed=42):
    rng = np.random.RandomState(seed)
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = parse_point(row["source_point"])
            target = parse_point(row["target_point"])
            if source is None or target is None:
                continue
            lon1, lat1 = source
            lon2, lat2 = target
            distance = np.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)
            if distance < 0.005 or distance > 0.15:
                continue
            pairs.append((np.array([lon1, lat1]), np.array([lon2, lat2])))
            if len(pairs) >= max_pairs * 3:
                break
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    print(f"  Loaded {len(pairs)} OD pairs from Porto Taxi")
    return pairs


def cluster_od_pairs(od_pairs, n_clusters=20, seed=42):
    from sklearn.cluster import KMeans

    od_flat = np.array([[s[0], s[1], e[0], e[1]] for s, e in od_pairs])
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(od_flat)
    clusters = defaultdict(list)
    for index, cluster_id in enumerate(cluster_labels):
        clusters[cluster_id].append(index)
    return clusters, kmeans.cluster_centers_


def generate_trajectory_from_od(start, end, seq_len=32, rng=None, route_template=None):
    if rng is None:
        rng = np.random.RandomState(42)
    t = np.linspace(0, 1, seq_len).reshape(-1, 1)
    path = start + t * (end - start)
    if route_template is not None:
        path += route_template * rng.uniform(0.8, 1.2)
    else:
        for _ in range(rng.randint(1, 3)):
            waypoint_t = rng.uniform(0.2, 0.8)
            waypoint_spread = rng.uniform(0.05, 0.15)
            direction = end - start
            perpendicular = np.array([-direction[1], direction[0]])
            perpendicular /= np.linalg.norm(perpendicular) + 1e-8
            offset = rng.normal(0, 0.08) * np.linalg.norm(direction)
            weights = np.exp(
                -0.5
                * ((np.linspace(0, 1, seq_len) - waypoint_t) / waypoint_spread) ** 2
            ).reshape(-1, 1)
            path += weights * perpendicular * offset
    path += rng.normal(0, 0.0001, path.shape)
    return path


def generate_route_template(center_start, center_end, seq_len=32, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    direction = center_end - center_start
    perpendicular = np.array([-direction[1], direction[0]])
    perpendicular /= np.linalg.norm(perpendicular) + 1e-8
    t = np.linspace(0, 1, seq_len)
    template = np.zeros((seq_len, 2))
    for _ in range(rng.randint(1, 4)):
        bend_t = rng.uniform(0.15, 0.85)
        bend_spread = rng.uniform(0.08, 0.2)
        bend_magnitude = rng.normal(0, 0.1) * np.linalg.norm(direction)
        weights = np.exp(-0.5 * ((t - bend_t) / bend_spread) ** 2)
        template += np.outer(weights, perpendicular * bend_magnitude)
    return template


def inject_anomalies_porto(trajectories, contamination=0.1, seed=42):
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
            if anomaly_type == 0:
                start, end = int(0.3 * seq_len), int(0.7 * seq_len)
                direction = trajectory[-1] - trajectory[0]
                perpendicular = np.array([-direction[1], direction[0]])
                perpendicular /= np.linalg.norm(perpendicular) + 1e-8
                offset = rng.uniform(0.15, 0.35) * od_distance
                sign = rng.choice([-1, 1])
                for point in range(start, end):
                    fraction = (point - start) / (end - start + 1e-8)
                    trajectory[point] += sign * perpendicular * offset * np.sin(
                        np.pi * fraction
                    )
            elif anomaly_type == 1:
                center = seq_len // 2
                radius = rng.uniform(0.08, 0.2) * od_distance
                n_loop = min(10, seq_len // 3)
                start = max(0, center - n_loop // 2)
                for point in range(n_loop):
                    index_point = start + point
                    if index_point < seq_len:
                        angle = 2 * np.pi * point / n_loop
                        trajectory[index_point] += np.array(
                            [radius * np.cos(angle), radius * np.sin(angle)]
                        )
            elif anomaly_type == 2:
                start = int(rng.uniform(0.2, 0.5) * seq_len)
                n_slow = min(8, seq_len // 4)
                center_point = trajectory[start].copy()
                for point in range(n_slow):
                    if start + point < seq_len:
                        trajectory[start + point] = center_point + rng.normal(
                            0, od_distance * 0.003, 2
                        )
            else:
                start = int(rng.uniform(0.4, 0.6) * seq_len)
                angle_deviation = rng.uniform(40, 80) * np.pi / 180 * rng.choice([-1, 1])
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
                        trajectory[point] = trajectory[point - 1] + rotated * 1.3
                    angle_deviation *= 0.92
            injected.append(trajectory)
            anomaly_types.append(anomaly_type)

    all_trajectories = list(trajectories) + injected
    all_labels = np.concatenate([np.zeros(n), np.ones(len(injected))])
    all_anomaly_types = np.concatenate([np.full(n, -1), np.array(anomaly_types)])
    permutation = rng.permutation(len(all_trajectories))
    return (
        [all_trajectories[index] for index in permutation],
        all_labels[permutation],
        all_anomaly_types[permutation],
    )
