"""
Semi-real trajectory generation mimicking real GPS data properties.
Uses grid-based road network with realistic noise, speed variation,
and trip diversity to create trajectories closer to real-world data.
"""

import numpy as np
from .data_generator import resample_trajectory


def generate_grid_network_trajectories(n_normal=5000, n_anomaly_per_type=125,
                                        seq_len=32, seed=42):
    """
    Generate trajectories on a grid-like road network with realistic properties:
    - Grid road network (10x10) with some diagonal shortcuts
    - Variable trip lengths and origins/destinations
    - Realistic GPS jitter (σ ~ 5-15m equivalent)
    - Speed variation along routes
    - Turn-by-turn path following

    Returns same format as generate_synthetic_trajectories.
    """
    rng = np.random.RandomState(seed)

    # Build a 10x10 grid road network
    grid_size = 10
    # Node positions (normalized to [0,1])
    nodes = {}
    for i in range(grid_size):
        for j in range(grid_size):
            # Add slight irregularity to grid (like real road networks)
            nodes[(i, j)] = np.array([
                i / (grid_size - 1) + rng.normal(0, 0.01),
                j / (grid_size - 1) + rng.normal(0, 0.01)
            ])

    # Edges: grid connections + some diagonals
    edges = {}
    for i in range(grid_size):
        for j in range(grid_size):
            neighbors = []
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    neighbors.append((ni, nj))
            # Add some diagonals (30% chance)
            for di, dj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size and rng.random() < 0.3:
                    neighbors.append((ni, nj))
            edges[(i, j)] = neighbors

    def shortest_path_bfs(start, end):
        """Simple BFS for shortest path on grid."""
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            for neighbor in edges.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def generate_road_trajectory(start_node, end_node):
        """Generate a trajectory following the road network."""
        path = shortest_path_bfs(start_node, end_node)
        if path is None or len(path) < 2:
            # Fallback: direct path
            path = [start_node, end_node]

        # Convert node path to continuous trajectory with intermediate points
        points = []
        for k in range(len(path) - 1):
            p1 = nodes[path[k]]
            p2 = nodes[path[k + 1]]
            # Add 3-5 intermediate points per segment
            n_inter = rng.randint(3, 6)
            for t in np.linspace(0, 1, n_inter, endpoint=False):
                pt = p1 + t * (p2 - p1)
                # Add GPS jitter (realistic: ~10m in normalized coords ≈ 0.005)
                pt += rng.normal(0, 0.005, 2)
                points.append(pt)
        points.append(nodes[path[-1]] + rng.normal(0, 0.005, 2))

        traj = np.array(points)
        return resample_trajectory(traj, seq_len)

    trajectories = []
    labels = []
    anomaly_types = []
    contexts = []

    # Generate normal trajectories with diverse OD pairs
    od_pairs = []
    for _ in range(min(50, n_normal)):
        si, sj = rng.randint(0, grid_size), rng.randint(0, grid_size)
        ei, ej = rng.randint(0, grid_size), rng.randint(0, grid_size)
        while abs(si - ei) + abs(sj - ej) < 3:  # minimum distance
            ei, ej = rng.randint(0, grid_size), rng.randint(0, grid_size)
        od_pairs.append(((si, sj), (ei, ej)))

    for i in range(n_normal):
        od = od_pairs[i % len(od_pairs)]
        traj = generate_road_trajectory(od[0], od[1])
        trajectories.append(traj)
        labels.append(0)
        anomaly_types.append(-1)
        contexts.append(i % len(od_pairs))

    # Generate anomalies
    for anom_type in range(4):
        for j in range(n_anomaly_per_type):
            od = od_pairs[rng.randint(len(od_pairs))]
            base_traj = generate_road_trajectory(od[0], od[1])
            od_dist = np.linalg.norm(base_traj[-1] - base_traj[0])

            if anom_type == 0:  # DETOUR - go off-road
                s = int(0.3 * seq_len)
                e = int(0.7 * seq_len)
                direction = base_traj[-1] - base_traj[0]
                perp = np.array([-direction[1], direction[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-8)
                offset_mag = rng.uniform(0.08, 0.2) * max(od_dist, 0.1)
                sign = rng.choice([-1, 1])
                for k in range(s, e):
                    frac = (k - s) / (e - s + 1e-8)
                    base_traj[k] += sign * perp * offset_mag * np.sin(np.pi * frac)

            elif anom_type == 1:  # LOOP
                center = seq_len // 2
                radius = rng.uniform(0.03, 0.1) * max(od_dist, 0.1)
                n_loop = min(8, seq_len // 4)
                start_idx = max(0, center - n_loop // 2)
                for k in range(n_loop):
                    idx = start_idx + k
                    if idx < seq_len:
                        angle = 2 * np.pi * k / n_loop
                        base_traj[idx] += np.array([radius * np.cos(angle), radius * np.sin(angle)])

            elif anom_type == 2:  # SPEED - sudden stop
                start = int(rng.uniform(0.2, 0.5) * seq_len)
                n_slow = min(6, seq_len // 5)
                center_pt = base_traj[start].copy()
                for k in range(n_slow):
                    if start + k < seq_len:
                        base_traj[start + k] = center_pt + rng.normal(0, 0.003, 2)

            elif anom_type == 3:  # ROUTE DEVIATION
                dev_start = int(rng.uniform(0.5, 0.7) * seq_len)
                angle_dev = rng.uniform(20, 50) * np.pi / 180 * rng.choice([-1, 1])
                for k in range(dev_start, seq_len):
                    if k > 0:
                        diff = base_traj[k] - base_traj[k-1]
                        cos_a, sin_a = np.cos(angle_dev), np.sin(angle_dev)
                        rotated = np.array([cos_a * diff[0] - sin_a * diff[1],
                                           sin_a * diff[0] + cos_a * diff[1]])
                        base_traj[k] = base_traj[k-1] + rotated
                    angle_dev *= 0.95

            trajectories.append(base_traj)
            labels.append(1)
            anomaly_types.append(anom_type)
            contexts.append(rng.randint(len(od_pairs)))

    # Shuffle
    indices = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in indices]
    labels = np.array(labels)[indices]
    anomaly_types = np.array(anomaly_types)[indices]
    contexts = np.array(contexts)[indices]

    return trajectories, labels, anomaly_types, contexts
