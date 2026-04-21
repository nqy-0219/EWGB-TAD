"""
Enhanced baselines including trajectory-aware methods.
"""

import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from .baselines import BaselineDetector, get_all_baselines


class DTWKNNDetector(BaselineDetector):
    """
    DTW-KNN: k-Nearest Neighbor anomaly detector using simplified
    trajectory distance (Euclidean on resampled trajectories).
    Since all trajectories are resampled to same length, we use
    point-wise Euclidean as a fast proxy for DTW.
    """
    def __init__(self, k=5):
        super().__init__('DTW-KNN')
        self.k = k

    def fit(self, features, trajectories=None):
        """Store trajectories for distance computation."""
        self._trajectories = trajectories
        if trajectories is not None:
            # Flatten trajectories for distance computation
            self._flat = np.array([t.flatten() for t in trajectories])
        else:
            self._flat = features

    def score(self, features, trajectories=None):
        if trajectories is not None:
            flat = np.array([t.flatten() for t in trajectories])
        else:
            flat = features

        n = len(flat)
        scores = np.zeros(n)

        # Batch compute distances
        for i in range(n):
            dists = np.linalg.norm(self._flat - flat[i], axis=1)
            dists[i] = np.inf  # exclude self
            knn_dists = np.sort(dists)[:self.k]
            scores[i] = np.mean(knn_dists)

        return scores

    def fit_score(self, features, trajectories=None):
        t0 = time.time()
        self.fit(features, trajectories)
        self.fit_time = time.time() - t0

        t0 = time.time()
        scores = self.score(features, trajectories)
        self.score_time = time.time() - t0
        return scores


class SegmentOutlierDetector(BaselineDetector):
    """
    Segment-based outlier detection inspired by TRAOD.
    Partitions trajectories into segments, detects outlying segments,
    aggregates segment outlier scores per trajectory.
    """
    def __init__(self, n_segments=4, k=10):
        super().__init__('SegmentOD')
        self.n_segments = n_segments
        self.k = k

    def fit(self, features, trajectories=None):
        self._trajectories = trajectories

    def score(self, features, trajectories=None):
        if trajectories is None:
            # Fallback to feature-based KNN
            from sklearn.neighbors import NearestNeighbors
            scaler = StandardScaler()
            X = scaler.fit_transform(features)
            nn = NearestNeighbors(n_neighbors=self.k + 1)
            nn.fit(X)
            dists, _ = nn.kneighbors(X)
            return dists[:, -1]

        trajs = trajectories
        n = len(trajs)
        seq_len = len(trajs[0])
        seg_len = seq_len // self.n_segments

        # Extract segment features (direction + length for each segment)
        all_seg_features = []
        for traj in trajs:
            seg_feats = []
            for s in range(self.n_segments):
                start_idx = s * seg_len
                end_idx = min((s + 1) * seg_len, seq_len)
                seg = traj[start_idx:end_idx]
                if len(seg) < 2:
                    seg_feats.extend([0, 0, 0, 0])
                    continue
                direction = seg[-1] - seg[0]
                length = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
                seg_feats.extend([direction[0], direction[1], length,
                                 np.linalg.norm(direction)])
            all_seg_features.append(seg_feats)

        seg_features = np.array(all_seg_features)
        scaler = StandardScaler()
        seg_features = scaler.fit_transform(seg_features)

        # KNN on segment features
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(self.k + 1, n))
        nn.fit(seg_features)
        dists, _ = nn.kneighbors(seg_features)
        scores = dists[:, -1]

        return scores

    def fit_score(self, features, trajectories=None):
        t0 = time.time()
        self.fit(features, trajectories)
        self.fit_time = time.time() - t0

        t0 = time.time()
        scores = self.score(features, trajectories)
        self.score_time = time.time() - t0
        return scores


class ClusterPrototypeDetector(BaselineDetector):
    """
    K-Means prototype detector (ablation baseline for granular-ball).
    Uses k-means clustering and scores by distance to nearest cluster center
    adjusted by cluster size.
    """
    def __init__(self, n_clusters=20):
        super().__init__('KMeans-Prototype')
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()

    def fit(self, features):
        from sklearn.cluster import KMeans
        X = self.scaler.fit_transform(features)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=3)
        self.kmeans.fit(X)
        # Compute cluster sizes
        labels = self.kmeans.labels_
        self.cluster_sizes = np.bincount(labels, minlength=self.n_clusters).astype(float)

    def score(self, features):
        X = self.scaler.transform(features)
        centers = self.kmeans.cluster_centers_
        n = len(X)
        scores = np.zeros(n)
        for i in range(n):
            dists = np.linalg.norm(X[i] - centers, axis=1)
            # Score = distance / log(1 + cluster_size), same formulation as GB
            density_factors = 1.0 / (np.log1p(self.cluster_sizes / (np.max(dists) + 1e-8)) + 1e-8)
            ball_scores = dists * density_factors
            scores[i] = np.min(ball_scores)
        return scores


def get_all_baselines_v2(contamination=0.1, seed=42):
    """Return all baselines including trajectory-aware ones."""
    baselines = get_all_baselines(contamination=contamination, seed=seed)
    baselines['DTW-KNN'] = DTWKNNDetector(k=5)
    baselines['SegmentOD'] = SegmentOutlierDetector(n_segments=4)
    baselines['KMeans-Prototype'] = ClusterPrototypeDetector(n_clusters=20)
    return baselines


if __name__ == '__main__':
    from data_generator import generate_synthetic_trajectories
    from feature_extraction import extract_all_features
    from sklearn.metrics import roc_auc_score

    trajs, labels, _, _ = generate_synthetic_trajectories(n_normal=1000, n_anomaly_per_type=25)
    _, _, _, features = extract_all_features(trajs)

    baselines = get_all_baselines_v2()
    for name, det in baselines.items():
        if hasattr(det, 'fit_score') and name in ('DTW-KNN', 'SegmentOD'):
            scores = det.fit_score(features, trajs)
        else:
            scores = det.fit_score(features)
        auc = roc_auc_score(labels, scores)
        print(f"{name:20s}: AUC={auc:.4f}")
