"""
Enhanced baselines including trajectory-aware methods.
"""

import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from baselines import BaselineDetector, get_all_baselines


class ShapeKNNDetector(BaselineDetector):
    """
    Shape-KNN: k-nearest-neighbor anomaly detector on resampled
    trajectory coordinates using point-wise Euclidean distance.
    """
    def __init__(self, k=5):
        super().__init__('Shape-KNN')
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
            # Fallback to feature-based nearest-neighbor scoring.
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

        # Nearest-neighbor scoring on segment features.
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


class SafetyDetector(BaselineDetector):
    """
    Safety-style spatial-feature mixed trajectory outlier detector.

    The implementation follows the Safety paper's two-source idea: spatial
    rarity is estimated from trajectory point occupancy over an adaptive grid,
    and feature-level local anomalies are estimated with LOF over handcrafted
    trajectory features. The final score is the average of normalized spatial
    and feature anomaly evidence.
    """
    def __init__(self, grid_size=24, n_neighbors=20):
        super().__init__('Safety')
        self.grid_size = grid_size
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()

    @staticmethod
    def _normalize(scores):
        scores = np.asarray(scores, dtype=float)
        smin, smax = np.min(scores), np.max(scores)
        if smax - smin < 1e-12:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)

    def _spatial_rarity(self, trajectories):
        points = np.vstack(trajectories)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        span = np.maximum(maxs - mins, 1e-12)

        grid_counts = np.zeros((self.grid_size, self.grid_size), dtype=float)
        traj_cells = []
        for traj in trajectories:
            idx = np.floor((traj - mins) / span * self.grid_size).astype(int)
            idx = np.clip(idx, 0, self.grid_size - 1)
            cells = idx[:, 0] * self.grid_size + idx[:, 1]
            traj_cells.append(cells)
            xs = idx[:, 0]
            ys = idx[:, 1]
            np.add.at(grid_counts, (xs, ys), 1.0)

        prob = grid_counts / (grid_counts.sum() + 1e-12)
        prob = np.maximum(prob, 1e-12)

        scores = []
        for cells in traj_cells:
            xs = cells // self.grid_size
            ys = cells % self.grid_size
            scores.append(float(np.mean(-np.log(prob[xs, ys]))))
        return np.asarray(scores)

    def fit(self, features, trajectories=None):
        self._features = features
        self._trajectories = trajectories
        return self

    def score(self, features, trajectories=None):
        if trajectories is None:
            trajectories = self._trajectories

        X = self.scaler.fit_transform(features)
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=min(self.n_neighbors, len(X) - 1))
        lof.fit_predict(X)
        feature_scores = -lof.negative_outlier_factor_

        if trajectories is None:
            return self._normalize(feature_scores)

        spatial_scores = self._spatial_rarity(trajectories)
        return 0.5 * self._normalize(spatial_scores) + 0.5 * self._normalize(feature_scores)

    def fit_score(self, features, trajectories=None):
        t0 = time.time()
        self.fit(features, trajectories)
        self.fit_time = time.time() - t0

        t0 = time.time()
        scores = self.score(features, trajectories)
        self.score_time = time.time() - t0
        return scores


class DTWMLDetector(BaselineDetector):
    """
    CPU-friendly DTW/ML trajectory anomaly detector.

    Trajectories are represented by banded-DTW distances to a small set of
    trajectory prototypes, then IsolationForest is applied to these DTW
    similarity features. This follows the DTW + classical ML principle while
    avoiding an O(N^2) full DTW distance matrix.
    """
    def __init__(self, n_prototypes=16, band=4, contamination=0.1, seed=42):
        super().__init__('DTW-ML')
        self.n_prototypes = n_prototypes
        self.band = band
        self.contamination = contamination
        self.seed = seed
        self.scaler = StandardScaler()

    @staticmethod
    def _dtw_distance(a, b, band=4):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n, m = len(a), len(b)
        band = max(band, abs(n - m))
        inf = np.inf
        prev = np.full(m + 1, inf)
        curr = np.full(m + 1, inf)
        prev[0] = 0.0
        for i in range(1, n + 1):
            curr.fill(inf)
            j_start = max(1, i - band)
            j_end = min(m, i + band)
            for j in range(j_start, j_end + 1):
                cost = np.linalg.norm(a[i - 1] - b[j - 1])
                curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev
        return prev[m] / (n + m)

    @staticmethod
    def _dtw_distance_batch(trajectories, prototype, band=4):
        """Vectorized banded DTW from many equal-length trajectories to one prototype."""
        trajectories = np.asarray(trajectories, dtype=float)
        prototype = np.asarray(prototype, dtype=float)
        n_items, n, _ = trajectories.shape
        m = len(prototype)
        band = max(band, abs(n - m))
        prev = np.full((n_items, m + 1), np.inf)
        curr = np.full((n_items, m + 1), np.inf)
        prev[:, 0] = 0.0
        for i in range(1, n + 1):
            curr.fill(np.inf)
            j_start = max(1, i - band)
            j_end = min(m, i + band)
            point = trajectories[:, i - 1, :]
            for j in range(j_start, j_end + 1):
                cost = np.linalg.norm(point - prototype[j - 1], axis=1)
                curr[:, j] = cost + np.minimum(np.minimum(prev[:, j], curr[:, j - 1]), prev[:, j - 1])
            prev, curr = curr, prev
        return prev[:, m] / (n + m)

    def fit(self, features, trajectories=None):
        if trajectories is None:
            raise ValueError("DTW-ML requires trajectory sequences.")

        from sklearn.cluster import MiniBatchKMeans
        from sklearn.ensemble import IsolationForest

        self._trajectories = trajectories
        flat = np.array([traj.flatten() for traj in trajectories])
        flat_scaled = self.scaler.fit_transform(flat)
        n_clusters = min(self.n_prototypes, len(flat_scaled))
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed, n_init=3, batch_size=512)
        labels = km.fit_predict(flat_scaled)

        prototypes = []
        for k in range(n_clusters):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            center = km.cluster_centers_[k]
            nearest = idx[np.argmin(np.linalg.norm(flat_scaled[idx] - center, axis=1))]
            prototypes.append(trajectories[int(nearest)])
        self.prototypes_ = prototypes

        dtw_features = self._dtw_features(trajectories)
        self.iforest_ = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.iforest_.fit(dtw_features)
        return self

    def _dtw_features(self, trajectories):
        feats = np.zeros((len(trajectories), len(self.prototypes_)), dtype=float)
        try:
            traj_arr = np.asarray(trajectories, dtype=float)
            equal_length = traj_arr.ndim == 3 and all(np.asarray(p).shape == traj_arr.shape[1:] for p in self.prototypes_)
        except ValueError:
            traj_arr = None
            equal_length = False

        if equal_length:
            for j, proto in enumerate(self.prototypes_):
                feats[:, j] = self._dtw_distance_batch(traj_arr, proto, band=self.band)
            return feats

        for i, traj in enumerate(trajectories):
            for j, proto in enumerate(self.prototypes_):
                feats[i, j] = self._dtw_distance(traj, proto, band=self.band)
        return feats

    def score(self, features, trajectories=None):
        if trajectories is None:
            trajectories = self._trajectories
        dtw_features = self._dtw_features(trajectories)
        return -self.iforest_.decision_function(dtw_features)

    def fit_score(self, features, trajectories=None):
        t0 = time.time()
        self.fit(features, trajectories)
        self.fit_time = time.time() - t0

        t0 = time.time()
        scores = self.score(features, trajectories)
        self.score_time = time.time() - t0
        return scores


class TADSDetector(BaselineDetector):
    """
    TADS-style stay-region and symbolic sub-trajectory detector.

    The original TADS method mines stay regions and detects anomalous
    sub-trajectories using symbolic similarity and adaptive clustering. This
    CPU adaptation uses the same idea under the unified resampled-trajectory
    input: trajectories are mapped to adaptive grid regions, local repetition
    and turning statistics are extracted, and route-pattern deviation is scored
    against symbolic route prototypes.
    """
    def __init__(self, grid_size=18, n_prototypes=18, contamination=0.1, seed=42):
        super().__init__('TADS')
        self.grid_size = grid_size
        self.n_prototypes = n_prototypes
        self.contamination = contamination
        self.seed = seed
        self.scaler = StandardScaler()

    @staticmethod
    def _normalize(scores):
        scores = np.asarray(scores, dtype=float)
        smin, smax = np.min(scores), np.max(scores)
        if smax - smin < 1e-12:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)

    def _cells(self, trajectories):
        cells = []
        for traj in trajectories:
            idx = np.floor((traj - self.mins_) / self.span_ * self.grid_size).astype(int)
            idx = np.clip(idx, 0, self.grid_size - 1)
            cells.append(idx[:, 0] * self.grid_size + idx[:, 1])
        return np.asarray(cells, dtype=int)

    @staticmethod
    def _compressed(seq):
        if len(seq) == 0:
            return seq
        keep = [seq[0]]
        for value in seq[1:]:
            if value != keep[-1]:
                keep.append(value)
        return np.asarray(keep, dtype=int)

    def _build_symbolic_model(self, cell_array):
        from collections import Counter
        from sklearn.cluster import MiniBatchKMeans

        unigram = Counter()
        bigram = Counter()
        for cells in cell_array:
            comp = self._compressed(cells)
            unigram.update(int(c) for c in comp)
            bigram.update((int(comp[i]), int(comp[i + 1])) for i in range(len(comp) - 1))
        total_uni = sum(unigram.values()) + self.grid_size * self.grid_size
        total_bi = sum(bigram.values()) + max(1, len(bigram))
        self.cell_prob_ = {cell: (count + 1.0) / total_uni for cell, count in unigram.items()}
        self.bigram_prob_ = {pair: (count + 1.0) / total_bi for pair, count in bigram.items()}
        self.default_cell_prob_ = 1.0 / total_uni
        self.default_bigram_prob_ = 1.0 / total_bi

        n_clusters = min(self.n_prototypes, len(cell_array))
        flat = np.asarray(cell_array, dtype=float)
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed, n_init=3, batch_size=512)
        labels = km.fit_predict(flat)
        prototypes = []
        for k in range(n_clusters):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            center = km.cluster_centers_[k]
            nearest = idx[np.argmin(np.linalg.norm(flat[idx] - center, axis=1))]
            prototypes.append(cell_array[int(nearest)])
        self.prototype_cells_ = np.asarray(prototypes, dtype=int)

    def _symbolic_features(self, trajectories):
        cell_array = self._cells(trajectories)
        out = []
        for traj, cells in zip(trajectories, cell_array):
            traj = np.asarray(traj, dtype=float)
            steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            total_len = float(np.sum(steps))
            od = float(np.linalg.norm(traj[-1] - traj[0]))
            directness = total_len / (od + 1e-8)
            low_step_ratio = float(np.mean(steps < (np.median(steps) * 0.35 + 1e-12))) if len(steps) else 0.0

            dirs = np.diff(traj, axis=0)
            angles = np.arctan2(dirs[:, 1], dirs[:, 0])
            if len(angles) > 1:
                turns = np.abs(np.diff(np.unwrap(angles)))
                mean_turn = float(np.mean(turns))
                max_turn = float(np.max(turns))
                large_turn_ratio = float(np.mean(turns > (np.pi / 3)))
            else:
                mean_turn = max_turn = large_turn_ratio = 0.0

            unique, counts = np.unique(cells, return_counts=True)
            repeat_ratio = 1.0 - len(unique) / max(1, len(cells))
            max_dwell = float(np.max(counts) / max(1, len(cells)))
            probs = counts / max(1, len(cells))
            cell_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

            comp = self._compressed(cells)
            cell_rarity = float(np.mean([
                -np.log(self.cell_prob_.get(int(c), self.default_cell_prob_))
                for c in comp
            ])) if len(comp) else 0.0
            if len(comp) > 1:
                bigram_rarity = float(np.mean([
                    -np.log(self.bigram_prob_.get((int(comp[i]), int(comp[i + 1])), self.default_bigram_prob_))
                    for i in range(len(comp) - 1)
                ]))
            else:
                bigram_rarity = cell_rarity

            if len(self.prototype_cells_) > 0:
                mismatches = np.mean(self.prototype_cells_ != cells.reshape(1, -1), axis=1)
                route_dist = float(np.min(mismatches))
            else:
                route_dist = 0.0

            out.append([
                route_dist,
                cell_rarity,
                bigram_rarity,
                repeat_ratio,
                max_dwell,
                low_step_ratio,
                directness,
                mean_turn,
                max_turn,
                large_turn_ratio,
                cell_entropy,
            ])
        return np.asarray(out, dtype=float)

    def fit(self, features, trajectories=None):
        if trajectories is None:
            from sklearn.ensemble import IsolationForest

            X = self.scaler.fit_transform(features)
            self.model_ = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=self.seed,
                n_jobs=-1,
            ).fit(X)
            self.mode_ = "feature"
            return self

        from sklearn.ensemble import IsolationForest

        self._trajectories = trajectories
        points = np.vstack(trajectories)
        self.mins_ = points.min(axis=0)
        self.span_ = np.maximum(points.max(axis=0) - self.mins_, 1e-12)
        cells = self._cells(trajectories)
        self._build_symbolic_model(cells)
        X = self._symbolic_features(trajectories)
        self._symbolic_train_ = X
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.model_ = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        ).fit(Xs)
        self.mode_ = "trajectory"
        return self

    def score(self, features, trajectories=None):
        if getattr(self, "mode_", None) == "feature" or trajectories is None:
            X = self.scaler.transform(features)
            return -self.model_.decision_function(X)

        X = self._symbolic_features(trajectories)
        Xs = self.scaler.transform(X)
        model_score = -self.model_.decision_function(Xs)
        route_score = self._normalize(X[:, 0])
        rarity_score = self._normalize(X[:, 1] + X[:, 2])
        stay_score = self._normalize(X[:, 3] + X[:, 4] + X[:, 5])
        turn_score = self._normalize(X[:, 7] + X[:, 8] + X[:, 9])
        return (
            0.35 * self._normalize(model_score)
            + 0.25 * route_score
            + 0.20 * rarity_score
            + 0.10 * stay_score
            + 0.10 * turn_score
        )

    def fit_score(self, features, trajectories=None):
        t0 = time.time()
        self.fit(features, trajectories)
        self.fit_time = time.time() - t0

        t0 = time.time()
        scores = self.score(features, trajectories)
        self.score_time = time.time() - t0
        return scores


class ProfileTADDetector(BaselineDetector):
    """
    Profile-monitoring trajectory anomaly detector.

    This CPU baseline follows the route-extraction and profile-monitoring idea:
    trajectories are clustered by OD route, transformed into route-normalized
    lateral, speed, and heading-change profiles, and scored by robust
    control-chart deviations from the route-specific normal profile.
    """
    def __init__(self, n_routes=20, seed=42):
        super().__init__('Profile-TAD')
        self.n_routes = n_routes
        self.seed = seed
        self.route_scaler = StandardScaler()

    @staticmethod
    def _normalize(scores):
        scores = np.asarray(scores, dtype=float)
        smin, smax = np.min(scores), np.max(scores)
        if smax - smin < 1e-12:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)

    @staticmethod
    def _route_features(trajectories):
        feats = []
        for traj in trajectories:
            traj = np.asarray(traj, dtype=float)
            start = traj[0]
            end = traj[-1]
            vec = end - start
            od = np.linalg.norm(vec)
            feats.append([start[0], start[1], end[0], end[1], vec[0], vec[1], od])
        return np.asarray(feats, dtype=float)

    @staticmethod
    def _profile(traj):
        traj = np.asarray(traj, dtype=float)
        start = traj[0]
        end = traj[-1]
        vec = end - start
        od = np.linalg.norm(vec)
        if od < 1e-8:
            dirs = np.diff(traj, axis=0)
            ref = dirs[np.argmax(np.linalg.norm(dirs, axis=1))] if len(dirs) else np.array([1.0, 0.0])
            od = max(np.linalg.norm(ref), 1e-8)
            vec = ref
        unit = vec / (np.linalg.norm(vec) + 1e-8)
        perp = np.array([-unit[1], unit[0]])
        centered = traj - start
        longitudinal = centered @ unit / od
        lateral = centered @ perp / od

        diffs = np.diff(traj, axis=0)
        speed = np.linalg.norm(diffs, axis=1) / od
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        rel_heading = np.unwrap(angles - np.arctan2(unit[1], unit[0])) / np.pi
        if len(rel_heading) > 1:
            heading_change = np.abs(np.diff(np.unwrap(angles))) / np.pi
        else:
            heading_change = np.zeros(0)

        def pad(values, length):
            values = np.asarray(values, dtype=float)
            if len(values) >= length:
                return values[:length]
            return np.pad(values, (0, length - len(values)), mode="edge") if len(values) else np.zeros(length)

        n = len(traj)
        return np.concatenate([
            pad(longitudinal, n),
            pad(lateral, n),
            pad(speed, n - 1),
            pad(rel_heading, n - 1),
            pad(heading_change, n - 2),
        ])

    def _profiles(self, trajectories):
        return np.asarray([self._profile(traj) for traj in trajectories], dtype=float)

    def fit(self, features, trajectories=None):
        if trajectories is None:
            from sklearn.ensemble import IsolationForest

            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(features)
            self.model_ = IsolationForest(n_estimators=100, random_state=self.seed, n_jobs=-1).fit(X)
            self.mode_ = "feature"
            return self

        from sklearn.cluster import KMeans

        self._trajectories = trajectories
        route_features = self._route_features(trajectories)
        route_scaled = self.route_scaler.fit_transform(route_features)
        n_clusters = min(self.n_routes, max(1, len(route_scaled) // 80), len(route_scaled))
        self.kmeans_ = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        labels = self.kmeans_.fit_predict(route_scaled)
        profiles = self._profiles(trajectories)

        self.cluster_stats_ = {}
        global_median = np.median(profiles, axis=0)
        global_mad = np.median(np.abs(profiles - global_median), axis=0)
        for cluster_id in range(n_clusters):
            idx = np.where(labels == cluster_id)[0]
            if len(idx) < 8:
                median = global_median
                mad = global_mad
            else:
                subset = profiles[idx]
                median = np.median(subset, axis=0)
                mad = np.median(np.abs(subset - median), axis=0)
            self.cluster_stats_[cluster_id] = (median, np.maximum(1.4826 * mad, 1e-6))
        self.mode_ = "trajectory"
        return self

    def score(self, features, trajectories=None):
        if getattr(self, "mode_", None) == "feature" or trajectories is None:
            X = self.scaler.transform(features)
            return -self.model_.decision_function(X)

        route_scaled = self.route_scaler.transform(self._route_features(trajectories))
        labels = self.kmeans_.predict(route_scaled)
        profiles = self._profiles(trajectories)
        scores = np.zeros(len(profiles), dtype=float)
        for i, (profile, cluster_id) in enumerate(zip(profiles, labels)):
            median, scale = self.cluster_stats_[int(cluster_id)]
            z = np.abs(profile - median) / scale
            top_k = max(5, int(0.12 * len(z)))
            top_mean = float(np.mean(np.partition(z, -top_k)[-top_k:]))
            max_z = float(np.max(z))
            exceed_ratio = float(np.mean(z > 3.0))
            scores[i] = 0.55 * top_mean + 0.25 * max_z + 0.20 * exceed_ratio * 10.0
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
    """Return the final CPU baselines used in the manuscript.

    KMeans-Prototype is intentionally excluded here because it is used only as
    a controlled ablation variant, not as a main baseline.
    """
    baselines = get_all_baselines(contamination=contamination, seed=seed)
    baselines['Shape-KNN'] = ShapeKNNDetector(k=5)
    baselines['SegmentOD'] = SegmentOutlierDetector(n_segments=4)
    baselines['TADS'] = TADSDetector(grid_size=18, n_prototypes=18, contamination=contamination, seed=seed)
    baselines['Profile-TAD'] = ProfileTADDetector(n_routes=20, seed=seed)
    return baselines


if __name__ == '__main__':
    from data_generator import generate_synthetic_trajectories
    from feature_extraction import extract_all_features
    from sklearn.metrics import roc_auc_score

    trajs, labels, _, _ = generate_synthetic_trajectories(n_normal=1000, n_anomaly_per_type=25)
    _, _, _, features = extract_all_features(trajs)

    baselines = get_all_baselines_v2()
    for name, det in baselines.items():
        if hasattr(det, 'fit_score') and name in ('Shape-KNN', 'SegmentOD', 'TADS', 'Profile-TAD'):
            scores = det.fit_score(features, trajs)
        else:
            scores = det.fit_score(features)
        auc = roc_auc_score(labels, scores)
        print(f"{name:20s}: AUC={auc:.4f}")
