"""
Granular-Ball construction and entropy-weighted scoring for EWGB-TAD.
"""

import numpy as np


class GranularBall:
    """A single granular ball in feature space."""

    def __init__(self, center, radius, indices, data):
        self.center = center
        self.radius = radius
        self.indices = indices
        self.size = len(indices)
        self.data = data[indices]

    def density(self):
        """Ball density = size / volume proxy."""
        return self.size / (self.radius + 1e-8)


def build_granular_balls(data, min_samples=8, purity_threshold=0.85, max_depth=15):
    """
    Build adaptive granular balls using hierarchical splitting.

    Uses a combined quality metric: the ball is split if it is too large
    (radius exceeds a data-adaptive threshold) or if internal variance
    is high (mean distance to center exceeds threshold).

    Args:
        data: (n, d) feature matrix (should be normalized)
        min_samples: minimum number of samples per ball
        purity_threshold: quality threshold for splitting (lower = more splitting)
        max_depth: maximum recursion depth

    Returns:
        list of GranularBall objects
    """
    balls = []

    # Compute global statistics for adaptive thresholds
    global_center = np.mean(data, axis=0)
    global_dists = np.linalg.norm(data - global_center, axis=1)
    global_mean_dist = np.mean(global_dists)

    # Target: each ball should cover a fraction of the global spread
    target_radius = global_mean_dist * purity_threshold

    def _split(indices, depth=0):
        if len(indices) < min_samples or depth >= max_depth:
            _create_ball(indices)
            return

        subset = data[indices]
        center = np.mean(subset, axis=0)
        dists = np.linalg.norm(subset - center, axis=1)
        radius = np.max(dists) if len(dists) > 0 else 0
        mean_dist = np.mean(dists)

        # Quality check: ball is compact enough if radius and mean distance are small
        if radius < target_radius and mean_dist < target_radius * 0.7 and len(indices) <= max(min_samples * 4, 50):
            _create_ball(indices)
            return

        # Also stop if very small
        if len(indices) < min_samples * 2:
            _create_ball(indices)
            return

        # Split using 2-means initialized by farthest pair
        idx1_local = np.argmax(dists)
        d1 = np.linalg.norm(subset - subset[idx1_local], axis=1)
        idx2_local = np.argmax(d1)

        c1, c2 = subset[idx1_local].copy(), subset[idx2_local].copy()

        for _ in range(10):  # k-means iterations
            d_to_c1 = np.linalg.norm(subset - c1, axis=1)
            d_to_c2 = np.linalg.norm(subset - c2, axis=1)
            mask1 = d_to_c1 <= d_to_c2

            if mask1.sum() == 0 or (~mask1).sum() == 0:
                break

            new_c1 = np.mean(subset[mask1], axis=0)
            new_c2 = np.mean(subset[~mask1], axis=0)

            if np.allclose(c1, new_c1) and np.allclose(c2, new_c2):
                break
            c1, c2 = new_c1, new_c2

        group1 = indices[mask1]
        group2 = indices[~mask1]

        if len(group1) < min_samples or len(group2) < min_samples:
            _create_ball(indices)
            return

        _split(group1, depth + 1)
        _split(group2, depth + 1)

    def _create_ball(indices):
        if len(indices) == 0:
            return
        center = np.mean(data[indices], axis=0)
        dists = np.linalg.norm(data[indices] - center, axis=1)
        radius = np.max(dists) if len(dists) > 0 else 0
        balls.append(GranularBall(center, radius, indices, data))

    all_indices = np.arange(len(data))
    _split(all_indices)

    return balls


def compute_feature_entropy_weights(data, n_bins=16):
    """
    Compute entropy-based weights for each feature dimension.
    Lower entropy = more discriminative = higher weight.

    Args:
        data: (n, d) feature matrix
        n_bins: number of histogram bins

    Returns:
        weights: (d,) normalized weights
    """
    n_features = data.shape[1]
    entropies = np.zeros(n_features)

    for dim in range(n_features):
        vals = data[:, dim]
        hist, _ = np.histogram(vals, bins=n_bins, density=False)
        hist = hist / (hist.sum() + 1e-8)
        hist = hist[hist > 0]
        entropies[dim] = -np.sum(hist * np.log2(hist + 1e-12))

    # Invert: low entropy = high weight
    max_ent = np.max(entropies) + 1e-8
    weights = (max_ent - entropies) / max_ent
    weights = weights / (weights.sum() + 1e-8)

    return weights


def compute_local_entropy_weights(ball_data, n_bins=8):
    """
    Compute entropy weights within a single granular ball.
    Uses the local distribution of samples in the ball.

    Args:
        ball_data: (k, d) samples within the ball
        n_bins: number of bins

    Returns:
        weights: (d,) local entropy weights
    """
    if len(ball_data) < 3:
        return np.ones(ball_data.shape[1]) / ball_data.shape[1]

    n_features = ball_data.shape[1]
    entropies = np.zeros(n_features)

    n_bins_local = min(n_bins, max(3, len(ball_data) // 2))

    for dim in range(n_features):
        vals = ball_data[:, dim]
        if np.std(vals) < 1e-10:
            entropies[dim] = 0.0
            continue
        hist, _ = np.histogram(vals, bins=n_bins_local, density=False)
        hist = hist / (hist.sum() + 1e-8)
        hist = hist[hist > 0]
        entropies[dim] = -np.sum(hist * np.log2(hist + 1e-12))

    max_ent = np.max(entropies) + 1e-8
    weights = (max_ent - entropies) / max_ent
    weights = weights / (weights.sum() + 1e-8)

    return weights


class EWGBDetector:
    """
    Entropy-Weighted Granular-Ball Anomaly Detector for a single view.

    Builds granular balls on training data, then scores new samples
    based on entropy-weighted distance to nearest ball adjusted by density.
    """

    def __init__(self, min_samples=8, purity_threshold=0.85,
                 n_entropy_bins=16, use_local_entropy=True):
        self.min_samples = min_samples
        self.purity_threshold = purity_threshold
        self.n_entropy_bins = n_entropy_bins
        self.use_local_entropy = use_local_entropy
        self.balls = None
        self.global_weights = None
        self.local_weights = None
        self.mean = None
        self.std = None

    def fit(self, data):
        """Build granular balls on the data."""
        # Normalize
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8
        norm_data = (data - self.mean) / self.std

        # Build balls
        self.balls = build_granular_balls(
            norm_data,
            min_samples=self.min_samples,
            purity_threshold=self.purity_threshold
        )

        # Compute global entropy weights
        self.global_weights = compute_feature_entropy_weights(
            norm_data, self.n_entropy_bins
        )

        # Compute local entropy weights per ball
        if self.use_local_entropy:
            self.local_weights = []
            for ball in self.balls:
                lw = compute_local_entropy_weights(ball.data, n_bins=8)
                self.local_weights.append(lw)
        else:
            self.local_weights = [self.global_weights] * len(self.balls)

        return self

    def score(self, data):
        """
        Compute anomaly scores using entropy-weighted distance and density.
        Higher score = more anomalous.
        Vectorized implementation for speed.
        """
        norm_data = (data - self.mean) / self.std
        n = len(data)
        n_balls = len(self.balls)

        # Precompute
        ball_centers = np.array([b.center for b in self.balls])  # (n_balls, d)
        ball_densities = np.array([b.density() for b in self.balls])
        ball_radii = np.array([b.radius for b in self.balls])
        density_factors = 1.0 / (np.log1p(ball_densities) + 1e-8)  # (n_balls,)
        weight_matrix = np.array(self.local_weights)  # (n_balls, d)

        # Process in batches for memory efficiency
        batch_size = 500
        scores = np.zeros(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = norm_data[start:end]  # (batch, d)
            bs = end - start

            # Compute diffs: (batch, n_balls, d)
            diffs = batch[:, np.newaxis, :] - ball_centers[np.newaxis, :, :]

            # Weighted squared distances: (batch, n_balls)
            weighted_sq = np.sum(weight_matrix[np.newaxis, :, :] * diffs ** 2, axis=2)
            weighted_dist = np.sqrt(weighted_sq)  # (batch, n_balls)

            # Inside-ball discount
            inside_mask = weighted_dist < ball_radii[np.newaxis, :]
            weighted_dist = np.where(inside_mask, weighted_dist * 0.5, weighted_dist)

            # Density-adjusted score: (batch, n_balls)
            ball_scores = weighted_dist * density_factors[np.newaxis, :]

            # Take minimum across balls
            scores[start:end] = np.min(ball_scores, axis=1)

        return scores

    def get_stats(self):
        """Return detector statistics."""
        if self.balls is None:
            return {}
        return {
            'n_balls': len(self.balls),
            'ball_sizes': [b.size for b in self.balls],
            'ball_radii': [b.radius for b in self.balls],
            'global_weights': self.global_weights.tolist(),
        }


class MultiViewEWGBDetector:
    """
    Multi-View EWGB-TAD: Entropy-Weighted Multi-View Granular-Ball
    Framework for Trajectory Anomaly Detection.

    Builds separate EWGB detectors for spatial, kinematic, and entropy views,
    then fuses scores using entropy-based view weights.
    """

    def __init__(self, min_samples=8, purity_threshold=0.85,
                 n_entropy_bins=16, use_local_entropy=True,
                 fusion='entropy'):
        """
        Args:
            fusion: 'entropy' | 'equal' | 'concat' | 'max' | 'spatial' | 'kinematic' | 'entropy_view'
        """
        self.min_samples = min_samples
        self.purity_threshold = purity_threshold
        self.n_entropy_bins = n_entropy_bins
        self.use_local_entropy = use_local_entropy
        self.fusion = fusion
        self.detectors = {}
        self.view_weights = None
        self.concat_detector = None

    def fit(self, spatial_features, kinematic_features, entropy_features):
        """Fit detectors on three views."""
        views = {
            'spatial': spatial_features,
            'kinematic': kinematic_features,
            'entropy': entropy_features,
        }

        if self.fusion == 'concat':
            # Single detector on concatenated features
            all_features = np.hstack([spatial_features, kinematic_features, entropy_features])
            self.concat_detector = EWGBDetector(
                min_samples=self.min_samples,
                purity_threshold=self.purity_threshold,
                n_entropy_bins=self.n_entropy_bins,
                use_local_entropy=self.use_local_entropy
            )
            self.concat_detector.fit(all_features)
            return self

        # Build per-view detectors
        for name, data in views.items():
            det = EWGBDetector(
                min_samples=self.min_samples,
                purity_threshold=self.purity_threshold,
                n_entropy_bins=self.n_entropy_bins,
                use_local_entropy=self.use_local_entropy
            )
            det.fit(data)
            self.detectors[name] = det

        return self

    def score(self, spatial_features, kinematic_features, entropy_features):
        """Compute fused anomaly scores."""
        if self.fusion == 'concat':
            all_features = np.hstack([spatial_features, kinematic_features, entropy_features])
            return self.concat_detector.score(all_features)

        # Single view modes
        if self.fusion in ('spatial', 'kinematic', 'entropy_view'):
            view_map = {'spatial': spatial_features, 'kinematic': kinematic_features,
                       'entropy_view': entropy_features}
            det_map = {'spatial': 'spatial', 'kinematic': 'kinematic',
                      'entropy_view': 'entropy'}
            view_key = det_map[self.fusion]
            return self.detectors[view_key].score(view_map[self.fusion])

        views_data = {
            'spatial': spatial_features,
            'kinematic': kinematic_features,
            'entropy': entropy_features,
        }

        # Get per-view scores
        view_scores = {}
        for name, data in views_data.items():
            raw_scores = self.detectors[name].score(data)
            # Min-max normalize
            smin, smax = raw_scores.min(), raw_scores.max()
            if smax - smin > 1e-8:
                raw_scores = (raw_scores - smin) / (smax - smin)
            view_scores[name] = raw_scores

        # Fusion
        if self.fusion == 'equal':
            fused = sum(view_scores.values()) / len(view_scores)

        elif self.fusion == 'max':
            score_matrix = np.array(list(view_scores.values()))
            fused = np.max(score_matrix, axis=0)

        elif self.fusion == 'entropy':
            # Entropy-based adaptive fusion
            view_entropies = {}
            for name, scores in view_scores.items():
                hist, _ = np.histogram(scores, bins=20, density=False)
                hist = hist / (hist.sum() + 1e-8)
                hist = hist[hist > 0]
                ent = -np.sum(hist * np.log2(hist + 1e-12))
                view_entropies[name] = ent

            # Lower entropy = more discriminative = higher weight
            max_ent = max(view_entropies.values()) + 1e-8
            weights = {}
            total_w = 0
            for name, ent in view_entropies.items():
                w = (max_ent - ent) / max_ent + 0.1  # add small base weight
                weights[name] = w
                total_w += w

            # Normalize
            for name in weights:
                weights[name] /= total_w

            self.view_weights = weights

            fused = np.zeros(len(spatial_features))
            for name, scores in view_scores.items():
                fused += weights[name] * scores

        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

        return fused

    def get_stats(self):
        """Return detector statistics."""
        stats = {'fusion': self.fusion}
        if self.concat_detector:
            stats['concat'] = self.concat_detector.get_stats()
        else:
            for name, det in self.detectors.items():
                stats[name] = det.get_stats()
            if self.view_weights:
                stats['view_weights'] = self.view_weights
        return stats


if __name__ == '__main__':
    # Quick test
    rng = np.random.RandomState(42)
    data = rng.randn(500, 5)
    balls = build_granular_balls(data, min_samples=10, purity_threshold=0.8)
    print(f"Built {len(balls)} granular balls from {len(data)} samples")
    for i, b in enumerate(balls[:5]):
        print(f"  Ball {i}: center={b.center[:3]}, radius={b.radius:.3f}, size={b.size}")

    weights = compute_feature_entropy_weights(data)
    print(f"Entropy weights: {weights}")
