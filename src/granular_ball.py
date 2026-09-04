"""Granular-ball construction and sample-adaptive entropy scoring.

This module contains the estimator used by the current EWGB-TAD release.
Each view is standardized on the fitting pool, partitioned into adaptive
granular balls, and scored using locally weighted distances to the nearest
ball. Local feature weights are obtained from a sample-size-adaptive
histogram entropy estimate.
"""

from __future__ import annotations

import numpy as np


ENTROPY_METHOD = "sample-adaptive-histogram"


class GranularBall:
    """A terminal granular ball in standardized feature space."""

    def __init__(self, center: np.ndarray, radius: float, indices: np.ndarray,
                 data: np.ndarray) -> None:
        self.center = center
        self.radius = float(radius)
        self.indices = indices
        self.size = len(indices)
        self.data = data[indices]

    def density(self) -> float:
        return self.size / (self.radius + 1e-8)


def build_granular_balls(data: np.ndarray, min_samples: int = 8,
                         purity_threshold: float = 0.85,
                         max_depth: int = 15) -> list[GranularBall]:
    """Build a deterministic binary granular-ball partition."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("data must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("data contains non-finite values")
    if not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    if not np.isfinite(purity_threshold) or not 0.0 < purity_threshold <= 1.0:
        raise ValueError("purity_threshold must be in (0, 1]")

    balls: list[GranularBall] = []
    global_center = np.mean(values, axis=0)
    global_dists = np.linalg.norm(values - global_center, axis=1)
    target_radius = float(np.mean(global_dists) * purity_threshold)

    def create_ball(indices: np.ndarray) -> None:
        if len(indices) == 0:
            return
        center = np.mean(values[indices], axis=0)
        dists = np.linalg.norm(values[indices] - center, axis=1)
        radius = float(np.max(dists)) if len(dists) else 0.0
        balls.append(GranularBall(center, radius, indices, values))

    def split(indices: np.ndarray, depth: int = 0) -> None:
        if len(indices) < min_samples or depth >= max_depth:
            create_ball(indices)
            return

        subset = values[indices]
        center = np.mean(subset, axis=0)
        dists = np.linalg.norm(subset - center, axis=1)
        radius = float(np.max(dists)) if len(dists) else 0.0
        mean_dist = float(np.mean(dists))
        if (radius < target_radius and
                mean_dist < target_radius * 0.7 and
                len(indices) <= max(min_samples * 4, 50)):
            create_ball(indices)
            return
        if len(indices) < min_samples * 2:
            create_ball(indices)
            return

        first_local = int(np.argmax(dists))
        first_dist = np.linalg.norm(subset - subset[first_local], axis=1)
        second_local = int(np.argmax(first_dist))
        c1 = subset[first_local].copy()
        c2 = subset[second_local].copy()
        mask1 = np.ones(len(subset), dtype=bool)
        for _ in range(10):
            d1 = np.linalg.norm(subset - c1, axis=1)
            d2 = np.linalg.norm(subset - c2, axis=1)
            mask1 = d1 <= d2
            if mask1.sum() == 0 or (~mask1).sum() == 0:
                create_ball(indices)
                return
            new_c1 = np.mean(subset[mask1], axis=0)
            new_c2 = np.mean(subset[~mask1], axis=0)
            if np.allclose(c1, new_c1) and np.allclose(c2, new_c2):
                break
            c1, c2 = new_c1, new_c2

        group1 = indices[mask1]
        group2 = indices[~mask1]
        if len(group1) < min_samples or len(group2) < min_samples:
            create_ball(indices)
            return
        split(group1, depth + 1)
        split(group2, depth + 1)

    split(np.arange(len(values)))
    return balls


def _uniform_weights(n_features: int, active_mask: np.ndarray | None = None) -> np.ndarray:
    if n_features == 0:
        return np.zeros(0, dtype=float)
    if active_mask is None:
        active = np.ones(n_features, dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != (n_features,) or not np.any(active):
            active = np.ones(n_features, dtype=bool)
    weights = np.zeros(n_features, dtype=float)
    weights[active] = 1.0 / float(np.sum(active))
    return weights


def _histogram_entropies(data: np.ndarray,
                         active_mask: np.ndarray | None = None,
                         constant_tol: float = 1e-10,
                         fixed_bins: int | None = None) -> np.ndarray:
    """Compute sample-size-adaptive per-feature Shannon entropy."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("data must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("data contains non-finite values")
    if active_mask is None:
        active = np.std(values, axis=0) > constant_tol
    else:
        active = np.asarray(active_mask, dtype=bool)
    if active.shape != (values.shape[1],):
        raise ValueError("active_mask has an incompatible shape")

    if fixed_bins is not None and (
            not isinstance(fixed_bins, (int, np.integer)) or fixed_bins < 2):
        raise ValueError("fixed_bins must be an integer greater than or equal to 2")
    bins = (
        max(2, int(fixed_bins))
        if fixed_bins is not None
        else max(2, int(np.ceil(np.log2(max(1, values.shape[0])) + 1.0)))
    )
    entropies = np.zeros(values.shape[1], dtype=float)
    for dimension in np.flatnonzero(active):
        feature = values[:, dimension]
        if np.std(feature) <= constant_tol:
            continue
        histogram, _ = np.histogram(feature, bins=bins, density=False)
        total = float(histogram.sum())
        if total <= 0.0 or not np.isfinite(total):
            continue
        probabilities = histogram.astype(float) / total
        probabilities = probabilities[probabilities > 0.0]
        entropies[dimension] = float(
            -np.sum(probabilities * np.log2(probabilities + 1e-12))
        )
    return entropies


def _histogram_weights(data: np.ndarray,
                       active_mask: np.ndarray | None = None,
                       constant_tol: float = 1e-10,
                       fixed_bins: int | None = None) -> np.ndarray:
    """Convert local entropy into reliability weights."""
    values = np.asarray(data, dtype=float)
    if active_mask is None:
        active = np.std(values, axis=0) > constant_tol
    else:
        active = np.asarray(active_mask, dtype=bool)
    entropies = _histogram_entropies(
        values,
        active_mask=active,
        constant_tol=constant_tol,
        fixed_bins=fixed_bins,
    )
    if not np.any(active):
        return _uniform_weights(values.shape[1])
    maximum = float(np.max(entropies[active]))
    reliability = np.zeros(values.shape[1], dtype=float)
    reliability[active] = maximum - entropies[active]
    if not np.isfinite(reliability).all() or float(reliability.sum()) <= 1e-12:
        return _uniform_weights(values.shape[1], active)
    return reliability / float(reliability.sum())


def compute_feature_entropy_weights(
    data: np.ndarray,
    active_mask: np.ndarray | None = None,
    constant_tol: float = 1e-10,
    fixed_bins: int | None = None,
) -> np.ndarray:
    """Compute global feature weights for the selected estimator."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("data must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("data contains non-finite values")
    if active_mask is None:
        active = np.std(values, axis=0) > constant_tol
    else:
        active = np.asarray(active_mask, dtype=bool)
    if active.shape != (values.shape[1],):
        raise ValueError("active_mask has an incompatible shape")
    return _histogram_weights(
        values,
        active_mask=active,
        constant_tol=constant_tol,
        fixed_bins=fixed_bins,
    )


def compute_local_entropy_weights(
    ball_data: np.ndarray,
    active_mask: np.ndarray | None = None,
    constant_tol: float = 1e-10,
    fixed_bins: int | None = None,
) -> np.ndarray:
    """Compute local feature weights inside one terminal ball."""
    values = np.asarray(ball_data, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("ball_data must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("ball_data contains non-finite values")
    if active_mask is None:
        active = np.std(values, axis=0) > constant_tol
    else:
        active = np.asarray(active_mask, dtype=bool)
    if active.shape != (values.shape[1],):
        raise ValueError("active_mask has an incompatible shape")
    return _histogram_weights(
        values,
        active_mask=active,
        constant_tol=constant_tol,
        fixed_bins=fixed_bins,
    )


class EWGBDetector:
    """Entropy-weighted granular-ball anomaly detector for one view."""

    def __init__(self, min_samples: int = 8, purity_threshold: float = 0.85,
                 use_local_entropy: bool = True,
                 constant_tol: float = 1e-10,
                 entropy_fixed_bins: int | None = None) -> None:
        if not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if not np.isfinite(purity_threshold) or not 0.0 < purity_threshold <= 1.0:
            raise ValueError("purity_threshold must be in (0, 1]")
        if not np.isfinite(constant_tol) or constant_tol < 0.0:
            raise ValueError("constant_tol must be finite and nonnegative")
        if entropy_fixed_bins is not None and (
                not isinstance(entropy_fixed_bins, (int, np.integer)) or
                entropy_fixed_bins < 2):
            raise ValueError("entropy_fixed_bins must be an integer greater than or equal to 2")
        self.min_samples = int(min_samples)
        self.purity_threshold = float(purity_threshold)
        self.use_local_entropy = bool(use_local_entropy)
        self.constant_tol = float(constant_tol)
        self.entropy_fixed_bins = (
            None if entropy_fixed_bins is None else int(entropy_fixed_bins)
        )
        self.balls = None
        self.global_weights = None
        self.local_weights = None
        self.mean = None
        self.std = None
        self.active_mask = None

    def fit(self, data: np.ndarray) -> "EWGBDetector":
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("data must be a non-empty 2-D array")
        if not np.isfinite(values).all():
            raise ValueError("data contains non-finite values")
        self.mean = np.mean(values, axis=0)
        raw_std = np.std(values, axis=0)
        self.active_mask = raw_std > self.constant_tol
        self.std = np.where(self.active_mask, raw_std, 1.0)
        normalized = (values - self.mean) / self.std
        self.balls = build_granular_balls(
            normalized, min_samples=self.min_samples,
            purity_threshold=self.purity_threshold,
        )
        self.global_weights = compute_feature_entropy_weights(
            normalized,
            active_mask=self.active_mask,
            constant_tol=self.constant_tol,
            fixed_bins=self.entropy_fixed_bins,
        )
        if self.use_local_entropy:
            self.local_weights = [
                compute_local_entropy_weights(
                    ball.data,
                    active_mask=self.active_mask,
                    constant_tol=self.constant_tol,
                    fixed_bins=self.entropy_fixed_bins,
                )
                for ball in self.balls
            ]
        else:
            self.local_weights = [self.global_weights] * len(self.balls)
        return self

    def score(self, data: np.ndarray) -> np.ndarray:
        if self.balls is None or self.mean is None or self.std is None:
            raise RuntimeError("Detector must be fitted before scoring")
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("data must be a non-empty 2-D array")
        if not np.isfinite(values).all():
            raise ValueError("data contains non-finite values")
        if values.shape[1] != len(self.mean):
            raise ValueError("data feature dimension does not match fitted data")
        normalized = (values - self.mean) / self.std
        centers = np.asarray([ball.center for ball in self.balls])
        densities = np.asarray([ball.density() for ball in self.balls])
        radii = np.asarray([ball.radius for ball in self.balls])
        density_factors = 1.0 / (np.log1p(densities) + 1e-8)
        weight_matrix = np.asarray(self.local_weights)
        scores = np.zeros(len(values), dtype=float)
        for start in range(0, len(values), 500):
            end = min(start + 500, len(values))
            differences = normalized[start:end, None, :] - centers[None, :, :]
            weighted_sq = np.sum(weight_matrix[None, :, :] * differences ** 2, axis=2)
            distance = np.sqrt(weighted_sq)
            inside = distance < radii[None, :]
            distance = np.where(inside, distance * 0.5, distance)
            scores[start:end] = np.min(distance * density_factors[None, :], axis=1)
        return scores

    def get_stats(self) -> dict:
        if self.balls is None:
            return {}
        return {
            "n_balls": len(self.balls),
            "ball_sizes": [ball.size for ball in self.balls],
            "ball_radii": [ball.radius for ball in self.balls],
            "global_weights": self.global_weights.tolist(),
            "entropy_method": ENTROPY_METHOD,
            "local_bin_rule": "max(2, ceil(log2(n_ball) + 1))",
            "constant_tol": self.constant_tol,
            "entropy_fixed_bins": self.entropy_fixed_bins,
            "use_local_entropy": self.use_local_entropy,
            "active_feature_count": int(np.sum(self.active_mask)),
        }


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    sample = rng.randn(500, 5)
    detector = EWGBDetector(min_samples=8).fit(sample)
    print(f"Built {len(detector.balls)} granular balls")
    print(f"Score range: {detector.score(sample).min():.4f}--{detector.score(sample).max():.4f}")
