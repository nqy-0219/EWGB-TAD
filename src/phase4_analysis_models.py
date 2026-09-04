"""Controlled component and view variants for Phase 4 analyses."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from granular_ball import (
    EWGBDetector,
    compute_feature_entropy_weights,
    compute_local_entropy_weights,
)


VIEW_NAMES = ("spatial_path", "kinematic", "trajectory_shape")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    span = float(scores.max() - scores.min())
    if span <= 1e-8:
        return np.zeros_like(scores)
    return (scores - scores.min()) / span


def entropy_fusion(
    view_scores: dict[str, np.ndarray],
    mode: str,
    bins: int = 20,
    base_weight: float = 0.1,
) -> tuple[np.ndarray, dict[str, float]]:
    normalized = {name: normalize_scores(scores) for name, scores in view_scores.items()}
    if mode == "equal" or len(normalized) == 1:
        weights = {name: 1.0 / len(normalized) for name in normalized}
    else:
        entropies: dict[str, float] = {}
        for name, scores in normalized.items():
            histogram, _ = np.histogram(scores, bins=bins, range=(0.0, 1.0))
            probabilities = histogram.astype(float) / max(1, int(histogram.sum()))
            probabilities = probabilities[probabilities > 0.0]
            entropies[name] = float(-np.sum(probabilities * np.log2(probabilities)))
        maximum = max(entropies.values())
        raw = {name: max(0.0, maximum - entropy) + base_weight for name, entropy in entropies.items()}
        total = sum(raw.values())
        weights = {name: value / total for name, value in raw.items()}
    fused = np.zeros(len(next(iter(normalized.values()))), dtype=float)
    for name, scores in normalized.items():
        fused += weights[name] * scores
    return fused, weights


class KMeansEntropyRegionDetector:
    """Fixed-prototype partition with the EWGB metric and density score."""

    def __init__(
        self,
        seed: int,
        use_local_entropy: bool,
        n_regions: int = 20,
        constant_tol: float = 1e-10,
    ) -> None:
        self.seed = seed
        self.use_local_entropy = use_local_entropy
        self.n_regions = n_regions
        self.constant_tol = constant_tol

    def fit(self, data: np.ndarray) -> "KMeansEntropyRegionDetector":
        data = np.asarray(data, dtype=float)
        self.mean_ = data.mean(axis=0)
        raw_std = data.std(axis=0)
        self.active_mask_ = raw_std > self.constant_tol
        self.std_ = np.where(self.active_mask_, raw_std, 1.0)
        normalized = (data - self.mean_) / self.std_
        self.global_weights_ = compute_feature_entropy_weights(
            normalized,
            active_mask=self.active_mask_,
            constant_tol=self.constant_tol,
        )
        n_regions = min(self.n_regions, len(normalized))
        labels = KMeans(n_clusters=n_regions, random_state=self.seed, n_init=10).fit_predict(normalized)
        centers = []
        radii = []
        densities = []
        weights = []
        sizes = []
        for region in range(n_regions):
            subset = normalized[labels == region]
            center = subset.mean(axis=0)
            distances = np.linalg.norm(subset - center, axis=1)
            radius = float(distances.max()) if len(distances) else 0.0
            centers.append(center)
            radii.append(radius)
            sizes.append(len(subset))
            densities.append(len(subset) / (radius + 1e-8))
            if self.use_local_entropy:
                weights.append(
                    compute_local_entropy_weights(
                        subset,
                        active_mask=self.active_mask_,
                        constant_tol=self.constant_tol,
                    )
                )
            else:
                weights.append(self.global_weights_)
        self.centers_ = np.asarray(centers)
        self.radii_ = np.asarray(radii)
        self.densities_ = np.asarray(densities)
        self.weights_ = np.asarray(weights)
        self.sizes_ = sizes
        return self

    def score(self, data: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(data, dtype=float) - self.mean_) / self.std_
        output = np.zeros(len(normalized), dtype=float)
        density_factor = 1.0 / (np.log1p(self.densities_) + 1e-8)
        for start in range(0, len(normalized), 500):
            batch = normalized[start : start + 500]
            differences = batch[:, None, :] - self.centers_[None, :, :]
            distance = np.sqrt(np.sum(self.weights_[None, :, :] * differences**2, axis=2))
            distance = np.where(distance < self.radii_[None, :], distance * 0.5, distance)
            output[start : start + len(batch)] = np.min(distance * density_factor[None, :], axis=1)
        return output

    def get_stats(self) -> dict:
        return {
            "partition": "fixed_kmeans_regions",
            "n_regions": len(self.centers_),
            "region_sizes": self.sizes_,
            "global_weights": self.global_weights_.tolist(),
            "local_weights": self.weights_.tolist(),
            "entropy_method": "sample-adaptive-histogram",
            "local_bin_rule": "max(2, ceil(log2(n_region) + 1))",
            "use_local_entropy": self.use_local_entropy,
        }


def fit_score_variant(
    views: dict[str, np.ndarray],
    selected_views: tuple[str, ...],
    seed: int,
    granular_partition: bool,
    local_metric: bool,
    fusion: str,
    min_samples: int = 8,
    entropy_fixed_bins: int | None = None,
) -> tuple[np.ndarray, dict]:
    view_scores: dict[str, np.ndarray] = {}
    detector_stats: dict[str, dict] = {}
    for name in selected_views:
        if granular_partition:
            detector = EWGBDetector(
                min_samples=min_samples,
                purity_threshold=0.85,
                use_local_entropy=local_metric,
                constant_tol=1e-10,
                entropy_fixed_bins=entropy_fixed_bins,
            )
        else:
            detector = KMeansEntropyRegionDetector(
                seed=seed,
                use_local_entropy=local_metric,
                n_regions=20,
            )
        detector.fit(views[name])
        view_scores[name] = detector.score(views[name])
        detector_stats[name] = detector.get_stats()
    scores, weights = entropy_fusion(view_scores, mode=fusion)
    return scores, {
        "selected_views": list(selected_views),
        "granular_partition": granular_partition,
        "local_metric": local_metric,
        "fusion": fusion,
        "min_samples": min_samples,
        "view_weights": weights,
        "detectors": detector_stats,
    }
