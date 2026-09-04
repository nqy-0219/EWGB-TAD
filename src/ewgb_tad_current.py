"""Canonical three-view EWGB-TAD implementation for the revision benchmark.

The standalone entropy-feature view is excluded from this method. The
three views are spatial-path, kinematic, and PCA trajectory-shape. The final
configuration uses sample-size-adaptive histogram entropy for local feature
weights.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from granular_ball import EWGBDetector


class ThreeViewEWGBDetector:
    """EWGB-TAD using the canonical three-view representation."""

    VIEW_NAMES = ("spatial_path", "kinematic", "trajectory_shape")

    def __init__(
        self,
        min_samples: int = 8,
        purity_threshold: float = 0.85,
        n_shape_dims: int = 10,
        view_fusion: str = "entropy",
        use_local_entropy: bool = True,
        constant_tol: float = 1e-10,
        fusion_score_bins: int = 20,
        fusion_base_weight: float = 0.1,
    ) -> None:
        if view_fusion not in {"entropy", "equal"}:
            raise ValueError("view_fusion must be 'entropy' or 'equal'")
        if not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if not np.isfinite(purity_threshold) or not 0.0 < purity_threshold <= 1.0:
            raise ValueError("purity_threshold must be in (0, 1]")
        if not isinstance(n_shape_dims, (int, np.integer)) or n_shape_dims < 1:
            raise ValueError("n_shape_dims must be a positive integer")
        if not np.isfinite(constant_tol) or constant_tol < 0.0:
            raise ValueError("constant_tol must be finite and nonnegative")
        if not isinstance(fusion_score_bins, (int, np.integer)) or fusion_score_bins < 1:
            raise ValueError("fusion_score_bins must be a positive integer")
        if not np.isfinite(fusion_base_weight) or fusion_base_weight <= 0.0:
            raise ValueError("fusion_base_weight must be finite and positive")
        self.min_samples = min_samples
        self.purity_threshold = purity_threshold
        self.n_shape_dims = n_shape_dims
        self.view_fusion = view_fusion
        self.use_local_entropy = use_local_entropy
        self.constant_tol = constant_tol
        self.fusion_score_bins = fusion_score_bins
        self.fusion_base_weight = fusion_base_weight
        self.detectors: dict[str, EWGBDetector] = {}
        self.pca: PCA | None = None
        self.view_weights: dict[str, float] = {}
        self.config = {
            "model": "EWGB-TAD",
            "views": list(self.VIEW_NAMES),
            "min_samples": min_samples,
            "purity_threshold": purity_threshold,
            "n_shape_dims": n_shape_dims,
            "view_fusion": view_fusion,
            "use_local_entropy": use_local_entropy,
            "entropy_method": "sample-adaptive-histogram",
            "local_bin_rule": "max(2, ceil(log2(n_ball) + 1))",
            "constant_tol": constant_tol,
            "fusion_score_bins": fusion_score_bins,
            "fusion_base_weight": fusion_base_weight,
        }

    def _make_views(
        self,
        spatial_path: np.ndarray,
        kinematic: np.ndarray,
        trajectories: list[np.ndarray],
        fit: bool,
    ) -> dict[str, np.ndarray]:
        spatial_path = np.asarray(spatial_path, dtype=float)
        kinematic = np.asarray(kinematic, dtype=float)
        if spatial_path.ndim != 2 or spatial_path.shape[0] == 0:
            raise ValueError("spatial_path must be a non-empty 2-D array")
        if kinematic.ndim != 2 or kinematic.shape[0] == 0:
            raise ValueError("kinematic must be a non-empty 2-D array")
        if not np.isfinite(spatial_path).all() or not np.isfinite(kinematic).all():
            raise ValueError("view features must contain only finite values")
        n_samples = spatial_path.shape[0]
        if kinematic.shape[0] != n_samples or len(trajectories) != n_samples:
            raise ValueError("all three views must contain the same number of samples")

        trajectory_arrays = []
        expected_shape = None
        for trajectory in trajectories:
            array = np.asarray(trajectory, dtype=float)
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError("each trajectory must be a non-empty 2-D array")
            if not np.isfinite(array).all():
                raise ValueError("trajectories must contain only finite values")
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise ValueError("all trajectories must have the same shape")
            trajectory_arrays.append(array.flatten())
        flat_trajs = np.stack(trajectory_arrays)
        if fit:
            n_components = min(self.n_shape_dims, flat_trajs.shape[0] - 1,
                               flat_trajs.shape[1])
            if n_components < 1:
                raise ValueError("at least two trajectories are required for PCA")
            self.pca = PCA(n_components=n_components)
            shape = self.pca.fit_transform(flat_trajs)
        else:
            if self.pca is None:
                raise RuntimeError("Detector must be fitted before scoring")
            shape = self.pca.transform(flat_trajs)
        return {
            "spatial_path": np.asarray(spatial_path, dtype=float),
            "kinematic": np.asarray(kinematic, dtype=float),
            "trajectory_shape": shape,
        }

    def fit(
        self,
        spatial_path: np.ndarray,
        kinematic: np.ndarray,
        trajectories: list[np.ndarray],
    ) -> "ThreeViewEWGBDetector":
        views = self._make_views(spatial_path, kinematic, trajectories, fit=True)
        self.detectors = {}
        for name in self.VIEW_NAMES:
            detector = EWGBDetector(
                min_samples=self.min_samples,
                purity_threshold=self.purity_threshold,
                use_local_entropy=self.use_local_entropy,
                constant_tol=self.constant_tol,
            )
            detector.fit(views[name])
            self.detectors[name] = detector
        return self

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError("scores must be a non-empty one-dimensional array")
        if not np.isfinite(scores).all():
            raise ValueError("scores must contain only finite values")
        span = float(np.max(scores) - np.min(scores))
        if span <= 1e-8:
            return np.zeros_like(scores)
        return (scores - np.min(scores)) / span

    @staticmethod
    def _score_entropy(scores: np.ndarray, n_bins: int = 20) -> float:
        """Full-pool score entropy used only for view fusion."""
        if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
            raise ValueError("scores must be a non-empty finite one-dimensional array")
        hist, _ = np.histogram(scores, bins=n_bins, range=(0.0, 1.0))
        probabilities = hist.astype(float) / max(1, int(hist.sum()))
        probabilities = probabilities[probabilities > 0.0]
        if len(probabilities) == 0:
            return 0.0
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def _fuse_view_scores(self, view_scores: dict[str, np.ndarray]) -> np.ndarray:
        """Fuse normalized per-view scores under the fixed view rule."""
        if set(view_scores) != set(self.VIEW_NAMES):
            raise ValueError("view_scores must contain the three canonical views")
        normalized = {
            name: self._normalize_scores(view_scores[name])
            for name in self.VIEW_NAMES
        }
        sample_counts = {len(scores) for scores in normalized.values()}
        if len(sample_counts) != 1:
            raise ValueError("all view scores must contain the same number of samples")
        if self.view_fusion == "equal":
            self.view_weights = {
                name: 1.0 / len(self.VIEW_NAMES) for name in self.VIEW_NAMES
            }
        else:
            entropies = {
                name: self._score_entropy(
                    normalized[name],
                    n_bins=self.fusion_score_bins,
                )
                for name in self.VIEW_NAMES
            }
            max_entropy = max(entropies.values())
            raw_weights = {
                name: max(0.0, max_entropy - entropies[name])
                + self.fusion_base_weight
                for name in self.VIEW_NAMES
            }
            total = sum(raw_weights.values())
            if not np.isfinite(total) or total <= 0.0:
                raise RuntimeError("view fusion produced invalid raw weights")
            self.view_weights = {
                name: raw_weights[name] / total for name in self.VIEW_NAMES
            }

        fused = np.zeros(sample_counts.pop(), dtype=float)
        for name in self.VIEW_NAMES:
            fused += self.view_weights[name] * normalized[name]
        if not np.isfinite(fused).all():
            raise RuntimeError("view fusion produced non-finite scores")
        return fused

    def score(
        self,
        spatial_path: np.ndarray,
        kinematic: np.ndarray,
        trajectories: list[np.ndarray],
    ) -> np.ndarray:
        if not self.detectors or self.pca is None:
            raise RuntimeError("Detector must be fitted before scoring")
        views = self._make_views(spatial_path, kinematic, trajectories, fit=False)
        view_scores = {
            name: self.detectors[name].score(views[name])
            for name in self.VIEW_NAMES
        }
        return self._fuse_view_scores(view_scores)

    def get_stats(self) -> dict:
        return {
            "config": dict(self.config),
            "views": list(self.VIEW_NAMES),
            "view_fusion": self.view_fusion,
            "view_weights": dict(self.view_weights),
            "detectors": {
                name: detector.get_stats()
                for name, detector in self.detectors.items()
            },
        }
