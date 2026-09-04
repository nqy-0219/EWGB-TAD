from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = SOURCE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ewgb_tad_current import ThreeViewEWGBDetector
from evaluation_protocol import evaluate_oracle_top_k, oracle_top_k_predictions
from phase4_analysis_models import fit_score_variant
from phase4_common import canonical_shape_view
from granular_ball import (
    EWGBDetector,
    _histogram_entropies,
    compute_feature_entropy_weights,
    compute_local_entropy_weights,
)


class SampleAdaptiveHistogramTests(unittest.TestCase):
    def assert_valid_weights(self, weights: np.ndarray) -> None:
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights >= 0.0).all())
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)

    def test_eight_sample_ball_has_finite_normalized_weights(self) -> None:
        rng = np.random.RandomState(42)
        weights = compute_local_entropy_weights(rng.normal(size=(8, 5)))
        self.assert_valid_weights(weights)

    def test_bin_count_grows_with_sample_size(self) -> None:
        original_histogram = np.histogram
        with patch("granular_ball.np.histogram", wraps=original_histogram) as histogram:
            _histogram_entropies(np.arange(8, dtype=float).reshape(-1, 1))
            self.assertEqual(histogram.call_args.kwargs["bins"], 4)
        with patch("granular_ball.np.histogram", wraps=original_histogram) as histogram:
            _histogram_entropies(np.arange(32, dtype=float).reshape(-1, 1))
            self.assertEqual(histogram.call_args.kwargs["bins"], 6)

    def test_local_entropy_api_exposes_only_effective_parameters(self) -> None:
        parameters = set(inspect.signature(compute_local_entropy_weights).parameters)
        self.assertEqual(
            parameters,
            {"ball_data", "active_mask", "constant_tol", "fixed_bins"},
        )

    def test_explicit_fixed_bins_are_used_only_when_requested(self) -> None:
        original_histogram = np.histogram
        data = np.arange(12, dtype=float).reshape(-1, 1)
        with patch("granular_ball.np.histogram", wraps=original_histogram) as histogram:
            _histogram_entropies(data, fixed_bins=8)
            self.assertEqual(histogram.call_args.kwargs["bins"], 8)
        adaptive = EWGBDetector(min_samples=4).fit(np.column_stack([data, data]))
        fixed = EWGBDetector(min_samples=4, entropy_fixed_bins=8).fit(
            np.column_stack([data, data])
        )
        self.assertIsNone(adaptive.entropy_fixed_bins)
        self.assertEqual(fixed.entropy_fixed_bins, 8)

    def test_all_global_constant_features_use_uniform_fallback(self) -> None:
        weights = compute_feature_entropy_weights(np.ones((16, 4), dtype=float))
        self.assert_valid_weights(weights)
        np.testing.assert_allclose(weights, np.full(4, 0.25), atol=1e-12)

    def test_locally_constant_globally_active_feature_is_reliable(self) -> None:
        local_ball = np.column_stack([np.zeros(8), np.linspace(-1.0, 1.0, 8)])
        weights = compute_local_entropy_weights(
            local_ball,
            active_mask=np.array([True, True]),
        )
        self.assert_valid_weights(weights)
        self.assertGreater(weights[0], weights[1])

    def test_equal_feature_entropies_use_uniform_fallback(self) -> None:
        feature = np.linspace(-1.0, 1.0, 16)
        weights = compute_local_entropy_weights(np.column_stack([feature, feature[::-1]]))
        self.assert_valid_weights(weights)
        np.testing.assert_allclose(weights, np.full(2, 0.5), atol=1e-12)

    def test_global_entropy_weights_are_deterministic(self) -> None:
        data = np.random.RandomState(202).normal(size=(80, 5))
        np.testing.assert_array_equal(
            compute_feature_entropy_weights(data),
            compute_feature_entropy_weights(data),
        )

    def test_single_view_detector_scores_and_weights_are_finite(self) -> None:
        rng = np.random.RandomState(7)
        data = rng.normal(size=(64, 6))
        data[:, -1] = 3.0
        detector = EWGBDetector(min_samples=8).fit(data)
        scores = detector.score(data)
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores.shape, (64,))
        for weights in detector.local_weights:
            self.assert_valid_weights(np.asarray(weights))
            self.assertEqual(float(weights[-1]), 0.0)

    def test_three_view_detector_smoke(self) -> None:
        rng = np.random.RandomState(11)
        n_samples = 48
        trajectories = [
            np.cumsum(rng.normal(scale=0.05, size=(32, 2)), axis=0)
            for _ in range(n_samples)
        ]
        spatial_path = rng.normal(size=(n_samples, 16))
        kinematic = rng.normal(size=(n_samples, 8))
        detector = ThreeViewEWGBDetector(min_samples=8).fit(
            spatial_path,
            kinematic,
            trajectories,
        )
        scores = detector.score(spatial_path, kinematic, trajectories)
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores.shape, (n_samples,))
        self.assertEqual(set(detector.view_weights), set(detector.VIEW_NAMES))
        self.assertAlmostEqual(sum(detector.view_weights.values()), 1.0, places=12)
        stats = detector.get_stats()
        self.assertEqual(stats["config"]["entropy_method"], "sample-adaptive-histogram")
        self.assertEqual(stats["config"]["constant_tol"], 1e-10)
        self.assertEqual(stats["config"]["fusion_score_bins"], 20)
        self.assertEqual(stats["config"]["fusion_base_weight"], 0.1)

    def test_constant_view_score_fusion_is_finite_and_uniform(self) -> None:
        detector = ThreeViewEWGBDetector(view_fusion="entropy")
        view_scores = {
            name: np.full(24, 7.0, dtype=float)
            for name in detector.VIEW_NAMES
        }
        fused = detector._fuse_view_scores(view_scores)
        self.assertTrue(np.isfinite(fused).all())
        np.testing.assert_array_equal(fused, np.zeros(24))
        np.testing.assert_allclose(
            [detector.view_weights[name] for name in detector.VIEW_NAMES],
            np.full(3, 1.0 / 3.0),
            atol=1e-12,
        )

    def test_complete_analysis_variant_matches_canonical_detector(self) -> None:
        rng = np.random.RandomState(23)
        n_samples = 48
        trajectory_array = np.asarray(
            [
                np.cumsum(rng.normal(scale=0.05, size=(32, 2)), axis=0)
                for _ in range(n_samples)
            ],
            dtype=np.float32,
        )
        trajectories = [trajectory for trajectory in trajectory_array]
        spatial_path = rng.normal(size=(n_samples, 16)).astype(np.float32)
        kinematic = rng.normal(size=(n_samples, 8)).astype(np.float32)

        detector = ThreeViewEWGBDetector(min_samples=8).fit(
            spatial_path,
            kinematic,
            trajectories,
        )
        canonical_scores = detector.score(spatial_path, kinematic, trajectories)

        shape = canonical_shape_view(trajectories)
        analysis_scores, _ = fit_score_variant(
            {
                "spatial_path": spatial_path,
                "kinematic": kinematic,
                "trajectory_shape": shape,
            },
            selected_views=detector.VIEW_NAMES,
            seed=23,
            granular_partition=True,
            local_metric=True,
            fusion="entropy",
            min_samples=8,
        )
        np.testing.assert_allclose(
            analysis_scores,
            canonical_scores,
            rtol=0.0,
            atol=1e-12,
        )

    def test_ewgb_analysis_entry_points_rebuild_the_shape_view(self) -> None:
        entry_points = (
            "phase4_run_analysis_job.py",
            "run_local_entropy_bin_sensitivity.py",
            "phase5_extended_sensitivity.py",
            "phase5_weight_stability.py",
        )
        for filename in entry_points:
            source = (SRC_DIR / filename).read_text(encoding="utf-8")
            self.assertIn("canonical_shape_view", source, filename)
            self.assertNotIn('arrays["trajectory_shape"]', source, filename)

    def test_detector_validates_state_dimensions_and_nonfinite_input(self) -> None:
        detector = EWGBDetector()
        with self.assertRaises(RuntimeError):
            detector.score(np.ones((4, 2)))
        fitted = EWGBDetector().fit(np.arange(24, dtype=float).reshape(12, 2))
        with self.assertRaises(ValueError):
            fitted.score(np.ones((4, 3)))
        invalid = np.ones((4, 2))
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fitted.score(invalid)

    def test_oracle_top_k_uses_exact_realized_count_with_stable_ties(self) -> None:
        labels = np.array([1, 0, 1, 0, 0])
        scores = np.ones(5, dtype=float)
        predictions = oracle_top_k_predictions(labels, scores)
        np.testing.assert_array_equal(predictions, np.array([1, 1, 0, 0, 0]))
        self.assertEqual(int(np.sum(predictions)), int(np.sum(labels)))
        metrics = evaluate_oracle_top_k(labels, scores)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
