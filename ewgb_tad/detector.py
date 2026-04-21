"""
EWGB-TAD: 4-view entropy-weighted granular-ball trajectory anomaly detector.

Views:
    spatial-path (16D) + kinematic (8D) + entropy (6D) + trajectory-shape PCA (10D)
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
)

from .granular_ball import EWGBDetector


def evaluate_detector(labels, scores, contamination):
    """Standard evaluation: AUC, AUPRC, F1/P/R at the contamination percentile."""
    auc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    threshold = np.percentile(scores, (1 - contamination) * 100)
    preds = (scores > threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    return {'AUC': auc, 'AUPRC': auprc, 'F1': f1, 'Precision': prec, 'Recall': rec}


class EWGBTAD:
    """
    Four-view granular-ball trajectory anomaly detector with entropy-weighted fusion.

    Parameters
    ----------
    min_samples : int
        Minimum number of samples per granular ball (default 8).
    purity_threshold : float
        Ball-quality threshold for hierarchical splitting (default 0.85).
    n_shape_dims : int
        PCA dimensionality for the trajectory-shape view (default 10).
    """

    def __init__(self, min_samples=8, purity_threshold=0.85, n_shape_dims=10):
        self.min_samples = min_samples
        self.purity_threshold = purity_threshold
        self.n_shape_dims = n_shape_dims
        self.detectors = {}
        self.view_names = []

    def fit(self, spatial_path, kinematic, entropy_feat, trajectories):
        flat_trajs = np.array([t.flatten() for t in trajectories])
        self.pca = PCA(n_components=self.n_shape_dims)
        shape_feat = self.pca.fit_transform(flat_trajs)

        views = {
            'spatial_path': spatial_path,
            'kinematic': kinematic,
            'entropy': entropy_feat,
            'shape': shape_feat,
        }
        self.view_names = list(views.keys())

        for name, data in views.items():
            det = EWGBDetector(
                min_samples=self.min_samples,
                purity_threshold=self.purity_threshold,
                use_local_entropy=False,
            )
            det.fit(data)
            self.detectors[name] = det
        return self

    def score(self, spatial_path, kinematic, entropy_feat, trajectories):
        flat_trajs = np.array([t.flatten() for t in trajectories])
        shape_feat = self.pca.transform(flat_trajs)

        views_data = {
            'spatial_path': spatial_path,
            'kinematic': kinematic,
            'entropy': entropy_feat,
            'shape': shape_feat,
        }

        view_scores = {}
        for name in self.view_names:
            raw = self.detectors[name].score(views_data[name])
            smin, smax = raw.min(), raw.max()
            if smax - smin > 1e-8:
                raw = (raw - smin) / (smax - smin)
            view_scores[name] = raw

        view_entropies = {}
        for name, scores in view_scores.items():
            hist, _ = np.histogram(scores, bins=20, density=False)
            hist = hist / (hist.sum() + 1e-8)
            hist = hist[hist > 0]
            view_entropies[name] = -np.sum(hist * np.log2(hist + 1e-12))

        max_ent = max(view_entropies.values()) + 1e-8
        weights = {name: (max_ent - ent) / max_ent + 0.1 for name, ent in view_entropies.items()}
        total = sum(weights.values())
        for name in weights:
            weights[name] /= total
        self.view_weights = weights

        fused = np.zeros(len(spatial_path))
        for name, scores in view_scores.items():
            fused += weights[name] * scores
        return fused
