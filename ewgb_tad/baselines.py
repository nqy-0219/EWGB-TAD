"""
Baseline methods for comparison with EWGB-TAD.
"""

import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class BaselineDetector:
    """Base class for baseline anomaly detectors."""

    def __init__(self, name):
        self.name = name
        self.fit_time = 0
        self.score_time = 0

    def fit(self, features):
        raise NotImplementedError

    def score(self, features):
        raise NotImplementedError

    def fit_score(self, features):
        t0 = time.time()
        self.fit(features)
        self.fit_time = time.time() - t0

        t0 = time.time()
        scores = self.score(features)
        self.score_time = time.time() - t0

        return scores


class IForestDetector(BaselineDetector):
    def __init__(self, n_estimators=200, contamination=0.1, seed=42):
        super().__init__('IsolationForest')
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def fit(self, features):
        X = self.scaler.fit_transform(features)
        self.model.fit(X)
        self._X = X

    def score(self, features):
        X = self.scaler.transform(features)
        return -self.model.score_samples(X)


class LOFDetector(BaselineDetector):
    def __init__(self, n_neighbors=20, contamination=0.1):
        super().__init__('LOF')
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.scaler = StandardScaler()

    def fit(self, features):
        self._features = self.scaler.fit_transform(features)

    def score(self, features):
        X = self.scaler.transform(features)
        clf = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False
        )
        clf.fit_predict(X)
        return -clf.negative_outlier_factor_


class KNNDetector(BaselineDetector):
    def __init__(self, k=10):
        super().__init__('KNN')
        self.k = k
        self.scaler = StandardScaler()

    def fit(self, features):
        self._X = self.scaler.fit_transform(features)
        self.nn = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1)
        self.nn.fit(self._X)

    def score(self, features):
        X = self.scaler.transform(features)
        distances, _ = self.nn.kneighbors(X)
        return distances[:, -1]  # distance to k-th neighbor


class OCSVMDetector(BaselineDetector):
    def __init__(self, nu=0.1, max_dim=16, seed=42):
        super().__init__('OCSVM')
        self.nu = nu
        self.max_dim = max_dim
        self.scaler = StandardScaler()
        self.pca = None

    def fit(self, features):
        X = self.scaler.fit_transform(features)
        if X.shape[1] > self.max_dim:
            self.pca = PCA(n_components=self.max_dim)
            X = self.pca.fit_transform(X)
        self.model = OneClassSVM(kernel='rbf', nu=self.nu)
        self.model.fit(X)

    def score(self, features):
        X = self.scaler.transform(features)
        if self.pca is not None:
            X = self.pca.transform(X)
        return -self.model.decision_function(X)


class DBSCANDetector(BaselineDetector):
    def __init__(self, min_samples=10, eps=None):
        super().__init__('DBSCAN')
        self.min_samples = min_samples
        self.eps = eps
        self.scaler = StandardScaler()

    def fit(self, features):
        self._X = self.scaler.fit_transform(features)
        if self.eps is None:
            # Estimate eps from k-distance
            nn = NearestNeighbors(n_neighbors=self.min_samples)
            nn.fit(self._X)
            distances, _ = nn.kneighbors(self._X)
            k_distances = np.sort(distances[:, -1])
            # Use knee point heuristic: 90th percentile
            self.eps = np.percentile(k_distances, 90)

    def score(self, features):
        X = self.scaler.transform(features)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = db.fit_predict(X)

        # Anomaly score: -1 labels are outliers
        # For continuous score, use distance to nearest core point
        core_mask = np.zeros(len(X), dtype=bool)
        if hasattr(db, 'core_sample_indices_') and len(db.core_sample_indices_) > 0:
            core_mask[db.core_sample_indices_] = True
            core_points = X[core_mask]
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(core_points)
            distances, _ = nn.kneighbors(X)
            scores = distances[:, 0]
        else:
            # Fallback: all points are outliers
            center = np.mean(X, axis=0)
            scores = np.linalg.norm(X - center, axis=1)

        return scores


class COPODDetector(BaselineDetector):
    """
    Copula-Based Outlier Detection (COPOD).
    Simplified implementation using empirical CDF.
    """
    def __init__(self):
        super().__init__('COPOD')

    def fit(self, features):
        self._features = features.copy()
        self.n_features = features.shape[1]

    def score(self, features):
        n = len(features)
        scores = np.zeros(n)

        for dim in range(self.n_features):
            vals = features[:, dim]
            # Left tail: P(X <= x)
            left_cdf = np.array([np.mean(vals <= v) for v in vals])
            # Right tail: P(X >= x)
            right_cdf = np.array([np.mean(vals >= v) for v in vals])

            # Negative log-likelihood of tails
            left_nll = -np.log(left_cdf + 1e-10)
            right_nll = -np.log(right_cdf + 1e-10)

            scores += np.maximum(left_nll, right_nll)

        return scores


class ECODDetector(BaselineDetector):
    """
    Empirical Cumulative Distribution-based Outlier Detection (ECOD).
    Uses empirical CDF on each dimension.
    """
    def __init__(self):
        super().__init__('ECOD')

    def fit(self, features):
        self._features = features.copy()

    def score(self, features):
        n, d = features.shape
        scores = np.zeros(n)

        for dim in range(d):
            vals = features[:, dim]
            sorted_vals = np.sort(vals)

            # Left tail ECDF
            left_ecdf = np.searchsorted(sorted_vals, vals, side='right') / n
            # Right tail ECDF
            right_ecdf = 1 - left_ecdf + 1.0 / n

            # Anomaly contribution from this dimension
            left_score = -np.log(left_ecdf + 1e-10)
            right_score = -np.log(right_ecdf + 1e-10)

            scores += np.maximum(left_score, right_score)

        return scores


def get_all_baselines(contamination=0.1, seed=42):
    """Return a dictionary of all baseline detectors."""
    return {
        'IsolationForest': IForestDetector(contamination=contamination, seed=seed),
        'LOF': LOFDetector(contamination=contamination),
        'KNN': KNNDetector(k=10),
        'OCSVM': OCSVMDetector(nu=contamination),
        'DBSCAN': DBSCANDetector(),
        'COPOD': COPODDetector(),
        'ECOD': ECODDetector(),
    }


if __name__ == '__main__':
    from data_generator import generate_synthetic_trajectories
    from feature_extraction import extract_all_features
    from sklearn.metrics import roc_auc_score

    trajs, labels, _, _ = generate_synthetic_trajectories(n_normal=1000, n_anomaly_per_type=25)
    _, _, _, features = extract_all_features(trajs)

    baselines = get_all_baselines(contamination=0.1)
    for name, det in baselines.items():
        scores = det.fit_score(features)
        auc = roc_auc_score(labels, scores)
        print(f"{name:20s}: AUC={auc:.4f}, fit={det.fit_time:.2f}s, score={det.score_time:.2f}s")
