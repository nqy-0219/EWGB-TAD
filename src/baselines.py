"""
Baseline methods for comparison with EWGB-TAD.
"""

import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
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


class IBoostODEDetector(BaselineDetector):
    """
    iBoost-ODE-style iterative unsupervised outlier-detector ensemble.

    This CPU implementation builds repeated random-subspace detectors and
    iteratively increases the influence of observations that are consistently
    assigned high anomaly scores. The final score is a robust rank-aggregated
    ensemble score over Isolation Forest, LOF, ECOD, and OCSVM-style evidence.
    """
    def __init__(self, contamination=0.1, seed=42, n_rounds=5, subspace_ratio=0.65):
        super().__init__('iBoost-ODE')
        self.contamination = contamination
        self.seed = seed
        self.n_rounds = n_rounds
        self.subspace_ratio = subspace_ratio
        self.scaler = StandardScaler()

    @staticmethod
    def _normalize(scores):
        scores = np.asarray(scores, dtype=float)
        smin, smax = np.min(scores), np.max(scores)
        if smax - smin < 1e-12:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)

    @staticmethod
    def _rank_score(scores):
        order = np.argsort(np.argsort(scores))
        return order.astype(float) / max(1, len(scores) - 1)

    @staticmethod
    def _ecod_score(X):
        n, d = X.shape
        scores = np.zeros(n)
        for dim in range(d):
            vals = X[:, dim]
            sorted_vals = np.sort(vals)
            left = np.searchsorted(sorted_vals, vals, side='right') / n
            right = 1 - left + 1.0 / n
            scores += np.maximum(-np.log(left + 1e-10), -np.log(right + 1e-10))
        return scores

    def fit(self, features):
        from sklearn.ensemble import IsolationForest

        X = self.scaler.fit_transform(features)
        n, d = X.shape
        rng = np.random.RandomState(self.seed)
        self.detectors_ = []
        self.ecod_subspaces_ = []
        self.round_weights_ = []

        target = np.zeros(n)
        sample_weights = np.ones(n) / n
        n_sub = max(2, min(d, int(np.ceil(d * self.subspace_ratio))))

        for r in range(self.n_rounds):
            subspace = rng.choice(d, size=n_sub, replace=False)
            Xs = X[:, subspace]

            iforest = IsolationForest(
                n_estimators=80,
                contamination=self.contamination,
                random_state=self.seed + r,
                n_jobs=-1,
                max_samples=min(512, n),
            )
            try:
                iforest.fit(Xs, sample_weight=sample_weights)
            except TypeError:
                iforest.fit(Xs)
            if_score = -iforest.score_samples(Xs)

            lof = LocalOutlierFactor(n_neighbors=min(25, n - 1), contamination=self.contamination)
            lof.fit_predict(Xs)
            lof_score = -lof.negative_outlier_factor_

            ecod_score = self._ecod_score(Xs)

            # OCSVM is comparatively expensive, so use it on a reduced PCA view
            # only when the sample size is manageable.
            if n <= 2500:
                try:
                    oc = OneClassSVM(nu=self.contamination, kernel='rbf', gamma='scale')
                    oc.fit(Xs)
                    oc_score = -oc.decision_function(Xs).ravel()
                except Exception:
                    oc_score = np.zeros(n)
            else:
                oc_score = np.zeros(n)

            combined = (
                0.35 * self._rank_score(if_score)
                + 0.25 * self._rank_score(lof_score)
                + 0.25 * self._rank_score(ecod_score)
                + 0.15 * self._rank_score(oc_score)
            )
            target = 0.65 * target + 0.35 * combined
            centered = target - np.median(target)
            sample_weights = np.exp(np.clip(centered, -2.0, 2.0))
            sample_weights = sample_weights / (sample_weights.sum() + 1e-12)

            # Later rounds are more aligned with the updated target.
            self.detectors_.append((subspace, iforest))
            self.ecod_subspaces_.append((subspace, Xs.copy()))
            self.round_weights_.append(1.0 + 0.25 * r)

        self.training_target_ = target
        return self

    def score(self, features):
        X = self.scaler.transform(features)
        n = len(X)
        scores = np.zeros(n)
        total_weight = 0.0
        for weight, (subspace, iforest), (_, train_subspace) in zip(
            self.round_weights_, self.detectors_, self.ecod_subspaces_
        ):
            Xs = X[:, subspace]
            if_score = -iforest.score_samples(Xs)

            # ECOD score against the current data distribution. Experiments in
            # this paper use transductive unsupervised scoring, matching other
            # tabular baselines in this codebase.
            ecod_score = self._ecod_score(Xs)
            combined = 0.60 * self._rank_score(if_score) + 0.40 * self._rank_score(ecod_score)
            scores += weight * combined
            total_weight += weight
        return scores / (total_weight + 1e-12)


class RSDPMDetector(BaselineDetector):
    """
    Random-subspace Dirichlet-process mixture ensemble for tabular outliers.

    Each ensemble member fits a Bayesian Gaussian mixture on a random feature
    subspace and random sample subset. Observations with low mixture likelihood
    and low maximum component responsibility receive high anomaly scores.
    """
    def __init__(self, seed=42, n_ensembles=12, subspace_ratio=0.55, max_components=12):
        super().__init__('RS-DPM')
        self.seed = seed
        self.n_ensembles = n_ensembles
        self.subspace_ratio = subspace_ratio
        self.max_components = max_components
        self.scaler = StandardScaler()

    @staticmethod
    def _normalize(scores):
        scores = np.asarray(scores, dtype=float)
        smin, smax = np.min(scores), np.max(scores)
        if smax - smin < 1e-12:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)

    @staticmethod
    def _rank_score(scores):
        order = np.argsort(np.argsort(scores))
        return order.astype(float) / max(1, len(scores) - 1)

    def fit(self, features):
        from sklearn.mixture import BayesianGaussianMixture

        X = self.scaler.fit_transform(features)
        n, d = X.shape
        rng = np.random.RandomState(self.seed)
        self.members_ = []
        n_sub = max(2, min(d, int(np.ceil(d * self.subspace_ratio))))
        sample_size = min(n, max(800, int(0.65 * n)))

        for i in range(self.n_ensembles):
            subspace = rng.choice(d, size=n_sub, replace=False)
            sample_idx = rng.choice(n, size=sample_size, replace=False)
            X_train = X[sample_idx][:, subspace]
            n_components = min(self.max_components, max(2, sample_size // 80))
            model = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.1,
                max_iter=250,
                n_init=1,
                reg_covar=1e-5,
                random_state=self.seed + i,
            )
            model.fit(X_train)
            self.members_.append((subspace, model))
        return self

    def score(self, features):
        X = self.scaler.transform(features)
        scores = np.zeros(len(X))
        for subspace, model in self.members_:
            Xs = X[:, subspace]
            nll = -model.score_samples(Xs)
            resp = model.predict_proba(Xs)
            resp_score = 1.0 - np.max(resp, axis=1)
            comp_weights = model.weights_
            comp = np.argmax(resp, axis=1)
            small_component_score = -np.log(comp_weights[comp] + 1e-12)
            combined = (
                0.60 * self._rank_score(nll)
                + 0.25 * self._rank_score(resp_score)
                + 0.15 * self._rank_score(small_component_score)
            )
            scores += combined
        return scores / max(1, len(self.members_))


class CoMadOutDetector(BaselineDetector):
    """
    CoMadOut-style robust tabular outlier detector.

    The implementation follows the core idea of CoMadOut: robust centering,
    co-median absolute dependency estimation, and a robust Mahalanobis-type
    outlier score. A PCA cap is used for high-dimensional trajectory features
    to keep the covariance inverse stable.
    """
    def __init__(self, max_dim=40, shrinkage=0.08):
        super().__init__('CoMadOut')
        self.max_dim = max_dim
        self.shrinkage = shrinkage

    @staticmethod
    def _rank_score(scores):
        order = np.argsort(np.argsort(scores))
        return order.astype(float) / max(1, len(scores) - 1)

    def _robust_project(self, features, fit=False):
        X = np.asarray(features, dtype=float)
        if fit:
            self.median_ = np.median(X, axis=0)
            self.mad_ = np.median(np.abs(X - self.median_), axis=0)
            self.mad_[self.mad_ < 1e-9] = 1.0
        Z = (X - self.median_) / self.mad_
        if fit:
            self.pca_ = None
            if Z.shape[1] > self.max_dim:
                n_components = min(self.max_dim, Z.shape[0] - 1, Z.shape[1])
                self.pca_ = PCA(n_components=n_components, random_state=0)
                Z = self.pca_.fit_transform(Z)
        elif self.pca_ is not None:
            Z = self.pca_.transform(Z)
        return Z

    def fit(self, features):
        Z = self._robust_project(features, fit=True)
        self.center_ = np.median(Z, axis=0)
        R = Z - self.center_
        d = R.shape[1]
        comad = np.zeros((d, d), dtype=float)
        for i in range(d):
            ri = R[:, i]
            for j in range(i, d):
                value = np.median(ri * R[:, j])
                comad[i, j] = value
                comad[j, i] = value
        diag_scale = np.median(np.diag(np.abs(comad))) + 1e-6
        cov = (1.0 - self.shrinkage) * comad + self.shrinkage * diag_scale * np.eye(d)
        cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(d)
        self.precision_ = np.linalg.pinv(cov)
        return self

    def score(self, features):
        Z = self._robust_project(features, fit=False)
        R = Z - self.center_
        mahal = np.sqrt(np.maximum(0.0, np.sum((R @ self.precision_) * R, axis=1)))
        max_robust = np.max(np.abs(R), axis=1)
        return 0.75 * self._rank_score(mahal) + 0.25 * self._rank_score(max_robust)


class GBFRDDetector(BaselineDetector):
    """
    Granular-ball fuzzy-rough-style detector for tabular outliers.

    It recursively forms data-adaptive balls, then scores samples by fuzzy
    membership to the nearest ball, ball sparsity, and boundary deviation.
    """
    def __init__(self, seed=42, min_samples=24, max_balls=96, max_dim=36, split_gain=0.88):
        super().__init__('GBFRD')
        self.seed = seed
        self.min_samples = min_samples
        self.max_balls = max_balls
        self.max_dim = max_dim
        self.split_gain = split_gain
        self.scaler = StandardScaler()

    @staticmethod
    def _rank_score(scores):
        order = np.argsort(np.argsort(scores))
        return order.astype(float) / max(1, len(scores) - 1)

    @staticmethod
    def _radius(X):
        if len(X) <= 1:
            return 1e-6
        center = X.mean(axis=0)
        return float(np.mean(np.linalg.norm(X - center, axis=1)) + 1e-6)

    def _transform(self, features, fit=False):
        X = self.scaler.fit_transform(features) if fit else self.scaler.transform(features)
        if fit:
            self.pca_ = None
            if X.shape[1] > self.max_dim:
                n_components = min(self.max_dim, X.shape[0] - 1, X.shape[1])
                self.pca_ = PCA(n_components=n_components, random_state=self.seed)
                X = self.pca_.fit_transform(X)
        elif self.pca_ is not None:
            X = self.pca_.transform(X)
        return X

    def fit(self, features):
        X = self._transform(features, fit=True)
        rng = np.random.RandomState(self.seed)
        balls = [np.arange(len(X))]

        changed = True
        while changed and len(balls) < self.max_balls:
            changed = False
            new_balls = []
            for idx in balls:
                if len(idx) < 2 * self.min_samples or len(new_balls) + len(balls) >= self.max_balls:
                    new_balls.append(idx)
                    continue
                Xi = X[idx]
                parent_radius = self._radius(Xi)
                try:
                    km = KMeans(n_clusters=2, n_init=3, random_state=int(rng.randint(0, 1_000_000)))
                    labels = km.fit_predict(Xi)
                except Exception:
                    new_balls.append(idx)
                    continue
                left = idx[labels == 0]
                right = idx[labels == 1]
                if len(left) < self.min_samples or len(right) < self.min_samples:
                    new_balls.append(idx)
                    continue
                child_radius = (
                    len(left) * self._radius(X[left]) + len(right) * self._radius(X[right])
                ) / len(idx)
                if child_radius < self.split_gain * parent_radius:
                    new_balls.extend([left, right])
                    changed = True
                else:
                    new_balls.append(idx)
            balls = new_balls

        centers, radii, sizes = [], [], []
        for idx in balls:
            Xi = X[idx]
            centers.append(Xi.mean(axis=0))
            radii.append(self._radius(Xi))
            sizes.append(len(idx))
        self.centers_ = np.asarray(centers)
        self.radii_ = np.asarray(radii) + 1e-6
        self.sizes_ = np.asarray(sizes, dtype=float)
        self.nn_ = NearestNeighbors(n_neighbors=1).fit(self.centers_)
        self.median_size_ = float(np.median(self.sizes_) + 1e-6)
        return self

    def score(self, features):
        X = self._transform(features, fit=False)
        dist, ind = self.nn_.kneighbors(X)
        dist = dist[:, 0]
        ball_idx = ind[:, 0]
        norm_dist = dist / self.radii_[ball_idx]
        membership_loss = 1.0 - np.exp(-(norm_dist ** 2))
        sparsity = np.sqrt(self.median_size_ / (self.sizes_[ball_idx] + 1e-6))
        boundary = np.maximum(0.0, norm_dist - 1.0)
        return (
            0.45 * self._rank_score(membership_loss)
            + 0.35 * self._rank_score(sparsity)
            + 0.20 * self._rank_score(boundary)
        )


class MFIODDetector(BaselineDetector):
    """
    Multi-scale fuzzy-information outlier detector.

    This CPU version combines multi-scale fuzzy neighborhood density, local
    distance entropy, and marginal tail evidence for tabular trajectory
    representations.
    """
    def __init__(self, max_dim=40, neighbors=(6, 12, 24, 48)):
        super().__init__('MFIOD')
        self.max_dim = max_dim
        self.neighbors = neighbors
        self.scaler = StandardScaler()

    @staticmethod
    def _rank_score(scores):
        order = np.argsort(np.argsort(scores))
        return order.astype(float) / max(1, len(scores) - 1)

    @staticmethod
    def _ecod_score(X):
        n, d = X.shape
        scores = np.zeros(n)
        for dim in range(d):
            vals = X[:, dim]
            sorted_vals = np.sort(vals)
            left = np.searchsorted(sorted_vals, vals, side='right') / n
            right = 1 - left + 1.0 / n
            scores += np.maximum(-np.log(left + 1e-10), -np.log(right + 1e-10))
        return scores

    def _transform(self, features, fit=False):
        X = self.scaler.fit_transform(features) if fit else self.scaler.transform(features)
        if fit:
            self.pca_ = None
            if X.shape[1] > self.max_dim:
                n_components = min(self.max_dim, X.shape[0] - 1, X.shape[1])
                self.pca_ = PCA(n_components=n_components, random_state=0)
                X = self.pca_.fit_transform(X)
        elif self.pca_ is not None:
            X = self.pca_.transform(X)
        return X

    def fit(self, features):
        self.X_ = self._transform(features, fit=True)
        return self

    def score(self, features):
        X = self._transform(features, fit=False)
        n = len(X)
        multi_density_scores = []
        entropy_scores = []
        for k in self.neighbors:
            k_eff = min(max(2, k), n - 1)
            nn = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=-1).fit(X)
            distances, _ = nn.kneighbors(X)
            dists = distances[:, 1:]
            sigma = np.median(dists[:, -1]) + 1e-9
            fuzzy = np.exp(-(dists / sigma) ** 2)
            multi_density_scores.append(-np.log(np.mean(fuzzy, axis=1) + 1e-10))
            prob = dists / (np.sum(dists, axis=1, keepdims=True) + 1e-12)
            entropy = -np.sum(prob * np.log(prob + 1e-12), axis=1) / np.log(k_eff)
            entropy_scores.append(entropy * (dists[:, -1] / sigma))
        density_score = np.mean(np.vstack(multi_density_scores), axis=0)
        entropy_score = np.mean(np.vstack(entropy_scores), axis=0)
        tail_score = self._ecod_score(X)
        return (
            0.50 * self._rank_score(density_score)
            + 0.30 * self._rank_score(entropy_score)
            + 0.20 * self._rank_score(tail_score)
        )


class FSODDetector(BaselineDetector):
    """
    Fuzzy-neighborhood and multi-information tabular outlier detector.

    It uses entropy-weighted marginal evidence, fuzzy neighborhood membership,
    and neighbor-relative deviation as complementary outlier information.
    """
    def __init__(self, max_dim=40, n_neighbors=24, n_bins=12):
        super().__init__('FSOD')
        self.max_dim = max_dim
        self.n_neighbors = n_neighbors
        self.n_bins = n_bins
        self.scaler = StandardScaler()

    @staticmethod
    def _rank_score(scores):
        order = np.argsort(np.argsort(scores))
        return order.astype(float) / max(1, len(scores) - 1)

    def _transform(self, features, fit=False):
        X = self.scaler.fit_transform(features) if fit else self.scaler.transform(features)
        if fit:
            self.pca_ = None
            if X.shape[1] > self.max_dim:
                n_components = min(self.max_dim, X.shape[0] - 1, X.shape[1])
                self.pca_ = PCA(n_components=n_components, random_state=0)
                X = self.pca_.fit_transform(X)
        elif self.pca_ is not None:
            X = self.pca_.transform(X)
        return X

    def _feature_weights(self, X):
        entropies = []
        for j in range(X.shape[1]):
            hist, _ = np.histogram(X[:, j], bins=self.n_bins)
            prob = hist.astype(float) / max(1, hist.sum())
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log(prob)) / max(1e-9, np.log(self.n_bins))
            entropies.append(entropy)
        entropies = np.asarray(entropies)
        weights = 1.0 / (entropies + 0.08)
        weights = weights / (np.mean(weights) + 1e-12)
        return weights

    @staticmethod
    def _weighted_tail_score(X, weights):
        n, d = X.shape
        scores = np.zeros(n)
        for dim in range(d):
            vals = X[:, dim]
            sorted_vals = np.sort(vals)
            left = np.searchsorted(sorted_vals, vals, side='right') / n
            right = 1 - left + 1.0 / n
            scores += weights[dim] * np.maximum(-np.log(left + 1e-10), -np.log(right + 1e-10))
        return scores

    def fit(self, features):
        self.X_ = self._transform(features, fit=True)
        self.weights_ = self._feature_weights(self.X_)
        self.X_weighted_ = self.X_ * np.sqrt(self.weights_)
        return self

    def score(self, features):
        X = self._transform(features, fit=False)
        Xw = X * np.sqrt(self.weights_)
        n = len(Xw)
        k_eff = min(max(2, self.n_neighbors), n - 1)
        nn = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=-1).fit(Xw)
        distances, indices = nn.kneighbors(Xw)
        dists = distances[:, 1:]
        neigh_idx = indices[:, 1:]
        sigma = np.median(dists[:, -1]) + 1e-9
        membership_loss = 1.0 - np.mean(np.exp(-(dists / sigma) ** 2), axis=1)
        local_mean = np.mean(Xw[neigh_idx], axis=1)
        rel_deviation = np.linalg.norm(Xw - local_mean, axis=1) / sigma
        tail_score = self._weighted_tail_score(X, self.weights_)
        return (
            0.40 * self._rank_score(tail_score)
            + 0.35 * self._rank_score(membership_loss)
            + 0.25 * self._rank_score(rel_deviation)
        )


def get_all_baselines(contamination=0.1, seed=42):
    """Return the final classical/tabular baselines used in the manuscript."""
    return {
        'IsolationForest': IForestDetector(contamination=contamination, seed=seed),
        'ECOD': ECODDetector(),
        'iBoost-ODE': IBoostODEDetector(contamination=contamination, seed=seed),
        'CoMadOut': CoMadOutDetector(),
        'GBFRD': GBFRDDetector(seed=seed),
        'MFIOD': MFIODDetector(),
        'FSOD': FSODDetector(),
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
