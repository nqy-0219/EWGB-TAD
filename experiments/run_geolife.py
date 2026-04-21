"""
GeoLife real-world GPS trajectory experiments for EWGB-TAD.

Uses actual GPS trajectory sequences from Microsoft GeoLife dataset (Beijing).
Segments long trajectories into trip-level sub-trajectories, clusters into
route groups, then injects anomalies for evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import glob
import time
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.cluster import KMeans
from scipy import stats

from ewgb_tad.feature_extraction_v2 import extract_all_features_v2
from ewgb_tad.granular_ball import EWGBDetector
from ewgb_tad.baselines_v2 import get_all_baselines_v2
from ewgb_tad.data_generator import resample_trajectory
from ewgb_tad.detector import EWGBTAD as EWGBTADv3, evaluate_detector


GEOLIFE_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'geolife',
                             'Geolife Trajectories 1.3', 'Data')


def load_geolife_trajectories(root=None, max_trajs=10000, seq_len=32,
                               min_points=20, max_points=2000, seed=42,
                               bbox=None, min_od_dist_deg=0.005):
    if root is None:
        root = GEOLIFE_ROOT
    if bbox is None:
        bbox = (116.1, 39.75, 116.65, 40.15)

    rng = np.random.RandomState(seed)
    all_trajs = []
    lon_min, lat_min, lon_max, lat_max = bbox

    user_dirs = sorted(glob.glob(os.path.join(root, '*')))
    print(f"  Found {len(user_dirs)} users in GeoLife")
    print(f"  Filtering to bbox: lon[{lon_min},{lon_max}] lat[{lat_min},{lat_max}]")

    for user_dir in user_dirs:
        traj_dir = os.path.join(user_dir, 'Trajectory')
        if not os.path.isdir(traj_dir):
            continue
        plt_files = sorted(glob.glob(os.path.join(traj_dir, '*.plt')))
        for plt_file in plt_files:
            try:
                points = []
                with open(plt_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i < 6:
                            continue
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            lat, lon = float(parts[0]), float(parts[1])
                            if lat_min < lat < lat_max and lon_min < lon < lon_max:
                                points.append([lon, lat])

                if len(points) < min_points:
                    continue

                points = np.array(points)
                segments = segment_trajectory(points, max_gap_meters=300,
                                              min_segment_len=min_points)
                for seg in segments:
                    if min_points <= len(seg) <= max_points:
                        od_dist = np.linalg.norm(seg[-1] - seg[0])
                        if od_dist >= min_od_dist_deg:
                            resampled = resample_trajectory(seg, seq_len)
                            all_trajs.append(resampled)

            except (ValueError, IndexError):
                continue

    rng.shuffle(all_trajs)
    all_trajs = all_trajs[:max_trajs]
    print(f"  Loaded {len(all_trajs)} trajectory segments from GeoLife")
    return all_trajs


def segment_trajectory(points, max_gap_meters=500, min_segment_len=15):
    if len(points) < min_segment_len:
        return []

    diffs = np.diff(points, axis=0)
    dists_deg = np.linalg.norm(diffs, axis=1)
    dists_m = dists_deg * 111000

    segments = []
    start = 0
    for i, d in enumerate(dists_m):
        if d > max_gap_meters:
            if i - start >= min_segment_len:
                segments.append(points[start:i+1])
            start = i + 1

    if len(points) - start >= min_segment_len:
        segments.append(points[start:])

    return segments


def cluster_geolife_trajectories(trajectories, n_clusters=15, seed=42):
    od_features = np.array([
        [t[0, 0], t[0, 1], t[-1, 0], t[-1, 1]]
        for t in trajectories
    ])
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(od_features)
    clusters = defaultdict(list)
    for i, cl in enumerate(labels):
        clusters[cl].append(i)
    return clusters, km.cluster_centers_


def inject_anomalies_geolife(trajectories, contamination=0.1, seed=42):
    rng = np.random.RandomState(seed)
    n = len(trajectories)
    n_anomaly = int(n * contamination / (1 - contamination))
    n_per_type = max(1, n_anomaly // 4)

    injected = []
    anom_types = []

    for anom_type in range(4):
        for _ in range(n_per_type):
            idx = rng.randint(0, n)
            traj = trajectories[idx].copy()
            seq_len = len(traj)
            od_dist = np.linalg.norm(traj[-1] - traj[0])

            if od_dist < 1e-6:
                od_dist = np.mean(np.linalg.norm(np.diff(traj, axis=0), axis=1)) * seq_len

            if anom_type == 0:  # DETOUR
                s = int(0.2 * seq_len)
                e = int(0.8 * seq_len)
                direction = traj[-1] - traj[0]
                if np.linalg.norm(direction) < 1e-8:
                    direction = traj[seq_len//2] - traj[0]
                perp = np.array([-direction[1], direction[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-8)
                offset_mag = rng.uniform(0.4, 0.7) * od_dist
                sign = rng.choice([-1, 1])
                for k in range(s, e):
                    frac = (k - s) / (e - s + 1e-8)
                    traj[k] += sign * perp * offset_mag * np.sin(np.pi * frac)

            elif anom_type == 1:  # LOOP
                center = seq_len // 2
                radius = rng.uniform(0.2, 0.4) * od_dist
                n_loop = min(12, seq_len // 3)
                s = max(0, center - n_loop // 2)
                for k in range(n_loop):
                    idx_k = s + k
                    if idx_k < seq_len:
                        angle = 2 * np.pi * k / n_loop
                        traj[idx_k] += np.array([radius * np.cos(angle),
                                                  radius * np.sin(angle)])

            elif anom_type == 2:  # SPEED ANOMALY
                start_pt = int(rng.uniform(0.2, 0.5) * seq_len)
                n_slow = min(10, seq_len // 3)
                center_pt = traj[start_pt].copy()
                for k in range(n_slow):
                    if start_pt + k < seq_len:
                        traj[start_pt + k] = center_pt + rng.normal(0, od_dist * 0.003, 2)

            elif anom_type == 3:  # ROUTE DEVIATION
                dev_start = int(rng.uniform(0.35, 0.55) * seq_len)
                angle_dev = rng.uniform(50, 90) * np.pi / 180 * rng.choice([-1, 1])
                for k in range(dev_start, seq_len):
                    if k > 0:
                        diff = traj[k] - traj[k-1]
                        if np.linalg.norm(diff) < 1e-10:
                            diff = (traj[-1] - traj[0]) / seq_len
                        cos_a, sin_a = np.cos(angle_dev), np.sin(angle_dev)
                        rotated = np.array([cos_a * diff[0] - sin_a * diff[1],
                                            sin_a * diff[0] + cos_a * diff[1]])
                        traj[k] = traj[k-1] + rotated * 1.5
                    angle_dev *= 0.90

            injected.append(traj)
            anom_types.append(anom_type)

    all_trajs = list(trajectories) + injected
    all_labels = np.concatenate([np.zeros(n), np.ones(len(injected))])
    all_anom_types = np.concatenate([np.full(n, -1), np.array(anom_types)])

    perm = rng.permutation(len(all_trajs))
    all_trajs = [all_trajs[i] for i in perm]
    all_labels = all_labels[perm]
    all_anom_types = all_anom_types[perm]

    return all_trajs, all_labels, all_anom_types


class EWGBTADv3:
    def __init__(self, min_samples=8, purity_threshold=0.85, n_shape_dims=10):
        self.min_samples = min_samples
        self.purity_threshold = purity_threshold
        self.n_shape_dims = n_shape_dims
        self.detectors = {}
        self.view_names = []

    def fit(self, spatial_path, kinematic, entropy_feat, trajectories):
        from sklearn.decomposition import PCA
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
                use_local_entropy=False
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
            ent = -np.sum(hist * np.log2(hist + 1e-12))
            view_entropies[name] = ent

        max_ent = max(view_entropies.values()) + 1e-8
        weights = {}
        total_w = 0
        for name, ent in view_entropies.items():
            w = (max_ent - ent) / max_ent + 0.1
            weights[name] = w
            total_w += w
        for name in weights:
            weights[name] /= total_w

        self.view_weights = weights
        fused = np.zeros(len(spatial_path))
        for name, scores in view_scores.items():
            fused += weights[name] * scores
        return fused


def evaluate_detector(labels, scores, contamination):
    auc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    threshold = np.percentile(scores, (1 - contamination) * 100)
    preds = (scores > threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    return {'AUC': auc, 'AUPRC': auprc, 'F1': f1, 'Precision': prec, 'Recall': rec}


def run_single_geolife(base_trajs, n_normal=3000, contamination=0.1, seed=42,
                       clusters=None):
    rng = np.random.RandomState(seed)

    if clusters is None:
        n_cl = min(15, len(base_trajs) // 100)
        clusters, _ = cluster_geolife_trajectories(base_trajs, n_clusters=n_cl, seed=seed)

    normal_trajs = []
    per_cluster = max(1, n_normal // len(clusters))
    for cl_id, indices in clusters.items():
        chosen = rng.choice(indices, min(per_cluster, len(indices)),
                           replace=len(indices) < per_cluster)
        for idx in chosen:
            normal_trajs.append(base_trajs[idx].copy())
        if len(normal_trajs) >= n_normal:
            break
    normal_trajs = normal_trajs[:n_normal]

    trajs, labels, anom_types = inject_anomalies_geolife(normal_trajs, contamination, seed)

    spatial, kinematic, entropy_feat, path_feat, all_features = extract_all_features_v2(trajs)
    spatial_path = np.hstack([spatial, path_feat])

    results = {}

    t0 = time.time()
    det = EWGBTADv3(min_samples=8, purity_threshold=0.85, n_shape_dims=10)
    det.fit(spatial_path, kinematic, entropy_feat, trajs)
    scores = det.score(spatial_path, kinematic, entropy_feat, trajs)
    elapsed = time.time() - t0
    metrics = evaluate_detector(labels, scores, contamination)
    metrics['Runtime'] = elapsed
    results['EWGB-TAD v3 (4-view)'] = metrics

    baselines = get_all_baselines_v2(contamination=contamination, seed=seed)
    for name, det_b in baselines.items():
        t0 = time.time()
        if name in ('DTW-KNN', 'SegmentOD'):
            scores_b = det_b.fit_score(all_features, trajs)
        else:
            scores_b = det_b.fit_score(all_features)
        elapsed = time.time() - t0
        metrics = evaluate_detector(labels, scores_b, contamination)
        metrics['Runtime'] = elapsed
        results[name] = metrics

    threshold = np.percentile(scores, (1 - contamination) * 100)
    per_type = {}
    type_names = {0: 'detour', 1: 'loop', 2: 'speed', 3: 'route_deviation'}
    for atype in range(4):
        mask = anom_types == atype
        if mask.sum() > 0:
            detected = (scores[mask] > threshold).sum()
            per_type[type_names[atype]] = {'recall': detected / mask.sum()}

    return results, per_type


def run_full_geolife(n_normal=3000, contamination_ratios=[0.05, 0.10, 0.15],
                     seeds=[42, 123, 456, 789, 1024], output_dir='../results'):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading GeoLife trajectories...")
    base_trajs = load_geolife_trajectories(max_trajs=8000, seq_len=32, seed=42)
    if len(base_trajs) < 500:
        print("ERROR: Not enough trajectories loaded from GeoLife.")
        return None

    print(f"Using {len(base_trajs)} base trajectories")

    n_cl = min(15, len(base_trajs) // 100)
    print(f"Clustering into {n_cl} route groups...")
    clusters, centers = cluster_geolife_trajectories(base_trajs, n_clusters=n_cl, seed=42)
    for cl_id, indices in sorted(clusters.items()):
        print(f"  Cluster {cl_id}: {len(indices)} trajectories")

    all_results = {}

    for cont in contamination_ratios:
        print(f"\n{'='*60}")
        print(f"GeoLife — Contamination ratio: {cont*100:.0f}%")
        print(f"{'='*60}")

        seed_results = defaultdict(lambda: defaultdict(list))
        all_per_type = defaultdict(lambda: defaultdict(list))
        raw_aucs = defaultdict(list)

        for seed in seeds:
            print(f"  Seed {seed}...", end=' ')
            t0 = time.time()
            results, per_type = run_single_geolife(base_trajs, n_normal, cont, seed,
                                                       clusters=clusters)
            print(f"done ({time.time()-t0:.1f}s)")

            for method, metrics in results.items():
                for metric, value in metrics.items():
                    seed_results[method][metric].append(value)
                raw_aucs[method].append(metrics['AUC'])

            for atype, s in per_type.items():
                all_per_type[atype]['recall'].append(s['recall'])

        aggregated = {}
        for method in seed_results:
            agg = {}
            for metric in seed_results[method]:
                values = seed_results[method][metric]
                agg[metric] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
            aggregated[method] = agg

        all_results[f'{cont*100:.0f}%'] = aggregated

        print(f"\n{'Method':<35} {'AUC':>12} {'AUPRC':>12} {'F1':>12} {'Time':>8}")
        print("-" * 85)
        methods_ours = sorted([m for m in aggregated if m.startswith('EWGB')])
        methods_base = sorted([m for m in aggregated if not m.startswith('EWGB')])

        for method in methods_ours + methods_base:
            m = aggregated[method]
            auc_str = f"{m['AUC']['mean']:.4f}±{m['AUC']['std']:.4f}"
            auprc_str = f"{m['AUPRC']['mean']:.4f}±{m['AUPRC']['std']:.4f}"
            f1_str = f"{m['F1']['mean']:.4f}±{m['F1']['std']:.4f}"
            time_str = f"{m['Runtime']['mean']:.3f}s"
            print(f"  {method:<33} {auc_str:>12} {auprc_str:>12} {f1_str:>12} {time_str:>8}")

        print(f"\n  Per-anomaly-type recall (EWGB-TAD v3):")
        for atype in ['detour', 'loop', 'speed', 'route_deviation']:
            if atype in all_per_type:
                vals = all_per_type[atype]['recall']
                print(f"    {atype:<20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        print(f"\n  Wilcoxon tests (EWGB-TAD v3 vs baselines):")
        our_aucs = raw_aucs.get('EWGB-TAD v3 (4-view)', [])
        for method in methods_base:
            base_aucs = raw_aucs[method]
            if len(our_aucs) >= 5:
                try:
                    _, p = stats.wilcoxon(our_aucs, base_aucs)
                    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
                    delta = np.mean(our_aucs) - np.mean(base_aucs)
                    print(f"    vs {method:<25}: Δ={delta:+.4f}, p={p:.4f} {sig}")
                except Exception:
                    print(f"    vs {method:<25}: n/a")

    output_path = os.path.join(output_dir, 'experiment_results_geolife.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return all_results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_full_geolife(n_normal=2000, contamination_ratios=[0.10],
                        seeds=[42, 123, 456], output_dir='../results')
    else:
        run_full_geolife(n_normal=3000, output_dir='../results')
