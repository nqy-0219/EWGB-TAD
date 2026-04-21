"""
Porto Taxi real-world dataset experiments for EWGB-TAD.

Strategy: Load real OD pairs from Porto Taxi (Figshare), generate realistic
trajectories between real OD coordinates using interpolation with road-like
curvature, then inject anomalies. This uses real-world spatial distribution
from 367K Porto taxi trips.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv
import re
import time
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

from ewgb_tad.feature_extraction_v2 import extract_all_features_v2
from ewgb_tad.granular_ball import EWGBDetector, build_granular_balls, compute_feature_entropy_weights
from ewgb_tad.baselines_v2 import get_all_baselines_v2
from ewgb_tad.data_generator import resample_trajectory
from ewgb_tad.detector import EWGBTAD as EWGBTADv3, evaluate_detector


def parse_point(s):
    m = re.match(r'POINT\(([-\d.]+)\s+([-\d.]+)\)', s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def load_porto_od_pairs(csv_path, max_pairs=50000, seed=42):
    rng = np.random.RandomState(seed)
    pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = parse_point(row['source_point'])
            tgt = parse_point(row['target_point'])
            if src is None or tgt is None:
                continue
            lon1, lat1 = src
            lon2, lat2 = tgt
            dist = np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2)
            if dist < 0.005 or dist > 0.15:
                continue
            pairs.append((np.array([lon1, lat1]), np.array([lon2, lat2])))
            if len(pairs) >= max_pairs * 3:
                break

    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    print(f"  Loaded {len(pairs)} OD pairs from Porto Taxi")
    return pairs


def cluster_od_pairs(od_pairs, n_clusters=20, seed=42):
    from sklearn.cluster import KMeans
    od_flat = np.array([[s[0], s[1], e[0], e[1]] for s, e in od_pairs])
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(od_flat)
    clusters = defaultdict(list)
    for i, cl in enumerate(cluster_labels):
        clusters[cl].append(i)
    return clusters, km.cluster_centers_


def generate_trajectory_from_od(start, end, seq_len=32, rng=None, route_template=None):
    if rng is None:
        rng = np.random.RandomState(42)
    t = np.linspace(0, 1, seq_len).reshape(-1, 1)
    path = start + t * (end - start)

    if route_template is not None:
        path += route_template * rng.uniform(0.8, 1.2)
    else:
        n_waypoints = rng.randint(1, 3)
        for _ in range(n_waypoints):
            wp_t = rng.uniform(0.2, 0.8)
            wp_spread = rng.uniform(0.05, 0.15)
            direction = end - start
            perp = np.array([-direction[1], direction[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            offset_mag = rng.normal(0, 0.08) * np.linalg.norm(direction)
            weights = np.exp(-0.5 * ((np.linspace(0, 1, seq_len) - wp_t) / wp_spread) ** 2)
            weights = weights.reshape(-1, 1)
            path += weights * perp * offset_mag

    gps_noise = rng.normal(0, 0.0001, path.shape)
    path += gps_noise

    return path


def generate_route_template(center_start, center_end, seq_len=32, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    direction = center_end - center_start
    perp = np.array([-direction[1], direction[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-8)
    t = np.linspace(0, 1, seq_len)
    template = np.zeros((seq_len, 2))
    n_bends = rng.randint(1, 4)
    for _ in range(n_bends):
        bend_t = rng.uniform(0.15, 0.85)
        bend_spread = rng.uniform(0.08, 0.2)
        bend_mag = rng.normal(0, 0.1) * np.linalg.norm(direction)
        weights = np.exp(-0.5 * ((t - bend_t) / bend_spread) ** 2)
        template += np.outer(weights, perp * bend_mag)
    return template


def inject_anomalies_porto(trajectories, contamination=0.1, seed=42):
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

            if anom_type == 0:  # DETOUR
                s = int(0.3 * seq_len)
                e = int(0.7 * seq_len)
                direction = traj[-1] - traj[0]
                perp = np.array([-direction[1], direction[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-8)
                offset_mag = rng.uniform(0.15, 0.35) * od_dist
                sign = rng.choice([-1, 1])
                for k in range(s, e):
                    frac = (k - s) / (e - s + 1e-8)
                    traj[k] += sign * perp * offset_mag * np.sin(np.pi * frac)

            elif anom_type == 1:  # LOOP
                center = seq_len // 2
                radius = rng.uniform(0.08, 0.2) * od_dist
                n_loop = min(10, seq_len // 3)
                s = max(0, center - n_loop // 2)
                for k in range(n_loop):
                    idx_k = s + k
                    if idx_k < seq_len:
                        angle = 2 * np.pi * k / n_loop
                        traj[idx_k] += np.array([radius * np.cos(angle),
                                                  radius * np.sin(angle)])

            elif anom_type == 2:  # SPEED ANOMALY
                start_pt = int(rng.uniform(0.2, 0.5) * seq_len)
                n_slow = min(8, seq_len // 4)
                center_pt = traj[start_pt].copy()
                for k in range(n_slow):
                    if start_pt + k < seq_len:
                        traj[start_pt + k] = center_pt + rng.normal(0, od_dist * 0.003, 2)

            elif anom_type == 3:  # ROUTE DEVIATION
                dev_start = int(rng.uniform(0.4, 0.6) * seq_len)
                angle_dev = rng.uniform(40, 80) * np.pi / 180 * rng.choice([-1, 1])
                for k in range(dev_start, seq_len):
                    if k > 0:
                        diff = traj[k] - traj[k-1]
                        step_len = np.linalg.norm(diff)
                        if step_len < 1e-10:
                            diff = (traj[-1] - traj[0]) / seq_len
                        cos_a, sin_a = np.cos(angle_dev), np.sin(angle_dev)
                        rotated = np.array([cos_a * diff[0] - sin_a * diff[1],
                                            sin_a * diff[0] + cos_a * diff[1]])
                        traj[k] = traj[k-1] + rotated * 1.3
                    angle_dev *= 0.92

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


class EWGBTADv3Porto:
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
            raw_scores = self.detectors[name].score(views_data[name])
            smin, smax = raw_scores.min(), raw_scores.max()
            if smax - smin > 1e-8:
                raw_scores = (raw_scores - smin) / (smax - smin)
            view_scores[name] = raw_scores

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


def run_single_porto(od_pairs, n_normal=5000, contamination=0.1, seq_len=32, seed=42,
                     clusters=None, cluster_centers=None):
    rng = np.random.RandomState(seed)

    if clusters is None:
        n_clusters = min(20, len(od_pairs) // 50)
        clusters, cluster_centers = cluster_od_pairs(od_pairs, n_clusters=n_clusters, seed=seed)

    print(f"    Generating {n_normal} trajectories from {len(clusters)} route clusters...")
    normal_trajs = []
    contexts = []
    route_templates = {}

    for cl_id in clusters:
        if cl_id not in route_templates:
            c = cluster_centers[cl_id]
            cs, ce = np.array([c[0], c[1]]), np.array([c[2], c[3]])
            route_templates[cl_id] = generate_route_template(cs, ce, seq_len, rng)

    per_cluster = max(1, n_normal // len(clusters))
    for cl_id, indices in clusters.items():
        n_from_cluster = min(per_cluster, len(indices))
        chosen = rng.choice(indices, n_from_cluster, replace=len(indices) < n_from_cluster)
        template = route_templates[cl_id]
        for idx in chosen:
            start, end = od_pairs[idx]
            traj = generate_trajectory_from_od(start, end, seq_len=seq_len, rng=rng,
                                               route_template=template)
            normal_trajs.append(traj)
            contexts.append(cl_id)
            if len(normal_trajs) >= n_normal:
                break
        if len(normal_trajs) >= n_normal:
            break

    normal_trajs = normal_trajs[:n_normal]

    print(f"    Injecting anomalies (contamination={contamination})...")
    trajs, labels, anom_types = inject_anomalies_porto(normal_trajs, contamination=contamination, seed=seed)

    print(f"    Extracting features...")
    spatial, kinematic, entropy_feat, path_feat, all_features = extract_all_features_v2(trajs)
    spatial_path = np.hstack([spatial, path_feat])

    results = {}

    # EWGB-TAD v3 (4-view)
    t0 = time.time()
    det = EWGBTADv3Porto(min_samples=8, purity_threshold=0.85, n_shape_dims=10)
    det.fit(spatial_path, kinematic, entropy_feat, trajs)
    scores = det.score(spatial_path, kinematic, entropy_feat, trajs)
    elapsed = time.time() - t0
    metrics = evaluate_detector(labels, scores, contamination)
    metrics['Runtime'] = elapsed
    results['EWGB-TAD v3 (4-view)'] = metrics

    # Baselines
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

    # Per-type recall for EWGB-TAD
    threshold = np.percentile(scores, (1 - contamination) * 100)
    per_type = {}
    type_names = {0: 'detour', 1: 'loop', 2: 'speed', 3: 'route_deviation'}
    for atype in range(4):
        mask = anom_types == atype
        if mask.sum() > 0:
            detected = (scores[mask] > threshold).sum()
            per_type[type_names[atype]] = {'recall': detected / mask.sum()}

    return results, per_type


def run_full_porto(csv_path='../data/porto_trajectories_all.csv',
                   n_normal=5000, contamination_ratios=[0.05, 0.10, 0.15],
                   seeds=[42, 123, 456, 789, 1024],
                   output_dir='../results'):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Porto Taxi OD pairs...")
    od_pairs = load_porto_od_pairs(csv_path, max_pairs=30000, seed=42)
    if len(od_pairs) < 100:
        print("ERROR: Not enough valid OD pairs. Check data file.")
        return None

    n_clusters = min(20, len(od_pairs) // 50)
    print(f"Clustering OD pairs into {n_clusters} route groups...")
    clusters, cluster_centers = cluster_od_pairs(od_pairs, n_clusters=n_clusters, seed=42)
    for cl_id, indices in sorted(clusters.items()):
        if len(indices) > 50:
            c = cluster_centers[cl_id]
            print(f"  Route {cl_id}: {len(indices)} trips, center: ({c[0]:.4f},{c[1]:.4f})->({c[2]:.4f},{c[3]:.4f})")

    all_results = {}

    for cont in contamination_ratios:
        print(f"\n{'='*60}")
        print(f"Porto Taxi — Contamination ratio: {cont*100:.0f}%")
        print(f"{'='*60}")

        seed_results = defaultdict(lambda: defaultdict(list))
        all_per_type = defaultdict(lambda: defaultdict(list))
        raw_aucs = defaultdict(list)

        for seed in seeds:
            print(f"  Seed {seed}...", end=' ')
            t0 = time.time()
            results, per_type = run_single_porto(
                od_pairs, n_normal=n_normal, contamination=cont, seed=seed,
                clusters=clusters, cluster_centers=cluster_centers
            )
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

        # Significance tests
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

    output_path = os.path.join(output_dir, 'experiment_results_porto.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return all_results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_full_porto(n_normal=2000, contamination_ratios=[0.10],
                      seeds=[42, 123, 456], output_dir='../results')
    else:
        run_full_porto(n_normal=5000, output_dir='../results')
