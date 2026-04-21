"""
Experiments on grid-network (semi-real) dataset.
Tests EWGB-TAD on trajectories following a road-network-like structure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy import stats

from ewgb_tad.data_generator_real import generate_grid_network_trajectories
from ewgb_tad.feature_extraction_v2 import extract_all_features_v2
from ewgb_tad.detector import EWGBTAD as EWGBTADv3, evaluate_detector
from ewgb_tad.baselines_v2 import get_all_baselines_v2


def run_grid_experiment(n_normal=5000, contamination=0.10, seed=42):
    n_anomaly_per_type = max(1, int(n_normal * contamination / (4 * (1 - contamination))))

    trajs, labels, anom_types, _ = generate_grid_network_trajectories(
        n_normal=n_normal, n_anomaly_per_type=n_anomaly_per_type, seed=seed
    )

    spatial, kinematic, entropy_feat, path_feat, all_features = extract_all_features_v2(trajs)
    spatial_path = np.hstack([spatial, path_feat])

    results = {}

    # EWGB-TAD v3
    t0 = time.time()
    det = EWGBTADv3(min_samples=8, purity_threshold=0.85, n_shape_dims=10)
    det.fit(spatial_path, kinematic, entropy_feat, trajs)
    scores = det.score(spatial_path, kinematic, entropy_feat, trajs)
    elapsed = time.time() - t0
    metrics = evaluate_detector(labels, scores, contamination)
    metrics['Runtime'] = elapsed
    results['EWGB-TAD'] = metrics

    # Key baselines
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

    # Per-type recall
    threshold = np.percentile(scores, (1 - contamination) * 100)
    per_type = {}
    type_names = {0: 'detour', 1: 'loop', 2: 'speed', 3: 'route_deviation'}
    for atype in range(4):
        mask = anom_types == atype
        if mask.sum() > 0:
            detected = (scores[mask] > threshold).sum()
            per_type[type_names[atype]] = {'recall': detected / mask.sum()}

    return results, per_type


def run_full_grid(seeds=[42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144],
                  n_normal=5000, contamination=0.10, output_dir='../results'):
    os.makedirs(output_dir, exist_ok=True)

    seed_results = defaultdict(lambda: defaultdict(list))
    all_per_type = defaultdict(lambda: defaultdict(list))
    raw_aucs = defaultdict(list)

    for seed in seeds:
        print(f"  Seed {seed}...", end=' ', flush=True)
        t0 = time.time()
        results, per_type = run_grid_experiment(n_normal=n_normal, contamination=contamination, seed=seed)
        print(f"done ({time.time()-t0:.1f}s)")

        for method, metrics in results.items():
            for metric, value in metrics.items():
                seed_results[method][metric].append(value)
            raw_aucs[method].append(metrics['AUC'])

        for atype, s in per_type.items():
            all_per_type[atype]['recall'].append(s['recall'])

    # Aggregate
    aggregated = {}
    for method in seed_results:
        agg = {}
        for metric in seed_results[method]:
            values = seed_results[method][metric]
            agg[metric] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
        aggregated[method] = agg

    # Print
    print(f"\n{'Method':<25} {'AUC':>14} {'AUPRC':>14} {'F1':>14} {'Time':>8}")
    print("-" * 80)
    for method in sorted(aggregated.keys()):
        m = aggregated[method]
        auc_str = f"{m['AUC']['mean']:.4f}±{m['AUC']['std']:.4f}"
        auprc_str = f"{m['AUPRC']['mean']:.4f}±{m['AUPRC']['std']:.4f}"
        f1_str = f"{m['F1']['mean']:.4f}±{m['F1']['std']:.4f}"
        time_str = f"{m['Runtime']['mean']:.2f}s"
        print(f"  {method:<23} {auc_str:>14} {auprc_str:>14} {f1_str:>14} {time_str:>8}")

    # Per-type
    print(f"\n  Per-anomaly-type recall (EWGB-TAD):")
    for atype in ['detour', 'loop', 'speed', 'route_deviation']:
        if atype in all_per_type:
            vals = all_per_type[atype]['recall']
            print(f"    {atype:<20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Wilcoxon
    print(f"\n  Wilcoxon tests (EWGB-TAD vs baselines):")
    our_aucs = raw_aucs['EWGB-TAD']
    for method in sorted(raw_aucs.keys()):
        if method == 'EWGB-TAD':
            continue
        base_aucs = raw_aucs[method]
        if len(our_aucs) >= 5:
            try:
                _, p = stats.wilcoxon(our_aucs, base_aucs)
                delta = np.mean(our_aucs) - np.mean(base_aucs)
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
                print(f"    vs {method:<20}: Δ={delta:+.4f}, p={p:.4f} {sig}")
            except Exception:
                print(f"    vs {method:<20}: n/a")

    # Save
    output = {'grid_network_10pct': aggregated}
    output_path = os.path.join(output_dir, 'experiment_results_grid.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return aggregated


if __name__ == '__main__':
    print("=" * 60)
    print("Grid-Network (Semi-Real) Dataset Evaluation")
    print("=" * 60)
    run_full_grid()
