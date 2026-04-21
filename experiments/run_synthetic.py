"""
EWGB-TAD v3: Enhanced version addressing all reviewer concerns.

Key changes from v2:
1. Add trajectory-shape features (flattened resampled coordinates as additional view)
2. Better fusion using all 4 views
3. Report full metrics including AUPRC
4. Statistical significance tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

from ewgb_tad.data_generator import generate_synthetic_trajectories
from ewgb_tad.feature_extraction_v2 import extract_all_features_v2
from ewgb_tad.granular_ball import EWGBDetector, build_granular_balls, compute_feature_entropy_weights
from ewgb_tad.baselines_v2 import get_all_baselines_v2
from ewgb_tad.detector import EWGBTAD as EWGBTADv3, evaluate_detector


def run_single_v3(n_normal=5000, contamination=0.1, seed=42):
    n_anomaly_per_type = max(1, int(n_normal * contamination / (4 * (1 - contamination))))

    trajs, labels, anom_types, _ = generate_synthetic_trajectories(
        n_normal=n_normal, n_anomaly_per_type=n_anomaly_per_type, seed=seed
    )

    spatial, kinematic, entropy_feat, path_feat, all_features = extract_all_features_v2(trajs)
    spatial_path = np.hstack([spatial, path_feat])

    results = {}

    # EWGB-TAD v3 (4-view with trajectory shape)
    t0 = time.time()
    det_v3 = EWGBTADv3(min_samples=8, purity_threshold=0.85, n_shape_dims=10)
    det_v3.fit(spatial_path, kinematic, entropy_feat, trajs)
    scores_v3 = det_v3.score(spatial_path, kinematic, entropy_feat, trajs)
    elapsed = time.time() - t0
    metrics = evaluate_detector(labels, scores_v3, contamination)
    metrics['Runtime'] = elapsed
    results['EWGB-TAD v3 (4-view)'] = metrics

    # EWGB-TAD v3 without shape view (ablation)
    t0 = time.time()
    det_no_shape = EWGBDetector(min_samples=8, purity_threshold=0.85, use_local_entropy=False)
    combined = np.hstack([spatial_path, kinematic, entropy_feat])
    det_no_shape.fit(combined)
    scores_no_shape = det_no_shape.score(combined)
    elapsed = time.time() - t0
    metrics = evaluate_detector(labels, scores_no_shape, contamination)
    metrics['Runtime'] = elapsed
    results['EWGB-TAD (no shape view)'] = metrics

    # All baselines
    baselines = get_all_baselines_v2(contamination=contamination, seed=seed)
    for name, det in baselines.items():
        t0 = time.time()
        if name in ('DTW-KNN', 'SegmentOD'):
            scores_b = det.fit_score(all_features, trajs)
        else:
            scores_b = det.fit_score(all_features)
        elapsed = time.time() - t0
        metrics = evaluate_detector(labels, scores_b, contamination)
        metrics['Runtime'] = elapsed
        results[name] = metrics

    # Per-type for v3
    threshold = np.percentile(scores_v3, (1 - contamination) * 100)
    per_type = {}
    type_names = {0: 'detour', 1: 'loop', 2: 'speed', 3: 'route_deviation'}
    for atype in range(4):
        mask = anom_types == atype
        if mask.sum() > 0:
            detected = (scores_v3[mask] > threshold).sum()
            per_type[type_names[atype]] = {'recall': detected / mask.sum()}

    return results, per_type


def run_full_v3(n_normal=5000, contamination_ratios=[0.05, 0.10, 0.15],
                seeds=[42, 123, 456, 789, 1024], output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for cont in contamination_ratios:
        print(f"\n{'='*60}")
        print(f"Contamination ratio: {cont*100:.0f}%")
        print(f"{'='*60}")

        seed_results = defaultdict(lambda: defaultdict(list))
        all_per_type = defaultdict(lambda: defaultdict(list))
        raw_aucs = defaultdict(list)

        for seed in seeds:
            print(f"  Seed {seed}...", end=' ')
            t0 = time.time()
            results, per_type = run_single_v3(n_normal=n_normal, contamination=cont, seed=seed)
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

        # Print table
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

        # Per-type
        print(f"\n  Per-anomaly-type recall (EWGB-TAD v3):")
        for atype in ['detour', 'loop', 'speed', 'route_deviation']:
            if atype in all_per_type:
                vals = all_per_type[atype]['recall']
                print(f"    {atype:<20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        # Significance: v3 vs each
        print(f"\n  Wilcoxon tests (EWGB-TAD v3 vs baselines):")
        our_aucs = raw_aucs['EWGB-TAD v3 (4-view)']
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

    output_path = os.path.join(output_dir, 'experiment_results_v3.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return all_results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_full_v3(n_normal=2000, contamination_ratios=[0.10], seeds=[42, 123, 456], output_dir='../results')
    else:
        run_full_v3(n_normal=5000, output_dir='../results')
