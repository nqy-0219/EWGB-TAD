"""Run cross-dataset ablation experiments for EWGB-TAD.

This script evaluates core mechanism variants on the four manuscript datasets under
the main 10% contamination setting:

1. EWGB-TAD with within-ball entropy weighting.
2. EWGB-TAD with global entropy weighting.
3. KMeans-Prototype fixed-prototype baseline.
4. EWGB-TAD with simple average view fusion.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines_v2 import ClusterPrototypeDetector
from data_generator import generate_synthetic_trajectories, resample_trajectory
from data_generator_real import generate_grid_network_trajectories
from geolife_data import cluster_geolife_trajectories, inject_anomalies_geolife, segment_trajectory
from porto_data import (
    cluster_od_pairs,
    generate_route_template,
    generate_trajectory_from_od,
    inject_anomalies_porto,
    load_porto_od_pairs,
)
from feature_extraction_v2 import extract_all_features_v2
from ewgb_tad_current import ThreeViewEWGBDetector
from evaluation_protocol import evaluate_oracle_top_k


RESULTS_DIR = ROOT / "results"
PAPER_DIR = ROOT / "paper_work"
def _locate_data_file(filename: str, *relative_candidates: str) -> Path:
    """Locate raw data in the source package or the workspace data directory."""
    candidates = [ROOT / filename]
    workspace_root = ROOT.parent.parent
    candidates.extend(workspace_root / candidate for candidate in relative_candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


PORTO_CSV = _locate_data_file(
    "porto_trajectories_all.csv",
    "data/porto_trajectories_all.csv",
    "data/eight_datasets_for_EWGB_TAD_20260603 (2)/eight_datasets_for_EWGB_TAD_20260603/03_Porto/porto_trajectories_all.csv",
)
GEOLIFE_ZIP = _locate_data_file(
    "geolife.zip",
    "data/geolife.zip",
    "data/eight_datasets_for_EWGB_TAD_20260603 (2)/eight_datasets_for_EWGB_TAD_20260603/04_GeoLife/geolife.zip",
)

CONTAMINATION = 0.10
SYN_GRID_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
PORTO_GEO_SEEDS = [42, 123, 456, 789, 1024]


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    contamination: float | None = None,
) -> dict[str, float]:
    """Apply the fixed oracle-top-k F1 rule.

    ``contamination`` remains as a compatibility keyword for older auxiliary
    scripts. It is intentionally ignored because the realized positive count
    is taken from ``labels`` after scoring.
    """
    del contamination
    return evaluate_oracle_top_k(labels, scores)


def _shape_features(trajectories: list[np.ndarray], n_dims: int = 10) -> np.ndarray:
    flat = np.array([traj.flatten() for traj in trajectories])
    pca = PCA(n_components=n_dims)
    return pca.fit_transform(flat)


def _balanced_cluster_samples(
    clusters: dict[int, list[int]],
    n_samples: int,
    rng: np.random.RandomState,
) -> list[tuple[int, int]]:
    """Draw an exact, seed-deterministic sample balanced over non-empty clusters."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    cluster_ids = sorted(cluster_id for cluster_id, indices in clusters.items() if indices)
    if not cluster_ids:
        raise ValueError("clusters must contain at least one non-empty cluster")

    base, remainder = divmod(n_samples, len(cluster_ids))
    sampled: list[tuple[int, int]] = []
    for position, cluster_id in enumerate(cluster_ids):
        count = base + (1 if position < remainder else 0)
        indices = np.asarray(clusters[cluster_id], dtype=int)
        chosen = rng.choice(indices, size=count, replace=len(indices) < count)
        sampled.extend((cluster_id, int(index)) for index in chosen)
    rng.shuffle(sampled)
    return sampled


def run_ablation_variants(trajs: list[np.ndarray], labels: np.ndarray) -> dict[str, dict[str, float]]:
    spatial, kinematic, _, path_feat, _ = extract_all_features_v2(trajs)
    spatial_path = np.hstack([spatial, path_feat])
    shape_feat = _shape_features(trajs, n_dims=10)
    concat_features = np.hstack([spatial_path, kinematic, shape_feat])

    results: dict[str, dict[str, float]] = {}

    def run_detector(name: str, view_fusion: str = "entropy", use_local_entropy: bool = True) -> None:
        t0 = time.perf_counter()
        detector = ThreeViewEWGBDetector(
            min_samples=8,
            purity_threshold=0.85,
            n_shape_dims=10,
            view_fusion=view_fusion,
            use_local_entropy=use_local_entropy,
        )
        detector.fit(spatial_path, kinematic, trajs)
        scores = detector.score(spatial_path, kinematic, trajs)
        metrics = evaluate(labels, scores)
        metrics["Runtime"] = float(time.perf_counter() - t0)
        results[name] = metrics

    run_detector("EWGB-TAD", use_local_entropy=True)
    run_detector("Global entropy weighting", use_local_entropy=False)

    t0 = time.perf_counter()
    kmeans_proto = ClusterPrototypeDetector(n_clusters=20)
    scores = kmeans_proto.fit_score(concat_features)
    metrics = evaluate(labels, scores)
    metrics["Runtime"] = float(time.perf_counter() - t0)
    results["KMeans-Prototype"] = metrics

    run_detector("Average fusion", view_fusion="equal", use_local_entropy=True)

    return results


def synthetic_dataset(seed: int, n_normal: int = 5000) -> tuple[list[np.ndarray], np.ndarray]:
    n_anomaly_per_type = max(1, int(n_normal * CONTAMINATION / (4 * (1 - CONTAMINATION))))
    trajs, labels, _, _ = generate_synthetic_trajectories(
        n_normal=n_normal,
        n_anomaly_per_type=n_anomaly_per_type,
        seed=seed,
    )
    return trajs, labels


def grid_dataset(seed: int, n_normal: int = 5000) -> tuple[list[np.ndarray], np.ndarray]:
    n_anomaly_per_type = max(1, int(n_normal * CONTAMINATION / (4 * (1 - CONTAMINATION))))
    trajs, labels, _, _ = generate_grid_network_trajectories(
        n_normal=n_normal,
        n_anomaly_per_type=n_anomaly_per_type,
        seed=seed,
    )
    return trajs, labels


def prepare_porto_context() -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[int, list[int]], np.ndarray]:
    od_pairs = load_porto_od_pairs(str(PORTO_CSV), max_pairs=30000, seed=42)
    clusters, centers = cluster_od_pairs(od_pairs, n_clusters=min(20, len(od_pairs) // 50), seed=42)
    return od_pairs, clusters, centers


def porto_dataset(
    seed: int,
    context: tuple[list[tuple[np.ndarray, np.ndarray]], dict[int, list[int]], np.ndarray],
    n_normal: int = 5000,
    seq_len: int = 32,
) -> tuple[list[np.ndarray], np.ndarray]:
    od_pairs, clusters, cluster_centers = context
    rng = np.random.RandomState(seed)
    normal_trajs: list[np.ndarray] = []
    route_templates: dict[int, np.ndarray] = {}

    for cl_id in clusters:
        c = cluster_centers[cl_id]
        route_templates[cl_id] = generate_route_template(np.array([c[0], c[1]]), np.array([c[2], c[3]]), seq_len, rng)

    sampled_pairs = _balanced_cluster_samples(clusters, n_normal, rng)
    for cl_id, idx in sampled_pairs:
        start, end = od_pairs[idx]
        normal_trajs.append(
            generate_trajectory_from_od(
                start,
                end,
                seq_len=seq_len,
                rng=rng,
                route_template=route_templates[cl_id],
            )
        )

    trajs, labels, _ = inject_anomalies_porto(normal_trajs, contamination=CONTAMINATION, seed=seed)
    return trajs, labels


def load_geolife_from_zip(max_trajs: int = 8000, seq_len: int = 32, seed: int = 42) -> list[np.ndarray]:
    cache_path = RESULTS_DIR / "geolife_base_trajs_ablation.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        arr = data["trajectories"]
        return [arr[i] for i in range(arr.shape[0])]

    rng = np.random.RandomState(seed)
    bbox = (116.1, 39.75, 116.65, 40.15)
    lon_min, lat_min, lon_max, lat_max = bbox
    all_trajs: list[np.ndarray] = []

    with zipfile.ZipFile(GEOLIFE_ZIP) as zf:
        plt_names = [n for n in zf.namelist() if n.endswith(".plt")]
        for name in plt_names:
            points = []
            try:
                with zf.open(name) as fh:
                    for i, raw in enumerate(fh):
                        if i < 6:
                            continue
                        parts = raw.decode("utf-8", errors="ignore").strip().split(",")
                        if len(parts) < 2:
                            continue
                        lat, lon = float(parts[0]), float(parts[1])
                        if lat_min < lat < lat_max and lon_min < lon < lon_max:
                            points.append([lon, lat])
            except (ValueError, IndexError, UnicodeDecodeError):
                continue

            if len(points) < 20:
                continue

            for seg in segment_trajectory(np.array(points), max_gap_meters=300, min_segment_len=20):
                if 20 <= len(seg) <= 2000 and np.linalg.norm(seg[-1] - seg[0]) >= 0.005:
                    all_trajs.append(resample_trajectory(seg, seq_len))
                    if len(all_trajs) >= max_trajs * 2:
                        break
            if len(all_trajs) >= max_trajs * 2:
                break

    rng.shuffle(all_trajs)
    all_trajs = all_trajs[:max_trajs]
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez_compressed(cache_path, trajectories=np.array(all_trajs))
    return all_trajs


def prepare_geolife_context() -> tuple[list[np.ndarray], dict[int, list[int]]]:
    base_trajs = load_geolife_from_zip(max_trajs=8000, seq_len=32, seed=42)
    n_clusters = min(15, len(base_trajs) // 100)
    clusters, _ = cluster_geolife_trajectories(base_trajs, n_clusters=n_clusters, seed=42)
    return base_trajs, clusters


def geolife_dataset(
    seed: int,
    context: tuple[list[np.ndarray], dict[int, list[int]]],
    n_normal: int = 3000,
) -> tuple[list[np.ndarray], np.ndarray]:
    base_trajs, clusters = context
    rng = np.random.RandomState(seed)
    normal_trajs: list[np.ndarray] = []
    sampled_pairs = _balanced_cluster_samples(clusters, n_normal, rng)
    normal_trajs.extend(base_trajs[idx].copy() for _, idx in sampled_pairs)
    trajs, labels, _ = inject_anomalies_geolife(normal_trajs, CONTAMINATION, seed)
    return trajs, labels


def aggregate(seed_results: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, dict[str, float]]]:
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for method, metric_values in seed_results.items():
        aggregated[method] = {}
        for metric, values in metric_values.items():
            aggregated[method][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return aggregated


def run_dataset(name: str, seeds: list[int], dataset_fn) -> dict[str, dict[str, dict[str, float]]]:
    print(f"\n{name}")
    print("-" * len(name))
    seed_results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        print(f"  seed={seed}", end=" ", flush=True)
        t0 = time.perf_counter()
        trajs, labels = dataset_fn(seed)
        results = run_ablation_variants(trajs, labels)
        for method, metrics in results.items():
            for metric, value in metrics.items():
                seed_results[method][metric].append(value)
        print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)
    aggregated = aggregate(seed_results)
    for method in ["EWGB-TAD", "Global entropy weighting", "KMeans-Prototype", "Average fusion"]:
        auc = aggregated[method]["AUC"]
        print(f"  {method:<23} AUC={auc['mean']:.4f}±{auc['std']:.4f}")
    return aggregated


def fmt_cell(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f}$\\pm${metric['std']:.3f}"


def write_table(all_results: dict[str, dict[str, dict[str, dict[str, float]]]]) -> None:
    datasets = ["Synthetic", "Grid-Network", "Porto-derived", "GeoLife"]
    dataset_labels = {
        "Synthetic": "Synthetic",
        "Grid-Network": "Grid-Network",
        "Porto-derived": "\\shortstack[l]{Porto-derived\\\\Taxi}",
        "GeoLife": "GeoLife",
    }
    methods = ["EWGB-TAD", "Global entropy weighting", "KMeans-Prototype", "Average fusion"]
    method_labels = {
        "EWGB-TAD": "\\textbf{EWGB-TAD}",
        "Global entropy weighting": "Global entropy weighting",
        "KMeans-Prototype": "KMeans-Prototype",
        "Average fusion": "Average fusion",
    }
    best = {
        dataset: {
            metric: max(all_results[dataset][method][metric]["mean"] for method in methods)
            for metric in ["AUC", "AUPRC", "F1"]
        }
        for dataset in datasets
    }

    def maybe_bold(dataset: str, method: str, metric: str) -> str:
        cell = fmt_cell(all_results[dataset][method][metric])
        if np.isclose(all_results[dataset][method][metric]["mean"], best[dataset][metric]):
            return f"\\textbf{{{cell}}}"
        return cell

    table_path = PAPER_DIR / "cross_dataset_ablation_table.tex"
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Cross-dataset ablation study results.}",
        "\\label{tab:cross_dataset_ablation}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\tablebodyfont",
        "\\renewcommand{\\arraystretch}{1.04}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcccc@{}}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Variant} & \\textbf{AUC} & \\textbf{AUPRC} & \\textbf{F1} & \\textbf{$\\Delta$AUC} \\\\",
        "\\midrule",
    ]
    for dataset_index, dataset in enumerate(datasets):
        base_auc = all_results[dataset]["EWGB-TAD"]["AUC"]["mean"]
        for method_index, method in enumerate(methods):
            dataset_cell = f"\\multirow{{4}}{{*}}{{{dataset_labels[dataset]}}}" if method_index == 0 else ""
            delta = all_results[dataset][method]["AUC"]["mean"] - base_auc
            delta_cell = "0.000" if method == "EWGB-TAD" else f"${0.0 if abs(delta) < 5e-4 else delta:+.3f}$"
            lines.append(
                f"{dataset_cell} & {method_labels[method]} & "
                f"{maybe_bold(dataset, method, 'AUC')} & "
                f"{maybe_bold(dataset, method, 'AUPRC')} & "
                f"{maybe_bold(dataset, method, 'F1')} & "
                f"{delta_cell} \\\\"
            )
        if dataset_index != len(datasets) - 1:
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular*}",
        "\\endgroup",
        "\\end{table}",
        "",
    ])
    table_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    PAPER_DIR.mkdir(exist_ok=True)

    porto_context = prepare_porto_context()
    geolife_context = prepare_geolife_context()

    all_results = {
        "Synthetic": run_dataset("Synthetic", SYN_GRID_SEEDS, synthetic_dataset),
        "Grid-Network": run_dataset("Grid-Network", SYN_GRID_SEEDS, grid_dataset),
        "Porto-derived": run_dataset("Porto-derived", PORTO_GEO_SEEDS, lambda seed: porto_dataset(seed, porto_context)),
        "GeoLife": run_dataset("GeoLife", PORTO_GEO_SEEDS, lambda seed: geolife_dataset(seed, geolife_context)),
    }

    out_json = RESULTS_DIR / "cross_dataset_ablation_results.json"
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    write_table(all_results)
    print(f"\nSaved {out_json}")
    print(f"Saved {PAPER_DIR / 'cross_dataset_ablation_table.tex'}")


if __name__ == "__main__":
    main()
