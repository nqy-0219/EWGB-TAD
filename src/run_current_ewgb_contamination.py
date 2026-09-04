"""Run current three-view EWGB-TAD contamination-ratio robustness experiments.

This script supports Table 6 in the manuscript. It evaluates only EWGB-TAD
under the current three-view setting: spatial-path, kinematic, and PCA shape.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_cross_dataset_ablation as cross
from feature_extraction_v2 import extract_all_features_v2
from ewgb_tad_current import ThreeViewEWGBDetector
from evaluation_protocol import evaluate_oracle_top_k

RESULTS_DIR = Path(os.environ.get("EWGB_AUX_RESULTS_DIR", str(ROOT / "results")))
PAPER_DIR = Path(os.environ.get("EWGB_AUX_PAPER_DIR", str(ROOT / "paper_work")))

CONTAMINATIONS = [0.05, 0.10, 0.15, 0.20]
SYN_GRID_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
PORTO_GEO_SEEDS = [42, 123, 456, 789, 1024]


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return evaluate_oracle_top_k(labels, scores)


def run_ewgb(trajectories: list[np.ndarray], labels: np.ndarray, contamination: float) -> dict[str, float]:
    # Match Phase 4 cache construction: extract from the generated precision,
    # then cast each detector input to float32.
    spatial, kinematic, _, path_feat, _ = extract_all_features_v2(trajectories)
    trajectories_array = np.asarray(trajectories, dtype=np.float32)
    spatial_path = np.hstack([spatial, path_feat]).astype(np.float32, copy=False)
    kinematic = np.asarray(kinematic, dtype=np.float32)

    t0 = time.perf_counter()
    detector = ThreeViewEWGBDetector(
        min_samples=8,
        purity_threshold=0.85,
        n_shape_dims=10,
        view_fusion="entropy",
        use_local_entropy=True,
        fusion_score_bins=20,
        fusion_base_weight=0.1,
    )
    detector.fit(spatial_path, kinematic, trajectories_array)
    scores = detector.score(spatial_path, kinematic, trajectories_array)
    metrics = evaluate(labels, scores)
    metrics["Runtime"] = float(time.perf_counter() - t0)
    return metrics


def aggregate(seed_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    by_metric: dict[str, list[float]] = defaultdict(list)
    for metrics in seed_metrics:
        for metric, value in metrics.items():
            by_metric[metric].append(float(value))
    return {
        metric: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for metric, values in by_metric.items()
    }


def dataset_jobs() -> dict[str, dict]:
    porto_context = cross.prepare_porto_context()
    geolife_context = cross.prepare_geolife_context()
    return {
        "Synthetic": {
            "seeds": SYN_GRID_SEEDS,
            "dataset_fn": lambda seed: cross.synthetic_dataset(seed),
        },
        "Grid": {
            "seeds": SYN_GRID_SEEDS,
            "dataset_fn": lambda seed: cross.grid_dataset(seed),
        },
        "Porto": {
            "seeds": PORTO_GEO_SEEDS,
            "dataset_fn": lambda seed: cross.porto_dataset(seed, porto_context),
        },
        "GeoLife": {
            "seeds": PORTO_GEO_SEEDS,
            "dataset_fn": lambda seed: cross.geolife_dataset(seed, geolife_context),
        },
    }


def run_dataset_ratio(dataset: str, contamination: float) -> tuple[str, str, dict[str, dict]]:
    """Run one dataset/contamination block in an isolated worker process."""
    cross.CONTAMINATION = contamination
    if dataset == "Synthetic":
        seeds = SYN_GRID_SEEDS
        dataset_fn = cross.synthetic_dataset
    elif dataset == "Grid":
        seeds = SYN_GRID_SEEDS
        dataset_fn = cross.grid_dataset
    elif dataset == "Porto":
        seeds = PORTO_GEO_SEEDS
        context = cross.prepare_porto_context()
        dataset_fn = lambda seed: cross.porto_dataset(seed, context)
    elif dataset == "GeoLife":
        seeds = PORTO_GEO_SEEDS
        context = cross.prepare_geolife_context()
        dataset_fn = lambda seed: cross.geolife_dataset(seed, context)
    else:
        raise ValueError(dataset)

    records = {}
    for seed in seeds:
        trajectories, labels = dataset_fn(seed)
        records[str(seed)] = run_ewgb(trajectories, labels, contamination)
    return dataset, f"{int(contamination * 100)}%", records


def fmt_metric(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f}$\\,\\pm\\,${metric['std']:.3f}"


def load_existing(path: Path) -> tuple[dict, dict]:
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", {}), data.get("raw", {})


def save_results(path: Path, summary: dict, raw: dict, contaminations: list[float]) -> None:
    payload = {
        "config": {
            "model": "EWGB-TAD current three-view",
            "views": ["spatial_path", "kinematic", "PCA shape"],
            "contaminations": contaminations,
            "f1_rule": "oracle_top_k_using_realized_injected_count",
            "synthetic_grid_seeds": SYN_GRID_SEEDS,
            "porto_geolife_seeds": PORTO_GEO_SEEDS,
        },
        "summary": summary,
        "raw": raw,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(summary: dict[str, dict[str, dict[str, dict[str, float]]]]) -> Path:
    out = RESULTS_DIR / "current_ewgb_contamination_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Dataset",
                "Contamination",
                "AUC_mean",
                "AUC_std",
                "AUPRC_mean",
                "AUPRC_std",
                "F1_mean",
                "F1_std",
                "Runtime_mean",
                "Runtime_std",
                "Seeds",
            ]
        )
        for dataset, ratios in summary.items():
            for ratio, metrics in ratios.items():
                writer.writerow(
                    [
                        dataset,
                        ratio,
                        metrics["AUC"]["mean"],
                        metrics["AUC"]["std"],
                        metrics["AUPRC"]["mean"],
                        metrics["AUPRC"]["std"],
                        metrics["F1"]["mean"],
                        metrics["F1"]["std"],
                        metrics["Runtime"]["mean"],
                        metrics["Runtime"]["std"],
                        metrics["n_seeds"],
                    ]
                )
    return out


def write_table(summary: dict[str, dict[str, dict[str, dict[str, float]]]]) -> Path:
    out = PAPER_DIR / "contamination_injected_table.tex"
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:contamination_injected}{Contamination-ratio robustness of EWGB-TAD. Values are AUC means $\\pm$ standard deviations over repeated seeds.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{5.0pt}",
        "\\tablebodyfont",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccc@{}}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{5\\%} & \\textbf{10\\%} & \\textbf{15\\%} & \\textbf{20\\%} & \\textbf{AUC range$\\downarrow$} \\\\",
        "\\midrule",
    ]
    table_contaminations = [0.05, 0.10, 0.15, 0.20]
    for dataset in ["Synthetic", "Grid", "Porto", "GeoLife"]:
        if dataset not in summary or any(f"{int(c * 100)}%" not in summary[dataset] for c in table_contaminations):
            continue
        aucs = [summary[dataset][f"{int(c * 100)}%"]["AUC"]["mean"] for c in table_contaminations]
        cells = [fmt_metric(summary[dataset][f"{int(c * 100)}%"]["AUC"]) for c in table_contaminations]
        auc_range = max(aucs) - min(aucs)
        lines.append(f"{dataset} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {auc_range:.3f} \\\\")
    lines.extend(
        [
            "\\botrule",
            "\\end{tabular*}",
            "\\endgroup",
            "\\end{center}",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contaminations",
        nargs="+",
        type=float,
        default=CONTAMINATIONS,
        help="Contamination ratios to run, e.g. --contaminations 0.20",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["Synthetic", "Grid", "Porto", "GeoLife"],
        default=["Synthetic", "Grid", "Porto", "GeoLife"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run existing seed-level results.",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    PAPER_DIR.mkdir(exist_ok=True)

    json_out = RESULTS_DIR / "current_ewgb_contamination_results.json"
    summary, raw = load_existing(json_out)
    if args.workers > 1:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        blocks = [
            (dataset, contamination)
            for contamination in args.contaminations
            for dataset in args.datasets
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_dataset_ratio, dataset, contamination): (dataset, contamination)
                for dataset, contamination in blocks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                dataset, ratio_key, records = future.result()
                raw.setdefault(dataset, {})[ratio_key] = records
                seed_metrics = list(records.values())
                summary.setdefault(dataset, {})[ratio_key] = aggregate(seed_metrics)
                summary[dataset][ratio_key]["n_seeds"] = len(seed_metrics)
                save_results(json_out, summary, raw, args.contaminations)
                print(
                    f"completed {completed}/{len(futures)} {dataset} {ratio_key}",
                    flush=True,
                )
        csv_out = write_csv(summary)
        table_out = write_table(summary)
        print(f"Saved {json_out}")
        print(f"Saved {csv_out}")
        print(f"Saved {table_out}")
        return

    jobs = {name: job for name, job in dataset_jobs().items() if name in set(args.datasets)}

    for contamination in args.contaminations:
        ratio_key = f"{int(contamination * 100)}%"
        cross.CONTAMINATION = contamination
        print(f"\nContamination={ratio_key}", flush=True)
        for dataset, job in jobs.items():
            print(f"  {dataset}", flush=True)
            raw.setdefault(dataset, {}).setdefault(ratio_key, {})
            seed_metrics = []
            for seed in job["seeds"]:
                seed_key = str(seed)
                if seed_key in raw[dataset][ratio_key] and not args.force:
                    metrics = raw[dataset][ratio_key][seed_key]
                    seed_metrics.append(metrics)
                    print(f"    seed={seed} existing; skip", flush=True)
                    continue
                print(f"    seed={seed}", end=" ", flush=True)
                trajectories, labels = job["dataset_fn"](seed)
                metrics = run_ewgb(trajectories, labels, contamination)
                raw[dataset][ratio_key][seed_key] = metrics
                seed_metrics.append(metrics)
                summary.setdefault(dataset, {})[ratio_key] = aggregate(seed_metrics)
                summary[dataset][ratio_key]["n_seeds"] = len(seed_metrics)
                save_results(json_out, summary, raw, args.contaminations)
                print(
                    f"AUC={metrics['AUC']:.4f}, AUPRC={metrics['AUPRC']:.4f}, "
                    f"F1={metrics['F1']:.4f}, time={metrics['Runtime']:.2f}s",
                    flush=True,
                )
            summary.setdefault(dataset, {})[ratio_key] = aggregate(seed_metrics)
            summary[dataset][ratio_key]["n_seeds"] = len(seed_metrics)
            save_results(json_out, summary, raw, args.contaminations)

    csv_out = write_csv(summary)
    table_out = write_table(summary)
    print(f"\nSaved {json_out}")
    print(f"Saved {csv_out}")
    print(f"Saved {table_out}")


if __name__ == "__main__":
    main()
