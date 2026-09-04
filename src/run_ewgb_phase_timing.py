"""Measure EWGB-TAD fit and score phases on the predefined main caches."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ewgb_tad_current import ThreeViewEWGBDetector
from phase4_common import (
    DATASET_SEEDS,
    PHASE4_ROOT,
    SUMMARY_ROOT,
    atomic_write_json,
    job_directory,
    load_cache,
    resolve_marker_artifact,
    set_reproducible,
)


def run_one(dataset: str, seed: int) -> dict:
    set_reproducible(seed, torch_threads=1)
    arrays, metadata = load_cache(dataset, seed)
    detector = ThreeViewEWGBDetector(
        min_samples=8,
        purity_threshold=0.85,
        n_shape_dims=10,
        view_fusion="entropy",
        use_local_entropy=True,
        fusion_score_bins=20,
        fusion_base_weight=0.1,
    )
    started = time.perf_counter()
    detector.fit(arrays["spatial_path"], arrays["kinematic"], arrays["trajectories"])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = detector.score(
        arrays["spatial_path"], arrays["kinematic"], arrays["trajectories"]
    )
    score_seconds = time.perf_counter() - started

    marker_path = job_directory(dataset, "EWGB-TAD", seed) / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    score_path = resolve_marker_artifact(marker_path, marker, "score")
    if score_path is None:
        raise FileNotFoundError(f"missing main scores: {dataset} seed={seed}")
    with np.load(score_path) as archive:
        main_scores = np.asarray(archive["scores"], dtype=float)
    scores = np.asarray(scores, dtype=float)
    stats = detector.get_stats()
    ball_counts = {
        name: int(stats["detectors"][name]["n_balls"])
        for name in detector.VIEW_NAMES
    }
    return {
        "dataset": dataset,
        "seed": seed,
        "n_samples": int(metadata["n_total"]),
        "fit_seconds": float(fit_seconds),
        "score_seconds": float(score_seconds),
        "fit_score_seconds": float(fit_seconds + score_seconds),
        "spatial_path_balls": ball_counts["spatial_path"],
        "kinematic_balls": ball_counts["kinematic"],
        "trajectory_shape_balls": ball_counts["trajectory_shape"],
        "total_balls": int(sum(ball_counts.values())),
        "score_exact_match": bool(np.array_equal(scores, main_scores)),
        "max_abs_score_difference": float(np.max(np.abs(scores - main_scores))),
        "labels_consumed_during_fit": False,
    }


def mean_std(rows: list[dict], field: str) -> tuple[float, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def write_table(aggregate: list[dict], table_dir: Path) -> None:
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:gb_complexity}{EWGB-TAD phase timing and terminal-ball counts on the 30 predefined dataset--seed caches. Fit includes PCA, granular-ball construction, and local entropy-weight estimation; score is post-fit inference.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc@{}}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{$N$} & \\textbf{Total balls} & \\textbf{Spatial} & \\textbf{Kinematic} & \\textbf{Shape} & \\textbf{Fit (s)} & \\textbf{Score (s)} \\\\ ",
        "\\midrule",
    ]
    for row in aggregate:
        lines.append(
            f"{row['dataset_label']} & {row['n_samples']:,} & "
            f"{row['total_balls_mean']:.1f} $\\pm$ {row['total_balls_std']:.1f} & "
            f"{row['spatial_path_balls_mean']:.1f} & {row['kinematic_balls_mean']:.1f} & "
            f"{row['trajectory_shape_balls_mean']:.1f} & "
            f"{row['fit_seconds_mean']:.2f} $\\pm$ {row['fit_seconds_std']:.2f} & "
            f"{row['score_seconds_mean']:.2f} $\\pm$ {row['score_seconds_std']:.2f} \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "gb_complexity_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=PHASE4_ROOT.parents[2]
        / "neurocomputing_submission"
        / "overleaf_neurocomputing"
        / "tables",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    jobs = [(dataset, seed) for dataset, seeds in DATASET_SEEDS.items() for seed in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, *job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 5 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (list(DATASET_SEEDS).index(row["dataset"]), row["seed"]))
    all_exact = all(row["score_exact_match"] for row in rows)

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (SUMMARY_ROOT / "ewgb_phase_timing_seed_level.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    labels = {
        "Synthetic": "Synthetic",
        "Grid-Network": "Grid",
        "Porto-derived": "Porto",
        "GeoLife": "GeoLife",
    }
    aggregate = []
    numeric_fields = (
        "total_balls",
        "spatial_path_balls",
        "kinematic_balls",
        "trajectory_shape_balls",
        "fit_seconds",
        "score_seconds",
        "fit_score_seconds",
    )
    for dataset in DATASET_SEEDS:
        subset = [row for row in rows if row["dataset"] == dataset]
        item = {
            "dataset": dataset,
            "dataset_label": labels[dataset],
            "n_samples": subset[0]["n_samples"],
            "n_seeds": len(subset),
        }
        for field in numeric_fields:
            mean, std = mean_std(subset, field)
            item[f"{field}_mean"] = mean
            item[f"{field}_std"] = std
        aggregate.append(item)

    atomic_write_json(SUMMARY_ROOT / "ewgb_phase_timing_seed_level.json", rows)
    atomic_write_json(SUMMARY_ROOT / "ewgb_phase_timing_aggregate.json", aggregate)
    atomic_write_json(
        SUMMARY_ROOT / "ewgb_phase_timing_status.json",
        {
            "completed_jobs": len(rows),
            "expected_jobs": len(jobs),
            "all_scores_exactly_match_main": all_exact,
            "max_abs_score_difference": max(
                row["max_abs_score_difference"] for row in rows
            ),
            "labels_consumed_during_fit": False,
            "cpu_processes": args.workers,
        },
    )
    write_table(aggregate, args.table_dir)
    if not all_exact:
        mismatches = [
            f"{row['dataset']} seed={row['seed']} max_abs={row['max_abs_score_difference']:.3e}"
            for row in rows
            if not row["score_exact_match"]
        ]
        print("\n".join(mismatches), flush=True)
        raise RuntimeError("phase timing scores do not exactly match the main experiment")
    print("EWGB_PHASE_TIMING_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
