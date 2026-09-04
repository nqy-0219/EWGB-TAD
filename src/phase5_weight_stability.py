"""Quantify local-weight stability from the predefined Phase 4 dataset caches."""

from __future__ import annotations

import io
import json
import os
import tarfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from granular_ball import EWGBDetector
from phase4_common import canonical_shape_view


ROOT = Path(__file__).resolve().parents[1]
PHASE4_ROOT = Path(
    os.environ.get(
        "EWGB_PHASE4_ROOT",
        str(ROOT / "paper_work" / "final_neurocomputing_results"),
    )
)
ARCHIVE = PHASE4_ROOT.parent / "final_neurocomputing_results.tar.gz"
OUTPUT_DIR = Path(
    os.environ.get(
        "EWGB_PHASE5_OUTPUT_DIR",
        str(ROOT / "paper_work" / "final_neurocomputing_results" / "sensitivity_stability"),
    )
)
TABLE_DIR = Path(
    os.environ.get(
        "EWGB_TABLE_DIR",
        str(ROOT.parent / "neurocomputing_submission" / "overleaf_neurocomputing" / "tables"),
    )
)

DATASET_SLUGS = {
    "Synthetic": "synthetic",
    "Grid-Network": "grid_network",
    "Porto-derived": "porto_derived",
    "GeoLife": "geolife",
}
SEEDS = {
    "Synthetic": (42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144),
    "Grid-Network": (42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144),
    "Porto-derived": (42, 123, 456, 789, 1024),
    "GeoLife": (42, 123, 456, 789, 1024),
}
VIEWS = ("spatial_path", "kinematic", "trajectory_shape")
MIN_SAMPLES = (4, 8, 16, 32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-15:
        return 1.0
    return float(np.dot(a, b) / denominator)


def weighted_profile(weights: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    if len(weights) == 0:
        raise ValueError("no terminal-ball weights")
    return np.average(weights, axis=0, weights=sizes)


def size_stratum_profiles(
    weights: np.ndarray, sizes: np.ndarray, min_samples: int
) -> dict[str, np.ndarray]:
    masks = {
        "small": sizes <= 2 * min_samples,
        "medium": (sizes > 2 * min_samples) & (sizes <= 4 * min_samples),
        "large": sizes > 4 * min_samples,
    }
    output = {}
    for name, mask in masks.items():
        if np.any(mask):
            output[name] = weighted_profile(weights[mask], sizes[mask])
    return output


def load_cache(tf: tarfile.TarFile, dataset: str, seed: int) -> dict[str, np.ndarray]:
    member_name = f"results/dataset_cache/{DATASET_SLUGS[dataset]}/seed_{seed}.npz"
    member = tf.getmember(member_name)
    payload = tf.extractfile(member)
    if payload is None:
        raise FileNotFoundError(member_name)
    with np.load(io.BytesIO(payload.read())) as cache:
        return {name: np.asarray(cache[name]) for name in cache.files}


def fit_view(data: np.ndarray, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    detector = EWGBDetector(
        min_samples=min_samples,
        purity_threshold=0.85,
        use_local_entropy=True,
        constant_tol=1e-10,
    )
    detector.fit(np.asarray(data, dtype=np.float32))
    return np.asarray(detector.local_weights, dtype=float), np.asarray(
        [ball.size for ball in detector.balls], dtype=float
    )


def build_records() -> tuple[pd.DataFrame, dict]:
    run_records = []
    profiles: dict[tuple[str, int, str], list[np.ndarray]] = {}
    cache_root = PHASE4_ROOT / "dataset_cache"
    if cache_root.exists():
        cache_source = None
    else:
        cache_source = tarfile.open(ARCHIVE, "r:gz")
    try:
        for dataset, seeds in SEEDS.items():
            for seed in seeds:
                if cache_source is None:
                    with np.load(cache_root / DATASET_SLUGS[dataset] / f"seed_{seed}.npz") as arrays:
                        cache = {name: np.asarray(arrays[name]) for name in arrays.files}
                else:
                    cache = load_cache(cache_source, dataset, seed)
                cache["trajectory_shape"] = canonical_shape_view(cache["trajectories"])
                for min_samples in MIN_SAMPLES:
                    for view in VIEWS:
                        weights, sizes = fit_view(cache[view], min_samples)
                        profile = weighted_profile(weights, sizes)
                        profiles.setdefault((dataset, min_samples, view), []).append(profile)
                        strata = size_stratum_profiles(weights, sizes, min_samples)
                        stratum_cosines = [
                            cosine(strata[left], strata[right])
                            for left, right in combinations(strata, 2)
                        ]
                        local_l1 = np.average(
                            np.sum(np.abs(weights - profile[None, :]), axis=1), weights=sizes
                        )
                        run_records.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "min_samples": min_samples,
                                "view": view,
                                "n_balls": len(sizes),
                                "mean_ball_size": float(np.mean(sizes)),
                                "size_strata_available": len(strata),
                                "size_stratum_cosine_mean": (
                                    float(np.mean(stratum_cosines)) if stratum_cosines else np.nan
                                ),
                                "weighted_local_l1_from_run_profile": float(local_l1),
                                "run_profile": profile.tolist(),
                            }
                        )

    finally:
        if cache_source is not None:
            cache_source.close()

    seed_records = []
    for (dataset, min_samples, view), group in profiles.items():
        similarities = [cosine(left, right) for left, right in combinations(group, 2)]
        seed_records.append(
            {
                "dataset": dataset,
                "min_samples": min_samples,
                "view": view,
                "n_seeds": len(group),
                "seed_profile_cosine_mean": float(np.mean(similarities)),
                "seed_profile_cosine_min": float(np.min(similarities)),
                "seed_profile_cosine_std": float(np.std(similarities, ddof=1)),
            }
        )
    return pd.DataFrame(run_records), {"seed_stability": seed_records}


def build_tables(run_df: pd.DataFrame, seed_df: pd.DataFrame) -> None:
    default_rows = []
    for dataset in DATASET_SLUGS:
        seed_values = seed_df[
            (seed_df.dataset == dataset) & (seed_df.min_samples == 8)
        ].seed_profile_cosine_mean
        size_values = run_df[
            (run_df.dataset == dataset) & (run_df.min_samples == 8)
        ].size_stratum_cosine_mean.dropna()
        ball_values = run_df[
            (run_df.dataset == dataset) & (run_df.min_samples == 8)
        ].n_balls
        default_rows.append(
            (
                dataset.replace("-Network", "").replace("-derived", ""),
                seed_values.mean(),
                seed_values.std(ddof=1),
                size_values.mean(),
                size_values.std(ddof=1),
                ball_values.mean(),
                ball_values.std(ddof=1),
            )
        )
    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:weight_stability}{Local-weight stability at the default $n_{min}=8$. Seed stability is the mean pairwise cosine similarity between sample-weighted view profiles across seeds. Size stability compares small, medium, and large terminal-ball profiles within each run.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{4.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccc@{}}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Seed cosine} & \\textbf{Size-stratum cosine} & \\textbf{Terminal balls} \\\\",
        "\\midrule",
    ]
    for dataset, seed_mean, seed_std, size_mean, size_std, balls_mean, balls_std in default_rows:
        lines.append(
            f"{dataset} & {seed_mean:.3f} $\\pm$ {seed_std:.3f} & "
            f"{size_mean:.3f} $\\pm$ {size_std:.3f} & {balls_mean:.1f} $\\pm$ {balls_std:.1f} \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    (TABLE_DIR / "weight_stability_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "\\begin{center}",
        "\\castablecaption[\\textwidth]{tab:weight_stability_nmin}{Aggregate local-weight stability across the tested terminal-ball sizes. Values summarize all dataset--view combinations.}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{5.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}cccc@{}}",
        "\\toprule",
        "\\textbf{$n_{min}$} & \\textbf{Seed cosine} & \\textbf{Size-stratum cosine} & \\textbf{Mean terminal balls} \\\\",
        "\\midrule",
    ]
    for min_samples in MIN_SAMPLES:
        seed_values = seed_df[seed_df.min_samples == min_samples].seed_profile_cosine_mean
        size_values = run_df[run_df.min_samples == min_samples].size_stratum_cosine_mean.dropna()
        balls = run_df[run_df.min_samples == min_samples].n_balls
        lines.append(
            f"{min_samples} & {seed_values.mean():.3f} $\\pm$ {seed_values.std(ddof=1):.3f} & "
            f"{size_values.mean():.3f} $\\pm$ {size_values.std(ddof=1):.3f} & "
            f"{balls.mean():.1f} $\\pm$ {balls.std(ddof=1):.1f} \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    (TABLE_DIR / "weight_stability_nmin_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    run_df, metadata = build_records()
    seed_df = pd.DataFrame(metadata["seed_stability"])
    csv_path = OUTPUT_DIR / "weight_stability_runs.csv"
    seed_csv_path = OUTPUT_DIR / "weight_stability_seeds.csv"
    json_path = OUTPUT_DIR / "weight_stability.json"
    run_export = run_df.drop(columns=["run_profile"])
    run_export.to_csv(csv_path, index=False)
    seed_df.to_csv(seed_csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "definition": {
                    "seed_stability": "pairwise cosine similarity of sample-weighted mean local-feature profiles across seeds",
                    "size_stability": "pairwise cosine similarity among small (<=2*n_min), medium (2*n_min to 4*n_min), and large (>4*n_min) terminal-ball profiles within a run",
                    "label_access": False,
                    "dataset_source": "Phase 4 predefined dataset-cache archive",
                },
                "run_records": run_df.to_dict(orient="records"),
                "seed_stability": seed_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_tables(run_df, seed_df)
    print(f"wrote {len(run_df)} run-view records and {len(seed_df)} seed-stability records")


if __name__ == "__main__":
    main()
