"""Run the reviewer-requested cross-dataset tau and fusion-bin sensitivity."""

from __future__ import annotations

import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evaluation_protocol import evaluate_oracle_top_k
from granular_ball import EWGBDetector
from phase4_analysis_models import entropy_fusion
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
CONFIGS = (
    ("tau0p70", 0.70, 20),
    ("tau0p80", 0.80, 20),
    ("tau0p85", 0.85, 20),
    ("tau0p90", 0.90, 20),
    ("bins10", 0.85, 10),
    ("bins30", 0.85, 30),
    ("bins40", 0.85, 40),
)
VIEWS = ("spatial_path", "kinematic", "trajectory_shape")


def canonical_views(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct the analysis views through the same PCA path as EWGB-TAD."""
    return {
        "spatial_path": np.asarray(arrays["spatial_path"], dtype=np.float32),
        "kinematic": np.asarray(arrays["kinematic"], dtype=np.float32),
        "trajectory_shape": canonical_shape_view(arrays["trajectories"]),
        "labels": np.asarray(arrays["labels"], dtype=np.int64),
    }


def load_all_caches() -> dict[tuple[str, int], dict[str, np.ndarray]]:
    caches = {}
    cache_root = PHASE4_ROOT / "dataset_cache"
    if cache_root.exists():
        for dataset, seeds in SEEDS.items():
            for seed in seeds:
                path = cache_root / DATASET_SLUGS[dataset] / f"seed_{seed}.npz"
                if not path.exists():
                    raise FileNotFoundError(path)
                with np.load(path) as arrays:
                    caches[(dataset, seed)] = canonical_views(
                        {name: np.asarray(arrays[name]) for name in arrays.files}
                    )
        return caches

    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for dataset, seeds in SEEDS.items():
            for seed in seeds:
                name = f"results/dataset_cache/{DATASET_SLUGS[dataset]}/seed_{seed}.npz"
                payload = tf.extractfile(name)
                if payload is None:
                    raise FileNotFoundError(name)
                with np.load(io.BytesIO(payload.read())) as arrays:
                    caches[(dataset, seed)] = canonical_views(
                        {name: np.asarray(arrays[name]) for name in arrays.files}
                    )
    return caches


def run_job(payload: tuple[str, int, str, float, int, dict[str, np.ndarray]]) -> dict:
    dataset, seed, variant, tau, bins, arrays = payload
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    np.random.seed(seed)
    view_scores = {}
    ball_counts = {}
    for view in VIEWS:
        detector = EWGBDetector(
            min_samples=8,
            purity_threshold=tau,
            use_local_entropy=True,
            constant_tol=1e-10,
        )
        detector.fit(arrays[view])
        view_scores[view] = detector.score(arrays[view])
        ball_counts[view] = len(detector.balls)
    scores, weights = entropy_fusion(view_scores, mode="entropy", bins=bins, base_weight=0.1)
    metrics = evaluate_oracle_top_k(arrays["labels"], scores)
    return {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "tau": tau,
        "fusion_bins": bins,
        **metrics,
        "spatial_balls": ball_counts["spatial_path"],
        "kinematic_balls": ball_counts["kinematic"],
        "shape_balls": ball_counts["trajectory_shape"],
        "spatial_view_weight": weights["spatial_path"],
        "kinematic_view_weight": weights["kinematic"],
        "shape_view_weight": weights["trajectory_shape"],
        "labels_consumed_during_fit": False,
    }


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, raw[key] * (m - index)))
        adjusted[key] = running
    return adjusted


def paired_statistics(df: pd.DataFrame, variants: list[str], reference: str) -> dict:
    output = {}
    for metric in ("AUC", "AUPRC", "F1"):
        pivot = df.pivot_table(index=["dataset", "seed"], columns="variant", values=metric)
        raw = {}
        details = {}
        for variant in variants:
            if variant == reference:
                continue
            difference = pivot[reference] - pivot[variant]
            test = wilcoxon(pivot[reference], pivot[variant], zero_method="wilcox", alternative="two-sided")
            raw[variant] = float(test.pvalue)
            details[variant] = {
                "mean_reference_minus_variant": float(difference.mean()),
                "wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "losses": int((difference < 0).sum()),
                "p_raw": float(test.pvalue),
            }
        adjusted = holm(raw)
        for variant in details:
            details[variant]["p_holm"] = adjusted[variant]
        output[metric] = details
    return output


def aggregate(df: pd.DataFrame) -> dict:
    output = {}
    for variant, variant_df in df.groupby("variant", sort=False):
        output[variant] = {}
        for dataset, dataset_df in variant_df.groupby("dataset", sort=False):
            output[variant][dataset] = {
                metric: {
                    "mean": float(dataset_df[metric].mean()),
                    "std": float(dataset_df[metric].std(ddof=1)),
                    "n": int(len(dataset_df)),
                }
                for metric in ("AUC", "AUPRC", "F1")
            }
    return output


def latex_p_value(value: float) -> str:
    if np.isclose(value, 1.0):
        return "1.000"
    if value >= 0.001:
        return f"{value:.3g}"
    exponent = int(np.floor(np.log10(value)))
    coefficient = value / (10 ** exponent)
    return f"${coefficient:.2f}\\times10^{{{exponent}}}$"


def build_table(
    aggregate_data: dict,
    statistics: dict,
    variants: list[str],
    reference: str,
    analysis: str,
) -> None:
    dataset_order = tuple(DATASET_SLUGS)
    if analysis == "tau":
        display = {variant: f"$\\tau={variant.removeprefix('tau').replace('p', '.')}$" for variant in variants}
        caption = "Cross-dataset sensitivity of AUC to the granular-ball quality threshold $\\tau$ under the sample-size-adaptive local entropy rule."
        label = "tau_cross_dataset"
    else:
        display = {
            variant: "$K=20$" if variant == reference else f"$K={variant.removeprefix('bins')}$"
            for variant in variants
        }
        caption = "Sensitivity of AUC to the number $K$ of fixed bins used only for full-pool view-score fusion. Local entropy uses the prescribed sample-size-adaptive rule."
        label = "fusion_bins"
    lines = [
        "\\begin{center}",
        f"\\castablecaption[\\textwidth]{{tab:{label}}}{{{caption}}}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{4.0pt}",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Setting} & \\textbf{Synthetic} & \\textbf{Grid} & \\textbf{Porto} & \\textbf{GeoLife} & \\textbf{Macro} & \\textbf{Adjusted $p$} \\\\",
        "\\midrule",
    ]
    for variant in variants:
        values = [aggregate_data[variant][dataset]["AUC"]["mean"] for dataset in dataset_order]
        p_text = "reference" if variant == reference else latex_p_value(
            statistics["AUC"][variant]["p_holm"]
        )
        lines.append(
            f"{display[variant]} & "
            + " & ".join([f"{value:.4f}" for value in values] + [f"{np.mean(values):.4f}", p_text])
            + " \\\\"
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\endgroup", "\\end{center}"]
    (TABLE_DIR / f"{label}_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    caches = load_all_caches()
    jobs = []
    for dataset, seeds in SEEDS.items():
        for seed in seeds:
            for variant, tau, bins in CONFIGS:
                jobs.append((dataset, seed, variant, tau, bins, caches[(dataset, seed)]))
    records = []
    workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if completed % 20 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)
    df = pd.DataFrame(records).sort_values(["variant", "dataset", "seed"])
    csv_path = OUTPUT_DIR / "extended_sensitivity_runs.csv"
    json_path = OUTPUT_DIR / "extended_sensitivity.json"
    df.to_csv(csv_path, index=False)
    aggregate_data = aggregate(df)
    tau_variants = ["tau0p70", "tau0p80", "tau0p85", "tau0p90"]
    bin_variants = ["bins10", "tau0p85", "bins30", "bins40"]
    tau_stats = paired_statistics(df[df.variant.isin(tau_variants)], tau_variants, "tau0p85")
    bin_stats = paired_statistics(df[df.variant.isin(bin_variants)], bin_variants, "tau0p85")
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "dataset_source": "Phase 4 predefined dataset-cache archive",
                    "labels_consumed_during_fit": False,
                    "default": {"tau": 0.85, "fusion_bins": 20},
                    "local_entropy_bin_count": "B_j=max(2, ceil(log2(n_j)+1)) for each terminal ball",
                },
                "aggregate": aggregate_data,
                "tau_statistics": tau_stats,
                "fusion_bin_statistics": bin_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_table(aggregate_data, tau_stats, tau_variants, "tau0p85", "tau")
    build_table(aggregate_data, bin_stats, bin_variants, "tau0p85", "bins")
    print(f"wrote {len(df)} sensitivity records")


if __name__ == "__main__":
    main()
