"""Verify that the component analysis is current, complete, and reproducible."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation_protocol import evaluate_oracle_top_k
from phase4_build_analysis_manifests import COMPONENT_VARIANTS
from phase4_common import (
    DATASET_SEEDS,
    PHASE4_ROOT,
    atomic_write_json,
    cache_path,
    resolve_marker_artifact,
    sha256_file,
    slug,
)
from phase4_run_analysis_job import analysis_job_dir


SOURCE_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = PHASE4_ROOT / "metadata" / "CURRENT_COMPONENT_RESULT_AUDIT.json"
OUTPUT_MD = PHASE4_ROOT / "metadata" / "CURRENT_COMPONENT_RESULT_AUDIT.md"
HASHED_SOURCES = (
    "granular_ball.py",
    "phase4_analysis_models.py",
    "phase4_common.py",
    "phase4_run_analysis_job.py",
)
DIRECT_VARIANTS = {
    "g0_l1_f1": "KMeans-Prototype",
    "g1_l0_f1": "Global entropy weighting",
    "g1_l1_f0": "Average fusion",
    "g1_l1_f1": "EWGB-TAD",
}
EXPECTED_SETTINGS = {
    "g0_l1_f1": (False, True, "entropy"),
    "g1_l0_f1": (True, False, "entropy"),
    "g1_l1_f0": (True, True, "equal"),
    "g1_l1_f1": (True, True, "entropy"),
}
METRIC_TOLERANCE = 1e-12


def marker_artifacts(marker_path: Path) -> tuple[dict, Path, Path]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    result_path = resolve_marker_artifact(marker_path, marker, "result")
    score_path = resolve_marker_artifact(marker_path, marker, "score")
    if result_path is None or score_path is None:
        raise FileNotFoundError(f"marker does not resolve both artifacts: {marker_path}")
    return marker, result_path, score_path


def main_score_marker(dataset: str, seed: int) -> Path:
    return (
        PHASE4_ROOT
        / "raw"
        / slug(dataset)
        / slug("EWGB-TAD")
        / f"seed_{seed}"
        / "COMPLETE.json"
    )


def add_error(errors: list[dict], category: str, message: str, **context: object) -> None:
    errors.append({"category": category, "message": message, **context})


def main() -> int:
    source_hashes = {name: sha256_file(SOURCE_DIR / name) for name in HASHED_SOURCES}
    errors: list[dict] = []
    records: list[dict] = []
    score_equivalence: list[dict] = []
    selected_attempts: set[Path] = set()

    for dataset, seeds in DATASET_SEEDS.items():
        with np.load(cache_path(dataset, seeds[0])) as archive:
            if "labels" not in archive.files:
                add_error(errors, "cache", "labels are absent from dataset cache", dataset=dataset)

        for seed in seeds:
            with np.load(cache_path(dataset, seed)) as archive:
                labels = np.asarray(archive["labels"], dtype=np.int64)

            for variant in COMPONENT_VARIANTS:
                marker_path = analysis_job_dir("component", variant, dataset, seed) / "COMPLETE.json"
                if not marker_path.exists():
                    add_error(
                        errors,
                        "missing",
                        "component marker is missing",
                        dataset=dataset,
                        seed=seed,
                        variant=variant,
                    )
                    continue
                try:
                    marker, result_path, score_path = marker_artifacts(marker_path)
                    selected_attempts.add(result_path.parent.resolve())
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    with np.load(score_path) as archive:
                        scores = np.asarray(archive["scores"], dtype=np.float64)
                except Exception as exc:  # report every damaged job in one pass
                    add_error(
                        errors,
                        "load",
                        repr(exc),
                        dataset=dataset,
                        seed=seed,
                        variant=variant,
                    )
                    continue

                context = {"dataset": dataset, "seed": seed, "variant": variant}
                if marker.get("status") != "complete" or result.get("status") != "complete":
                    add_error(errors, "status", "job is not marked complete", **context)
                if result.get("analysis") != "component":
                    add_error(errors, "identity", "analysis identifier is incorrect", **context)
                if result.get("variant") != variant or result.get("dataset") != dataset or result.get("seed") != seed:
                    add_error(errors, "identity", "result identity does not match its directory", **context)
                if sha256_file(result_path) != marker.get("result_sha256"):
                    add_error(errors, "hash", "result JSON hash mismatch", **context)
                if sha256_file(score_path) != result.get("score_sha256"):
                    add_error(errors, "hash", "score archive hash mismatch", **context)
                if scores.shape != labels.shape or not np.isfinite(scores).all():
                    add_error(errors, "scores", "score shape or finiteness check failed", **context)
                    continue

                recorded_hashes = result.get("implementation", {}).get("source_hashes_sha256", {})
                if recorded_hashes != source_hashes:
                    add_error(errors, "implementation", "recorded source hashes are not current", **context)
                if result.get("implementation", {}).get("shape_view") != "canonical_shape_view from phase4_common.py":
                    add_error(errors, "implementation", "canonical PCA shape-view marker is absent", **context)

                recalculated = evaluate_oracle_top_k(labels, scores)
                for metric in ("AUC", "AUPRC", "F1"):
                    difference = abs(float(recalculated[metric]) - float(result["metrics"][metric]))
                    if difference > METRIC_TOLERANCE:
                        add_error(
                            errors,
                            "metrics",
                            f"recalculated {metric} differs by {difference:.3e}",
                            **context,
                        )

                if variant in EXPECTED_SETTINGS:
                    expected_g, expected_l, expected_f = EXPECTED_SETTINGS[variant]
                    stats = result.get("model_stats", {})
                    actual = (
                        stats.get("granular_partition"),
                        stats.get("local_metric"),
                        stats.get("fusion"),
                    )
                    if actual != (expected_g, expected_l, expected_f):
                        add_error(errors, "definition", f"variant settings are {actual}", **context)
                    for view, detector in stats.get("detectors", {}).items():
                        if detector.get("entropy_method") != "sample-adaptive-histogram":
                            add_error(
                                errors,
                                "entropy",
                                f"{view} does not report sample-adaptive histogram entropy",
                                **context,
                            )
                        if detector.get("local_bin_rule") != "max(2, ceil(log2(n_ball) + 1))" and expected_g:
                            add_error(errors, "entropy", f"{view} has an unexpected ball bin rule", **context)
                        if detector.get("local_bin_rule") != "max(2, ceil(log2(n_region) + 1))" and not expected_g:
                            add_error(errors, "entropy", f"{view} has an unexpected region bin rule", **context)

                records.append(
                    {
                        **context,
                        "result_path": str(result_path.resolve()),
                        "score_path": str(score_path.resolve()),
                        "result_modified_utc": datetime.fromtimestamp(
                            result_path.stat().st_mtime, tz=timezone.utc
                        ).isoformat(),
                        **{metric: float(recalculated[metric]) for metric in ("AUC", "AUPRC", "F1")},
                    }
                )

                if variant == "g1_l1_f1":
                    main_marker_path = main_score_marker(dataset, seed)
                    try:
                        _, _, main_score_path = marker_artifacts(main_marker_path)
                        with np.load(main_score_path) as archive:
                            main_scores = np.asarray(archive["scores"], dtype=np.float64)
                        same = bool(np.array_equal(scores, main_scores))
                        max_abs = float(np.max(np.abs(scores - main_scores)))
                        score_equivalence.append(
                            {**context, "array_equal": same, "maximum_absolute_difference": max_abs}
                        )
                        if not same:
                            add_error(
                                errors,
                                "main_equivalence",
                                f"complete component scores differ from main EWGB-TAD by {max_abs:.3e}",
                                **context,
                            )
                    except Exception as exc:
                        add_error(errors, "main_equivalence", repr(exc), **context)

    component_root = PHASE4_ROOT / "analysis_raw" / "component"
    all_attempts = {path.resolve() for path in component_root.rglob("attempt_*") if path.is_dir()}
    obsolete_attempts = sorted(str(path) for path in all_attempts - selected_attempts)

    direct_summary: dict[str, dict] = {}
    for variant, display_name in DIRECT_VARIANTS.items():
        subset = [record for record in records if record["variant"] == variant]
        dataset_means = {
            dataset: {
                metric: float(
                    np.mean(
                        [
                            record[metric]
                            for record in subset
                            if record["dataset"] == dataset
                        ]
                    )
                )
                for metric in ("AUC", "AUPRC", "F1")
            }
            for dataset in DATASET_SEEDS
        }
        direct_summary[variant] = {
            "name": display_name,
            "n_dataset_seed_blocks": len(subset),
            "macro_across_four_dataset_means": {
                metric: {
                    "mean": float(np.mean([values[metric] for values in dataset_means.values()])),
                    "std": float(np.std([values[metric] for values in dataset_means.values()], ddof=1)),
                }
                for metric in ("AUC", "AUPRC", "F1")
            },
        }

    expected_jobs = len(COMPONENT_VARIANTS) * sum(len(seeds) for seeds in DATASET_SEEDS.values())
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_result_root": "paper_work/final_neurocomputing_results",
        "expected_jobs": expected_jobs,
        "validated_jobs": len(records),
        "source_hashes_sha256": source_hashes,
        "all_selected_results_record_current_hashes": not any(
            error["category"] == "implementation" for error in errors
        ),
        "all_scores_finite_and_metrics_reproduced": not any(
            error["category"] in {"scores", "metrics", "hash", "load"} for error in errors
        ),
        "complete_variant_matches_main_scores": bool(score_equivalence)
        and all(item["array_equal"] for item in score_equivalence),
        "direct_variant_summary": direct_summary,
        "selected_result_time_range_utc": {
            "earliest": min((record["result_modified_utc"] for record in records), default=None),
            "latest": max((record["result_modified_utc"] for record in records), default=None),
        },
        "obsolete_attempt_count": len(obsolete_attempts),
        "obsolete_attempts": obsolete_attempts,
        "errors": errors,
        "ok": len(records) == expected_jobs and not errors,
    }
    atomic_write_json(OUTPUT_JSON, report)

    lines = [
        "# Current component-result audit",
        "",
        f"- Status: {'PASS' if report['ok'] else 'FAIL'}",
        f"- Validated jobs: {len(records)}/{expected_jobs}",
        f"- Current source hashes recorded by every job: {report['all_selected_results_record_current_hashes']}",
        f"- Scores finite and metrics reproduced: {report['all_scores_finite_and_metrics_reproduced']}",
        f"- Complete component scores equal main EWGB-TAD scores: {report['complete_variant_matches_main_scores']}",
        f"- Unreferenced attempts awaiting cleanup: {len(obsolete_attempts)}",
        "",
        "## Direct comparisons used in Table 8",
        "",
        "| Variant | AUC | AUPRC | F1 |",
        "|---|---:|---:|---:|",
    ]
    for item in direct_summary.values():
        cells = []
        for metric in ("AUC", "AUPRC", "F1"):
            summary = item["macro_across_four_dataset_means"][metric]
            cells.append(f"{summary['mean']:.4f} +/- {summary['std']:.4f}")
        lines.append(f"| {item['name']} | " + " | ".join(cells) + " |")
    if errors:
        lines += ["", "## Errors", ""]
        lines.extend(f"- {error}" for error in errors)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({key: report[key] for key in (
        "ok",
        "expected_jobs",
        "validated_jobs",
        "all_selected_results_record_current_hashes",
        "all_scores_finite_and_metrics_reproduced",
        "complete_variant_matches_main_scores",
        "obsolete_attempt_count",
        "errors",
    )}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
