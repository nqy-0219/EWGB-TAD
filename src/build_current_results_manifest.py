"""Build and validate the canonical manifest for the current result set."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase4_common import atomic_write_json, sha256_file


SOURCE_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path(
    os.environ.get(
        "EWGB_PHASE4_ROOT",
        SOURCE_ROOT / "paper_work" / "final_neurocomputing_results",
    )
).resolve()
OUTPUT_PATH = RESULT_ROOT / "metadata" / "CURRENT_RESULT_SET.json"

MARKER_GROUPS = {
    "main": (RESULT_ROOT / "raw", 390),
    "analysis": (RESULT_ROOT / "analysis_raw", 570),
    "deep_timing": (RESULT_ROOT / "deep_timing_raw", 16),
    "local_entropy_bins": (
        RESULT_ROOT / "sensitivity_stability" / "local_entropy_bins" / "raw",
        120,
    ),
}

SUMMARY_FILES = (
    "summary/main_aggregate.json",
    "summary/main_statistics.json",
    "summary/main_seed_level.csv",
    "summary/analysis_aggregate.json",
    "summary/analysis_statistics.json",
    "summary/analysis_seed_level.csv",
    "summary/runtime_seed_level.csv",
    "summary/runtime_by_dataset_method.json",
    "sensitivity_stability/extended_sensitivity.json",
    "sensitivity_stability/extended_sensitivity_runs.csv",
    "sensitivity_stability/weight_stability.json",
    "sensitivity_stability/weight_stability_runs.csv",
    "sensitivity_stability/weight_stability_seeds.csv",
    "sensitivity_stability/local_entropy_bins/local_entropy_bins_summary.json",
    "sensitivity_stability/local_entropy_bins/local_entropy_bins_seed_level.csv",
    "protocol/unlabeled_inner_split_manifest.json",
    "protocol/unlabeled_inner_splits.npz",
)

SOURCE_FILES = tuple(
    path.relative_to(SOURCE_ROOT).as_posix()
    for path in sorted((SOURCE_ROOT / "src").rglob("*.py"))
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(RESULT_ROOT).as_posix()


def artifact_path(marker_path: Path, marker: dict[str, Any], key: str) -> Path:
    recorded = marker.get(key)
    if not recorded:
        raise RuntimeError(f"{marker_path} does not record {key}")
    path = Path(str(recorded))
    if not path.is_absolute():
        path = marker_path.parent / path
    path = path.resolve()
    path.relative_to(RESULT_ROOT)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def marker_records(root: Path, expected: int) -> list[dict[str, Any]]:
    records = []
    markers = sorted(root.rglob("COMPLETE.json"))
    if len(markers) != expected:
        raise RuntimeError(f"{root} has {len(markers)} markers; expected {expected}")
    for marker_path in markers:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        result_path = artifact_path(marker_path, marker, "result")
        score_path = artifact_path(marker_path, marker, "score")
        result_hash = sha256_file(result_path)
        if result_hash != marker.get("result_sha256"):
            raise RuntimeError(f"result hash mismatch: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        score_hash = sha256_file(score_path)
        if score_hash != result.get("score_sha256"):
            raise RuntimeError(f"score hash mismatch: {score_path}")
        records.append(
            {
                "marker": relative(marker_path),
                "attempt": marker["attempt"],
                "result": relative(result_path),
                "result_sha256": result_hash,
                "score": relative(score_path),
                "score_sha256": score_hash,
            }
        )
    return records


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def dataset_caches() -> list[dict[str, Any]]:
    records = []
    for path in sorted((RESULT_ROOT / "dataset_cache").rglob("seed_*.npz")):
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if digest != metadata.get("sha256"):
            raise RuntimeError(f"dataset-cache hash mismatch: {path}")
        records.append(
            {
                "cache": relative(path),
                "metadata": relative(metadata_path),
                "sha256": digest,
                "dataset": metadata["dataset"],
                "seed": metadata["seed"],
            }
        )
    if len(records) != 30:
        raise RuntimeError(f"found {len(records)} dataset caches; expected 30")
    return records


def hash_records(base: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    records = []
    for name in names:
        path = (base / name).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        records.append({"path": path.relative_to(base.resolve()).as_posix(), "sha256": sha256_file(path)})
    return records


def main() -> int:
    marker_payload = {
        name: marker_records(root, expected)
        for name, (root, expected) in MARKER_GROUPS.items()
    }
    counts = {
        "main_jobs": len(marker_payload["main"]),
        "component_view_nmin_jobs": len(marker_payload["analysis"]),
        "deep_timing_jobs": len(marker_payload["deep_timing"]),
        "local_entropy_bin_jobs": len(marker_payload["local_entropy_bins"]),
        "extended_sensitivity_rows": csv_rows(
            RESULT_ROOT / "sensitivity_stability" / "extended_sensitivity_runs.csv"
        ),
        "weight_stability_run_view_rows": csv_rows(
            RESULT_ROOT / "sensitivity_stability" / "weight_stability_runs.csv"
        ),
        "weight_stability_seed_rows": csv_rows(
            RESULT_ROOT / "sensitivity_stability" / "weight_stability_seeds.csv"
        ),
    }
    expected_counts = {
        "main_jobs": 390,
        "component_view_nmin_jobs": 570,
        "deep_timing_jobs": 16,
        "local_entropy_bin_jobs": 120,
        "extended_sensitivity_rows": 210,
        "weight_stability_run_view_rows": 360,
        "weight_stability_seed_rows": 48,
    }
    if counts != expected_counts:
        raise RuntimeError(f"current result counts do not match the predefined protocol: {counts}")

    payload = {
        "result_set": "EWGB-TAD Neurocomputing revision, canonical PCA-aligned results",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": "paper_work/final_neurocomputing_results",
        "shape_view_policy": (
            "Every EWGB-TAD main, component, sensitivity, stability, timing, and figure path "
            "fits the trajectory-shape PCA from the predefined trajectories through canonical_shape_view "
            "or the fitted ThreeViewEWGBDetector. The cached trajectory_shape array belongs only to "
            "the shared 34-dimensional generic-baseline representation."
        ),
        "counts": counts,
        "dataset_caches": dataset_caches(),
        "marker_groups": marker_payload,
        "summary_artifacts": hash_records(RESULT_ROOT, SUMMARY_FILES),
        "source_artifacts": hash_records(SOURCE_ROOT, SOURCE_FILES),
    }
    atomic_write_json(OUTPUT_PATH, payload)
    print(f"wrote {OUTPUT_PATH}")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
