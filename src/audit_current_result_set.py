"""Audit the canonical result tree and write a machine-readable report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = SOURCE_ROOT / "paper_work" / "final_neurocomputing_results"
LOCK_PATH = RESULT_ROOT / "metadata" / "CURRENT_RESULT_SET.json"
OUTPUT_PATH = RESULT_ROOT / "metadata" / "RESULT_SET_AUDIT.json"
GROUPS = {
    "main": (RESULT_ROOT / "raw", 390),
    "analysis": (RESULT_ROOT / "analysis_raw", 570),
    "deep_timing": (RESULT_ROOT / "deep_timing_raw", 16),
    "local_entropy_bins": (
        RESULT_ROOT / "sensitivity_stability" / "local_entropy_bins" / "raw",
        120,
    ),
}


def audit_group(root: Path, expected: int) -> dict:
    markers = sorted(root.rglob("COMPLETE.json"))
    selected: set[Path] = set()
    missing_selected: list[str] = []
    for marker_path in markers:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        attempt = (marker_path.parent / marker["attempt"]).resolve()
        selected.add(attempt)
        if not attempt.is_dir():
            missing_selected.append(str(attempt))
    attempts = {path.resolve() for path in root.rglob("attempt_*") if path.is_dir()}
    obsolete = sorted(str(path) for path in attempts - selected)
    return {
        "expected_markers": expected,
        "actual_markers": len(markers),
        "selected_attempts": len(selected),
        "missing_selected_attempts": missing_selected,
        "unreferenced_attempts": obsolete,
        "ok": len(markers) == expected and not missing_selected and not obsolete,
    }


def main() -> int:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    groups = {name: audit_group(root, expected) for name, (root, expected) in GROUPS.items()}
    cache_files = sorted((RESULT_ROOT / "dataset_cache").rglob("seed_*.npz"))
    statuses = {}
    for path in sorted((RESULT_ROOT / "summary").glob("*_status.json")):
        statuses[path.name] = json.loads(path.read_text(encoding="utf-8"))
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_result_root": "paper_work/final_neurocomputing_results",
        "canonical_lock": "paper_work/final_neurocomputing_results/metadata/CURRENT_RESULT_SET.json",
        "lock_counts": lock["counts"],
        "dataset_cache_count": len(cache_files),
        "groups": groups,
        "summary_statuses": statuses,
        "ok": len(cache_files) == 30 and all(item["ok"] for item in groups.values()),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "groups": groups}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
