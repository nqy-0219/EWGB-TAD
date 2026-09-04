"""Remove result attempts that are not referenced by a COMPLETE marker."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path(
    os.environ.get(
        "EWGB_PHASE4_ROOT",
        ROOT / "paper_work" / "final_neurocomputing_results",
    )
).resolve()
ATTEMPT_ROOTS = (
    RESULT_ROOT / "raw",
    RESULT_ROOT / "analysis_raw",
    RESULT_ROOT / "deep_timing_raw",
    RESULT_ROOT / "sensitivity_stability" / "local_entropy_bins" / "raw",
)


def marker_attempts(root: Path) -> set[Path]:
    referenced: set[Path] = set()
    for marker_path in root.rglob("COMPLETE.json"):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        attempt = marker.get("attempt")
        if attempt:
            referenced.add((marker_path.parent / str(attempt)).resolve())
    return referenced


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    candidates: list[Path] = []
    referenced_count = 0
    for root in ATTEMPT_ROOTS:
        resolved_root = root.resolve()
        if not is_within(resolved_root, RESULT_ROOT):
            raise RuntimeError(f"attempt root escapes result root: {resolved_root}")
        if not resolved_root.exists():
            continue
        referenced = marker_attempts(resolved_root)
        referenced_count += len(referenced)
        for path in resolved_root.rglob("attempt_*"):
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if not is_within(resolved, resolved_root):
                raise RuntimeError(f"attempt path escapes approved root: {resolved}")
            if resolved not in referenced:
                candidates.append(resolved)

    if args.apply:
        for path in candidates:
            shutil.rmtree(path)
    print(
        json.dumps(
            {
                "result_root": str(RESULT_ROOT),
                "referenced_attempts": referenced_count,
                "obsolete_attempts": len(candidates),
                "removed": len(candidates) if args.apply else 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
