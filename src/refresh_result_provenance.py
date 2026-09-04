"""Refresh source hashes and provenance for the canonical result set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = SOURCE_ROOT / "paper_work" / "final_neurocomputing_results"
LOCK_PATH = RESULT_ROOT / "metadata" / "CURRENT_RESULT_SET.json"
SOURCE_HASH_PATH = RESULT_ROOT / "protocol" / "source_hashes.json"
PROVENANCE_PATH = RESULT_ROOT / "PROVENANCE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return path.relative_to(SOURCE_ROOT).as_posix()


def source_files() -> list[Path]:
    files = sorted((SOURCE_ROOT / "src").rglob("*.py"))
    files.extend(sorted((SOURCE_ROOT / "configs").glob("*.json")))
    for path in (
        SOURCE_ROOT / "external_baselines_phase3" / "LMTAD" / "LICENSE.txt",
        SOURCE_ROOT / "external_baselines_phase3" / "MST-OATD" / "LICENSE",
    ):
        if path.exists():
            files.append(path)
    return files


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-only", action="store_true")
    args = parser.parse_args()
    generated = datetime.now(timezone.utc).isoformat()
    if not args.provenance_only:
        hashes = {relative(path): sha256(path) for path in source_files()}
        source_payload = {
            "generated_for": "canonical EWGB-TAD Neurocomputing revision",
            "generated_utc": generated,
            "hash_algorithm": "SHA-256",
            "source_files": hashes,
            "external_repositories": {
                "LMTAD": "80bb89a8ea108db8f13cb9959826424e9c45f41c",
                "MST-OATD": "db94b41c1b6fd333d6776ef2dcbbbeec39d16c02",
            },
        }
        write_json(SOURCE_HASH_PATH, source_payload)

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    provenance.update(
        {
            "status": "complete_and_locked",
            "generated_utc": generated,
            "canonical_result_lock": "metadata/CURRENT_RESULT_SET.json",
            "canonical_result_lock_sha256": sha256(LOCK_PATH),
            "source_hash_manifest": "protocol/source_hashes.json",
            "source_hash_manifest_sha256": sha256(SOURCE_HASH_PATH),
            "shape_view_policy": lock["shape_view_policy"],
            "source_policy": (
                "Only the COMPLETE-selected successful attempt is included for each job; "
                "unreferenced and failed attempts are excluded."
            ),
        }
    )
    write_json(PROVENANCE_PATH, provenance)
    if not args.provenance_only:
        print(f"wrote {SOURCE_HASH_PATH}")
    print(f"wrote {PROVENANCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
