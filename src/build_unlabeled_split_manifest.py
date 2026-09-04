"""Build exact, label-free inner-split records for reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RESULT_ROOT = SOURCE_ROOT / "paper_work" / "final_neurocomputing_results"
PUBLIC_RESULT_ROOT = SOURCE_ROOT / "results" / "phase4"
SPLIT_METHODS = ("LSTM-AE", "USAD", "LM-TAD")
FULL_POOL_METHODS = (
    "EWGB-TAD",
    "IForest",
    "ECOD",
    "iBoost-ODE",
    "CoMadOut",
    "Shape-KNN",
    "SegmentOD",
    "TADS",
    "Profile-TAD",
    "MST-OATD",
)


def sha256_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_default_root() -> Path:
    if CANONICAL_RESULT_ROOT.is_dir():
        return CANONICAL_RESULT_ROOT
    return PUBLIC_RESULT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=resolve_default_root())
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    dataset_manifest_path = result_root / "metadata" / "dataset_seed_manifest_latest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8-sig"))

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    for item in dataset_manifest["records"]:
        dataset = str(item["dataset"])
        seed = int(item["seed"])
        n_total = int(item["n_total"])
        order = np.random.RandomState(seed).permutation(n_total)
        n_validation = max(1, int(round(n_total * 0.1)))
        n_validation = min(n_validation, n_total - 1)
        validation = np.asarray(order[:n_validation], dtype="<i8")
        training = np.asarray(order[n_validation:], dtype="<i8")
        split_id = f"{dataset.lower().replace('-', '_')}_seed_{seed}"
        arrays[f"{split_id}__training"] = training
        arrays[f"{split_id}__validation"] = validation
        records.append(
            {
                "split_id": split_id,
                "dataset": dataset,
                "seed": seed,
                "n_total": n_total,
                "n_training": int(len(training)),
                "n_validation": int(len(validation)),
                "validation_fraction": 0.1,
                "permutation": "numpy RandomState(seed).permutation(n_total)",
                "assignment": "validation is the first round(0.1*n_total) indices; training is the remainder",
                "training_indices_sha256": sha256_array(training),
                "validation_indices_sha256": sha256_array(validation),
                "applies_to_formal_methods": list(SPLIT_METHODS),
            }
        )

    protocol_root = result_root / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    archive_path = protocol_root / "unlabeled_inner_splits.npz"
    np.savez_compressed(archive_path, **arrays)
    manifest_path = protocol_root / "unlabeled_inner_split_manifest.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Exact label-free inner train/validation splits used by neural baselines with validation",
        "labels_used": False,
        "array_dtype": "little-endian int64",
        "archive": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "split_methods": list(SPLIT_METHODS),
        "full_unlabeled_pool_methods_without_inner_split": list(FULL_POOL_METHODS),
        "tuning_rule": "LSTM-AE and USAD use the dataset seed-42 split for candidate selection; selected configurations are fixed before formal repeated-seed evaluation.",
        "records": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"wrote {archive_path}")
    print(f"split records: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
