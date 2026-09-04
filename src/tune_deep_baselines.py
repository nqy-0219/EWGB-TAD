"""Unsupervised, dataset-specific tuning for the local deep baselines.

The tuner deliberately uses only an inner split of the unlabeled trajectory
pool.  Labels are never loaded.  The selected configuration is fixed per
dataset and can then be used for every seed of that dataset in the formal
benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from baselines_dl import TrajLSTMAEDetector, USADDetector
from phase4_common import DATASET_SEEDS, load_cache, set_reproducible, sha256_file, slug


TUNING_SEED = 42
MAX_EPOCHS = 80
VALIDATION_FRACTION = 0.10
PATIENCE = 8
MIN_EPOCHS = 10
MIN_DELTA = 1e-5


LSTM_CANDIDATES = [
    {"hidden_dim": 32, "latent_dim": 16, "num_layers": 1, "dropout": 0.0, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 32, "latent_dim": 16, "num_layers": 1, "dropout": 0.0, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 16, "num_layers": 1, "dropout": 0.1, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 32, "num_layers": 1, "dropout": 0.1, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 32, "num_layers": 2, "dropout": 0.1, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 32, "num_layers": 2, "dropout": 0.1, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 128, "latent_dim": 32, "num_layers": 1, "dropout": 0.1, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 128, "latent_dim": 32, "num_layers": 2, "dropout": 0.1, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 32, "latent_dim": 32, "num_layers": 2, "dropout": 0.0, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 16, "num_layers": 2, "dropout": 0.0, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 128, "latent_dim": 16, "num_layers": 1, "dropout": 0.0, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 128, "latent_dim": 32, "num_layers": 2, "dropout": 0.1, "learning_rate": 1e-3, "batch_size": 128},
]

USAD_CANDIDATES = [
    {"hidden_dim": 32, "latent_dim": 8, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 32, "latent_dim": 8, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 32, "latent_dim": 16, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 8, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 16, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 16, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 32, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 128, "latent_dim": 16, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 128, "latent_dim": 32, "learning_rate": 3e-4, "batch_size": 128},
    {"hidden_dim": 128, "latent_dim": 32, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 32, "latent_dim": 32, "learning_rate": 1e-3, "batch_size": 128},
    {"hidden_dim": 64, "latent_dim": 8, "learning_rate": 3e-4, "batch_size": 128},
]


def detector_for(method: str, config: dict, device: str):
    common = {
        "epochs": MAX_EPOCHS,
        "batch_size": int(config["batch_size"]),
        "lr": float(config["learning_rate"]),
        "device": device,
        "validation_fraction": VALIDATION_FRACTION,
        "patience": PATIENCE,
        "min_epochs": MIN_EPOCHS,
        "min_delta": MIN_DELTA,
    }
    if method == "LSTM-AE":
        return TrajLSTMAEDetector(
            **common,
            hidden_dim=int(config["hidden_dim"]),
            latent_dim=int(config["latent_dim"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
        )
    if method == "USAD":
        return USADDetector(
            **common,
            hidden_dim=int(config["hidden_dim"]),
            latent_dim=int(config["latent_dim"]),
        )
    raise ValueError(method)


def run_tuning(dataset: str, method: str, output_root: Path, device: str, limit: int | None = None) -> dict:
    arrays, cache_metadata = load_cache(dataset, TUNING_SEED)
    trajectories = np.asarray(arrays["trajectories"], dtype=np.float32)
    features = np.asarray(arrays["canonical_features"], dtype=np.float32)
    candidates = (LSTM_CANDIDATES if method == "LSTM-AE" else USAD_CANDIDATES)
    if limit is not None:
        candidates = candidates[:limit]
    records = []
    for index, config in enumerate(candidates, start=1):
        started = time.perf_counter()
        set_reproducible(TUNING_SEED, torch_threads=2)
        detector = detector_for(method, config, device)
        if method == "LSTM-AE":
            detector.fit_score(features, trajectories, seed=TUNING_SEED, score_population=True)
        else:
            detector.fit_score(features, seed=TUNING_SEED, score_population=True)
        elapsed = time.perf_counter() - started
        timing = detector.timing_
        records.append(
            {
                "dataset": dataset,
                "method": method,
                "tuning_seed": TUNING_SEED,
                "candidate_id": index,
                "configuration": config,
                "best_validation_loss": float(timing["best_validation_loss"]),
                "best_epoch": int(timing["best_epoch"]),
                "epochs_completed": int(timing["epochs_completed"]),
                "early_stopped": bool(timing["early_stopped"]),
                "training_seconds": float(timing["training_seconds"]),
                "elapsed_seconds": float(elapsed),
                "validation_fraction": VALIDATION_FRACTION,
                "patience": PATIENCE,
                "min_epochs": MIN_EPOCHS,
                "min_delta": MIN_DELTA,
                "labels_loaded": False,
            }
        )
        print(
            f"TUNE {dataset} {method} candidate={index}/{len(candidates)} "
            f"val_loss={timing['best_validation_loss']:.8f} "
            f"best_epoch={timing['best_epoch']} time={elapsed:.1f}s",
            flush=True,
        )

    records.sort(key=lambda item: (item["best_validation_loss"], item["candidate_id"]))
    selected = records[0]
    destination = output_root / "deep_tuning" / slug(dataset) / slug(method)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "selection_rule": "minimum inner validation reconstruction loss; tie-break by candidate id",
        "dataset": dataset,
        "method": method,
        "tuning_seed": TUNING_SEED,
        "cache_metadata": cache_metadata,
        "cache_sha256": cache_metadata.get("sha256"),
        "labels_loaded": False,
        "validation_split": {
            "fraction": VALIDATION_FRACTION,
            "source": "deterministic permutation from numpy RandomState(seed=42)",
            "fit_scaler_on": "inner training partition only",
        },
        "budget": {
            "candidate_count": len(records),
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "min_epochs": MIN_EPOCHS,
            "min_delta": MIN_DELTA,
        },
        "candidate_records_sorted_by_validation_loss": records,
        "selected_configuration": selected,
    }
    json_path = destination / "tuning_result.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    csv_path = destination / "candidate_results.csv"
    fieldnames = [
        "dataset", "method", "tuning_seed", "candidate_id", "hidden_dim", "latent_dim",
        "num_layers", "dropout", "learning_rate", "batch_size", "best_validation_loss",
        "best_epoch", "epochs_completed", "early_stopped", "training_seconds", "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fieldnames}
            row.update(record["configuration"])
            writer.writerow(row)
    return payload


def write_summary(payloads: list[dict], output_root: Path) -> None:
    summary_dir = output_root / "deep_tuning"
    summary_dir.mkdir(parents=True, exist_ok=True)
    selected_path = summary_dir / "selected_configurations.json"
    selected: dict[str, dict[str, dict]] = {}
    for payload in sorted(payloads, key=lambda item: (item["dataset"], item["method"])):
        selected.setdefault(payload["dataset"], {})[payload["method"]] = payload[
            "selected_configuration"
        ]
    selected_path.write_text(json.dumps(selected, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    rows = []
    for payload in sorted(payloads, key=lambda item: (item["dataset"], item["method"])):
        chosen = payload["selected_configuration"]
        row = {
            "dataset": payload["dataset"],
            "method": payload["method"],
            "tuning_seed": payload["tuning_seed"],
            "candidate_count": payload["budget"]["candidate_count"],
            "best_validation_loss": chosen["best_validation_loss"],
            "best_epoch": chosen["best_epoch"],
            "epochs_completed": chosen["epochs_completed"],
            "early_stopped": chosen["early_stopped"],
            **chosen["configuration"],
        }
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (summary_dir / "selected_configurations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_existing_payloads(output_root: Path) -> list[dict]:
    payloads = []
    for dataset in DATASET_SEEDS:
        for method in ("LSTM-AE", "USAD"):
            path = output_root / "deep_tuning" / slug(dataset) / slug(method) / "tuning_result.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete" or payload.get("labels_loaded") is not False:
                raise ValueError(f"invalid tuning result: {path}")
            payloads.append(payload)
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_SEEDS), default=list(DATASET_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=["LSTM-AE", "USAD"], default=["LSTM-AE", "USAD"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="rebuild the aggregate files from completed tuning_result.json files",
    )
    args = parser.parse_args()
    if args.limit_candidates is not None and args.limit_candidates < 1:
        raise ValueError("--limit-candidates must be positive")
    payloads = []
    if not args.summarize_existing:
        for dataset in args.datasets:
            for method in args.methods:
                payloads.append(
                    run_tuning(dataset, method, args.output_root, args.device, args.limit_candidates)
                )
    existing = collect_existing_payloads(args.output_root)
    if existing:
        payloads = existing
    if not payloads:
        raise RuntimeError("no completed tuning results were found")
    write_summary(payloads, args.output_root)
    print(f"TUNING_SUMMARY_COMPLETE records={len(payloads)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
