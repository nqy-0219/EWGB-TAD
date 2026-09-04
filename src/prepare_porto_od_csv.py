"""Convert the official Porto taxi trajectory CSV to EWGB-TAD OD records."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {"TAXI_ID", "TRIP_ID", "TIMESTAMP", "POLYLINE"}
OUTPUT_COLUMNS = (
    "taxi_id",
    "trajectory_id",
    "timestamp",
    "source_point",
    "target_point",
)


def wkt_point(point: list[float]) -> str:
    longitude, latitude = (float(point[0]), float(point[1]))
    return f"POINT({longitude:.6f} {latitude:.6f})"


def convert_porto_csv(source: Path, destination: Path) -> tuple[int, int]:
    """Write valid first/last trajectory points in deterministic source-row order."""
    csv.field_size_limit(1024 * 1024 * 16)
    destination.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with source.open("r", newline="", encoding="utf-8") as input_handle:
        reader = csv.DictReader(input_handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Porto CSV is missing columns: {sorted(missing)}")
        with destination.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for row in reader:
                try:
                    polyline = json.loads(row["POLYLINE"])
                    if len(polyline) < 2:
                        raise ValueError("trajectory has fewer than two points")
                    endpoints = np.asarray([polyline[0], polyline[-1]], dtype=float)
                    if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
                        raise ValueError("trajectory endpoints are invalid")
                    timestamp = datetime.fromtimestamp(
                        int(row["TIMESTAMP"]), tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except (TypeError, ValueError, json.JSONDecodeError):
                    skipped += 1
                    continue
                writer.writerow(
                    {
                        "taxi_id": row["TAXI_ID"],
                        "trajectory_id": row["TRIP_ID"],
                        "timestamp": timestamp,
                        "source_point": wkt_point(endpoints[0].tolist()),
                        "target_point": wkt_point(endpoints[1].tolist()),
                    }
                )
                kept += 1
    return kept, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kept, skipped = convert_porto_csv(args.input, args.output)
    print(f"wrote {kept} OD records; skipped {skipped} invalid rows")


if __name__ == "__main__":
    main()
