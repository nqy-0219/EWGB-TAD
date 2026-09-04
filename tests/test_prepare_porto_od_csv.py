from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = SOURCE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prepare_porto_od_csv import convert_porto_csv


class PortoODConversionTests(unittest.TestCase):
    def test_converter_preserves_valid_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.csv"
            destination = root / "porto_trajectories_all.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["TAXI_ID", "TRIP_ID", "TIMESTAMP", "POLYLINE"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "TAXI_ID": "7",
                        "TRIP_ID": "11",
                        "TIMESTAMP": "1372636858",
                        "POLYLINE": json.dumps([[-8.6, 41.1], [-8.7, 41.2]]),
                    }
                )
                writer.writerow(
                    {
                        "TAXI_ID": "8",
                        "TRIP_ID": "12",
                        "TIMESTAMP": "1372636859",
                        "POLYLINE": "[]",
                    }
                )
            kept, skipped = convert_porto_csv(source, destination)
            self.assertEqual((kept, skipped), (1, 1))
            with destination.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["source_point"], "POINT(-8.600000 41.100000)")
            self.assertEqual(rows[0]["target_point"], "POINT(-8.700000 41.200000)")


if __name__ == "__main__":
    unittest.main()
