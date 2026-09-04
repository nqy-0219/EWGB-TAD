"""Execute one Phase 4 analysis manifest with process isolation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))
    runner = Path(__file__).resolve().with_name("phase4_run_analysis_job.py")
    failures = 0
    for index, job in enumerate(jobs, start=1):
        command = [
            sys.executable,
            str(runner),
            "--analysis",
            job["analysis"],
            "--variant",
            job["variant"],
            "--dataset",
            job["dataset"],
            "--seed",
            str(job["seed"]),
        ]
        if args.force:
            command.append("--force")
        print(f"ANALYSIS_WORKER {index}/{len(jobs)} {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures += 1
            if not args.keep_going:
                return completed.returncode
    print(f"ANALYSIS_WORKER_DONE jobs={len(jobs)} failures={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
