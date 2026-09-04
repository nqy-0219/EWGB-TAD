"""Execute a Phase 4 manifest sequentially with per-job process isolation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", choices=("preflight", "full"), default="full")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))
    runner = Path(__file__).resolve().with_name("phase4_run_job.py")
    failures = 0
    for index, job in enumerate(jobs, start=1):
        command = [
            sys.executable,
            str(runner),
            "--dataset",
            job["dataset"],
            "--method",
            job["method"],
            "--seed",
            str(job["seed"]),
            "--device",
            args.device,
            "--profile",
            args.profile,
        ]
        print(f"WORKER {index}/{len(jobs)} {' '.join(command)}", flush=True)
        completed = subprocess.run(command, env=os.environ.copy(), check=False)
        if completed.returncode != 0:
            failures += 1
            if not args.keep_going:
                return completed.returncode
    print(f"WORKER_DONE jobs={len(jobs)} failures={failures}", flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

