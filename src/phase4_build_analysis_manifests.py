"""Build deterministic manifests for Phase 4 component and robustness jobs."""

from __future__ import annotations

import argparse
import itertools

from phase4_common import DATASET_SEEDS, MANIFEST_ROOT, atomic_write_json


COMPONENT_VARIANTS = [
    f"g{granular}_l{local}_f{fusion}"
    for granular, local, fusion in itertools.product((0, 1), repeat=3)
]
VIEW_VARIANTS = [
    "spatial_path",
    "kinematic",
    "trajectory_shape",
    "spatial_path+kinematic",
    "spatial_path+trajectory_shape",
    "kinematic+trajectory_shape",
    "spatial_path+kinematic+trajectory_shape",
]
SENSITIVITY_VARIANTS = [
    "min4",
    "min8",
    "min16",
    "min32",
]


def build_jobs(analysis: str, variants: list[str]) -> list[dict]:
    return [
        {"analysis": analysis, "variant": variant, "dataset": dataset, "seed": seed}
        for dataset, seeds in DATASET_SEEDS.items()
        for seed in seeds
        for variant in variants
    ]


def shard(jobs: list[dict], workers: int) -> list[list[dict]]:
    output = [[] for _ in range(workers)]
    for index, job in enumerate(jobs):
        output[index % workers].append(job)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    all_jobs = []
    for analysis, variants in (
        ("component", COMPONENT_VARIANTS),
        ("view", VIEW_VARIANTS),
        ("sensitivity", SENSITIVITY_VARIANTS),
    ):
        jobs = build_jobs(analysis, variants)
        all_jobs.extend(jobs)
        atomic_write_json(MANIFEST_ROOT / f"{analysis}_all.json", jobs)
    for index, worker_jobs in enumerate(shard(all_jobs, args.workers)):
        atomic_write_json(MANIFEST_ROOT / f"analysis_worker_{index}.json", worker_jobs)
    atomic_write_json(
        MANIFEST_ROOT / "analysis_manifest_summary.json",
        {
            "component_jobs": len(build_jobs("component", COMPONENT_VARIANTS)),
            "view_jobs": len(build_jobs("view", VIEW_VARIANTS)),
            "sensitivity_jobs": len(build_jobs("sensitivity", SENSITIVITY_VARIANTS)),
            "total_jobs": len(all_jobs),
            "workers": args.workers,
        },
    )
    print(f"wrote {len(all_jobs)} analysis jobs")


if __name__ == "__main__":
    main()
