"""Build deterministic Phase 4 main-benchmark worker manifests."""

from __future__ import annotations

import argparse

from phase4_common import (
    CPU_METHODS,
    DATASET_SEEDS,
    GPU_METHODS,
    MANIFEST_ROOT,
    atomic_write_json,
)


def jobs_for(methods: list[str]) -> list[dict]:
    return [
        {"dataset": dataset, "method": method, "seed": seed}
        for dataset, seeds in DATASET_SEEDS.items()
        for seed in seeds
        for method in methods
    ]


def shard_jobs(jobs: list[dict], n_shards: int) -> list[list[dict]]:
    shards = [[] for _ in range(n_shards)]
    for index, job in enumerate(jobs):
        shards[index % n_shards].append(job)
    return shards


GPU_JOB_WEIGHTS = {
    "LM-TAD": 50,
    "MST-OATD": 8,
    "USAD": 5,
    "LSTM-AE": 4,
}


def weighted_gpu_shards(jobs: list[dict], n_shards: int) -> tuple[list[list[dict]], list[int]]:
    """Greedily balance deterministic GPU shards by expected training cost."""
    shards = [[] for _ in range(n_shards)]
    loads = [0 for _ in range(n_shards)]
    ordered = sorted(
        enumerate(jobs),
        key=lambda item: (-GPU_JOB_WEIGHTS[item[1]["method"]], item[0]),
    )
    for _, job in ordered:
        worker = min(range(n_shards), key=lambda index: (loads[index], index))
        shards[worker].append(job)
        loads[worker] += GPU_JOB_WEIGHTS[job["method"]]
    return shards, loads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--gpu-workers", type=int, default=8)
    args = parser.parse_args()
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    cpu_jobs = jobs_for(CPU_METHODS)
    gpu_jobs = jobs_for(GPU_METHODS)
    atomic_write_json(MANIFEST_ROOT / "main_cpu_all.json", cpu_jobs)
    atomic_write_json(MANIFEST_ROOT / "main_gpu_all.json", gpu_jobs)
    for index, shard in enumerate(shard_jobs(cpu_jobs, args.cpu_workers)):
        atomic_write_json(MANIFEST_ROOT / f"main_cpu_worker_{index}.json", shard)
    gpu_shards, gpu_estimated_loads = weighted_gpu_shards(gpu_jobs, args.gpu_workers)
    for index, shard in enumerate(gpu_shards):
        atomic_write_json(MANIFEST_ROOT / f"main_gpu_worker_{index}.json", shard)

    preflight = [
        {"dataset": "Synthetic", "method": method, "seed": 42}
        for method in CPU_METHODS + GPU_METHODS
    ]
    atomic_write_json(MANIFEST_ROOT / "preflight.json", preflight)
    atomic_write_json(
        MANIFEST_ROOT / "manifest_summary.json",
        {
            "cpu_jobs": len(cpu_jobs),
            "gpu_jobs": len(gpu_jobs),
            "total_jobs": len(cpu_jobs) + len(gpu_jobs),
            "cpu_workers": args.cpu_workers,
            "gpu_workers": args.gpu_workers,
            "gpu_job_weights": GPU_JOB_WEIGHTS,
            "gpu_estimated_worker_loads": gpu_estimated_loads,
        },
    )
    print(f"wrote {len(cpu_jobs) + len(gpu_jobs)} main jobs")


if __name__ == "__main__":
    main()
