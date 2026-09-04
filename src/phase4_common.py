"""Shared paths and immutable-artifact helpers for Phase 4 experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PHASE4_ROOT = Path(os.environ.get("EWGB_PHASE4_ROOT", SOURCE_ROOT / "results" / "phase4"))
CACHE_ROOT = PHASE4_ROOT / "dataset_cache"
RAW_ROOT = PHASE4_ROOT / "raw"
MANIFEST_ROOT = PHASE4_ROOT / "manifests"
SUMMARY_ROOT = PHASE4_ROOT / "summary"

DATASET_SEEDS = {
    "Synthetic": [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144],
    "Grid-Network": [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144],
    "Porto-derived": [42, 123, 456, 789, 1024],
    "GeoLife": [42, 123, 456, 789, 1024],
}

CPU_METHODS = [
    "EWGB-TAD",
    "IForest",
    "ECOD",
    "iBoost-ODE",
    "CoMadOut",
    "Shape-KNN",
    "SegmentOD",
    "TADS",
    "Profile-TAD",
]
GPU_METHODS = ["LSTM-AE", "USAD", "LM-TAD", "MST-OATD"]
MAIN_METHODS = CPU_METHODS + GPU_METHODS


def canonical_shape_view(trajectories: np.ndarray | list[np.ndarray], n_components: int = 10) -> np.ndarray:
    """Build the trajectory-shape view with the same PCA path as the detector."""
    flat = np.stack(
        [np.asarray(trajectory, dtype=float).flatten() for trajectory in trajectories]
    )
    components = min(int(n_components), flat.shape[0] - 1, flat.shape[1])
    if components < 1:
        raise ValueError("at least two trajectories are required for the shape view")
    return PCA(n_components=components).fit_transform(flat)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def cache_path(dataset: str, seed: int) -> Path:
    return CACHE_ROOT / slug(dataset) / f"seed_{seed}.npz"


def cache_metadata_path(dataset: str, seed: int) -> Path:
    return CACHE_ROOT / slug(dataset) / f"seed_{seed}.json"


def job_directory(dataset: str, method: str, seed: int) -> Path:
    return RAW_ROOT / slug(dataset) / slug(method) / f"seed_{seed}"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_marker_artifact(marker_path: Path, marker: dict[str, Any], key: str) -> Path | None:
    """Resolve an artifact even when a copied server marker has an old absolute path."""
    recorded = marker.get(key)
    if recorded:
        path = Path(str(recorded))
        if path.exists():
            return path

    attempt = marker.get("attempt")
    if attempt:
        candidate = marker_path.parent / str(attempt) / Path(str(recorded or key)).name
        if candidate.exists():
            return candidate

    name = Path(str(recorded or key)).name
    candidates = sorted(
        marker_path.parent.glob(f"attempt_*/{name}"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_reproducible(seed: int, torch_threads: int = 1) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment_record(device: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_requested": device,
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        requested = torch.device(device)
        gpu_index = requested.index if requested.index is not None else torch.cuda.current_device()
        record["gpu_index"] = int(gpu_index)
        record["gpu_name"] = torch.cuda.get_device_name(gpu_index)
        record["gpu_count_visible"] = torch.cuda.device_count()
    return record


def load_cache(dataset: str, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = cache_path(dataset, seed)
    metadata_path = cache_metadata_path(dataset, seed)
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"missing Phase 4 dataset cache for {dataset}, seed={seed}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    current_hash = sha256_file(npz_path)
    if current_hash != metadata.get("sha256"):
        raise RuntimeError(f"dataset cache hash mismatch: {npz_path}")
    with np.load(npz_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return arrays, metadata
