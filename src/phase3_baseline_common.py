"""Shared deterministic utilities for Phase 3 trajectory baselines."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_BASELINES_ROOT = SOURCE_ROOT / "external_baselines_phase3"


def set_deterministic_seed(seed: int, torch_threads: int = 1) -> None:
    """Set all locally relevant random generators without using labels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_trajectories(trajectories: Any) -> np.ndarray:
    array = np.asarray(trajectories, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError("trajectories must have shape (n_samples, sequence_length, 2)")
    if array.shape[0] < 4 or array.shape[1] < 4:
        raise ValueError("at least four trajectories and four points per trajectory are required")
    if not np.isfinite(array).all():
        raise ValueError("trajectories contain non-finite coordinates")
    return array


def grid_tokenize(
    trajectories: Any,
    grid_side: int,
    token_offset: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map all coordinates to one globally fitted square grid."""
    array = validate_trajectories(trajectories)
    if grid_side < 2:
        raise ValueError("grid_side must be at least 2")

    flat = array.reshape(-1, 2).astype(np.float64)
    lower = flat.min(axis=0)
    upper = flat.max(axis=0)
    span = upper - lower
    constant = span <= 1e-12
    span[constant] = 1.0

    normalized = (array.astype(np.float64) - lower) / span
    indices = np.floor(normalized * grid_side).astype(np.int64)
    indices = np.clip(indices, 0, grid_side - 1)
    tokens = token_offset + indices[:, :, 0] * grid_side + indices[:, :, 1]

    metadata = {
        "grid_side": int(grid_side),
        "token_offset": int(token_offset),
        "coordinate_min": lower.tolist(),
        "coordinate_max": upper.tolist(),
        "constant_coordinate_axes": np.flatnonzero(constant).astype(int).tolist(),
        "n_spatial_tokens": int(grid_side * grid_side),
    }
    return tokens, metadata


def serializable_config(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"unsupported configuration type: {type(config)!r}")

