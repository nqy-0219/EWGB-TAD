"""Thin, label-free adapter around the official MST-OATD implementation."""

from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phase3_baseline_common import (
    EXTERNAL_BASELINES_ROOT,
    grid_tokenize,
    serializable_config,
    set_deterministic_seed,
    validate_trajectories,
)


@dataclass(frozen=True)
class MSTOATDAdapterConfig:
    seed: int = 42
    grid_side: int = 8
    pretrain_epochs: int = 1
    epochs: int = 1
    batch_size: int = 32
    embedding_size: int = 128
    hidden_size: int = 32
    n_cluster: int = 3
    pretrain_lr_s: float = 2e-3
    pretrain_lr_t: float = 2e-3
    lr_s: float = 3e-4
    lr_t: float = 3e-4
    s1_size: int = 2
    s2_size: int = 4
    sampling_interval_seconds: int = 15
    torch_threads: int = 1
    device: str = "cpu"
    runtime_parent: str | None = None


def _load_official_module() -> tuple[Any, str]:
    repo = EXTERNAL_BASELINES_ROOT / "MST-OATD"
    if not (repo / "mst_oatd_trainer.py").exists():
        raise FileNotFoundError(f"official MST-OATD source not found: {repo}")

    for name in ("utils", "temporal", "mst_oatd", "mst_oatd_trainer", "logging_set"):
        sys.modules.pop(name, None)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    try:
        trainer_module = importlib.import_module("mst_oatd_trainer")
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous_bytecode_setting
    return trainer_module, str(repo)


def _grid_graph_with_padding(grid_side: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    rows: list[int] = []
    cols: list[int] = []
    for row in range(grid_side):
        for col in range(grid_side):
            node = row * grid_side + col
            rows.append(node)
            cols.append(node)
            if row + 1 < grid_side:
                other = (row + 1) * grid_side + col
                rows.extend((node, other))
                cols.extend((other, node))
            if col + 1 < grid_side:
                other = row * grid_side + col + 1
                rows.extend((node, other))
                cols.extend((other, node))
    n_cells = grid_side * grid_side
    adjacency_cells = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_cells),
    ).tocsr()
    degree = np.asarray(adjacency_cells.sum(axis=1)).reshape(-1)
    degree_norm = sparse.diags(1.0 / np.sqrt(np.maximum(degree, 1e-10))).tocsr()

    adjacency = sparse.block_diag((sparse.csr_matrix([[1.0]]), adjacency_cells), format="csr")
    d_norm = sparse.block_diag((sparse.csr_matrix([[1.0]]), degree_norm), format="csr")
    return adjacency, d_norm


def _position_time_vectors(n_samples: int, sequence_length: int, interval_seconds: int) -> np.ndarray:
    seconds = np.arange(sequence_length, dtype=np.int64) * interval_seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    one = np.stack(
        (
            hours,
            minutes,
            secs,
            np.full(sequence_length, 2013, dtype=np.int64),
            np.ones(sequence_length, dtype=np.int64),
            np.ones(sequence_length, dtype=np.int64),
        ),
        axis=1,
    )
    return np.repeat(one[None, :, :], n_samples, axis=0)


def _to_official_sequences(tokens: np.ndarray, times: np.ndarray) -> list[list[list[Any]]]:
    return [
        [[int(tokens[i, j]), times[i, j].astype(int).tolist()] for j in range(tokens.shape[1])]
        for i in range(tokens.shape[0])
    ]


@contextlib.contextmanager
def _runtime_workspace(repo: Path, grid_side: int, runtime_parent: str | None) -> Iterator[Path]:
    if runtime_parent is not None:
        Path(runtime_parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ewgb_phase3_mstoatd_",
        dir=runtime_parent,
    ) as temp_dir:
        root = Path(temp_dir)
        (root / "data" / "phase3_smoke").mkdir(parents=True)
        (root / "temporal_model").mkdir(parents=True)
        (root / "models").mkdir()
        (root / "logs").mkdir()
        adjacency, d_norm = _grid_graph_with_padding(grid_side)
        sparse.save_npz(root / "data" / "phase3_smoke" / "adj.npz", adjacency)
        sparse.save_npz(root / "data" / "phase3_smoke" / "d_norm.npz", d_norm)
        shutil.copy2(repo / "temporal_model" / "emb_128.pth", root / "temporal_model" / "emb_128.pth")
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield root
        finally:
            os.chdir(previous)


def _official_scores(trainer: Any, loader: DataLoader, module: Any) -> np.ndarray:
    trainer.MST_OATD_S.eval()
    trainer.MST_OATD_T.eval()
    detector_loss = nn.CrossEntropyLoss(reduction="none")
    all_spatial: list[torch.Tensor] = []
    all_temporal: list[torch.Tensor] = []

    with torch.no_grad():
        for trajs, times, seq_lengths in loader:
            batch_size = len(trajs)
            mask = module.make_mask(module.make_len_mask(trajs)).to(trainer.device)
            time_tokens = module.time_convert(times, trainer.time_interval)
            cluster_spatial: list[torch.Tensor] = []
            cluster_temporal: list[torch.Tensor] = []
            for cluster in range(trainer.n_cluster):
                output_s, _, _, _ = trainer.MST_OATD_S(
                    trajs, times, seq_lengths, batch_size, "test", cluster
                )
                log_likelihood_s = -detector_loss(
                    output_s.reshape(-1, output_s.shape[-1]),
                    trajs.to(trainer.device).reshape(-1),
                )
                likelihood_s = torch.exp(
                    torch.sum(mask * log_likelihood_s.reshape(batch_size, -1), dim=-1)
                    / torch.sum(mask, dim=1)
                )

                output_t, _, _, _ = trainer.MST_OATD_T(
                    trajs, times, seq_lengths, batch_size, "test", cluster
                )
                log_likelihood_t = -detector_loss(
                    output_t.reshape(-1, output_t.shape[-1]),
                    time_tokens.to(trainer.device).reshape(-1),
                )
                likelihood_t = torch.exp(
                    torch.sum(mask * log_likelihood_t.reshape(batch_size, -1), dim=-1)
                    / torch.sum(mask, dim=1)
                )
                cluster_spatial.append(likelihood_s.unsqueeze(0))
                cluster_temporal.append(likelihood_t.unsqueeze(0))

            all_spatial.append(torch.cat(cluster_spatial).max(0)[0])
            all_temporal.append(torch.cat(cluster_temporal).max(0)[0])

    spatial = torch.cat(all_spatial, dim=0)
    temporal = torch.cat(all_temporal, dim=0)
    return (1.0 - spatial * temporal).cpu().numpy().astype(np.float64)


def fit_score_mstoatd(
    trajectories: Any,
    config: MSTOATDAdapterConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit on unlabeled trajectories and return official-orientation scores."""
    overall_started = time.perf_counter()
    cfg = config or MSTOATDAdapterConfig()
    array = validate_trajectories(trajectories)
    if array.shape[1] % cfg.s1_size or array.shape[1] % cfg.s2_size:
        raise ValueError("sequence length must be divisible by both MST-OATD scale sizes")
    if cfg.embedding_size != 128:
        raise ValueError("the official temporal embedding checkpoint requires embedding_size=128")
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MST-OATD requested CUDA but CUDA is unavailable")
    set_deterministic_seed(cfg.seed, cfg.torch_threads)
    module, repo_string = _load_official_module()
    repo = Path(repo_string)

    spatial_tokens, tokenization = grid_tokenize(array, cfg.grid_side, token_offset=1)
    time_vectors = _position_time_vectors(
        len(array), array.shape[1], cfg.sampling_interval_seconds
    )
    sequences = _to_official_sequences(spatial_tokens, time_vectors)
    dataset = module.MyDataset(sequences)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        collate_fn=module.collate_fn,
        generator=generator,
        pin_memory=cfg.device.startswith("cuda"),
        num_workers=0,
    )
    score_loader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=False,
        collate_fn=module.collate_fn,
        pin_memory=cfg.device.startswith("cuda"),
        num_workers=0,
    )

    seconds = np.arange(array.shape[1], dtype=np.int64) * cfg.sampling_interval_seconds
    time_token_size = int(seconds.max() // cfg.sampling_interval_seconds + 1)
    args = SimpleNamespace(
        embedding_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        n_cluster=cfg.n_cluster,
        pretrain_lr_s=cfg.pretrain_lr_s,
        pretrain_lr_t=cfg.pretrain_lr_t,
        lr_s=cfg.lr_s,
        lr_t=cfg.lr_t,
        device=cfg.device,
        dataset="phase3_smoke",
        s1_size=cfg.s1_size,
        s2_size=cfg.s2_size,
    )
    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()
    data_preparation_seconds = time.perf_counter() - overall_started
    started = time.perf_counter()
    with _runtime_workspace(repo, cfg.grid_side, cfg.runtime_parent):
        model_setup_started = time.perf_counter()
        trainer = module.train_mst_oatd(
            s_token_size=1 + cfg.grid_side * cfg.grid_side,
            t_token_size=time_token_size,
            labels=np.zeros(len(array), dtype=np.int64),
            train_loader=train_loader,
            outliers_loader=score_loader,
            args=args,
        )
        trainer.time_interval = cfg.sampling_interval_seconds
        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        model_setup_seconds = time.perf_counter() - model_setup_started
        pretraining_started = time.perf_counter()
        for epoch in range(cfg.pretrain_epochs):
            trainer.pretrain(epoch)
        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        pretraining_seconds = time.perf_counter() - pretraining_started
        restore_started = time.perf_counter()
        if cfg.pretrain_epochs:
            trainer.load_pretrained()
        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        checkpoint_restore_seconds = time.perf_counter() - restore_started
        training_started = time.perf_counter()
        for epoch in range(cfg.epochs):
            trainer.train(epoch)
        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        scoring_started = time.perf_counter()
        score_array = _official_scores(trainer, score_loader, module).reshape(-1)
        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        scoring_seconds = time.perf_counter() - scoring_started
        parameter_count = int(
            sum(parameter.numel() for parameter in trainer.MST_OATD_S.parameters())
            + sum(parameter.numel() for parameter in trainer.MST_OATD_T.parameters())
        )
        for handler in list(trainer.logger.handlers):
            handler.flush()
            handler.close()
            trainer.logger.removeHandler(handler)
    runtime = time.perf_counter() - started

    if score_array.shape != (len(array),):
        raise RuntimeError(f"MST-OATD returned {score_array.shape}, expected {(len(array),)}")
    if not np.isfinite(score_array).all():
        raise RuntimeError("MST-OATD returned non-finite anomaly scores")

    metadata = {
        "method": "MST-OATD",
        "official_repository": repo_string,
        "official_model_file": str(repo / "mst_oatd.py"),
        "official_training_file": str(repo / "mst_oatd_trainer.py"),
        "labels_consumed_during_fit": False,
        "score_direction": "higher_is_more_anomalous",
        "score_definition": "official 1 - max-cluster spatial likelihood * max-cluster temporal likelihood",
        "n_samples": int(len(array)),
        "sequence_length": int(array.shape[1]),
        "pretrain_epochs_completed": int(cfg.pretrain_epochs),
        "epochs_completed": int(cfg.epochs),
        "runtime_seconds": runtime,
        "data_preparation_seconds": data_preparation_seconds,
        "model_setup_seconds": model_setup_seconds,
        "pretraining_seconds": pretraining_seconds,
        "checkpoint_restore_seconds": checkpoint_restore_seconds,
        "training_seconds": training_seconds,
        "scoring_seconds": scoring_seconds,
        "total_fit_score_seconds": time.perf_counter() - overall_started,
        "parameter_count": parameter_count,
        "device": cfg.device,
        "adapter_config": serializable_config(cfg),
        "tokenization": tokenization,
        "temporal_input": {
            "type": "position-derived pseudo-time",
            "sampling_interval_seconds": int(cfg.sampling_interval_seconds),
            "shared_across_trajectories": True,
            "reason": "the predefined four-dataset coordinate protocol does not expose comparable timestamps",
        },
    }
    return score_array, metadata
