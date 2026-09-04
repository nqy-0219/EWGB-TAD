"""Thin, label-free adapter around the official LM-TAD implementation."""

from __future__ import annotations

import contextlib
import importlib
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from phase3_baseline_common import (
    EXTERNAL_BASELINES_ROOT,
    grid_tokenize,
    serializable_config,
    set_deterministic_seed,
    validate_trajectories,
)


@dataclass(frozen=True)
class LMTADAdapterConfig:
    seed: int = 42
    grid_side: int = 12
    validation_fraction: float = 0.2
    epochs: int = 3
    patience: int | None = 2
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    use_cosine_decay: bool = False
    warmup_steps: int = 0
    lr_decay_steps: int = 0
    min_learning_rate: float = 3e-5
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1
    torch_threads: int = 1
    device: str = "cpu"
    mixed_precision: bool = False
    score_batch_size: int = 128


def _load_official_modules() -> tuple[Any, Any, str]:
    repo = EXTERNAL_BASELINES_ROOT / "LMTAD"
    code_dir = repo / "code"
    if not (code_dir / "models" / "LMTAD.py").exists():
        raise FileNotFoundError(f"official LM-TAD source not found: {code_dir}")

    for name in ("utils", "models", "models.LMTAD", "eval_lm", "plot_utils"):
        sys.modules.pop(name, None)
    plot_stub = types.ModuleType("plot_utils")
    for function_name in (
        "plot_agent_surprisal_rate",
        "plot_metrics_pattern_of_life",
        "plot_agent_perlexity_over_date",
    ):
        setattr(plot_stub, function_name, lambda *args, **kwargs: None)
    sys.modules["plot_utils"] = plot_stub
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_dir))
    try:
        model_module = importlib.import_module("models.LMTAD")
        eval_module = importlib.import_module("eval_lm")
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous_bytecode_setting
    return model_module, eval_module, str(repo)


def _split_indices(n_samples: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_samples)
    n_validation = max(1, int(round(n_samples * validation_fraction)))
    n_validation = min(n_validation, n_samples - 1)
    return order[n_validation:], order[:n_validation]


def _learning_rate(step: int, config: LMTADAdapterConfig) -> float:
    if not config.use_cosine_decay:
        return config.learning_rate
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if config.lr_decay_steps <= config.warmup_steps:
        raise ValueError("lr_decay_steps must exceed warmup_steps when cosine decay is enabled")
    if step > config.lr_decay_steps:
        return config.min_learning_rate
    ratio = (step - config.warmup_steps) / (config.lr_decay_steps - config.warmup_steps)
    ratio = min(max(ratio, 0.0), 1.0)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_learning_rate + coefficient * (
        config.learning_rate - config.min_learning_rate
    )


def fit_score_lmtad(
    trajectories: Any,
    config: LMTADAdapterConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit on unlabeled trajectories and return higher-is-more-anomalous scores."""
    overall_started = time.perf_counter()
    cfg = config or LMTADAdapterConfig()
    array = validate_trajectories(trajectories)
    set_deterministic_seed(cfg.seed, cfg.torch_threads)
    model_module, eval_module, repo = _load_official_modules()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("LM-TAD requested CUDA but CUDA is unavailable")
    device = torch.device(cfg.device)
    use_amp = bool(cfg.mixed_precision and device.type == "cuda")

    cell_tokens, tokenization = grid_tokenize(array, cfg.grid_side, token_offset=3)
    start = np.full((len(array), 1), 1, dtype=np.int64)
    end = np.full((len(array), 1), 2, dtype=np.int64)
    sequences = np.concatenate((start, cell_tokens, end), axis=1)
    train_idx, validation_idx = _split_indices(len(array), cfg.validation_fraction, cfg.seed)

    model_config = model_module.LMTADConfig(
        block_size=int(sequences.shape[1]),
        vocab_size=int(3 + cfg.grid_side * cfg.grid_side),
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=False,
        pad_token=0,
        log_file="",
        logging=False,
        integer_poe=False,
    )
    model = model_module.LMTAD(model_config).to(device)
    optimizer = model.configure_optimizers(
        cfg.weight_decay,
        cfg.learning_rate,
        (cfg.beta1, cfg.beta2),
        device.type,
    )

    train_tensor = torch.as_tensor(sequences[train_idx], dtype=torch.long)
    validation_tensor = torch.as_tensor(sequences[validation_idx], dtype=torch.long)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=min(cfg.batch_size, len(train_tensor)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        TensorDataset(validation_tensor),
        batch_size=min(cfg.batch_size, len(validation_tensor)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def amp_context():
        if not use_amp:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    global_step = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    data_preparation_seconds = time.perf_counter() - overall_started
    started = time.perf_counter()
    training_seconds = 0.0
    validation_seconds = 0.0

    for epoch in range(cfg.epochs):
        training_phase_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_items = 0
        for (batch,) in loader:
            batch = batch.to(device)
            current_lr = _learning_rate(global_step, cfg)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            with amp_context():
                _, loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(batch)
            total_items += len(batch)
            global_step += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - training_phase_started
        validation_phase_started = time.perf_counter()
        model.eval()
        validation_total = 0.0
        validation_items = 0
        with torch.no_grad():
            for (validation_batch,) in validation_loader:
                validation_batch = validation_batch.to(device, non_blocking=True)
                with amp_context():
                    _, validation_loss = model(
                        validation_batch[:, :-1].contiguous(),
                        validation_batch[:, 1:].contiguous(),
                    )
                validation_total += float(validation_loss.item()) * len(validation_batch)
                validation_items += len(validation_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_seconds += time.perf_counter() - validation_phase_started
        train_loss = total_loss / max(1, total_items)
        val_loss = validation_total / max(1, validation_items)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if cfg.patience is not None and stale_epochs >= cfg.patience:
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_validation_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("LM-TAD training did not produce a finite validation checkpoint")
    restore_started = time.perf_counter()
    model.load_state_dict(best_state)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    checkpoint_restore_seconds = time.perf_counter() - restore_started

    score_loader = DataLoader(
        TensorDataset(torch.as_tensor(sequences, dtype=torch.long)),
        batch_size=min(cfg.score_batch_size, len(sequences)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    score_batches: list[np.ndarray] = []
    scoring_started = time.perf_counter()
    with torch.no_grad():
        for (score_batch,) in score_loader:
            score_batch = score_batch.to(device, non_blocking=True)
            duplicated = len(score_batch) == 1
            if duplicated:
                score_batch = torch.cat((score_batch, score_batch), dim=0)
            mask = torch.ones_like(score_batch, dtype=torch.long, device=device)
            with amp_context():
                batch_scores, _ = eval_module.get_perplexity_fast(score_batch, model, mask)
            batch_array = np.asarray(batch_scores.float().cpu(), dtype=np.float64).reshape(-1)
            score_batches.append(batch_array[:1] if duplicated else batch_array)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    score_array = np.concatenate(score_batches)
    scoring_seconds = time.perf_counter() - scoring_started
    runtime = time.perf_counter() - started

    if score_array.shape != (len(array),):
        raise RuntimeError(f"LM-TAD returned {score_array.shape}, expected {(len(array),)}")
    if not np.isfinite(score_array).all():
        raise RuntimeError("LM-TAD returned non-finite anomaly scores")

    metadata = {
        "method": "LM-TAD",
        "official_repository": repo,
        "official_model_file": str(Path(repo) / "code" / "models" / "LMTAD.py"),
        "official_scoring_file": str(Path(repo) / "code" / "eval_lm.py"),
        "labels_consumed_during_fit": False,
        "score_direction": "higher_is_more_anomalous",
        "score_definition": "official mean negative log next-token probability (log perplexity)",
        "n_samples": int(len(array)),
        "sequence_length": int(array.shape[1]),
        "train_size": int(len(train_idx)),
        "validation_size": int(len(validation_idx)),
        "epochs_completed": int(len(history)),
        "best_validation_loss": best_loss,
        "runtime_seconds": runtime,
        "data_preparation_seconds": data_preparation_seconds,
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "training_validation_seconds": training_validation_seconds,
        "checkpoint_restore_seconds": checkpoint_restore_seconds,
        "scoring_seconds": scoring_seconds,
        "total_fit_score_seconds": time.perf_counter() - overall_started,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(device),
        "mixed_precision": use_amp,
        "adapter_config": serializable_config(cfg),
        "tokenization": tokenization,
        "training_history": history,
    }
    return score_array, metadata
