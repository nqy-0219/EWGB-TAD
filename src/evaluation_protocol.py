"""Evaluation helpers for the EWGB-TAD revision benchmark."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validated_labels_and_scores(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("labels and scores must be non-empty and equally sized")
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise ValueError("labels and scores must contain only finite values")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary values in {0, 1}")
    return labels.astype(int, copy=False), scores


def oracle_top_k_predictions(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Select exactly ``sum(labels)`` highest scores with stable tie handling.

    This is an evaluation-only oracle threshold. Labels determine only the
    number of predicted anomalies after all model fitting and scoring have
    finished. Stable mergesort preserves input order when scores are tied.
    """
    labels, scores = _validated_labels_and_scores(labels, scores)
    n_positive = int(np.sum(labels))
    predictions = np.zeros(len(labels), dtype=int)
    if n_positive == 0:
        return predictions
    order = np.argsort(-scores, kind="mergesort")
    predictions[order[:n_positive]] = 1
    return predictions


def evaluate_oracle_top_k(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Evaluate ranking metrics and oracle-top-k classification metrics."""
    labels, scores = _validated_labels_and_scores(labels, scores)
    if len(np.unique(labels)) != 2:
        raise ValueError("AUC and AUPRC require both normal and anomaly labels")
    predictions = oracle_top_k_predictions(labels, scores)
    return {
        "AUC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
    }
