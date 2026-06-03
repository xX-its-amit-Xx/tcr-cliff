# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 tcr-cliff contributors
"""Standard classification and regression metrics with degenerate-case handling."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tcr_cliff._logging import get_logger

_log = get_logger("eval.metrics")


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(np.asarray(a, dtype=float))
    return mask


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float]:
    """Classification metrics. AUROC/AUPRC are NaN when only one class is present.

    Parameters
    ----------
    y_true:
        Binary labels ``{0, 1}``.
    y_score:
        Predicted probabilities/scores for the positive class.
    threshold:
        Decision threshold for the hard-label metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = _finite_mask(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    n_pos = int(y_true.sum())
    out: dict[str, float] = {
        "n": float(n),
        "n_pos": float(n_pos),
        "pos_rate": float(n_pos / n) if n else float("nan"),
    }
    if n == 0:
        return {
            **out,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "accuracy": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    both_classes = 0 < n_pos < n
    y_pred = (y_score >= threshold).astype(int)
    out["auroc"] = float(roc_auc_score(y_true, y_score)) if both_classes else float("nan")
    out["auprc"] = float(average_precision_score(y_true, y_score)) if both_classes else float("nan")
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Affinity-regression metrics: Pearson r, Spearman rho, RMSE, MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = _finite_mask(y_true, y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    out: dict[str, float] = {"n": float(n)}
    if n < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return {
            **out,
            "pearson": float("nan"),
            "spearman": float("nan"),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if n else float("nan"),
            "mae": float(np.mean(np.abs(y_true - y_pred))) if n else float("nan"),
        }
    out["pearson"] = float(pearsonr(y_true, y_pred)[0])
    out["spearman"] = float(spearmanr(y_true, y_pred)[0])
    out["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    out["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "auroc",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap a metric's point estimate and ``(1-alpha)`` confidence interval.

    Returns ``(point, lo, hi)``. NaN bounds if the metric is undefined on resamples.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    fn = {
        "auroc": lambda t, s: binary_metrics(t, s).get("auroc", np.nan),
        "auprc": lambda t, s: binary_metrics(t, s).get("auprc", np.nan),
    }[metric]
    point = fn(y_true, y_score)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        val = fn(y_true[idx], y_score[idx])
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi
