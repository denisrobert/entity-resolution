"""Additional evaluation metrics for the entity-resolution papers.

These complement F1 by measuring dimensions F1 ignores: probabilistic calibration
(Brier score), threshold-free discrimination (average precision), and asymmetric
false-positive/false-negative cost (expected cost and its optimal threshold).

All functions take a binary label array ``y`` (1 = match) and a score/probability
array ``p`` (posterior match probability). They are cheap to compute from the per-pair
probability frames already produced by the experiment scripts.
"""

from __future__ import annotations

import numpy as np


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the predicted match probability vs. the true label."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Mean cross-entropy of the predicted probabilities."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(np.asarray(y) * np.log(p) + (1 - np.asarray(y)) * np.log(1 - p)))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / (tp + fp)
    prec[np.isnan(prec)] = 0.0
    return float(np.sum(prec * y) / max(np.sum(y), 1))


def confusion_at(y: np.ndarray, p: np.ndarray, tau: float):
    pred = np.asarray(p) >= tau
    yt = np.asarray(y) == 1
    tp = int(((pred) & (yt)).sum())
    fp = int(((pred) & (~yt)).sum())
    fn = int(((~pred) & (yt)).sum())
    tn = int(((~pred) & (~yt)).sum())
    return tp, fp, fn, tn


def f1_at(y: np.ndarray, p: np.ndarray, tau: float) -> float:
    tp, fp, fn, _ = confusion_at(y, p, tau)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def f_beta_at(y: np.ndarray, p: np.ndarray, tau: float, beta: float) -> float:
    tp, fp, fn, _ = confusion_at(y, p, tau)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    den = (b2 * prec + rec)
    return (1 + b2) * prec * rec / den if den else 0.0


def expected_cost(
    y: np.ndarray,
    p: np.ndarray,
    tau: float,
    c_fp: float,
    c_fn: float,
) -> float:
    """Expected cost C_FP*FP + C_FN*FN (per-pair average)."""
    tp, fp, fn, _ = confusion_at(y, p, tau)
    return (c_fp * fp + c_fn * fn) / max(len(y), 1)


def cost_optimal_tau(
    y: np.ndarray,
    p: np.ndarray,
    c_fp: float,
    c_fn: float,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Threshold minimising expected cost; returns (tau*, min expected cost)."""
    if grid is None:
        grid = np.linspace(0.01, 0.999, 400)
    best_tau, best_cost = 0.5, float("inf")
    for tau in grid:
        c = expected_cost(y, p, tau, c_fp, c_fn)
        if c < best_cost:
            best_cost, best_tau = c, tau
    return float(best_tau), float(best_cost)


def summarize(
    y: np.ndarray,
    p: np.ndarray,
    tau: float = 0.85,
    c_fp: float = 1.0,
    c_fn: float = 1.0,
    beta: float = 1.0,
) -> dict:
    """Compact summary of all additional metrics at a decision threshold."""
    tp, fp, fn, tn = confusion_at(y, p, tau)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    tau_star, cost_star = cost_optimal_tau(y, p, c_fp, c_fn)
    return {
        "n_pairs": int(len(y)),
        "threshold": float(tau),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_at(y, p, tau)),
        "f_beta": float(f_beta_at(y, p, tau, beta)),
        "brier": float(brier_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "average_precision": float(average_precision(y, p)),
        "expected_cost": float(expected_cost(y, p, tau, c_fp, c_fn)),
        "cost_optimal_tau": tau_star,
        "cost_optimal_cost": float(cost_star),
    }