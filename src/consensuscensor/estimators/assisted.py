"""Cross-fitted generalized-difference estimates with conditional SRS weights."""

from dataclasses import dataclass

import numpy as np

from consensuscensor.data.views import Audit, FloatArray, IntArray, Observed
from consensuscensor.estimators.likelihood import STATES, fit, kernel
from consensuscensor.estimators.residual import (
    AuditDesign,
    ResidualEstimate,
    contributions,
    residual_estimate,
    unavailable,
    validate,
)


@dataclass
class CrossFitOutput:
    estimate: ResidualEstimate
    prediction: FloatArray
    conditional_pi: FloatArray
    trace: list[dict]


def prompt_folds(prompt_ids: list[str], seed: int = 20261022) -> IntArray:
    unique = np.array(sorted(set(prompt_ids)))
    if len(unique) < 2:
        raise ValueError("Cross-fitting requires at least two distinct prompts")
    np.random.default_rng(seed).shuffle(unique)
    membership = {p: j % 2 for j, p in enumerate(unique)}
    return np.array([membership[p] for p in prompt_ids], dtype=np.int64)


def simple_missing_prediction(
    train: Observed, audit: Audit, target: Observed, alpha: float = 0.5
) -> FloatArray:
    if alpha <= 0:
        raise ValueError("Smoothing must be positive")
    counts = np.full((2, 3), alpha)
    hidden = train.labels[audit.indices] < 0
    for g in range(2):
        for c in range(3):
            counts[g, c] += (
                (audit.labels[:, 2 * g : 2 * g + 2] == c) & hidden[:, 2 * g : 2 * g + 2]
            ).sum()
    p = counts / counts.sum(axis=1, keepdims=True)
    missing_mass = (target.labels.reshape(-1, 2, 2) < 0).sum(axis=2) / 2
    return missing_mass[:, :, None] * p[None, :, :]


def posterior_missing_prediction(target: Observed, p: FloatArray) -> FloatArray:
    if p.shape != (2, 3) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("B3 posterior requires strictly positive group probabilities")
    if not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Invalid B3 probabilities")
    joint = np.prod(np.stack([p[j // 2, STATES[:, j]] for j in range(4)], axis=1), axis=1)
    events, matrix = kernel(target.mechanism)
    predictions: dict[tuple[int, ...], FloatArray] = {}
    for j, event in enumerate(events):
        weights = matrix[j] * joint
        if weights.sum() <= 0:
            raise ValueError("Impossible conditioning event")
        weights /= weights.sum()
        latent_missing = STATES.copy()
        latent_missing[:, np.array(event) >= 0] = -1
        predictions[event] = np.einsum("n,ngc->gc", weights, contributions(latent_missing))
    return np.array([predictions[tuple(row)] for row in target.labels])


def cross_fitted(
    observed: Observed, design: AuditDesign, folds: IntArray, method: str, alpha: float = 0.5
) -> CrossFitOutput:
    validate(observed, design)
    n = len(observed.labels)
    if method not in ("B5a", "B5b"):
        raise ValueError(method)
    if folds.shape != (n,) or len(np.unique(folds)) < 2:
        raise ValueError("At least two non-overlapping target folds required")
    if not np.allclose(design.pi, len(design.audit.indices) / n):
        raise ValueError("Cross-fitting implemented only for global fixed-size SRS")
    predictions = np.zeros((n, 2, 3))
    pi = np.zeros(n)
    trace: list[dict] = []
    if not (observed.labels < 0).any():
        result = ResidualEstimate(contributions(observed.labels).mean(axis=0))
        return CrossFitOutput(result, predictions, pi, trace)
    if not len(design.audit.indices):
        return CrossFitOutput(unavailable("Zero audit with unknown labels"), predictions, pi, trace)
    training_failure = False
    for f in np.unique(folds):
        target_idx = np.flatnonzero(folds == f)
        train_idx = np.flatnonzero(folds != f)
        target = Observed(
            observed.labels[target_idx], observed.retained[target_idx], observed.mechanism
        )
        train = Observed(
            observed.labels[train_idx], observed.retained[train_idx], observed.mechanism
        )
        train_pick = folds[design.audit.indices] != f
        train_global = design.audit.indices[train_pick]
        train_audit = Audit(
            np.searchsorted(train_idx, train_global), design.audit.labels[train_pick].copy()
        )
        k_fold = int((~train_pick).sum())
        pi[target_idx] = k_fold / len(target_idx)
        record = {
            "fold": int(f),
            "target_n": len(target_idx),
            "audit_n": k_fold,
            "train_indices": train_idx.tolist(),
            "train_audit_indices": train_global.tolist(),
            "target_indices": target_idx.tolist(),
            "training_success": True,
        }
        if method == "B5a":
            predictions[target_idx] = simple_missing_prediction(train, train_audit, target, alpha)
        else:
            result = fit(train, train_audit)
            record.update(
                {
                    "training_success": result.success,
                    "fit_message": result.message,
                    "boundary": result.boundary,
                    "failed_starts": result.failed_starts,
                }
            )
            if not result.success or result.boundary or not np.isfinite(result.probabilities).all():
                training_failure = True
            else:
                predictions[target_idx] = posterior_missing_prediction(target, result.probabilities)
        trace.append(record)
    if training_failure:
        result = unavailable("B3 training fold failed or reached logit boundary", "numeric_failure")
    elif np.any(((observed.labels < 0).any(axis=1)) & (pi <= 0)):
        result = unavailable("Unknown target fold has zero audited units")
    else:
        result = residual_estimate(
            observed,
            AuditDesign(design.audit, pi, "SRS conditional on fold audit counts"),
            predictions,
        )
    result.diagnostics.update(
        {
            "fold_counts": [t["audit_n"] for t in trace],
            "fit_failures": sum(not t["training_success"] for t in trace),
        }
    )
    return CrossFitOutput(result, predictions, pi, trace)
