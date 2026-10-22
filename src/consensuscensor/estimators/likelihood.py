"""Exact 81-state observed likelihood; classic selection-aware MLE."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from consensuscensor.censoring.rules import outcomes
from consensuscensor.data.views import Audit, FloatArray, Observed

STATES = np.array(list(product(range(3), repeat=4)), dtype=np.int64)
COUNTS = np.stack(
    [(STATES[:, :2] == c).sum(axis=1) for c in range(3)]
    + [(STATES[:, 2:] == c).sum(axis=1) for c in range(3)],
    axis=1,
)


def probabilities(z: FloatArray) -> FloatArray:
    return softmax(np.column_stack([z.reshape(2, 2), np.zeros(2)]), axis=1)


def masses(z: FloatArray) -> tuple[FloatArray, FloatArray]:
    p = probabilities(z)
    logw = COUNTS @ np.log(p.reshape(-1))
    w = np.exp(logw)
    derivative = np.column_stack(
        [COUNTS[:, g * 3 + c] - 2 * p[g, c] for g in range(2) for c in range(2)]
    )
    return w, w[:, None] * derivative


@lru_cache(maxsize=None)
def kernel(mechanism: str) -> tuple[tuple[tuple[int, ...], ...], FloatArray]:
    events = sorted({event for y in STATES for event, _ in outcomes(tuple(y), mechanism)})
    lookup = {e: k for k, e in enumerate(events)}
    matrix = np.zeros((len(events), 81))
    for j, y in enumerate(STATES):
        for event, probability in outcomes(tuple(y), mechanism):
            matrix[lookup[event], j] += probability
    return tuple(events), matrix


def design(observed: Observed, audit: Audit) -> tuple[FloatArray, FloatArray]:
    if len(np.unique(audit.indices)) != len(audit.indices):
        raise ValueError("Duplicate audit index")
    if np.any(audit.indices < 0) or np.any(audit.indices >= len(observed.labels)):
        raise ValueError("Audit index out of range")
    if audit.labels.shape != (len(audit.indices), 4):
        raise ValueError("Invalid audit shape")
    if not np.isin(audit.labels, [0, 1, 2]).all():
        raise ValueError("Invalid audit label")
    exposed = observed.labels[audit.indices]
    if np.any((exposed >= 0) & (exposed != audit.labels)):
        raise ValueError("Audit contradicts observed")
    events, matrix = kernel(observed.mechanism)
    lookup = {e: i for i, e in enumerate(events)}
    mask = np.ones(len(observed.labels), dtype=bool)
    mask[audit.indices] = False
    counts = np.zeros(len(events))
    for row in observed.labels[mask]:
        counts[lookup[tuple(row)]] += 1
    full_events, identity = kernel("NONE")
    full_lookup = {e: i for i, e in enumerate(full_events)}
    full_counts = np.zeros(81)
    for row in audit.labels:
        full_counts[full_lookup[tuple(row)]] += 1
    positive = counts > 0
    full_positive = full_counts > 0
    return (
        np.vstack([matrix[positive], identity[full_positive]]),
        np.concatenate([counts[positive], full_counts[full_positive]]),
    )


def objective(z: FloatArray, matrix: FloatArray, counts: FloatArray) -> tuple[float, FloatArray]:
    w, dw = masses(z)
    event_p = matrix @ w
    if np.any(event_p <= 0):
        return float("inf"), np.zeros(4)
    return -float(counts @ np.log(event_p)), -(counts / event_p) @ (matrix @ dw)


@dataclass
class Fit:
    probabilities: FloatArray
    success: bool
    seconds: float
    nll: float
    message: str
    local_rank: int
    minimum_singular: float
    multistart_spread: float
    boundary: bool
    failed_starts: int


def fit(observed: Observed, audit: Audit) -> Fit:
    start = perf_counter()
    matrix, counts = design(observed, audit)
    if counts.sum() == 0:
        return Fit(
            np.full((2, 3), np.nan),
            False,
            perf_counter() - start,
            float("nan"),
            "empty data",
            0,
            0.0,
            float("nan"),
            False,
            0,
        )
    starts = [np.zeros(4), np.array([1.0, -1.0, -0.5, 0.5]), np.array([-1.0, 0.5, 1.0, -0.5])]
    results = [
        minimize(
            objective,
            z,
            args=(matrix, counts),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-20, 20)] * 4,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7},
        )
        for z in starts
    ]
    finite = [r for r in results if np.isfinite(r.fun)]
    if not finite:
        return Fit(
            np.full((2, 3), np.nan),
            False,
            perf_counter() - start,
            float("nan"),
            "all starts failed",
            0,
            0.0,
            float("nan"),
            False,
            len(results),
        )
    best = min(finite, key=lambda r: r.fun)
    _, derivative = masses(best.x)
    # Local structural rank of ALL observable outcomes, not just occupied outcomes.
    _, all_matrix = kernel(observed.mechanism)
    jacobian = all_matrix @ derivative
    if len(audit.indices):
        jacobian = np.vstack([jacobian, derivative])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    near = [probabilities(r.x) for r in finite if abs(r.fun - best.fun) < 1e-5]
    spread = max(float(np.max(np.abs(p - probabilities(best.x)))) for p in near)
    return Fit(
        probabilities(best.x),
        bool(best.success),
        perf_counter() - start,
        float(best.fun),
        str(best.message),
        int(np.linalg.matrix_rank(jacobian, tol=1e-8)),
        float(singular[-1]),
        spread,
        bool(np.max(abs(best.x)) > 19.9),
        sum(not r.success for r in results),
    )
