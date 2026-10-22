"""Finite-slot targets and distributional errors."""

import numpy as np

from consensuscensor.data.views import FloatArray, IntArray

SCORES = np.array([1.0, 0.5, 0.0])


def distribution(labels: IntArray) -> FloatArray:
    rows = []
    for group in (labels[:, :2], labels[:, 2:]):
        valid = group[group >= 0]
        rows.append(
            np.bincount(valid, minlength=3) / len(valid) if len(valid) else np.full(3, np.nan)
        )
    return np.array(rows)


def tv(p: FloatArray, q: FloatArray) -> float:
    return float(np.abs(p - q).sum() / 2)


def errors(
    p: FloatArray, reference: FloatArray, overall: FloatArray | None = None
) -> dict[str, float]:
    total = p.mean(axis=0) if overall is None else overall
    target = reference.mean(axis=0)
    return {
        "tv": tv(total, target),
        "score_error": float(abs((total - target) @ SCORES)),
        "normal_tv": tv(p[0], reference[0]),
        "expert_tv": tv(p[1], reference[1]),
        "group_gap_error": float(abs((p[1] - p[0] - reference[1] + reference[0]) @ SCORES)),
        "group_tv_gap_error": abs(tv(p[0], p[1]) - tv(reference[0], reference[1])),
    }
