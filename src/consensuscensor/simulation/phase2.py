"""Specified S3 heterogeneity; Phase 1 S1/S2 generator remains untouched."""

import numpy as np

from consensuscensor.data.views import FloatArray, Oracle
from consensuscensor.simulation.generate import generate

S3_MIXTURE = np.array([0.55, 0.45])
S3_PROBABILITIES = np.array(
    [
        [[0.90, 0.07, 0.03], [0.85, 0.05, 0.10]],
        [[0.20, 0.35, 0.45], [0.35, 0.40, 0.25]],
    ]
)


def generate_phase2(n: int, scenario: str, seed: int) -> tuple[Oracle, FloatArray]:
    if scenario in ("S1", "S2"):
        return generate(n, scenario, seed)
    if scenario != "S3":
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    difficulty = rng.choice(2, size=n, p=S3_MIXTURE)
    labels = np.empty((n, 4), dtype=np.int64)
    for j in range(4):
        p = S3_PROBABILITIES[difficulty, j // 2]
        labels[:, j] = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
    truth = np.einsum("h,hgc->gc", S3_MIXTURE, S3_PROBABILITIES)
    return Oracle(labels), truth
