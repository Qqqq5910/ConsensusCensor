"""Specified independent and correlated scenarios; no latent covariate release."""

import numpy as np

from consensuscensor.data.views import FloatArray, Oracle

P = np.array([[0.60, 0.25, 0.15], [0.40, 0.15, 0.45]])
DELTA = np.array([[0.30, -0.15, -0.15], [0.30, -0.10, -0.20]])


def generate(n: int, scenario: str, seed: int) -> tuple[Oracle, FloatArray]:
    if scenario not in ("S1", "S2"):
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    latent = rng.choice([-1, 1], size=n) if scenario == "S2" else np.zeros(n)
    labels = np.empty((n, 4), dtype=np.int64)
    for j in range(4):
        p = P[j // 2] + latent[:, None] * DELTA[j // 2]
        labels[:, j] = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
    return Oracle(labels), P.copy()
