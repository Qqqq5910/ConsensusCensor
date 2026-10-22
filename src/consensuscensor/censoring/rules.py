"""Known stochastic selection kernels, version 1."""

from functools import lru_cache
from itertools import combinations

import numpy as np

from consensuscensor.data.views import IntArray, Observed, Oracle

SUBSETS = tuple(combinations(range(4), 3))
VERSION = "four-slot-v1-uniform-min-cost"


def cost(y: tuple[int, ...], subset: tuple[int, ...]) -> int:
    return sum(abs(y[a] - y[b]) for a, b in combinations(subset, 2))


@lru_cache(maxsize=None)
def outcomes(y: tuple[int, ...], mechanism: str) -> tuple[tuple[tuple[int, ...], float], ...]:
    if mechanism == "NONE":
        return ((y, 1.0),)
    if mechanism == "M2":
        keep = max(y.count(c) for c in range(3)) >= 3
        return ((y if keep else (-1,) * 4, 1.0),)
    if mechanism not in ("M0", "M1"):
        raise ValueError("Unknown mechanism")
    choices = SUBSETS
    if mechanism == "M1":
        costs = [cost(y, subset) for subset in SUBSETS]
        choices = tuple(s for s, c in zip(SUBSETS, costs, strict=True) if c == min(costs))
    return tuple(
        (tuple(y[j] if j in s else -1 for j in range(4)), 1 / len(choices)) for s in choices
    )


def censor(oracle: Oracle, mechanism: str, seed: int) -> tuple[Observed, list[dict]]:
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, ...]] = []
    ledger: list[dict] = []
    for i, labels in enumerate(oracle.labels):
        possibilities = outcomes(tuple(map(int, labels)), mechanism)
        selected = int(rng.integers(len(possibilities)))
        row = possibilities[selected][0]
        rows.append(row)
        ledger.append(
            {
                "index": i,
                "kept": [j for j, v in enumerate(row) if v >= 0],
                "deleted": [j for j, v in enumerate(row) if v < 0],
                "tie": mechanism == "M1" and len(possibilities) > 1,
                "version": VERSION,
                "seed": seed,
            }
        )
    visible: IntArray = np.array(rows, dtype=np.int64).reshape(-1, 4)
    return Observed(visible, (visible >= 0).any(axis=1), mechanism), ledger
