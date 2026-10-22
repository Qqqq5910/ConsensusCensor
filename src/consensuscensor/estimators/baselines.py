"""B0--B2, including explicit unavailable results at zero audit."""

import numpy as np

from consensuscensor.data.views import Audit, FloatArray, Observed
from consensuscensor.estimands.distribution import distribution


def baseline(observed: Observed, audit: Audit, method: str) -> tuple[FloatArray, FloatArray]:
    if method == "B0":
        labels = observed.labels
    elif method in ("B1", "B2"):
        labels = audit.labels
    else:
        raise ValueError(method)
    groups = distribution(labels)
    if method == "B2":
        total = groups.mean(axis=0)
    else:
        valid = labels[labels >= 0]
        total = np.bincount(valid, minlength=3) / len(valid) if len(valid) else np.full(3, np.nan)
    return groups, total
