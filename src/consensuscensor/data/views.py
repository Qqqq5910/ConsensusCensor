"""Separate oracle and permitted observations; no oracle reference in Observed."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def readonly(x: NDArray) -> NDArray:
    """Copy so that changing a view cannot change its source."""
    result = x.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Oracle:
    labels: IntArray

    def __post_init__(self) -> None:
        if self.labels.ndim != 2 or self.labels.shape[1] != 4:
            raise ValueError("Expected n by 4 labels")
        if not np.isin(self.labels, [0, 1, 2]).all():
            raise ValueError("Invalid class")
        object.__setattr__(self, "labels", readonly(self.labels))

    def release_audit(self, indices: IntArray) -> "Audit":
        return Audit(readonly(indices), readonly(self.labels[indices]))


@dataclass(frozen=True)
class Audit:
    indices: IntArray
    labels: IntArray


@dataclass(frozen=True)
class Observed:
    labels: IntArray  # -1 is hidden; slots remain normal,normal,expert,expert
    retained: BoolArray
    mechanism: str

    def __post_init__(self) -> None:
        if self.labels.shape != (len(self.retained), 4):
            raise ValueError("Invalid observed shape")
        if not np.isin(self.labels, [-1, 0, 1, 2]).all():
            raise ValueError("Invalid observed class")
        object.__setattr__(self, "labels", readonly(self.labels))
        object.__setattr__(self, "retained", readonly(self.retained))
