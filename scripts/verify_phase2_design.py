"""Standalone exhaustive finite-population design expectation verification."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from consensuscensor.censoring.rules import censor
from consensuscensor.data.views import Oracle
from consensuscensor.estimators.assisted import cross_fitted
from consensuscensor.estimators.residual import AuditDesign, contributions, residual_estimate

ROOT = Path(__file__).resolve().parents[1]


def verify() -> list[dict]:
    oracle = Oracle(
        np.array(
            [[0, 0, 0, 2], [1, 1, 1, 1], [0, 2, 1, 2], [2, 2, 0, 0], [2, 1, 2, 2]], dtype=np.int64
        )
    )
    truth = contributions(oracle.labels).mean(axis=0)
    records = []
    for mechanism in ["NONE", "M0", "M1", "M2"]:
        observed, _ = censor(oracle, mechanism, 23)
        for k in [1, 2, 5]:
            values = []
            for ids in combinations(range(5), k):
                design = AuditDesign(oracle.release_audit(np.array(ids)), np.full(5, k / 5))
                values.append(residual_estimate(observed, design).groups)
            error = float(np.max(abs(np.mean(values, axis=0) - truth)))
            assert error < 1e-12
            records.append(
                {
                    "method": "B4",
                    "mechanism": mechanism,
                    "N": 5,
                    "audit_n": k,
                    "subsets_enumerated": len(values),
                    "max_design_expectation_error": error,
                    "conditioning": "fixed finite population and fixed observed view",
                }
            )
    # Fixed size one, unequal selection probabilities; expectation must use design weights.
    observed, _ = censor(oracle, "M1", 23)
    pi = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    expectation = np.zeros((2, 3))
    for j in range(5):
        design = AuditDesign(oracle.release_audit(np.array([j])), pi, "unequal-size-one")
        expectation += pi[j] * residual_estimate(observed, design).groups
    error = float(np.max(abs(expectation - truth)))
    assert error < 1e-12
    records.append(
        {
            "method": "B4",
            "mechanism": "M1",
            "N": 5,
            "audit_n": 1,
            "subsets_enumerated": 5,
            "max_design_expectation_error": error,
            "conditioning": "known unequal size-one probabilities",
        }
    )
    # CF predictor is random, but target selection independent of training conditional counts.
    small = Oracle(oracle.labels[:4])
    obs, _ = censor(small, "M1", 23)
    values = []
    for left in [0, 1]:
        for right in [2, 3]:
            design = AuditDesign(small.release_audit(np.array([left, right])), np.full(4, 0.5))
            values.append(cross_fitted(obs, design, np.array([0, 0, 1, 1]), "B5a").estimate.groups)
    error = float(np.max(abs(np.mean(values, axis=0) - contributions(small.labels).mean(axis=0))))
    assert error < 1e-12
    records.append(
        {
            "method": "B5a",
            "mechanism": "M1",
            "N": 4,
            "audit_n": 2,
            "subsets_enumerated": 4,
            "max_design_expectation_error": error,
            "conditioning": "fixed fold audit counts (1,1); outside-fold training only",
        }
    )
    return records


def main() -> None:
    out = ROOT / "experiments/phase2"
    out.mkdir(exist_ok=True)
    records = verify()
    pd.DataFrame(records).to_csv(out / "design_enumeration.csv", index=False)
    (out / "design_enumeration.json").write_text(json.dumps(records, indent=2))
    print(
        f"{len(records)} exhaustive design checks passed; "
        f"max error={max(r['max_design_expectation_error'] for r in records):.3g}"
    )


if __name__ == "__main__":
    main()
