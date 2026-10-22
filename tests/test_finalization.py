"""Regression checks for failure-aware final summaries; no estimator changes."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


def audit_module() -> ModuleType:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    # Only load the reporting utility; no archived experiments are rerun.
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("final_audit_test", scripts / "final_audit.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def fixture_frame() -> pd.DataFrame:
    rows = []
    for seed, status, loss in [(11, "ok", 0.2), (23, "numeric_failure", np.nan)]:
        rows.append(
            {
                "experiment": "fixed_audit",
                "dataset": "S2",
                "split": "full",
                "mechanism": "M2",
                "audit_rate": 0.2,
                "method": "B5b",
                "status": status,
                "population_seed": 1,
                "audit_seed": seed,
                "audit_hash": str(seed),
                "tv": loss,
                "overall_A": 0.5,
                "overall_T": 0.25,
                "overall_B": 0.25,
            }
        )
    return pd.DataFrame(rows)


def test_failure_is_not_silently_dropped_from_denominator() -> None:
    module = audit_module()
    result = module.failure_sensitivity(fixture_frame()).iloc[0]
    assert result.success_n == 1 and result.numeric_failure_n == 1
    assert result.failure_rate_eligible == 0.5
    assert result.success_conditional_half_l1 == 0.2
    assert result.failure_loss_0_lower == 0.1
    assert result.failure_loss_1_upper == 0.6


def test_invalid_raw_vector_not_treated_as_tv_probability() -> None:
    module = audit_module()
    frame = fixture_frame()
    frame.loc[0, "overall_A"] = 0.7
    assert not module.probability_validity(frame, "overall")[0]
    row = module.failure_sensitivity(frame).iloc[0]
    assert np.isnan(row.failure_loss_1_upper)


def test_pairing_uses_identical_successful_randomizations() -> None:
    module = audit_module()
    frame = fixture_frame()
    other = frame.copy()
    other["method"] = "B3"
    other["status"] = "ok"
    other["tv"] = [0.1, 0.9]
    result = module.paired_complete_cases(pd.concat([frame, other])).iloc[0]
    assert result.paired_n == 1
    assert result.rhs_same_pairs_mean == 0.1
    assert result.paired_mean_difference == 0.1
