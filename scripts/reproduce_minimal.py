"""One-command offline correctness checks and a small comparison of existing estimators."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import process_time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from verify_phase2_design import verify

from consensuscensor.censoring.rules import censor
from consensuscensor.estimands.distribution import errors
from consensuscensor.estimators.assisted import cross_fitted, prompt_folds
from consensuscensor.estimators.baselines import baseline
from consensuscensor.estimators.likelihood import fit
from consensuscensor.estimators.residual import (
    contributions,
    post_stratified_m2,
    residual_estimate,
    srs,
)
from consensuscensor.simulation.phase2 import generate_phase2

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/minimal")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output must be empty; use --output with a new directory")
    out.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, text=True, capture_output=True
    )
    (out / "tests.txt").write_text(result.stdout + result.stderr)
    result.check_returncode()
    pd.DataFrame(verify()).to_csv(out / "design_enumeration.csv", index=False)
    rows = []
    protocol = {
        "N": 240,
        "population_seed": 20261022,
        "censor_seed": 314159,
        "audit_seeds": [11, 23, 37],
        "audit_rate": 0.2,
        "methods": ["B1", "B3", "B4", "B4-M2", "B5a", "B5b"],
        "scope": "offline small reproducibility check, not new research evidence",
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for scenario in ["S1", "S2", "S3"]:
        oracle, _ = generate_phase2(240, scenario, 20261022)
        folds = prompt_folds([str(j) for j in range(240)])
        for mechanism in ["M0", "M1", "M2"]:
            observed, _ = censor(oracle, mechanism, 314159)
            for seed in protocol["audit_seeds"]:
                design = srs(oracle, 48, seed)
                for method in protocol["methods"]:
                    status = "ok"
                    cpu = process_time()
                    if method == "B1":
                        groups, overall = baseline(observed, design.audit, "B1")
                    elif method == "B3":
                        fitted = fit(observed, design.audit)
                        groups, overall = fitted.probabilities, fitted.probabilities.mean(axis=0)
                        if not fitted.success:
                            status = "numeric_failure"
                    else:
                        if method == "B4":
                            estimate = residual_estimate(observed, design)
                        elif method == "B4-M2":
                            estimate = post_stratified_m2(observed, design)
                        else:
                            estimate = cross_fitted(observed, design, folds, method).estimate
                        groups, overall, status = estimate.groups, estimate.overall, estimate.status
                    elapsed = process_time() - cpu
                    reference = contributions(oracle.labels).mean(axis=0)
                    values = np.vstack([groups, overall])
                    valid = bool(
                        np.isfinite(values).all()
                        and (values >= -1e-10).all()
                        and (values <= 1 + 1e-10).all()
                        and np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-10)
                    )
                    row = {
                        "scenario": scenario,
                        "mechanism": mechanism,
                        "audit_seed": seed,
                        "method": method,
                        "status": status,
                        "raw_valid_all_vectors": valid,
                        "cpu_seconds": elapsed,
                        "audit_hash": hashlib.sha256(design.audit.indices.tobytes()).hexdigest(),
                        **errors(groups, reference, overall),
                    }
                    row["overall_half_l1"] = row.pop("tv")
                    for name, p in zip(
                        ["normal", "expert", "overall"], [*groups, overall], strict=True
                    ):
                        row.update({f"{name}_{c}": float(p[j]) for j, c in enumerate("ATB")})
                    rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "results.csv", index=False)
    summary = frame.groupby(["scenario", "mechanism", "method"]).agg(
        runs=("method", "size"),
        success=("status", lambda x: int((x == "ok").sum())),
        finite_raw_half_l1_mean=("overall_half_l1", "mean"),
        cpu_seconds_mean=("cpu_seconds", "mean"),
    )
    summary.to_csv(out / "summary.csv")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (out / "summary.md").write_text(
        "# Minimal CPU reproduction\n\n" + result.stdout + "\n"
        "No real data were downloaded. These are small smoke comparisons, not replacements "
        "for the frozen Phase 1/2 experiments. Loss is half L1 for raw vectors; it is TV only "
        "when the relevant vector is a probability distribution. Failed and unavailable "
        "runs remain explicit in results.csv.\n\n" + summary.to_string() + "\n"
    )
    print(result.stdout.strip())
    print(f"Completed {len(frame)} small method runs. Outputs: {out}")


if __name__ == "__main__":
    main()
