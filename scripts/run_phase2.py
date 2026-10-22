"""Locked Phase 2 experiments. Does not modify Phase 1 outputs or raw data."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd
import scipy

from consensuscensor.censoring.rules import VERSION, censor
from consensuscensor.data.multipref import MAPPING
from consensuscensor.data.views import Oracle
from consensuscensor.estimands.distribution import SCORES, errors
from consensuscensor.estimators.assisted import cross_fitted, prompt_folds
from consensuscensor.estimators.baselines import baseline
from consensuscensor.estimators.likelihood import fit
from consensuscensor.estimators.residual import (
    contributions,
    post_stratified_m2,
    residual_estimate,
    srs,
)
from consensuscensor.simulation.generate import DELTA, P
from consensuscensor.simulation.phase2 import S3_MIXTURE, S3_PROBABILITIES, generate_phase2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase2"
AUDIT_SEEDS = list(range(11001, 11051))
REAL_AUDIT_SEEDS = list(range(31001, 31031))
POPULATION_SEEDS = list(range(21001, 21021))
RATES = [0.0, 0.05, 0.10, 0.20]
METHODS = ["B0", "B1", "B2", "B3", "B4", "B4-M2", "B5a", "B5b"]
CENSOR_SEED = 314159


def evaluate_target(
    oracle: Oracle,
    prompt_ids: list[str],
    dataset: str,
    experiment: str,
    population_seed: int,
    audit_seeds: list[int],
    split: str,
    generating_truth: np.ndarray | None = None,
) -> tuple[list[dict], list[dict]]:
    n = len(oracle.labels)
    folds = prompt_folds(prompt_ids)
    observations = {m: censor(oracle, m, CENSOR_SEED)[0] for m in ["M0", "M1", "M2"]}
    records, audits = [], []
    for seed in audit_seeds:
        for rate in RATES:
            design = srs(oracle, int(np.floor(n * rate)), seed)
            audit_hash = hashlib.sha256(design.audit.indices.tobytes()).hexdigest()
            audits.append(
                {
                    "dataset": dataset,
                    "experiment": experiment,
                    "split": split,
                    "population_seed": population_seed,
                    "audit_seed": seed,
                    "audit_rate": rate,
                    "audit_hash": audit_hash,
                    "indices": design.audit.indices.tolist(),
                    "fold_counts": np.bincount(folds[design.audit.indices], minlength=2).tolist(),
                    "N": n,
                    "k": len(design.audit.indices),
                    "unconditional_pi": len(design.audit.indices) / n,
                }
            )
            for mechanism, observed in observations.items():
                extra = int((observed.labels[design.audit.indices] < 0).sum())
                for method in METHODS:
                    status, message, diagnostics = "ok", "", {}
                    cpu, wall = process_time(), perf_counter()
                    if method in ["B0", "B1", "B2"]:
                        groups, overall = baseline(observed, design.audit, method)
                        if not np.isfinite(groups).all():
                            status, message = "not_estimable", "No audited labels"
                    elif method == "B3":
                        fitted = fit(observed, design.audit)
                        groups = fitted.probabilities
                        overall = groups.mean(axis=0)
                        if not fitted.success:
                            status, message = "numeric_failure", fitted.message
                        diagnostics = {
                            "local_rank": fitted.local_rank,
                            "min_singular": fitted.minimum_singular,
                            "boundary": fitted.boundary,
                            "failed_starts": fitted.failed_starts,
                        }
                    elif method == "B4":
                        result = residual_estimate(observed, design)
                        groups, overall = result.groups, result.overall
                        status, message = result.status, result.message
                    elif method == "B4-M2":
                        result = post_stratified_m2(observed, design)
                        groups, overall = result.groups, result.overall
                        status, message = result.status, result.message
                        diagnostics.update(result.diagnostics)
                    else:
                        output = cross_fitted(observed, design, folds, method)
                        result = output.estimate
                        groups, overall = result.groups, result.overall
                        status, message = result.status, result.message
                        diagnostics.update(result.diagnostics)
                        # Full fold assignment is persisted once; no label-dependent trace input.
                        diagnostics["conditional_pi_min"] = float(output.conditional_pi.min())
                        diagnostics["conditional_pi_max"] = float(output.conditional_pi.max())
                    elapsed_cpu, elapsed_wall = process_time() - cpu, perf_counter() - wall
                    # Oracle reference enters only HERE, after the estimate has been produced.
                    reference = contributions(oracle.labels).mean(axis=0)
                    ref_total = reference.mean(axis=0)
                    difference = overall - ref_total
                    group_difference = groups - reference
                    gap_error = float(
                        (groups[1] - groups[0] - reference[1] + reference[0]) @ SCORES
                    )
                    probabilities = np.vstack([groups, overall])
                    is_finite = bool(np.isfinite(probabilities).all())
                    raw_valid = bool(
                        is_finite
                        and (probabilities >= -1e-10).all()
                        and (probabilities <= 1 + 1e-10).all()
                        and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
                    )
                    row = {
                        "experiment": experiment,
                        "dataset": dataset,
                        "split": split,
                        "population_seed": population_seed,
                        "audit_seed": seed,
                        "mechanism": mechanism,
                        "audit_rate": rate,
                        "method": method,
                        "N": n,
                        "audit_n": len(design.audit.indices),
                        "audit_hash": audit_hash,
                        "extra_labels": extra,
                        "audit_full_slots": 4 * len(design.audit.indices),
                        "cpu_seconds": elapsed_cpu,
                        "wall_seconds": elapsed_wall,
                        "status": status,
                        "message": message,
                        "raw_valid": raw_valid,
                        "finite_raw": is_finite,
                        "raw_min": float(probabilities.min()),
                        "max_mass_error": float(abs(probabilities.sum(axis=1) - 1).max()),
                        "mse": float(np.mean(difference**2)),
                        "normal_mse": float(np.mean(group_difference[0] ** 2)),
                        "expert_mse": float(np.mean(group_difference[1] ** 2)),
                        "group_gap_signed_error": gap_error,
                        "group_gap_squared_error": gap_error**2,
                        "diagnostics": json.dumps(diagnostics),
                        **errors(groups, reference, overall),
                    }
                    for name, values, ref in zip(
                        ["normal", "expert", "overall"],
                        [*groups, overall],
                        [*reference, ref_total],
                        strict=True,
                    ):
                        for j, c in enumerate("ATB"):
                            row[f"{name}_{c}"] = float(values[j])
                            row[f"{name}_{c}_error"] = float(values[j] - ref[j])
                            row[f"{name}_{c}_reference"] = float(ref[j])
                    if generating_truth is not None:
                        row.update(
                            {
                                "truth_" + k: v
                                for k, v in errors(groups, generating_truth, overall).items()
                            }
                        )
                    records.append(row)
        if seed == audit_seeds[-1] or (audit_seeds.index(seed) + 1) % 10 == 0:
            print(
                f"{experiment} {dataset}/{split} audit {audit_seeds.index(seed) + 1}"
                f"/{len(audit_seeds)}",
                flush=True,
            )
    return records, audits


def real_targets() -> tuple[Oracle, pd.DataFrame]:
    raw = ROOT / "data/raw/multipref.json"
    if not raw.exists():
        raise FileNotFoundError("Phase 1 raw data missing: no replacement download is performed")
    audit = json.loads((ROOT / "experiments/phase1/data_audit.json").read_text())
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if digest != audit["sha256"]:
        raise ValueError("Phase 1 raw data hash mismatch")
    rows = json.loads(raw.read_text())
    if len(rows) != 2000:
        raise ValueError("Expected preserved 2000 comparison pilot")
    labels = []
    for row in rows:
        normal, expert = row["normal_worker_annotations"], row["expert_worker_annotations"]
        if len(normal) != 2 or len(expert) != 2:
            raise ValueError("Original valid slots changed")
        labels.append([MAPPING[a["overall_pref"]] for a in normal + expert])
    meta = pd.DataFrame([{k: row[k] for k in ["comparison_id", "prompt_id"]} for row in rows])
    manifest = pd.read_csv(ROOT / "experiments/phase1/split_manifest.csv")
    if not meta.comparison_id.is_unique or not manifest.comparison_id.is_unique:
        raise ValueError("Duplicate comparison ids")
    meta = meta.merge(
        manifest, on=["comparison_id", "prompt_id"], validate="one_to_one", how="left", sort=False
    )
    if len(meta) != 2000 or meta.split.isna().any():
        raise ValueError("Original prompt split cannot be reconstructed")
    if meta.groupby("prompt_id").split.nunique().max() != 1:
        raise ValueError("Prompt split leakage")
    return Oracle(np.array(labels)), meta


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if (OUT / "results.csv").exists():
        raise FileExistsError("Refusing to overwrite Phase 2 results; archive them before rerun")
    for command, file in [
        ([sys.executable, "-m", "pytest", "-q"], "pytest.txt"),
        ([sys.executable, str(ROOT / "scripts/verify_phase2_design.py")], "design_check.txt"),
    ]:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (OUT / file).write_text(result.stdout + result.stderr)
        result.check_returncode()
    protocol = {
        "synthetic_n": 800,
        "fixed_population_seed": 20261022,
        "fixed_audit_seeds": AUDIT_SEEDS,
        "population_seeds": POPULATION_SEEDS,
        "regenerated_audit_seed": 7007,
        "real_audit_seeds": REAL_AUDIT_SEEDS,
        "censor_seed": CENSOR_SEED,
        "mechanism_version": VERSION,
        "audit_rates": RATES,
        "methods": METHODS,
        "folds": 2,
        "fold_seed": 20261022,
        "smoothing_alpha": 0.5,
        "prediction_coefficient": 1.0,
        "postprocessing": "none",
        "S1": P.tolist(),
        "S2_delta": DELTA.tolist(),
        "S3_mixture": S3_MIXTURE.tolist(),
        "S3_probabilities": S3_PROBABILITIES.tolist(),
        "B5_weights": "conditional inclusion pi=k_fold/N_fold under global SRS",
        "audit_design": "fixed-size global SRS, same selected indices for all methods",
        "regeneration": "20 independent populations, common audit-index and selection seeds; "
        "not pooled with fixed-population audit randomizations",
        "test": "original split retained; test truth used for scoring after estimators only",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (OUT / "frozen_protocol.json").write_text(json.dumps(protocol, indent=2))
    (OUT / "protocol.sha256").write_text(
        hashlib.sha256((OUT / "frozen_protocol.json").read_bytes()).hexdigest()
    )
    accumulated, audit_records = [], []

    def append_run(records: list[dict], audits: list[dict]) -> None:
        accumulated.extend(records)
        audit_records.extend(audits)
        pd.DataFrame(accumulated).to_csv(OUT / "results.csv", index=False)
        (OUT / "audit_manifest.json").write_text(json.dumps(audit_records, separators=(",", ":")))

    for scenario in ["S1", "S2", "S3"]:
        oracle, truth = generate_phase2(800, scenario, 20261022)
        prompts = [f"synthetic-{j}" for j in range(800)]
        np.savez(
            OUT / f"{scenario}_fixed_reference.npz",
            labels=oracle.labels,
            truth=truth,
            folds=prompt_folds(prompts),
        )
        append_run(
            *evaluate_target(
                oracle, prompts, scenario, "fixed_audit", 20261022, AUDIT_SEEDS, "full", truth
            )
        )
    for scenario in ["S1", "S2", "S3"]:
        for seed in POPULATION_SEEDS:
            oracle, truth = generate_phase2(800, scenario, seed)
            prompts = [f"synthetic-{j}" for j in range(800)]
            append_run(
                *evaluate_target(
                    oracle, prompts, scenario, "regenerated", seed, [7007], "full", truth
                )
            )
    oracle, metadata = real_targets()
    (OUT / "original_split_counts.json").write_text(
        json.dumps(metadata.groupby("split").size().to_dict(), indent=2)
    )
    for split in ["train", "validation", "test"]:
        indices = np.flatnonzero((metadata.split == split).to_numpy())
        subset = Oracle(oracle.labels[indices])
        prompts = metadata.iloc[indices].prompt_id.tolist()
        pd.DataFrame(
            {
                "comparison_id": metadata.iloc[indices].comparison_id.to_numpy(),
                "prompt_id": prompts,
                "fold": prompt_folds(prompts),
            }
        ).to_csv(OUT / f"{split}_fold_manifest.csv", index=False)
        append_run(
            *evaluate_target(subset, prompts, "MultiPref", "real_audit", 0, REAL_AUDIT_SEEDS, split)
        )
    subprocess.run([sys.executable, str(ROOT / "scripts/report_phase2.py")], cwd=ROOT, check=True)
    print(f"Completed {len(accumulated)} paired method runs", flush=True)


if __name__ == "__main__":
    main()
